from gpytorch.likelihoods import likelihood
import torch
import gpytorch
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import time
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from torch.utils import data
# from autograd_minimize import minimize


def Fast_RVM(K, targets, N, config, align_thr, gamma, eps, tol, max_itr=1000, device='cuda', verbose=True, task_id=None):
    

    M = K.shape[1]
    logMarginalLog = []

    targets[targets==-1]= 0

    targets = targets.to(device)
    K = K.to(device)
    targets_pseudo_linear	= 2 * targets - 1
    # targets_pseudo_linear	= targets 
    # KK = K.T @ K  # all K_j @ K_i
    Kt = K.T @ targets_pseudo_linear
    start_k = torch.argmax(abs(Kt)) #torch.argmax(kt_k)
    active_m = torch.tensor([start_k]).to(device)
    K_m         = K[:, active_m]
    # KK_m        = KK[:, active_m]
    Kt          = K.T @ targets

    #  Heuristic initialisation based on log-odds
    LogOut	= (targets_pseudo_linear * 0.9 + 1) / 2
    # mu_m	=  K_m.pinverse() @ (torch.log(LogOut / (1 - LogOut))) #
    mu_m = torch.linalg.lstsq(K_m, (torch.log(LogOut / (1 - LogOut)))).solution
    mu_m = mu_m.to(device).to(torch.float64)
    alpha_init = 1 / (mu_m + 1e-12).pow(2)
    # print(f'alpha_init {alpha_init.item()}')
    alpha_m = torch.tensor([alpha_init], dtype=torch.float64).to(device)
    if alpha_m < 1e-3:
        alpha_m = torch.tensor([1e-3], dtype=torch.float64).to(device)
    if alpha_m > 1e3:
        alpha_m = torch.tensor([1e3], dtype=torch.float64).to(device)
    
    aligned_out		= torch.tensor([], dtype=torch.int).to(device)
    aligned_in		= torch.tensor([], dtype=torch.int).to(device)
    low_gamma       = []
    # Sigma_m = torch.inverse(A_m + beta * KK_mm)
    

    Sigma_m, mu_m, S, Q, s, q, beta, beta_KK_m, logML, Gamma, U = Statistics(K, K_m, mu_m, alpha_m, active_m, targets, N, device)

    delete_priority = config[0]=="1"
    add_priority    = config[1]=="1"
    alignment_test  = config[2]=="1"
    align_zero      = align_thr
    check_gamma = gamma
    gm = 0.1
    add_count = 0
    del_count = 0
    recomp_count = 0
    count = 0
    deltaML = torch.zeros(M, dtype=torch.float64).to(device)  
    action = torch.zeros(M)
    tic = time.time()
    print_freq = 10
    for itr in range(max_itr):

        # 'Relevance Factor' (q^2-s) values for basis functions in model
        Factor = q*q - s 
        if (Factor<0).all():
            print(f'#{itr} Factor <0 all -------------------------')
        #compute delta in marginal log likelihood for new action selection
        deltaML = deltaML * 0
        action = action * 0
        active_factor = Factor[active_m]

        # RECOMPUTE: must be a POSITIVE 'factor' and already IN the model
        idx                 = active_factor > 1e-12
        recompute           = active_m[idx]
        alpha_prim          =  s[recompute]**2 / (Factor[recompute])
        delta_alpha         = (1/(alpha_prim) - 1/(alpha_m[idx]))
        d_alpha_S           = delta_alpha * S[recompute] + 1 
        deltaML[recompute]  = ((delta_alpha * Q[recompute]**2) / (d_alpha_S) - torch.log(d_alpha_S))
     
        # DELETION: if NEGATIVE factor and IN model
        idx = ~idx #active_factor <= 1e-12
        delete = active_m[idx]
        anyToDelete = len(delete) > 0
        if anyToDelete and active_m.shape[0] > 1:
            deltaML[delete] = -(q[delete]**2 / (s[delete] + alpha_m[idx]) - torch.log(1 + s[delete] / alpha_m[idx])) 
            # deltaML[delete] = (Q[delete]**2 / (S[delete] - alpha_m[idx]) - torch.log(1 - S[delete] / alpha_m[idx]))
            action[delete]  = -1

        # ADDITION: must be a POSITIVE factor and OUT of the model
        good_factor = Factor > 1e-12
        good_factor[active_m] = False
        
        if alignment_test and len(aligned_out) > 0:
            good_factor[aligned_out] = False

        add = torch.where(good_factor)[0]
        anyToAdd = len(add) > 0
        if anyToAdd:
            Q_S             = Q[add]**2 / (S[add] +1e-12)
            deltaML[add]    = (Q_S - 1 - torch.log(Q_S))
            # alpha_tilde     = s[add]**2 / (Factor[add])
            # deltaML[add]    = (q[add]**2 / (alpha_tilde + s[add])- torch.log(alpha_tilde/(alpha_tilde + s[add])))
                            # = (Q[add]**2 / (alpha_tilde + S[add])- torch.log(alpha_tilde/(alpha_tilde + S[add])))
            action[add]     = 1
            
        # Priority of Deletion   
        if anyToDelete and delete_priority and not add_priority:
            deltaML[recompute] = 0
            deltaML[add] = 0
        # Priority of Addition       
        if anyToAdd and add_priority and (deltaML[add]>0).any():
            deltaML[recompute] = 0
            deltaML[delete] = 0

        #  choose the action that results in the greatest change in likelihood 
        max_idx = torch.argmax(deltaML)[None]
        deltaLogMarginal = deltaML[max_idx]
        selected_action		= action[max_idx]
        anyWorthwhileAction	= deltaLogMarginal > 0 

        # already in the model
        if selected_action != 1:
            j = torch.where(active_m==max_idx)[0]

        alpha_new = s[max_idx]**2 / (Factor[max_idx])
        terminate = False

        if not anyWorthwhileAction:
            if verbose and (task_id%print_freq==0):
                print(f'{itr:3}, No positive action, m={active_m.shape[0]:3}')
            selected_action = torch.tensor(10)
            terminate = True

        elif (selected_action==0) and (not anyToDelete):
            no_change_in_alpha = torch.abs(torch.log(alpha_new) - torch.log(alpha_m[j])) < tol
            if no_change_in_alpha:
                if verbose and (task_id%print_freq==0):
                    print(f'{itr:3}, No change in alpha, m={active_m.shape[0]:3}')
                selected_action = torch.tensor(11)
                terminate = True

        
        if alignment_test:
            # Addition - rule out addition (from now onwards) if the new basis
            # vector is aligned too closely to one or more already in the model
            if selected_action== 1:
            # Basic test for correlated basis vectors
            # (note, Phi and columns of PHI are normalised)
                k_new = K[:, max_idx]
                k_new_K_m         = k_new.T @ K_m 
                aligned_idx = torch.where(k_new_K_m > (1- align_zero))[0]
                num_aligned	= len(aligned_idx)
                if num_aligned > 0:
                    # The added basis function is effectively indistinguishable from one present already
                    selected_action	= torch.tensor(12)
                    # Make a note so we don't try this next time. May be more than one in the model, which we need to note was
                    # the cause of function 'nu' being rejected
                    aligned_out = torch.cat([aligned_out, max_idx * torch.ones(num_aligned, dtype=torch.int).to(device)])
                    aligned_in = torch.cat([aligned_in, active_m[aligned_idx]])
            # Deletion: reinstate any previously deferred basis functions
            # resulting from this basis function
            if selected_action== -1:
                aligned_idx = torch.where(aligned_in==max_idx)[0]
                # findAligned	= find(Aligned_in==max_idx);
                num_aligned	= len(aligned_idx)
                if num_aligned > 0:
                    reinstated					= aligned_out[aligned_idx]
                    aligned_in = aligned_in[torch.arange(aligned_in.size(0)).to(device)!=aligned_idx]
                    aligned_out = aligned_out[torch.arange(aligned_out.size(0)).to(device)!=aligned_idx]
                

        update_required = False
        if selected_action==0:   #recompute
            recomp_count += 1
            alpha_j_old     = alpha_m[j]
            alpha_m[j]      = alpha_new
            s_j             = Sigma_m[:, j]
            delta_inv       = 1 / (alpha_new - alpha_j_old)
            kappa           = 1 / (Sigma_m[j, j] + delta_inv)
            tmp             = kappa * s_j
            # Sigma_new       = Sigma_m - tmp @ s_j.T
            delta_mu        = -mu_m[j] * tmp
            mu_m            = mu_m + delta_mu.squeeze()
          
            # S	= S + kappa * (beta_KK_m @ s_j).squeeze()**2
            # Q	= Q - (beta_KK_m @ delta_mu).squeeze()
            update_required = True
        
        if selected_action==-1:  #delete
            del_count += 1
            # print(f'itr= {itr}, selected_action={selected_action.item()}')
            active_m        = active_m[active_m!=active_m[j]]
            alpha_m         = alpha_m[alpha_m!=alpha_m[j]]

            s_jj			= Sigma_m[j, j]
            s_j				= Sigma_m[:, j]
            tmp				= s_j/s_jj
            # Sigma_		    = Sigma_m - tmp @ s_j.T
            # Sigma_          = Sigma_[torch.arange(Sigma_.size(0)).to(device)!=j]
            # Sigma_new       = Sigma_[:, torch.arange(Sigma_.size(1)).to(device)!=j]
            delta_mu		= -mu_m[j] * tmp
            mu_j			= mu_m[j]
            mu_m			= mu_m + delta_mu.squeeze()
            mu_m			= mu_m[torch.arange(mu_m.size(0)).to(device)!=j]
            
            # jPm	            = (beta_KK_m @ s_j).squeeze()
            # S	            = S + jPm.pow(2) / s_jj
            # Q	            = Q + jPm * mu_j / s_jj

            K_m             = K[:, active_m]
            # KK_m            = KK[:, active_m]
            # beta_KK_m       = beta * KK_m
            update_required = True
        
        if selected_action==1:  #add
            add_count += 1
            active_m = torch.cat([active_m, max_idx])
            alpha_m = torch.cat([alpha_m, alpha_new])

            k_new           = K[:, max_idx]
            # K_k_new         = K.T @ k_new
            beta_k_new      = beta * k_new.squeeze()
            # beta_K_k_new  	= K.T @ beta_k_new

            tmp		        = ((beta_k_new.T @ K_m) @ Sigma_m).T
            s_ii		    = 1/ (alpha_new + S[max_idx] + 1e-8)
            # s_i			    = -s_ii * tmp
            # tau			    = -s_i @ tmp.T
            # Sigma_          = torch.cat([Sigma_m + tau, s_i.unsqueeze(-1)], axis=1)
            # Sigma_new       = torch.cat([Sigma_, torch.cat([s_i.T, s_ii]).unsqueeze(0)], axis=0)
            mu_i		    = (s_ii * Q[max_idx])
            delta_mu        = torch.cat([-(mu_i * tmp) , mu_i], axis=0)
            mu_m			= torch.cat([mu_m, torch.tensor([0.0]).to(device)], axis=0) + delta_mu
        
            # mCi	            = beta_K_k_new - beta_KK_m @ tmp
            # S   	        = S - mCi.pow(2) * s_ii
            # Q   	        = Q - mCi * mu_i

            K_m             = K[:, active_m]
            # KK_m            = KK[:, active_m]
            # beta_KK_m       = beta * KK_m
            update_required = True
        
            
        # UPDATE STATISTICS
        if update_required:
            count += 1
            Sigma_m, mu_m, S, Q, s, q, beta, beta_KK_m, new_logML, Gamma, U = Statistics(K, K_m, mu_m, alpha_m, active_m, targets, N, device)
            deltaLogMarginal	= new_logML - logML
            
            logML = logML + deltaLogMarginal
            logMarginalLog.append(logML.item())

        if terminate:
           
            # if verbose and (task_id%print_freq==0):
            #     if selected_action!=11:
            #         print(f'Finished at {itr:3}, m= {active_m.shape[0]:3}')
            toc = time.time()
            # print(f'elapsed time: {toc-tic}')
            # if count > 0:
            #     print(f'add: {add_count:3d} ({add_count/count:.1%}), delete: {del_count:3d} ({del_count/count:.1%}), recompute: {recomp_count:3d} ({recomp_count/count:.1%})')
            return active_m.cpu().numpy(), alpha_m, Gamma, beta, mu_m, U

        if ((itr+1)%25==0) and False:
            print(f'#{itr+1:3},     m={active_m.shape[0]}, selected_action= {selected_action.item():.0f}, logML= {logML.item()/N:.5f}')


    # print(f'logML= {logML/N}\n{logMarginalLog}')
    print(f'End, m= {active_m.shape[0]:3}')
    toc = time.time()
    # print(f'elapsed time: {toc-tic}')
    # if count > 0:
    #     print(f'add: {add_count:3d} ({add_count/count:.1%}), delete: {del_count:3d} ({del_count/count:.1%}), recompute: {recomp_count:3d} ({recomp_count/count:.1%})')


    return active_m.cpu().numpy(), alpha_m, Gamma, beta, mu_m, U 


def Statistics(K, K_m, mu_m, alpha_m, active_m, targets, N, device):
        
        
        mu_m, U, beta, dataLikely, bad_Hess = posterior_mode(K_m, targets, alpha_m, mu_m, max_itr=25, device=device)
        
        if bad_Hess: raise ValueError('bad Hessian')

        #  Compute covariance approximation
        U_inv	= torch.linalg.inv(U)
        Sigma_m	= U_inv @ U_inv.T
        #  Compute posterior-mean-based outputs and errors
        K_mu_m = (K_m @ mu_m).squeeze()
        y	= torch.sigmoid(K_mu_m)
        e	= (targets-y)
        logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        # if torch.isinf(dataLikely):
        #     logML			=  - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        # else:
        logML			= dataLikely - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        #  Well-determinedness factors
        DiagC	= torch.sum(U_inv**2, axis=1)
        Gamma	= 1 - alpha_m * DiagC
        # COMPUTE THE Q & S VALUES
        # KK_Sigma_m = KK_m @ Sigma_m
        # M = active_m.shape[0]
        beta_KK_m = K.T @  (torch.diag(beta) @ K_m) #(K_m * (beta.repeat([1, M])))
        # Q: "quality" factor - related to how well the basis function contributes
        # to reducing the error
        # 
        # S: "sparsity factor" - related to how orthogonal a given basis function
        # is to the currently used set of basis functions
        S = (beta.T @ K.pow(2)).squeeze() - torch.sum((beta_KK_m @ U_inv)**2, axis=1)
        Q =	(K.T @ e).squeeze()
        s =  S.clone()
        q =  Q.clone()
        S_active = S[active_m]
        s[active_m] = alpha_m * S_active / (alpha_m - S_active)
        q[active_m] = alpha_m * Q[active_m] / (alpha_m - S_active)

        
        return Sigma_m, mu_m, S, Q, s, q, beta, beta_KK_m, logML, Gamma, U  

def posterior_mode(K_m, targets, alpha_m, mu_m, max_itr, device):

    # Termination criterion for each gradient dimension
    grad_min = 1e-6
    # Minimum fraction of the full Newton step considered
    step_min = 1 / (2**8)


    def compute_data_error(K_mu_m, targets):
        y	= torch.sigmoid(K_mu_m)
        #  Handle probability zero cases
        y0	=(y==0) # (y<1e-6)
        y1	= (y==1) #(y>=(1-1e-6))
        if (y0[targets>0]).any() or (y1[targets<1]).any():
        #     #  Error infinite when model gives zero probability in
        #     #  contradiction to data
            data_error	= torch.tensor(np.inf)
            print(f'data_error inf------------------')
        # else:
            # Any y=0 or y=1 cases must now be accompanied by appropriate
            # output=0 or output=1 values, so can be excluded.
        # data_error	= -(targets[~y0].T @ torch.log(y[~y0]+1e-12) + ((1-targets[~y1]).T @ torch.log(1-y[~y1]+1e-12)))
        data_error = -((targets[targets==1]).T @ torch.log(y[targets==1]+1e-12) + ((1-targets[targets==0]).T @ torch.log(1-y[targets==0]+1e-12)))
        # data_error = -(1/2*(1+targets[targets==1]).T @ torch.log(y[targets==1]+1e-12) + (1/2*(1-targets[targets==-1]).T @ torch.log(1-y[targets==-1]+1e-12)))
        # data_error	= -(targets[~y0].T @ torch.log(y[~y0]) + ((1-targets[~y1]).T @ torch.log(1-y[~y1])))
        
        return y, data_error
    
    K_mu_m = K_m @ mu_m
    y, data_error = compute_data_error(K_mu_m.squeeze(), targets)
    #  Add on the weight penalty
    regulariser		= (alpha_m @ (mu_m.pow(2)))/2
    new_total_error	= data_error + regulariser

    bad_Hess	= False
    error_log	= torch.zeros([max_itr])

    for itr in range(max_itr):
    
        #  Log the error value each iteration
        error_log[itr]	= new_total_error

        #  Construct the gradient
        e	= (targets-y)
        g	= K_m.T @ e - (alpha_m * mu_m)
        #  Compute the likelihood-dependent analogue of the noise precision.
        #  NB: Beta now a vector.
        beta	= y * (1-y)
        # beta = beta.unsqueeze(1)
        #   Compute the Hessian
        beta_K_m	= (torch.diag(beta) @ K_m)  #K_m * (beta.unsqueeze(1).repeat([1, alpha_m.shape[0]]))
        H			= (K_m.T @ beta_K_m + torch.diag(alpha_m))
        #  Invert Hessian via Cholesky, watching out for ill-conditioning
        # try:
        #     U	=  torch.linalg.cholesky((H).transpose(-2, -1).conj()).transpose(-2, -1).conj()
        # except:
        #     print('pd_err of Hessian')
        #     return None, H, None, None, True
        # U =  torch.linalg.cholesky(H, upper=True)
        U, info =  torch.linalg.cholesky_ex(H, upper=True)
        # if info>0:
        #     print('pd_err of Hessian')
        #     return None, H, None, None, True
            #  Make sure its positive definite
        #  Before progressing, check for termination based on the gradient norm
        if all(abs(g)< grad_min): 
            break

        #  If all OK, compute full Newton step: H^{-1} * g
        # U_g = U.T.pinverse() @ g  #
        U_g = torch.linalg.lstsq(U.T, g).solution
        # delta_mu = U.pinverse() @ U_g #
        delta_mu = torch.linalg.lstsq(U, U_g).solution
        step		= 1
        while step > step_min:
            #  Follow gradient to get new value of parameters
            mu_new		= mu_m + step * delta_mu
            K_mu_m	    = K_m @ mu_new
            #  Compute outputs and error at new point
            y, data_error = compute_data_error(K_mu_m.squeeze(), targets)

            regulariser		= (alpha_m @ (mu_new.pow(2)))/2
            new_total_error	= data_error + regulariser
            #  Test that we haven't made things worse
            if new_total_error>=error_log[itr]:
                #  If so, back off!
                step	= step/2
            else:
                mu_m	= mu_new
                step	= 0	# this will force exit from the "while" loop
            
        if step>0:
            break
    #  Simple computation of return value of log likelihood at mode
    dataLikely	= -data_error

    return mu_m, U, beta, dataLikely, bad_Hess 


def posterior_mode_scipy(K_m, targets, alpha_m, mu_m, max_itr, device):

    # Termination criterion for each gradient dimension
    grad_min = 1e-6
    # Minimum fraction of the full Newton step considered
    step_min = 1 / (2**9)


    def compute_log_post(mu_m, K_m, targets, alpha_m):

        y	= expit(K_m @ mu_m)  
        # log_p_y = - (np.log(y[targets==1]+1e-12).sum() + np.log(y[targets==0]+1e-12).sum())
        log_p_y = -1 * (np.sum(np.log(y[targets == 1]), 0) +
                      np.sum(np.log(1-y[targets == 0]), 0))
        log_p_y = log_p_y + (alpha_m @ (mu_m**2))/2
        jacobian =  (alpha_m * mu_m) - K_m.T @ (targets-y)
        return log_p_y, jacobian

    def hessian(mu_m, K_m, targets, alpha_m):
        y	= expit(K_m @ mu_m)  
        beta   = y * (1-y)
        H = (K_m.T @ np.diag(beta) @ K_m + np.diag(alpha_m))
        return H

    tic = time.time()
    K_m_ = K_m.detach().cpu().numpy()
    mu_m_ = mu_m.detach().cpu().numpy()
    targets_ = targets.detach().cpu().numpy()
    alpha_m_ = alpha_m.detach().cpu().numpy()
    result = minimize(
            fun=compute_log_post,
            hess=hessian,
            x0=mu_m_,
            args=(K_m_, targets_, alpha_m_),
            # method='BFGS',
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': 50,
                'tol': 1e-6, 'disp': False
            }
        )
    mu_m_ = result.x
    H = hessian(mu_m_, K_m_, targets_, alpha_m_)
    mu_m = torch.tensor(mu_m_).to(device=device)
    H = torch.tensor(H).to(device=device)
    # U =  torch.linalg.cholesky(H, upper=True)
    U, info =  torch.linalg.cholesky_ex(H, upper=True)
    if info>0:
        print('pd_err of Hessian')
        return None, H, None, None, True

    data_error, _ = compute_log_post(mu_m_, K_m_, targets_, alpha_m_)
    dataLikely	= - torch.tensor(data_error).to(device=device)
    y	= torch.sigmoid(K_m @ mu_m)
    beta	= y * (1-y)
    # res = minimize(compute_log_post, [K_m, mu_m, targets, alpha_m], method='Newton-CG', 
    #             precision='float64', tol=1e-6, backend='torch', torch_device='cuda')
    toc = time.time()
    print(toc-tic)
    tic = time.time()
    # mu_m_old, U_old, beta_old, dataLikely_old, bad_Hess = posterior_mode_old(K_m, targets, alpha_m, mu_m, max_itr, device)
    # toc = time.time()
    # print(toc-tic)
    return mu_m, U, beta, dataLikely, False 



def generate_data():
    np.random.seed(8)
    rng = np.random.RandomState(0)
    # Generate sample data
    n = 100
    X_org = 4 * np.pi * np.random.random(n) - 2 * np.pi
    X_org = np.sort(X_org)
    y_org = np.sinc(X_org)
    y_org += 0.25 * (0.5 + 5 * rng.rand(X_org.shape[0]))  # add noise
    # y_org = (np.random.rand(X_org.shape[0]) < (1/(1+np.exp(-X_org))))
    y_org = (rng.rand(n) > y_org)
    y_org = y_org.astype(np.float64)
    normalized = True
    if normalized:
        X = np.copy(X_org) - np.mean(X_org, axis=0)
        X = X / np.std(X_org)
        # y = np.copy(y_org) - np.mean(y_org)
        # y = y / np.std(y_org)
        y = y_org
    else: 
        X = np.copy(X_org)
        y = np.copy(y_org)

    return X, y
    # X, y


def plot_result(rv, X_test, y_test, covar_module, K, N, mu_m, active):
    
    # rv = torch.tensor(X_org[active])
    """Evaluate the RVR model at x."""
    # K = covar_module(X_test, rv).evaluate()
   
    y_pred = K @ mu_m
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).to(int)
    # err_var = (1/beta) + K @ Sigma_m @ K.T
    # y_std = torch.sqrt(torch.diag(err_var))
    targets[targets==-1]=0
    acc = torch.sum(y_pred==targets)
    print(f'FRVM ACC:{(acc/N):.1%}')

    y_pred_ = y_pred.detach().numpy()
    # y_std_ = y_std.detach().numpy()
    lw=2
    plt.scatter(X_test, y_test, marker=".", c="k", label="data")
    # plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

    plt.plot(X_test, y_pred_, color="darkorange", lw=lw, label="RVR")
    # plt.fill_between(X_test, y_pred_ - y_std_, y_pred_ + y_std_, color="darkorange", alpha=0.4)
    plt.scatter(X_test[active], y_test[active], s=80, facecolors="none", edgecolors="r", lw=2,
                label="relevance vectors")

    plt.xlabel("data")
    plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
    plt.show()


if __name__=='__main__':


    X, y = generate_data()
    # X = X[:, None]
    
    inputs = torch.tensor(X[:])
    targets = torch.tensor(y[:])
    # import torch.nn.functional as F
    # inputs = F.normalize(inputs, p=2, dim=0)
    N   = inputs.shape[0]
    tol = 1e-6
    eps = torch.finfo(torch.float32).eps
    # sigma = model.likelihood.raw_noise.detach().clone()
    
    # B = torch.eye(N) * beta
    center= False
    scale = True
    bias = False
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    covar_module.base_kernel.lengthscale = 0.05 
    covar_module.outputscale = 1
    kernel_matrix = covar_module(inputs).evaluate()
    
    import scipy.io
    X = scipy.io.loadmat('./methods/X.mat')['X']
    inputs = torch.from_numpy(X).to(dtype=torch.float64)
    N = inputs.shape[0]
    kernel_matrix = scipy.io.loadmat('./methods/K.mat')['BASIS']
    kernel_matrix = torch.from_numpy(kernel_matrix).to(dtype=torch.float64)
    targets = scipy.io.loadmat('./methods/targets.mat')['Targets']
    targets = targets.astype(np.float64)
    targets = 2 * targets -1
    targets = torch.from_numpy(targets).to(dtype=torch.float64)
    targets = targets.squeeze()
    N = targets.shape[0]
    sigma = torch.var(targets)  #sigma^2
    # sigma = torch.tensor([0.01])
    update_sigma = True
    beta = 1 /sigma
    #centering in feature space
    unit = torch.ones([N, N], dtype=float)/N
    if center:
        kernel_matrix = kernel_matrix - unit @ kernel_matrix - kernel_matrix @ unit + unit @ kernel_matrix @ unit
    Scales = torch.ones(N)
    if scale:
        # scale = torch.sqrt(torch.sum(kernel_matrix) / N ** 2)
        # kernel_matrix = kernel_matrix / scale
        # normalize k(x,z) vector
        Scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
        K = kernel_matrix.clone() / Scales

    config = "001"
    align_thr = 1e-3
    gamma = False
    # scale = torch.sqrt(torch.sum(K) / N ** 2)
    # K = K / scale                                 K, targets, N, config, align_thr
    target = targets.clone()
    active_m, alpha_m, gamma_m, beta, mu_m, U = Fast_RVM(K, target, N, config, align_thr, gamma, eps, tol, device='cpu', task_id=0)
    print(f'relevant index \n {active_m}')
    print(f'relevant alpha \n {alpha_m}')
    print(f'relevant Gamma \n {gamma_m}')

    index = np.argsort(active_m)
    ss = Scales[active_m]
    ss = ss[index]
    active_m = active_m[index]
    alpha_m = alpha_m[index] / ss.pow(2)
    mu_m = mu_m[index] / ss
    
    plot_result(inputs[active_m], inputs, target, covar_module, kernel_matrix[:, active_m], N, mu_m, active_m)














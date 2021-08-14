from gpytorch.likelihoods import likelihood
import torch
import gpytorch
import numpy as np

import time
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from torch.utils import data


def Fast_RVM(K, targets, N, config, align_thr, eps, tol, max_itr=3000, device='cuda', verbose=True):
    

    M = K.shape[1]
    logMarginalLog = []
    targets = targets.to(device)
    K = K.to(device)
    targets_pseudo_linear	= 2 * targets - 1
    KK = K.T @ K  # all K_j @ K_i
    Kt = K.T @ targets_pseudo_linear
    start_k = torch.argmax(abs(Kt)) #torch.argmax(kt_k)
    active_m = torch.tensor([start_k]).to(device)
    K_m         = K[:, active_m]
    KK_m        = KK[:, active_m]
    Kt          = K.T @ targets

    #  Heuristic initialisation based on log-odds
    LogOut	= (targets_pseudo_linear * 0.9 + 1) / 2
    mu_m	=  K_m.pinverse() @ (torch.log(LogOut / (1 - LogOut))) #torch.linalg.lstsq(K_m, (torch.log(LogOut / (1 - LogOut)))).solution
    mu_m = mu_m.to(device)
    alpha_init = 1 / (mu_m + 1e-1).pow(2)
    alpha_m = torch.tensor([alpha_init], dtype=torch.float64).to(device)
    if alpha_m < 1e-3:
        alpha_m = torch.tensor([1e-3], dtype=torch.float64).to(device)
    if alpha_m > 1e3:
        alpha_m = torch.tensor([1e3], dtype=torch.float64).to(device)
    
    aligned_out		= torch.tensor([], dtype=torch.int).to(device)
    aligned_in		= torch.tensor([], dtype=torch.int).to(device)
    # Sigma_m = torch.inverse(A_m + beta * KK_mm)
    

    Sigma_m, mu_m, S, Q, s, q, beta, beta_KK_m, logML, Gamma = Statistics(K, K_m, mu_m, alpha_m, active_m, targets, N, device)

    delete_priority = config[0]=="1"
    add_priority    = config[1]=="1"
    alignment_test  = config[2]=="1"
    align_zero      = align_thr
    
    for itr in range(max_itr):

        # 'Relevance Factor' (q^2-s) values for basis functions in model
        Factor = q*q - s 
        #compute delta in marginal log likelihood for new action selection
        deltaML = torch.zeros(M, dtype=torch.float64).to(device)  
        action = torch.zeros(M)
        active_factor = Factor[active_m]

        # RECOMPUTE: must be a POSITIVE 'factor' and already IN the model
        idx                 = active_factor > 1e-12
        recompute           = active_m[idx]
        alpha_prim          =  s[recompute]**2 / (Factor[recompute])
        delta_alpha         = (1/alpha_prim - 1/alpha_m[idx])
        # d_alpha =  ((alpha_m[idx] - alpha_prim)/(alpha_prim * alpha_m[idx]))
        d_alpha_S           = delta_alpha * S[recompute] + 1 
        # deltaML[recompute] = ((delta_alpha * Q[recompute]**2) / (S[recompute] + 1/delta_alpha) - torch.log(d_alpha_S)) /2
        deltaML[recompute]  = ((delta_alpha * Q[recompute]**2) / (delta_alpha + eps) - torch.log(d_alpha_S)) /2
        
        # DELETION: if NEGATIVE factor and IN model
        idx = ~idx #active_factor <= 1e-12
        delete = active_m[idx]
        anyToDelete = len(delete) > 0
        if anyToDelete and active_m.shape[0] > 1:
            deltaML[delete] = -(q[delete]**2 / (s[delete] + alpha_m[idx]) - torch.log(1 + s[delete] / alpha_m[idx])) /2
            # deltaML[delete] = -(Q[delete]**2 / (S[delete] + alpha_m[idx]) - torch.log(1 + S[delete] / alpha_m[idx])) /2
            action[delete]  = -1

        # ADDITION: must be a POSITIVE factor and OUT of the model
        good_factor = Factor > 1e-12
        good_factor[active_m] = False
        
        if alignment_test and len(aligned_out) > 0:
            
            good_factor[aligned_out] = False

        add = torch.where(good_factor)[0]
        anyToAdd = len(add) > 0
        if anyToAdd:
            Q_S             = Q[add]**2 / S[add]
            deltaML[add]    = (Q_S - 1 - torch.log(Q_S)) /2
            action[add]     = 1

        # Priority of Deletion   
        if anyToDelete and delete_priority:
            deltaML[recompute] = 0
            deltaML[add] = 0
        # Priority of Addition       
        if anyToAdd and add_priority:
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

        if ~anyWorthwhileAction:
            print(f'{itr:03}, No positive action, m={active_m.shape[0]}')
            selected_action = torch.tensor(10)
            terminate = True

        if selected_action==0 and ~anyToDelete:
            no_change_in_alpha = torch.abs(torch.log(alpha_new) - torch.log(alpha_m[j])) < tol
            no_change_in_alpha = len(no_change_in_alpha) == 1
            if no_change_in_alpha:
                # print(selected_action)
                print(f'{itr:03}, No change in alpha, m={active_m.shape[0]}')
                selected_action = torch.tensor(11)
                terminate = True
        
        
        
        if alignment_test:
            #
            # Addition - rule out addition (from now onwards) if the new basis
            # vector is aligned too closely to one or more already in the model
            # 
            if selected_action== 1:
            # Basic test for correlated basis vectors
            # (note, Phi and columns of PHI are normalised)
            # 
                k_new = K[:, max_idx]
                k_new_K_m         = k_new.T @ K_m 
                # p				= Phi'*PHI;
                aligned_idx = torch.where(k_new_K_m > (1- align_zero))[0]
                num_aligned	= len(aligned_idx)
                if num_aligned > 0:
                    # The added basis function is effectively indistinguishable from
                    # one present already
                    selected_action	= torch.tensor(12)
                    # act_			= 'alignment-deferred addition';
                    # alignDeferCount	= alignDeferCount+1;
                    # Make a note so we don't try this next time
                    # May be more than one in the model, which we need to note was
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

            alpha_j_old     = alpha_m[j]
            alpha_m[j]      = alpha_new
            s_j             = Sigma_m[:, j]
            delta_inv       = 1 / (alpha_new - alpha_j_old)
            kappa           = 1 / (Sigma_m[j, j] + delta_inv)
            tmp             = kappa * s_j
            Sigma_new       = Sigma_m - tmp @ s_j.T
            delta_mu        = -mu_m[j] * tmp
            mu_m            = mu_m + delta_mu.squeeze()
          
            S	= S + kappa * (beta_KK_m @ s_j).squeeze()**2
            Q	= Q - (beta_KK_m @ delta_mu).squeeze()
            update_required = True
        
        if selected_action==-1:  #delete
            # print(f'itr= {itr}, selected_action={selected_action.item()}')
            active_m        = active_m[active_m!=active_m[j]]
            alpha_m         = alpha_m[alpha_m!=alpha_m[j]]

            s_jj			= Sigma_m[j, j]
            s_j				= Sigma_m[:, j]
            tmp				= s_j/s_jj
            Sigma_		    = Sigma_m - tmp @ s_j.T
            Sigma_          = Sigma_[torch.arange(Sigma_.size(0)).to(device)!=j]
            Sigma_new       = Sigma_[:, torch.arange(Sigma_.size(1)).to(device)!=j]
            delta_mu		= -mu_m[j] * tmp
            mu_j			= mu_m[j]
            mu_m			= mu_m + delta_mu.squeeze()
            mu_m			= mu_m[torch.arange(mu_m.size(0)).to(device)!=j]
            
            jPm	            = (beta_KK_m @ s_j).squeeze()
            S	            = S + jPm.pow(2) / s_jj
            Q	            = Q + jPm * mu_j / s_jj

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]
            beta_KK_m       = beta * KK_m
            update_required = True
        
        if selected_action==1:  #add

            active_m = torch.cat([active_m, max_idx])
            alpha_m = torch.cat([alpha_m, alpha_new])

            k_new           = K[:, max_idx]
            # K_k_new         = K.T @ k_new
            beta_k_new      = beta * k_new
            beta_K_k_new  	= K.T @ beta_k_new

            tmp		        = ((beta_k_new.T @ K_m) @ Sigma_m).T
            s_ii		    = 1/ (alpha_new +S[max_idx])
            s_i			    = -tmp @ s_ii
            tau			    = -tmp @ s_i.unsqueeze(0)
            Sigma_          = torch.cat([Sigma_m + tau, s_i.unsqueeze(-1)], axis=1)
            Sigma_new       = torch.cat([Sigma_, torch.cat([s_i.T, s_ii]).unsqueeze(0)], axis=0)
            mu_i		    = (s_ii @ Q[max_idx]).unsqueeze(0)
            delta_mu        = torch.cat([-tmp @ mu_i , mu_i], axis=0)
            mu_m			= torch.cat([mu_m, torch.tensor([0.0]).to(device)], axis=0) + delta_mu
        
            mCi	            = beta_K_k_new - beta_KK_m @ tmp
            S   	        = S - mCi.pow(2) @ s_ii
            Q   	        = Q - mCi @ mu_i

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]

            beta_KK_m       = beta * KK_m
            update_required = True
        
            
        # UPDATE STATISTICS
        if update_required:
            s = S.clone()
            q = Q.clone()
            tmp = alpha_m / (alpha_m -S[active_m])
            s[active_m] = tmp * S[active_m] 
            q[active_m] = tmp * Q[active_m]
            Sigma_m = Sigma_new
            #quantity Gamma_i measures how well the corresponding parameter mu_i is determined by the data
            Gamma = 1 - alpha_m * torch.diag(Sigma_m)
            logML = logML + deltaLogMarginal
            logMarginalLog.append(logML.item())

            beta_KK_m = K @ (K_m * (beta * torch.ones(1,active_m.shape[0]).to(device)))


        if terminate:
            # print(f'sigma2={1/beta:.4f}')
            print(f'Finished at {itr:03}, m= {active_m.shape[0]}')
            return active_m.cpu().numpy(), alpha_m, Gamma, beta 

        if ((itr+1)%50==0) and verbose:
            print(f'#{itr+1:03},     m={active_m.shape[0]}, selected_action= {selected_action.item():.0f}, logML= {logML.item()/N:.5f}, sigma2={1/beta:.4f}')


    print(f'logML= {logML/N}\n{logMarginalLog}')

    return active_m.cpu().numpy(), alpha_m, Gamma, beta 


def Statistics(K, K_m, mu_m, alpha_m, active_m, targets, N, device):
        
        
        mu_m, U, beta, dataLikely, bad_Hess = posterior_mode(K_m, targets, alpha_m, mu_m, max_itr=25, device=device)
        
        if bad_Hess: ValueError('bad Hessian')

        #  Compute covariance approximation
        U_inv	= torch.inverse(U)
        Sigma_m	= U_inv @ U_inv.T
        #  Compute posterior-mean-based outputs and errors
        K_mu_m = (K_m @ mu_m).unsqueeze(1)
        y	= torch.sigmoid(K_mu_m)
        e	= (targets-y)
        logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        logML			= dataLikely - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        #  Well-determinedness factors
        DiagC	= torch.sum(U_inv**2, axis=1)
        Gamma	= 1 - alpha_m * DiagC
        # COMPUTE THE Q & S VALUES
        # KK_Sigma_m = KK_m @ Sigma_m
        M = active_m.shape[0]
        beta_KK_m = K.T @ (K_m * (beta * torch.ones(1,M).to(device)))
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

        return Sigma_m, mu_m, S, Q, s, q, beta, beta_KK_m, logML, Gamma  


def posterior_mode(K_m, targets, alpha_m, mu_m, max_itr, device):

    # Termination criterion for each gradient dimension
    grad_min = 1e-6
    # Minimum fraction of the full Newton step considered
    step_min = 1 / (2**8)

    K_mu_m = K_m @ mu_m

    def compute_data_error(K_mu_m, targets):
        y	= torch.sigmoid(K_mu_m)
        #  Handle probability zero cases
        y0	= (y==0)
        y1	= (y==1)
        if (y0[targets>0]).any() or (y1[targets<1]).any():
            #  Error infinite when model gives zero probability in
            #  contradiction to data
            data_error	= np.inf
        else:
            # Any y=0 or y=1 cases must now be accompanied by appropriate
            # output=0 or output=1 values, so can be excluded.
            data_error	= -(targets[~y0].T @ torch.log(y[~y0])) + ((1-targets[~y1]).T @ torch.log(1-y[~y1]))
        return y, data_error
    
    y, data_error = compute_data_error(K_mu_m, targets)
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
        g	= K_m.T @ e -alpha_m * mu_m
        #  Compute the likelihood-dependent analogue of the noise precision.
        #  NB: Beta now a vector.
        beta	= y * (1-y)
        #   Compute the Hessian
        beta_K_m	= K_m * (beta * torch.ones(1, K_m.shape[1]).to(device))
        H			= (beta_K_m.T @ K_m + torch.diag(alpha_m))
        #  Invert Hessian via Cholesky, watching out for ill-conditioning
        try:
            U	=  torch.linalg.cholesky((H).transpose(-2, -1).conj()).transpose(-2, -1).conj()
            #  Make sure its positive definite
        except:
            print('pd_err of Hessian')
            return None, None, None, None, True
        
        #  Before progressing, check for termination based on the gradient norm
        if all(abs(g)< grad_min): break

        #  If all OK, compute full Newton step: H^{-1} * g
        U_g = U.T.pinverse() @ g  #torch.linalg.lstsq(U.T, g).solution
        delta_mu = U.pinverse() @ U_g #torch.linalg.lstsq(U, U_g).solution
        step		= 1
        while step > step_min:
            #  Follow gradient to get new value of parameters
            mu_new		= mu_m + step * delta_mu
            K_mu_m	= K_m @ mu_new
            #  Compute outputs and error at new point
            y, data_error = compute_data_error(K_mu_m, targets)

            regulariser		= (alpha_m @ (mu_m.pow(2)))/2
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

    return mu_m.squeeze(1), U, beta, dataLikely, bad_Hess 

def Fast_RVM_Full_out(K, targets, bias, beta, N, update_sigma, eps, tol, max_itr=3000, device='cpu', verbose=True):
    
    if bias:
        K = torch.cat((torch.ones(N, 1), K), dim=1)
    M = K.shape[1]

    logMarginalLog = []
    active = []
    alpha = []
    if bias:
        active.append(0)
        alpha.append(eps)

    KK = K.T @ K  # all K_j @ K_i
    Kt = K.T @ targets
    KK_diag = torch.diag(KK) # all K_j @ K_j

    # kt_k = Kt**2 / KK_diag
    start_k = torch.argmax(Kt) #torch.argmax(kt_k)
    p = beta * KK_diag[start_k]
    q = beta * Kt[start_k]
    alpha_start_k = p.pow(2) / (q.pow(2) - p)

    alpha.append(alpha_start_k)
    active.append(start_k)

    alpha_m = torch.tensor(alpha, dtype=torch.float64).to(device)
    active_m = torch.tensor(active).to(device)

    # Sigma_m = torch.inverse(A_m + beta * KK_mm)
    
    K_m = K[:, active_m]
    KK_m = KK[:, active_m]
    KK_mm = KK[active_m, :][:, active_m]
    K_mt = Kt[active_m] 
    beta_KK_m = beta * KK_m
    Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)

    for itr in range(max_itr):

        # 'Relevance Factor' (q^2-s) values for basis functions in model
        Factor = q*q - s 
        #compute delta in marginal log likelihood for new action selection
        deltaML = torch.zeros(M, dtype=torch.float64).to(device)  
        action = torch.zeros(M)
        active_factor = Factor[active_m]

        # RECOMPUTE: must be a POSITIVE 'factor' and already IN the model
        idx = active_factor > 1e-12
        recompute = active_m[idx]
        alpha_prim =  s[recompute]**2 / (Factor[recompute])
        delta_alpha    = (1./alpha_prim - 1./alpha_m[idx])
        # d_alpha =  ((alpha_m[idx] - alpha_prim)/(alpha_prim * alpha_m[idx]))
        d_alpha_S = delta_alpha * S[recompute] + 1 
        # deltaML[recompute] = ((delta_alpha * Q[recompute]**2) / (S[recompute] + 1/delta_alpha) - torch.log(d_alpha_S)) /2
        deltaML[recompute] = ((delta_alpha * Q[recompute]**2) / (delta_alpha + eps) - torch.log(d_alpha_S)) /2
        
        # DELETION: if NEGATIVE factor and IN model
        idx = ~idx #active_factor <= 1e-12
        delete = active_m[idx]
        anyToDelete = len(delete) > 0
        if anyToDelete:
            deltaML[delete] = -(q[delete]**2 / (s[delete] + alpha_m[idx]) - torch.log(1 + s[delete] / alpha_m[idx])) /2
            # deltaML[delete] = -(Q[delete]**2 / (S[delete] + alpha_m[idx]) - torch.log(1 + S[delete] / alpha_m[idx])) /2
            action[delete] = -1

        # ADDITION: must be a POSITIVE factor and OUT of the model
        good_factor = Factor > 0
        good_factor[active_m] = False
        add = torch.where(good_factor)[0]
        anyToAdd = len(add) > 0
        if anyToAdd:
            Q_S = Q[add]**2 / S[add]
            deltaML[add] = (Q_S - 1 - torch.log(Q_S)) /2
            action[add] = 1

        # Priority of Deletion
        delete_priority = True
        if anyToDelete and delete_priority:

            deltaML[recompute] = 0
            deltaML[add] = 0

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
        if ~anyWorthwhileAction:
            print(f'No positive action, m={active_m.shape[0]}')
            terminate = True
        if selected_action==0 and ~anyToDelete:
            no_change_in_alpha = torch.abs(torch.log(alpha_new) - torch.log(alpha_m[j])) < tol
            no_change_in_alpha = len(no_change_in_alpha) == 1
            if no_change_in_alpha:
                print(selected_action)
                print(f'Finish, nochange in alpha, m={active_m.shape[0]}')
                terminate = True
        if terminate:
            print(f'before terminate try beta update')
            beta_old = beta
            y_      = K_m @ mu_m  
            e       = (targets.squeeze() - y_)
            beta	= (N - torch.sum(Gamma))/(e.T @ e)
            delta_beta	= torch.log(beta)-torch.log(beta_old)
            beta_KK_m       = beta * KK_m
            if torch.abs(delta_beta) > 1e-6:
                print(f'update statistics after beta update')
                Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)
                logMarginalLog.append(logML.item())
            else:
                print(f'no change in beta update')
                return active_m, alpha_m, K_m, Sigma_m, mu_m, Gamma, beta 

        if selected_action==0:   #recompute

            alpha_j_old     = alpha_m[j]
            alpha_m[j]      = alpha_new
            s_j             = Sigma_m[:, j]
            delta_inv       = 1 / (alpha_new - alpha_j_old)
            kappa           = 1 / (Sigma_m[j, j] + delta_inv)
            tmp             = kappa * s_j
            Sigma_new       = Sigma_m - tmp @ s_j.T
            delta_mu        = -mu_m[j] * tmp
            mu_m            = mu_m + delta_mu.squeeze()
          
            S	= S + kappa * (beta_KK_m @ s_j).squeeze()**2
            Q	= Q - (beta_KK_m @ delta_mu).squeeze()

        elif selected_action==-1:  #delete

            active_m        = active_m[active_m!=active_m[j]]
            alpha_m         = alpha_m[alpha_m!=alpha_m[j]]

            s_jj			= Sigma_m[j, j]
            s_j				= Sigma_m[:, j]
            tmp				= s_j/s_jj
            Sigma_		    = Sigma_m - tmp @ s_j.T
            Sigma_          = Sigma_[torch.arange(Sigma_.size(0))!=j]
            Sigma_new       = Sigma_[:, torch.arange(Sigma_.size(1))!=j]
            delta_mu		= -mu_m[j] * tmp
            mu_j			= mu_m[j]
            mu_m			= mu_m + delta_mu.squeeze()
            mu_m			= mu_m[torch.arange(mu_m.size(0))!=j]
            
            jPm	            = (beta_KK_m @ s_j).squeeze()
            S	            = S + jPm.pow(2) / s_jj
            Q	            = Q + jPm * mu_j / s_jj

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m

        else:  #add

            active_m = torch.cat([active_m, max_idx])
            alpha_m = torch.cat([alpha_m, alpha_new])

            k_new           = K[:, max_idx]
            K_k_new         = K.T @ k_new
            beta_k_new      = beta * k_new

            tmp		        = ((beta_k_new.T @ K_m) @ Sigma_m).T
            s_ii		    = 1/ (alpha_new +S[max_idx])
            s_i			    = -tmp @ s_ii
            tau			    = -tmp @ s_i.unsqueeze(0)
            Sigma_          = torch.cat([Sigma_m + tau, s_i.unsqueeze(-1)], axis=1)
            Sigma_new       = torch.cat([Sigma_, torch.cat([s_i.T, s_ii]).unsqueeze(0)], axis=0)
            mu_i		    = (s_ii @ Q[max_idx]).unsqueeze(0)
            delta_mu        = torch.cat([-tmp @ mu_i , mu_i], axis=0)
            mu_m			= torch.cat([mu_m, torch.tensor([0.0])], axis=0) + delta_mu
        
            mCi	            = beta * K_k_new - beta_KK_m @ tmp
            S   	        = S - mCi.pow(2) @ s_ii
            Q   	        = Q - mCi @ mu_i

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m

        # UPDATE STATISTICS
        s = S.clone()
        q = Q.clone()
        tmp = alpha_m / (alpha_m -S[active_m])
        s[active_m] = tmp * S[active_m] 
        q[active_m] = tmp * Q[active_m]
        Sigma_m = Sigma_new
        Gamma = 1 - alpha_m * torch.diag(Sigma_m)
        logML = logML + deltaLogMarginal
        logMarginalLog.append(logML.item())


        #compute mu and beta
        if update_sigma and ((itr%5==0) or (itr <=10)):
            
            beta_old = beta
            y_      = K_m @ mu_m  
            e       = (targets.squeeze() - y_)
            beta	= (N - torch.sum(Gamma))/(e.T @ e)
            delta_beta	= torch.log(beta)-torch.log(beta_old)
            beta_KK_m       = beta * KK_m
            if torch.abs(delta_beta) > 1e-6:
                print(f'update statistics after beta update')
                Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)
                logMarginalLog.append(logML.item())
     
        if ((itr+1)%1==0) and verbose:
            print(f'#{itr+1:03},     m={active_m.shape[0]}, selected_action= {selected_action.item():.0f}, sigma={1/beta:.4f}')


    print(f'logML= {logML}\n{logMarginalLog}')
    return active_m, alpha_m, K_m, Sigma_m, mu_m, Gamma, beta 


def generate_data():
    np.random.seed(8)
    rng = np.random.RandomState(0)
    # Generate sample data
    X_org = 4 * np.pi * np.random.random(500) - 2 * np.pi
    y_org = np.sinc(X_org)
    y_org += 0.25 * (0.5 - rng.rand(X_org.shape[0])) + 100  # add noise
    y_org = (np.random.rand(X_org.shape[0]) < (1/(1+np.exp(-X_org))))
    y_org = y_org.astype(np.float)
    normalized = False
    if normalized:
        X = np.copy(X_org) - np.mean(X_org, axis=0)
        X = X / np.std(X_org)
        y = np.copy(y_org) - np.mean(y_org)
        y = y / np.std(y_org)
    else: 
        X = np.copy(X_org)
        y = np.copy(y_org)

    return X, y
    # X, y

if __name__=='__main__':


    X, y = generate_data()
    # X = X[:, None]
    
    inputs = torch.tensor(X[:, None])
    targets = torch.tensor(y[:, None])
    N   = inputs.shape[0]
    tol = 1e-3
    eps = torch.finfo(torch.float32).eps
    # sigma = model.likelihood.raw_noise.detach().clone()
    
    # B = torch.eye(N) * beta
    center= False
    scale = True
    bias = False
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    covar_module.base_kernel.lengthscale = 0.2  #0.23
    covar_module.outputscale = 1
    kernel_matrix = covar_module(inputs).evaluate()
    N = inputs.shape[0]
    # import scipy.io
    # kernel_matrix = scipy.io.loadmat('./K.mat')['BASIS']
    # kernel_matrix = torch.from_numpy(kernel_matrix).to(dtype=torch.float64)
    # targets = scipy.io.loadmat('./targets.mat')['Targets']
    # targets = torch.from_numpy(targets).to(dtype=torch.float64)

    sigma = torch.var(targets)  #sigma^2
    # sigma = torch.tensor([0.01])
    update_sigma = True
    beta = 1 /sigma
    #centering in feature space
    unit = torch.ones([N, N], dtype=float)/N
    if center:
        kernel_matrix = kernel_matrix - unit @ kernel_matrix - kernel_matrix @ unit + unit @ kernel_matrix @ unit
    if scale:
        # scale = torch.sqrt(torch.sum(kernel_matrix) / N ** 2)
        # kernel_matrix = kernel_matrix / scale
        # normalize k(x,z) vector
        Scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
        kernel_matrix = kernel_matrix / Scales

    K = kernel_matrix
    # scale = torch.sqrt(torch.sum(K) / N ** 2)
    # K = K / scale
    active_m, alpha_m, gamma_m, beta = Fast_RVM(K, targets, beta, N, update_sigma, eps, tol)













# kernel_matrix = torch.tensor([
    # [0.9241,    0.2553,    0.0060,    0.0000,    0.2553,    0.0705,    0.0017,    0.0000,    0.0060,    0.0017,    0.0000,    0.0000,     0.0000,    0.0000,    0.0000,    0.0000],
    # [0.2648,    0.8909,    0.2553,    0.0062,    0.0731,    0.2461,    0.0705,    0.0017,    0.0017,    0.0058,    0.0017,    0.0000,     0.0000,    0.0000,    0.0000,    0.0000],
    # [0.0062,    0.2553,    0.8909,    0.2648,    0.0017,    0.0705,    0.2461,    0.0731,    0.0000,    0.0017,    0.0058,    0.0017,     0.0000,    0.0000,    0.0000,    0.0000],
    # [0.0000,    0.0060,    0.2553,    0.9241,    0.0000,    0.0017,    0.0705,    0.2553,    0.0000,    0.0000,    0.0017,    0.0060,     0.0000,    0.0000,    0.0000,    0.0000],
    # [0.2648,    0.0731,    0.0017,    0.0000,    0.8909,    0.2461,    0.0058,    0.0000,    0.2553,    0.0705,    0.0017,    0.0000,     0.0062,    0.0017,    0.0000,    0.0000],
    # [0.0759,    0.2553,    0.0731,    0.0018,    0.2553,    0.8589,    0.2461,    0.0060,    0.0731,    0.2461,    0.0705,    0.0017,     0.0018,    0.0060,    0.0017,    0.0000],
    # [0.0018,    0.0731,    0.2553,    0.0759,    0.0060,    0.2461,    0.8589,    0.2553,    0.0017,    0.0705,    0.2461,    0.0731,     0.0000,    0.0017,    0.0060,    0.0018],
    # [0.0000,    0.0017,    0.0731,    0.2648,    0.0000,    0.0058,    0.2461,    0.8909,    0.0000,    0.0017,    0.0705,    0.2553,     0.0000,    0.0000,    0.0017,    0.0062],
    # [0.0062,    0.0017,    0.0000,    0.0000,    0.2553,    0.0705,    0.0017,    0.0000,    0.8909,    0.2461,    0.0058,    0.0000,     0.2648,    0.0731,    0.0017,    0.0000],
    # [0.0018,    0.0060,    0.0017,    0.0000,    0.0731,    0.2461,    0.0705,    0.0017,    0.2553,    0.8589,    0.2461,    0.0060,     0.0759,    0.2553,    0.0731,    0.0018],
    # [0.0000,    0.0017,    0.0060,    0.0018,    0.0017,    0.0705,    0.2461,    0.0731,    0.0060,    0.2461,    0.8589,    0.2553,     0.0018,    0.0731,    0.2553,    0.0759],
    # [0.0000,    0.0000,    0.0017,    0.0062,    0.0000,    0.0017,    0.0705,    0.2553,    0.0000,    0.0058,    0.2461,    0.8909,     0.0000,    0.0017,    0.0731,    0.2648],
    # [0.0000,    0.0000,    0.0000,    0.0000,    0.0060,    0.0017,    0.0000,    0.0000,    0.2553,    0.0705,    0.0017,    0.0000,     0.9241,    0.2553,    0.0060,    0.0000],
    # [0.0000,    0.0000,    0.0000,    0.0000,    0.0017,    0.0058,    0.0017,    0.0000,    0.0731,    0.2461,    0.0705,    0.0017,     0.2648,    0.8909,    0.2553,    0.0062],
    # [0.0000,    0.0000,    0.0000,    0.0000,    0.0000,    0.0017,    0.0058,    0.0017,    0.0017,    0.0705,    0.2461,    0.0731,     0.0062,    0.2553,    0.8909,    0.2648],
    # [0.0000,    0.0000,    0.0000,    0.0000,    0.0000,    0.0000,    0.0017,    0.0060,    0.0000,    0.0017,    0.0705,    0.2553,     0.0000,    0.0060,    0.2553,    0.9241],
    # ], dtype=torch.float64)
    # targets = torch.tensor([ 52.5118,  -0.9995, -60.4597, -23.7588,  17.9821,  -3.8639, -24.8190,
    #      -9.0013,   1.3258,   2.0937,  -3.0549,  -9.7661,   2.5847,  -7.6229,
    #      -8.0955, -35.3434], dtype=torch.float64)
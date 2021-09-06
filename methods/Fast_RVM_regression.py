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


def Fast_RVM_regression(K, targets, beta, N, config, align_thr, eps, tol, max_itr=3000, device='cuda', verbose=True):
    

    M = K.shape[1]
    logMarginalLog = []

    KK = K.T @ K  # all K_j @ K_i
    Kt = K.T @ targets
    KK_diag = torch.diag(KK) # all K_j @ K_j

    start_k = torch.argmax(abs(Kt)) #torch.argmax(kt_k)
    p = beta * KK_diag[start_k]
    q = beta * Kt[start_k]
    alpha_start_k = p.pow(2) / (q.pow(2) - p)

    alpha_m = torch.tensor([alpha_start_k], dtype=torch.float64).to(device)
    if alpha_m < 0:
        alpha_m = torch.tensor([1000], dtype=torch.float64).to(device)
    active_m = torch.tensor([start_k]).to(device)
    aligned_out		= torch.tensor([], dtype=torch.int).to(device)
    aligned_in		= torch.tensor([], dtype=torch.int).to(device)
    # Sigma_m = torch.inverse(A_m + beta * KK_mm)
    
    K_m = K[:, active_m]
    KK_m = KK[:, active_m]
    KK_mm = KK[active_m, :][:, active_m]
    K_mt = Kt[active_m] 
    beta_KK_m = beta * KK_m
    Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)

    update_sigma    = config[0]=="1"
    delete_priority = config[1]=="1"
    add_priority    = config[2]=="1"
    alignment_test  = config[3]=="1"
    align_zero      = align_thr
    add_count = 0
    del_count = 0
    recomp_count = 0
    count = 0
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
        deltaML[recompute]  = ((delta_alpha * Q[recompute]**2) / (d_alpha_S) - torch.log(d_alpha_S)) /2
        
        # DELETION: if NEGATIVE factor and IN model
        idx = ~idx #active_factor <= 1e-12
        delete = active_m[idx]
        anyToDelete = len(delete) > 0
        if anyToDelete and alpha_m.shape[0] > 1: 	   # if there is only one basis function (M=1) (In practice, this latter event ought only to happen with the Gaussian
                                                        #    likelihood when initial noise is too high. In that case, a later beta
                                                        #    update should 'cure' this.)
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
            if verbose:
                print(f'{itr:3}, No positive action, m={active_m.shape[0]:3}')
            selected_action = torch.tensor(10)
            terminate = True

        elif selected_action==0 and ~anyToDelete:
            no_change_in_alpha = torch.abs(torch.log(alpha_new) - torch.log(alpha_m[j])) < tol
            
            if no_change_in_alpha:
                # print(selected_action)
                if verbose:
                    print(f'{itr:3}, No change in alpha, m={active_m.shape[0]:3}')
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
         
            #
            # Deletion: reinstate any previously deferred basis functions
            # resulting from this basis function
            # 
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
            Sigma_new       = Sigma_m - tmp @ s_j.T
            delta_mu        = -mu_m[j] * tmp
            mu_m            = mu_m + delta_mu.squeeze()
          
            S	= S + kappa * (beta_KK_m @ s_j).squeeze()**2
            Q	= Q - (beta_KK_m @ delta_mu).squeeze()
            update_required = True
        
        elif selected_action==-1:  #delete
            del_count += 1
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
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m
            update_required = True
        
        elif selected_action==1:  #add
            add_count += 1
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
            mu_m			= torch.cat([mu_m, torch.tensor([0.0]).to(device)], axis=0) + delta_mu
        
            mCi	            = beta * K_k_new - beta_KK_m @ tmp
            S   	        = S - mCi.pow(2) @ s_ii
            Q   	        = Q - mCi @ mu_i

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m
            update_required = True
        


            
        # UPDATE STATISTICS
        if update_required:
            count += 1
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
            beta_KK_m = beta * KK_m

        min_index = torch.where(Gamma < 0.5)[0]
        if min_index.shape[0] >0:
            del_from_active = active_m[min_index]
            print(f'remove low Gamma: {Gamma[min_index].detach().cpu().numpy()} correspond to {del_from_active.detach().cpu().numpy()} data index')
            for j in min_index:
                del_count += 1
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
                # jPm	            = (beta_KK_m @ s_j).squeeze()
                # S	            = S + jPm.pow(2) / s_jj
                # Q	            = Q + jPm * mu_j / s_jj
                Sigma_m         = Sigma_new
                K_m             = K[:, active_m]
                KK_m            = KK[:, active_m]
                KK_mm           = KK[active_m, :][:, active_m]
                K_mt            = Kt[active_m]
                # beta_KK_m       = beta * KK_m
                # update_required = True
                count += 1

            
            #quantity Gamma_i measures how well the corresponding parameter mu_i is determined by the data
            Gamma = 1 - alpha_m * torch.diag(Sigma_m)
           
            beta_KK_m = beta * KK_m
            
            terminate = True

        #compute mu and beta
        if update_sigma and ((itr%5==0) or (itr <=10) or terminate):
            
            beta_old = beta
            y_      = K_m @ mu_m  
            e       = (targets.squeeze() - y_)
            beta	= (N - torch.sum(Gamma))/(e.T @ e)
            beta	= torch.min(torch.tensor([beta, 1e6/torch.var(targets)]).to(device))
            delta_beta	= torch.log(beta)-torch.log(beta_old)
            beta_KK_m       = beta * KK_m
            if torch.abs(delta_beta) > 1e-6:
                if verbose:
                    print(f'{itr:3}, update statistics after beta update')
                Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)
                count = count + 1
                logMarginalLog.append(logML.item())
                terminate = False
            # else: 
            #     print(f'No change in beta')



        if terminate:
            # print(f'sigma2={1/beta:.4f}')
            # if verbose:
            # if active_m.shape[0] < 3:
            print(f'Finished at {itr:3}, m= {active_m.shape[0]:3} sigma2= {1/beta:4.4f}')
            # if count > 0:
            #     print(f'add: {add_count:3d} ({add_count/count:.1%}), delete: {del_count:3d} ({del_count/count:.1%}), recompute: {recomp_count:3d} ({recomp_count/count:.1%})')
            return active_m.cpu().numpy(), alpha_m, Gamma, beta, mu_m

        if ((itr+1)%50==0) and verbose:
            print(f'#{itr+1:3},     m={active_m.shape[0]}, selected_action= {selected_action.item():.0f}, logML= {logML.item()/N:.5f}, sigma2= {1/beta:.4f}')

    if verbose:
        print(f'logML= {logML/N}\n{logMarginalLog}')

    return active_m.cpu().numpy(), alpha_m, Gamma, beta, mu_m


def Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N):
        A_m = torch.diag(alpha_m)
        
        U = torch.linalg.cholesky((A_m + beta * KK_mm).transpose(-2, -1).conj()).transpose(-2, -1).conj()
        U_inv = torch.linalg.inv(U)
        Sigma_m = U_inv @ U_inv.T
                 
        mu_m = beta * (Sigma_m @ K_mt)
        # mu_m = mu_m.squeeze(0)
        y_ = K_m @ mu_m  
        e = (targets - y_)
        ED = e.T @ e
        dataLikely	= (N * torch.log(beta) - beta * ED)/2
        logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        if torch.isinf(dataLikely):
            logML			=  - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        else:
            logML			= dataLikely - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        DiagC	= torch.sum(U_inv**2, axis=1)
        Gamma	= 1 - alpha_m * DiagC
        # COMPUTE THE Q & S VALUES
        # KK_Sigma_m = KK_m @ Sigma_m
        beta_KK_m = beta * KK_m
        # Q: "quality" factor - related to how well the basis function contributes
        # to reducing the error
        # 
        # S: "sparsity factor" - related to how orthogonal a given basis function
        # is to the currently used set of basis functions
        S = beta - torch.sum((beta_KK_m @ U_inv)**2, axis=1)
        Q = beta * (Kt.squeeze() - KK_m @ mu_m)
        #S_j = (beta * KK_diag - (beta**2) * torch.sum(KK_Sigma_m * KK_m, axis=1)).squeeze()
        #Q_j = (beta * Kt - (beta**2) * KK_Sigma_m @ K_mt).squeeze()
        s =  S.clone()
        q =  Q.clone()
        S_active = S[active_m]
        s[active_m] = alpha_m * S_active / (alpha_m - S_active)
        q[active_m] = alpha_m * Q[active_m] / (alpha_m - S_active)

        return Sigma_m, mu_m, S, Q, s, q, logML, Gamma  

def Fast_RVM_regression_fullout(K, targets, beta, N, config, align_thr, eps, tol, max_itr=3000, device='cuda', verbose=True):
    

    M = K.shape[1]
    logMarginalLog = []

    KK = K.T @ K  # all K_j @ K_i
    Kt = K.T @ targets
    KK_diag = torch.diag(KK) # all K_j @ K_j

    start_k = torch.argmax(abs(Kt)) #torch.argmax(kt_k)
    p = beta * KK_diag[start_k]
    q = beta * Kt[start_k]
    alpha_start_k = p.pow(2) / (q.pow(2) - p)

    alpha_m = torch.tensor([alpha_start_k], dtype=torch.float64).to(device)
    if alpha_m < 0:
        alpha_m = torch.tensor([1000], dtype=torch.float64).to(device)
    active_m = torch.tensor([start_k]).to(device)
    aligned_out		= torch.tensor([], dtype=torch.int).to(device)
    aligned_in		= torch.tensor([], dtype=torch.int).to(device)
    # Sigma_m = torch.inverse(A_m + beta * KK_mm)
    
    K_m = K[:, active_m]
    KK_m = KK[:, active_m]
    KK_mm = KK[active_m, :][:, active_m]
    K_mt = Kt[active_m] 
    beta_KK_m = beta * KK_m
    Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)

    update_sigma    = config[0]=="1"
    delete_priority = config[1]=="1"
    add_priority    = config[2]=="1"
    alignment_test  = config[3]=="1"
    align_zero      = align_thr
    add_count = 0
    del_count = 0
    recomp_count = 0
    count = 0
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
        deltaML[recompute]  = ((delta_alpha * Q[recompute]**2) / (d_alpha_S) - torch.log(d_alpha_S)) /2
        
        # DELETION: if NEGATIVE factor and IN model
        idx = ~idx #active_factor <= 1e-12
        delete = active_m[idx]
        anyToDelete = len(delete) > 0
        if anyToDelete and alpha_m.shape[0] > 1: 	   # if there is only one basis function (M=1) (In practice, this latter event ought only to happen with the Gaussian
                                                        #    likelihood when initial noise is too high. In that case, a later beta
                                                        #    update should 'cure' this.)
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
            if verbose:
                print(f'{itr:3}, No positive action, m={active_m.shape[0]:3}')
            selected_action = torch.tensor(10)
            terminate = True

        elif selected_action==0 and ~anyToDelete:
            no_change_in_alpha = torch.abs(torch.log(alpha_new) - torch.log(alpha_m[j])) < tol
            
            if no_change_in_alpha:
                # print(selected_action)
                if verbose:
                    print(f'{itr:3}, No change in alpha, m={active_m.shape[0]:3}')
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
         
            #
            # Deletion: reinstate any previously deferred basis functions
            # resulting from this basis function
            # 
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
            Sigma_new       = Sigma_m - tmp @ s_j.T
            delta_mu        = -mu_m[j] * tmp
            mu_m            = mu_m + delta_mu.squeeze()
          
            S	= S + kappa * (beta_KK_m @ s_j).squeeze()**2
            Q	= Q - (beta_KK_m @ delta_mu).squeeze()
            update_required = True
        
        elif selected_action==-1:  #delete
            del_count += 1
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
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m
            update_required = True
        
        elif selected_action==1:  #add
            add_count += 1
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
            mu_m			= torch.cat([mu_m, torch.tensor([0.0]).to(device)], axis=0) + delta_mu
        
            mCi	            = beta * K_k_new - beta_KK_m @ tmp
            S   	        = S - mCi.pow(2) @ s_ii
            Q   	        = Q - mCi @ mu_i

            K_m             = K[:, active_m]
            KK_m            = KK[:, active_m]
            KK_mm           = KK[active_m, :][:, active_m]
            K_mt            = Kt[active_m]
            beta_KK_m       = beta * KK_m
            update_required = True
      
        # UPDATE STATISTICS
        if update_required:
            count += 1
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
            beta_KK_m = beta * KK_m

        min_index = torch.where(Gamma < 0.5)[0]
        if min_index.shape[0] >0:
            del_from_active = active_m[min_index]
            print(f'remove low Gamma: {Gamma[min_index].detach().cpu().numpy()} correspond to {del_from_active.detach().cpu().numpy()} data index')
            for j in min_index:
                del_count += 1
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
                
                # jPm	            = (beta_KK_m @ s_j).squeeze()
                # S	            = S + jPm.pow(2) / s_jj
                # Q	            = Q + jPm * mu_j / s_jj
                Sigma_m = Sigma_new
                K_m             = K[:, active_m]
                KK_m            = KK[:, active_m]
                KK_mm           = KK[active_m, :][:, active_m]
                K_mt            = Kt[active_m]
                # beta_KK_m       = beta * KK_m
                # update_required = True
                count += 1
                #quantity Gamma_i measures how well the corresponding parameter mu_i is determined by the data
            Gamma = 1 - alpha_m * torch.diag(Sigma_m)
            
            terminate = True

        #compute mu and beta
        if update_sigma and ((itr%5==0) or (itr <=10) or terminate):
            
            beta_old = beta
            y_      = K_m @ mu_m  
            e       = (targets.squeeze() - y_)
            beta	= (N - torch.sum(Gamma))/(e.T @ e)
            beta	= torch.min(torch.tensor([beta, 1e6/torch.var(targets)]).to(device))
            delta_beta	= torch.log(beta)-torch.log(beta_old)
            beta_KK_m       = beta * KK_m
            if torch.abs(delta_beta) > 1e-6:
                if verbose:
                    print(f'{itr:3}, update statistics after beta update')
                Sigma_m, mu_m, S, Q, s, q, logML, Gamma = Statistics(K_m, KK_m, KK_mm, Kt, K_mt, alpha_m, active_m, beta, targets, N)
                count = count + 1
                logMarginalLog.append(logML.item())
                terminate = False
            # else: 
            #     print(f'No change in beta')

        if terminate:
            # print(f'sigma2={1/beta:.4f}')
            # if verbose:
            if active_m.shape[0] < 3:
                print(f'Finished at {itr:3}, m= {active_m.shape[0]:3} sigma2= {1/beta:4.4f}')
            if count > 0:
                print(f'add: {add_count:3d} ({add_count/count:.1%}), delete: {del_count:3d} ({del_count/count:.1%}), recompute: {recomp_count:3d} ({recomp_count/count:.1%})')
            return active_m.cpu().numpy(), alpha_m, Gamma, beta, mu_m, Sigma_m, K_m 

        if ((itr+1)%1==0) and verbose:
            print(f'#{itr+1:3},     m={active_m.shape[0]}, selected_action= {selected_action.item():.0f}, logML= {logML.item()/N:.5f}, sigma2= {1/beta:.4f}')


    print(f'logML= {logML/N}\n{logMarginalLog}')

    return active_m.cpu().numpy(), alpha_m, Gamma, beta,  mu_m, Sigma_m, K_m  


def generate_data():
    np.random.seed(8)
    rng = np.random.RandomState(0)
    # Generate sample data
    X_org = 4 * np.pi * np.random.random(500) - 2 * np.pi
    X_org = np.sort(X_org)
    y_org = np.sinc(X_org)
    y_org += 0.25 * (0.5 - rng.rand(X_org.shape[0])) + 100  # add noise
    normalized = True
    if normalized:
        X = np.copy(X_org)
        X = np.copy(X_org) - np.mean(X_org, axis=0)
        X = X / np.std(X_org)
        y = np.copy(y_org) - np.mean(y_org)
        # y = y / np.std(y_org)
    else: 
        X = np.copy(X_org)
        y = np.copy(y_org)

    return X, y
    # X, y


def plot_result(rv, X_test, y_test, covar_module, mu_m, Sigma_m, K_m, active):
    
    # rv = torch.tensor(X_org[active])
    """Evaluate the RVR model at x."""
    K = covar_module(X_test, rv).evaluate()
   
    y_pred = K @ mu_m

    err_var = (1/beta.cpu()) + K @ Sigma_m @ K.T
    y_std = torch.sqrt(torch.diag(err_var))
    print(torch.nn.MSELoss()(y_pred, y_test))

    y_pred_ = y_pred.detach().numpy()
    y_std_ = y_std.detach().numpy()
    lw=2
    plt.scatter(X_test, y_test, marker=".", c="k", label="data")
    # plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

    plt.plot(X_test, y_pred_, color="darkorange", lw=lw, label="RVR")
    plt.fill_between(X_test, y_pred_ - y_std_, y_pred_ + y_std_, color="darkorange", alpha=0.4)
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
    tol = 1e-4
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
    # kernel_matrix = scipy.io.loadmat('./methods/K2.mat')['BASIS']
    # kernel_matrix = torch.from_numpy(kernel_matrix).to(dtype=torch.float64)
    # targets = scipy.io.loadmat('./methods/targets2.mat')['Targets']
    # targets = torch.from_numpy(targets).to(dtype=torch.float64).squeeze()
    # N = targets.shape[0]
    sigma = torch.var(targets)  #sigma^2
    sigma = torch.tensor([0.01])
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
    config = "1001"
    align_thr = 1e-3
    device= 'cuda'
    active_m, alpha_m, gamma_m, beta, mu_m, Sigma_m, K_m = Fast_RVM_regression_fullout(K.to(device), targets.to(device), beta.to(device), N,  config, align_thr, eps, tol)
    print(f'relevant index \n {active_m}')
    print(f'relevant alpha \n {alpha_m}')
    print(f'relevant Gamma \n {gamma_m}')

    index = np.argsort(active_m)
    ss = Scales[active_m]
    ss = ss[index]
    active_m = active_m[index]
    alpha_m = alpha_m[index].cpu() / ss.pow(2)
    mu_m = mu_m[index].cpu() / ss
    Sigma_m = Sigma_m[:, index]
    Sigma_m = Sigma_m[index, :].cpu()
    K_m = K_m[index].cpu()

    
    plot_result(inputs[active_m], inputs, targets, covar_module, mu_m, Sigma_m, K_m, active_m)














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
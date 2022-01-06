## Original packages

import torch

import numpy as np
import torch.nn.functional as F

from fast_pytorch_kmeans import KMeans as Fast_KMeans
from collections import namedtuple
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
## Our packages
import gpytorch
import os
from time import gmtime, strftime
import random
from colorama import Fore
from methods.Fast_RVM import Fast_RVM, posterior_mode

IP = namedtuple("inducing_points", "z_values index count alpha gamma mu U")

def rvm_ML(K_m, targets, alpha_m, mu_m, U):
        
        N = targets.shape[0]
        alpha_m = alpha_m.to(torch.float64)
        targets = targets.to(torch.float64)
        targets[targets==-1]= 0
        # targets_pseudo_linear	= 2 * targets - 1
        K_m = K_m.to(torch.float64)
        # LogOut	= (targets_pseudo_linear * 0.9 + 1) / 2
        # mu_m	=  K_m.pinverse() @ (torch.log(LogOut / (1 - LogOut))) #
        # with torch.no_grad():
            # mu_m = torch.linalg.lstsq(K_m, (torch.log(LogOut / (1 - LogOut)))).solution
            # mu_m = mu_m.to(torch.float64)
            # mu_m, U, beta, dataLikely, bad_Hess = posterior_mode(K_m, targets, alpha_m, mu_m, max_itr=25, device='cuda')
       
        K_mu_m = K_m @ mu_m
        y	= torch.sigmoid(K_mu_m)
        beta	= y * (1-y)
        #   Compute the Hessian
        beta_K_m	= (torch.diag(beta) @ K_m) 
        H			= (K_m.T @ beta_K_m + torch.diag(alpha_m))
        U, info =  torch.linalg.cholesky_ex(H, upper=True)
        dataLikely = (targets[targets==1].T @ torch.log(y[targets==1]+1e-12) + ((1-targets[targets==0]).T @ torch.log(1-y[targets==0]+1e-12)))
        logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        
        logML			= dataLikely - (mu_m**2) @ alpha_m /2  - logdetHOver2  #+ torch.sum(torch.log(alpha_m))/2
        return logML/N


def get_inducing_points(base_covar_module, inputs, targets, sparse_method, scale,
                        config='000', align_threshold='0', gamma=False, 
                        num_inducing_points=None, verbose=True, device='cuda'):

        
    IP_index = np.array([])
    acc = None
    mu_m = None
    U = None
    if sparse_method=='Random':
        if num_inducing_points is not None:
            num_IP = num_inducing_points
            # random selection of inducing points
            idx_zero = torch.where((targets==-1) | (targets==0))[0]
            idx_one = torch.where(targets==-1)[0]
            inducing_points_zero = list(np.random.choice(idx_zero.cpu().numpy(), replace=False, size=num_inducing_points//2))
            inducing_points_one = list(np.random.choice(idx_one.cpu().numpy(), replace=False, size=num_inducing_points//2))

            IP_index = inducing_points_one + inducing_points_zero
            inducing_points = inputs[IP_index, :]
            alpha = None
            gamma = None

    elif sparse_method=='KMeans':
        if num_inducing_points is not None:
            num_IP = num_inducing_points
            # self.kmeans_clustering = KMeans(n_clusters=num_IP, init='k-means++',  n_init=10, max_iter=1000).fit(inputs.cpu().numpy())
            # inducing_points = self.kmeans_clustering.cluster_centers_
            # inducing_points = torch.from_numpy(inducing_points).to(torch.float)

            kmeans_clustering = Fast_KMeans(n_clusters=num_IP, max_iter=1000)
            kmeans_clustering.fit(inputs.cuda())
            inducing_points = kmeans_clustering.centroids
            # print(inducing_points.shape[0])

            IP_index = None
            alpha = None
            gamma = None


    elif sparse_method=='FRVM':
        # with sigma and updating sigma converges to more sparse solution
        N   = inputs.shape[0]
        tol = 1e-6
        eps = torch.finfo(torch.float32).eps
        max_itr = 2000
        
        kernel_matrix = base_covar_module(inputs).evaluate()
        # normalize kernel
        scales = torch.ones(kernel_matrix.shape[1]).to(device)
        if scale:
            scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
            # print(f'scale: {Scales}')
            scales[scales==0] = 1
            kernel_matrix = kernel_matrix / scales

        kernel_matrix = kernel_matrix.to(torch.float64)
        # targets[targets==-1]= 0
        target = targets.clone().to(torch.float64)
        active, alpha, gamma, beta, mu_m, U = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]
        ss = scales[active]
        alpha = alpha[index] #/ ss**2
        mu_m = mu_m[index] #/ss
        num_IP = active.shape[0]
        IP_index = active

        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_r = mu_m / ss
            mu_r = mu_r.to(torch.float)
            y_pred = K @ mu_r
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).to(int)
            
            acc = (torch.sum(y_pred==target) / N).item()  * 100 # targets is zero and one (after FRVM)
            if verbose:
                print(f'FRVM ACC on IPs: {(acc):.2f}%')
            
            # self.frvm_acc.append(acc.item())
    elif sparse_method=='augmFRVM':
         
        # with sigma and updating sigma converges to more sparse solution
        N   = inputs.shape[0]
        tol = 1e-6
        eps = torch.finfo(torch.float32).eps
        max_itr = 1000
        
        kernel_matrix = base_covar_module(inputs).evaluate()
        # normalize kernel
        scales = torch.ones(kernel_matrix.shape[1]).to(device)
        if scale:
            scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
            # print(f'scale: {Scales}')
            scales[scales==0] = 1
            kernel_matrix = kernel_matrix / scales

        kernel_matrix = kernel_matrix.to(torch.float64)
        # targets[targets==-1]= 0
        target = targets.clone().to(torch.float64)
        active, alpha, gamma, beta, mu_m, U = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]

        ss = scales[active]
        alpha = alpha[index] #/ ss**2
        mu_m = mu_m[index] #/ss
        num_IP = active.shape[0]
        IP_index = active
        y_ip = target[active]
        ones = y_ip==1
        zeros = y_ip==0
        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_r = mu_m / ss
            mu_r = mu_r.to(torch.float)
            y_pred = K @ mu_r
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).to(int)
            
            acc = (torch.sum(y_pred==target) / N).item()  * 100 # targets is zero and one (after FRVM)
            if verbose:
                print(f'FRVM [augm] ACC on IPs: {(acc):.2f}%  class one #{ones.sum()}, zero #{zeros.sum()}')

        if num_inducing_points is not None:
            if num_IP < num_inducing_points:
                num_IP = num_inducing_points + num_IP
                # random selection of inducing points
                idx_zero = torch.where((targets==-1) | (targets==0))[0]
                idx_one = torch.where(targets==-1)[0]
                inducing_points_zero = list(np.random.choice(idx_zero.cpu().numpy(), replace=False, size=num_inducing_points//2))
                inducing_points_one = list(np.random.choice(idx_one.cpu().numpy(), replace=False, size=num_inducing_points//2))

                IP_index_rand = inducing_points_one + inducing_points_zero
                for ix in IP_index:
                    if ix in IP_index_rand:
                        IP_index_rand.remove(ix)
        
                random_inducing_points = inputs[IP_index_rand, :]
                IP_index = np.concatenate([IP_index, IP_index_rand])
                inducing_points = torch.cat([inducing_points, random_inducing_points])
                alpha = None
                gamma = None
                print(f'   augmented IP, m={num_IP:3}')
    
    elif sparse_method=='constFRVM':
         
        # with sigma and updating sigma converges to more sparse solution
        N   = inputs.shape[0]
        tol = 1e-6
        eps = torch.finfo(torch.float32).eps
        max_itr = 1000
        
        kernel_matrix = base_covar_module(inputs).evaluate()
        # normalize kernel
        scales = torch.ones(kernel_matrix.shape[1]).to(device)
        if scale:
            scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
            # print(f'scale: {Scales}')
            scales[scales==0] = 1
            kernel_matrix = kernel_matrix / scales

        kernel_matrix = kernel_matrix.to(torch.float64)
        # targets[targets==-1]= 0
        target = targets.clone().to(torch.float64)
        active, alpha, gamma, beta, mu_m, U = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]
        ss = scales[active]
        alpha = alpha[index] #/ ss**2
        mu_m = mu_m[index] #/ss
        num_IP = active.shape[0]
        IP_index = active
        y_ip = target[active]
        ones = y_ip==1
        zeros = y_ip==0
        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_r = mu_m / ss
            mu_r = mu_r.to(torch.float)
            y_pred = K @ mu_r
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).to(int)
            
            acc = (torch.sum(y_pred==target) / N).item()  * 100 # targets is zero and one (after FRVM)
            if verbose:
                print(f'FRVM [const] ACC on IPs: {(acc):.2f}%  class one #{ones.sum()}, zero #{zeros.sum()}')
        if num_inducing_points is not None:
            if num_IP < num_inducing_points:
                num_IP = num_inducing_points + num_IP
                # random selection of inducing points
                idx_zero = torch.where((targets==-1) | (targets==0))[0]
                idx_one = torch.where(targets==-1)[0]
                inducing_points_zero = list(np.random.choice(idx_zero.cpu().numpy(), replace=False, size=num_inducing_points//2))
                inducing_points_one = list(np.random.choice(idx_one.cpu().numpy(), replace=False, size=num_inducing_points//2))

                IP_index_rand = inducing_points_one + inducing_points_zero
                for ix in IP_index:
                    if ix in IP_index_rand:
                        IP_index_rand.remove(ix)
        
                random_inducing_points = inputs[IP_index_rand, :]
                IP_index = np.concatenate([IP_index, IP_index_rand])
                inducing_points = torch.cat([inducing_points, random_inducing_points])
                alpha = None
                gamma = None
                print(f'   augmented IP, m={num_IP:3}')
    
    else:
        print(f'No method')

    return IP(inducing_points, IP_index, num_IP, alpha, gamma, mu_m, U), acc
  
 
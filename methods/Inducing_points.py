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
from methods.Fast_RVM import Fast_RVM

IP = namedtuple("inducing_points", "z_values index count alpha gamma")

def get_inducing_points(base_covar_module, inputs, targets, sparse_method, scale,
                        config='000', align_threshold='0', gamma=False, 
                        num_inducing_points=10, verbose=True, device='cuda'):

        
    IP_index = np.array([])
    acc = None
    if sparse_method=='Random':
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
        active, alpha, gamma, beta, mu_m = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]
        ss = scales[index]
        alpha = alpha[index] / ss
        num_IP = active.shape[0]
        IP_index = active

        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_m = mu_m[index] / ss
            mu_m = mu_m.to(torch.float)
            y_pred = K @ mu_m
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
        active, alpha, gamma, beta, mu_m = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]
        ss = scales[index]
        alpha = alpha[index] / ss
        num_IP = active.shape[0]
        IP_index = active
        y_ip = target[active]
        ones = y_ip==1
        zeros = y_ip==0
        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_m = mu_m[index] / ss
            mu_m = mu_m.to(torch.float)
            y_pred = K @ mu_m
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).to(int)
            
            acc = (torch.sum(y_pred==target) / N).item()  * 100 # targets is zero and one (after FRVM)
            if verbose:
                print(f'FRVM [augm] ACC on IPs: {(acc):.2f}%  class one #{ones.sum()}, zero #{zeros.sum()}')

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
        active, alpha, gamma, beta, mu_m = Fast_RVM(kernel_matrix, target, N, config, align_threshold, gamma,
                                                eps, tol, max_itr, device, verbose)

        index = np.argsort(active)
        active = active[index]
        inducing_points = inputs[active]
        gamma = gamma[index]
        ss = scales[index]
        alpha = alpha[index] / ss
        num_IP = active.shape[0]
        IP_index = active
        y_ip = target[active]
        ones = y_ip==1
        zeros = y_ip==0
        if True:
            ss = scales[index]
            K = base_covar_module(inputs, inducing_points).evaluate()
            mu_m = mu_m[index] / ss
            mu_m = mu_m.to(torch.float)
            y_pred = K @ mu_m
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).to(int)
            
            acc = (torch.sum(y_pred==target) / N).item()  * 100 # targets is zero and one (after FRVM)
            if verbose:
                print(f'FRVM [const] ACC on IPs: {(acc):.2f}%  class one #{ones.sum()}, zero #{zeros.sum()}')

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

    return IP(inducing_points, IP_index, num_IP, alpha, gamma), acc
  
 
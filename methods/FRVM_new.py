## Original packages
# from torch._C import ShortTensor


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
# matplotlib.use('Agg')
from colorama import Fore
from sklearn.manifold import TSNE
import torchvision.transforms as transforms

from time import gmtime, strftime
import torch.nn.functional as F
import random
## Our packages
import gpytorch

from data.qmul_loader import get_batch, train_people, val_people, test_people, get_unnormalized_label
from configs import kernel_type
from collections import namedtuple
import torch.optim

class FRVM_new(nn.Module):
    def __init__(self):
        super(FRVM_new, self).__init__()

        self.device = 'cuda'
      
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        
    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(self.num_induce_points, 2916).cuda() #2916: size of feature z
        # if(train_x is None): train_x=torch.rand(19, 3, 100, 100).cuda()
        if(train_y is None): train_y=torch.ones(self.num_induce_points).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.1
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel='rbf', induce_point=train_x)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll
    

    def generate_data(self):
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
        y_org = y_org.astype(np.float)
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
   
    def get_inducing_points(self, optimizer):
        
        X,y = self.generate_data()
        self.model.train()
        self.likelihood.train()
        self.model.set_train_data(inputs=X, targets=y, strict=False)
        for i in range(5):
            print(i)
            out = self.model(*self.model.train_inputs)
            mll = -self.mll(out, self.model.train_targets)
            mll.backward()
            optimizer.step()
      
model = FRVM_new()
optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.01}])
model.get_inducing_points(optimizer)


from gpytorch.lazy import DiagLazyTensor, LowRankRootAddedDiagLazyTensor, LowRankRootLazyTensor, MatmulLazyTensor, delazify
import copy
import math 
from gpytorch.distributions  import MultivariateNormal
from gpytorch.mlls import InducingPointKernelAddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky

class SparseKernel(gpytorch.kernels.InducingPointKernel):

    def __init__(self, base_kernel, inducing_points, likelihood, N, active_dims=None):
        
        super(SparseKernel, self).__init__(base_kernel, inducing_points, likelihood, active_dims=active_dims)
        self.register_parameter(name="A", parameter=torch.nn.Parameter(torch.eye(N)))

    @property
    def _A_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self.A, upper=True)
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            # if not self.training:
            #     self._cached_kernel_inv_root = res
            return res
    
    
    def _get_covariance(self, x1, x2):
        k_ux1 = delazify(self.base_kernel(x1, self.inducing_points))
        if torch.equal(x1, x2):
            covar = LowRankRootLazyTensor(k_ux1.matmul(self._A_inv_root))
            covar = LowRankRootAddedDiagLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)

            # Diagonal correction for predictive posterior
            # if not self.training:
            #     correction = (self.base_kernel(x1, x2, diag=True) - covar.diag()).clamp(0, math.inf)
            #     covar = LowRankRootAddedDiagLazyTensor(covar, DiagLazyTensor(correction))
        else:
            k_ux2 = delazify(self.base_kernel(x2, self.inducing_points))
            covar = MatmulLazyTensor(
                k_ux1.matmul(self._A_inv_root), k_ux2.matmul(self._A_inv_root).transpose(-1, -2)
            )
            covar = LowRankRootAddedDiagLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)
            # S = torch.inv((1/self.likelihood.noise) * k_ux1.transpose(-1, -2).matmul(k_ux1) + (1/self.A).pow(2))
            # k_s = k_ux1.matmul(S)
            # covar = MatmulLazyTensor(
            #     k_s.transpose(-1, -2), k_ux2
            # )

        return covar
    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
            zero_mean = torch.zeros_like(x1.select(-1, 0))
            new_added_loss_term = InducingPointKernelAddedLossTerm(
                MultivariateNormal(zero_mean, self._covar_diag(x1)),
                MultivariateNormal(zero_mean, covar),
                self.likelihood,
            )
            self.update_added_loss_term("inducing_point_loss_term", new_added_loss_term)

        if diag:
            return covar.diag()
        else:
            return covar


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear', induce_point=None):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.base_covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=16, ard_num_dims=2916)
            self.base_covar_module.initialize_from_data_empspect(train_x, train_y)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")
        
        self.covar_module = SparseKernel(self.base_covar_module, inducing_points=induce_point , likelihood=likelihood, N=train_y.shape[0])
    
    def forward(self, x):
        mean_x  = self.mean_module(x)
        # covar_x = self.base_covar_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

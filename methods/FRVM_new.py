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
# from data.qmul_loader import get_batch, train_people, val_people, test_people, get_unnormalized_label
# from configs import kernel_type
from collections import namedtuple
import torch.optim

from gpytorch.lazy import DiagLazyTensor, LowRankRootAddedDiagLazyTensor, LowRankRootLazyTensor, MatmulLazyTensor, delazify, SumLazyTensor
import copy
import math 
from gpytorch.distributions  import MultivariateNormal
from gpytorch.mlls import InducingPointKernelAddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky

# torch.manual_seed(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class SparseKernel(gpytorch.kernels.Kernel):

    def __init__(self, base_kernel, y, likelihood, N, active_dims=None):
        
        super(SparseKernel, self).__init__()
        self.base_kernel = base_kernel
        self.likelihood = likelihood
        B = y * (1-y)
        self.B = torch.eye(B.shape[0]).cuda() * B
        A_init = torch.randn(N) # * 0.1
        # A_init[0] = A_init[0] * -1
        self.register_parameter(name="A", parameter=torch.nn.Parameter(A_init))

    @property
    def _A_inv_root(self):
        
        clip_A = self.A.clone()
        mask = clip_A < 0
        clip_A[mask] = 1e-12
        # mask = clip_A > 1
        # clip_A[mask] = 1
        A = torch.eye(self.A.shape[0]).cuda() * clip_A #self.A.clamp(min=1e-12)
        # chol = psd_safe_cholesky(A, upper=True)
        # eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
        # inv_root = torch.triangular_solve(eye, chol)[0]

        res = A  #torch.linalg.inv(A)
        # if not self.training:
        #     self._cached_kernel_inv_root = res
        return res
    
    
    def _get_covariance(self, x1, x2):
        k_ux1 = delazify(self.base_kernel(x1, x1))
        if torch.equal(x1, x2):
            # A = torch.eye(self.A.shape[0]).cuda() * self.A
            # A_inv = torch.linalg.inv(A)
            # covar = LowRankRootLazyTensor(k_ux1.matmul(self._A_inv_root))
            covar =  MatmulLazyTensor(k_ux1.matmul(self._A_inv_root), k_ux1.transpose(-1, -2))

            # covar = MatmulLazyTensor(
            #     k_ux1.matmul(A_inv), k_ux1.transpose(-1, -2)
            # )
            # covar = SumLazyTensor(self.B,  covar)
            covar = SumLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)
            # psd_safe_cholesky(delazify(covar), True)
        else:
            k_ux2 = delazify(self.base_kernel(x2, x1))
            covar = MatmulLazyTensor(
                k_ux1.matmul(self._A_inv_root), k_ux2.matmul(self._A_inv_root).transpose(-1, -2)
            )
            covar = SumLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)
            # S = torch.inv((1/self.likelihood.noise) * k_ux1.transpose(-1, -2).matmul(k_ux1) + (1/self.A).pow(2))
            # k_s = k_ux1.matmul(S)
            # covar = MatmulLazyTensor(
            #     k_s.transpose(-1, -2), k_ux2
            # )

        return covar
   
    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)
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
        
        self.covar_module = SparseKernel(self.base_covar_module, y=train_y , likelihood=likelihood, N=train_y.shape[0])
    
    def forward(self, x):
        mean_x  = self.mean_module(x)
        # covar_x = self.base_covar_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FRVM_new(nn.Module):
    def __init__(self, N):
        super(FRVM_new, self).__init__()

        self.device = 'cuda'
        self.num_induce_points = 10
        self.get_model_likelihood_mll(N) #Init model, likelihood, and mll
        
    def get_model_likelihood_mll(self, N, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(N, 2916).cuda() #2916: size of feature z
        # if(train_x is None): train_x=torch.rand(19, 3, 100, 100).cuda()
        if(train_y is None): train_y=torch.ones(N).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.001
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel='rbf', induce_point=train_x)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll
    

    def generate_data(self, N):
        np.random.seed(8)
        rng = np.random.RandomState(0)
        # Generate sample data
        n = N
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

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        # X, y
   
    def get_inducing_points(self, optimizer, N):
        
        X,y = self.generate_data(N)
        X = X.to('cuda')
        y = y.to('cuda')
        y[y==0]= -1
        self.model.train()
        self.likelihood.train()
        self.model.set_train_data(inputs=X, targets=y, strict=False)
        # self.model.covar_module.inducing_points = nn.Parameter(X[:self.num_induce_points], requires_grad=False)
        mll_list = []
        min_num_sv = X.shape[0]
        mll_prev = 10e7
        for i in range(5000):
            
            out = self.model(*self.model.train_inputs)
            mll = -self.mll(out, self.model.train_targets)
            mll.backward()
            optimizer.step()
            # if i%50==0:
            #     print(f'{i:3}, {self.model.covar_module.A}')
            mll_list.append(mll.item())
            num_near_zero = ((self.model.covar_module.A > 0) & (self.model.covar_module.A < 1)).sum()
            if num_near_zero == 0:
                num_sv = (self.model.covar_module.A > 0).sum()
                if num_sv < min_num_sv: min_num_sv = num_sv
                print(f'Finished at itr {i:3}, num_sv: {num_sv}, mll: {mll.item():6.6f} noise: {self.likelihood.noise.item()}')
                #break
                if abs(mll_prev - mll) < 0.000001:
                    num_sv = (self.model.covar_module.A > 0).sum()
                    print(f'itr {i:3}, num_sv: {num_sv}')
                    # break
            mll_prev = mll
        plt.plot(mll_list)
        # plt.show()
        num_sv = (self.model.covar_module.A > 0).sum()
        print(f'num_sv: {num_sv}')

N = 500
model = FRVM_new(N)
optimizer = torch.optim.Adam([{'params': model.model.covar_module.A, 'lr': 0.001},
                                {'params': model.likelihood.parameters(), 'lr': 0.001}])
# optimizer = torch.optim.Adam([{'params': model.model.covar_module.A, 'lr': 0.0001},
#                             {'params': model.model.mean_module.parameters(), 'lr': 0.001}])
model.get_inducing_points(optimizer, N)



class SparseKernel2(gpytorch.kernels.Kernel):

    def __init__(self, base_kernel, inducing_points, likelihood, N, active_dims=None):
        
        super(SparseKernel, self).__init__()
        self.base_kernel = base_kernel
        self.likelihood = likelihood
        self.register_parameter(name="A", parameter=torch.nn.Parameter(torch.ones(N)))

    @property
    def _A_inv_root(self):
      
        A = torch.eye(self.A.shape[0]).cuda() * self.A
        chol = psd_safe_cholesky(A, upper=True)
        eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
        inv_root = torch.triangular_solve(eye, chol)[0]

        res = inv_root  #torch.linalg.inv(A)
        # if not self.training:
        #     self._cached_kernel_inv_root = res
        return res
    
    
    def _get_covariance(self, x1, x2):
        k_ux1 = delazify(self.base_kernel(x1, x1))
        if torch.equal(x1, x2):
            # A = torch.eye(self.A.shape[0]).cuda() * self.A
            # A_inv = torch.linalg.inv(A)
            covar = LowRankRootLazyTensor(k_ux1.matmul(self._A_inv_root))
            

            # covar = MatmulLazyTensor(
            #     k_ux1.matmul(A_inv), k_ux1.transpose(-1, -2)
            # )
            covar = SumLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)
        else:
            k_ux2 = delazify(self.base_kernel(x2, x1))
            covar = MatmulLazyTensor(
                k_ux1.matmul(self._A_inv_root), k_ux2.matmul(self._A_inv_root).transpose(-1, -2)
            )
            covar = SumLazyTensor(torch.eye(covar.shape[0]).to('cuda') * (self.likelihood.noise),  covar)
            # S = torch.inv((1/self.likelihood.noise) * k_ux1.transpose(-1, -2).matmul(k_ux1) + (1/self.A).pow(2))
            # k_s = k_ux1.matmul(S)
            # covar = MatmulLazyTensor(
            #     k_s.transpose(-1, -2), k_ux2
            # )

        return covar
    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)

  

        if diag:
            return covar.diag()
        else:
            return covar

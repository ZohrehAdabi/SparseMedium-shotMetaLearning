## Original packages
from sklearn.base import RegressorMixin
from torchvision.transforms.transforms import ColorJitter
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

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
from configs import kernel_type
from methods.Fast_RVM import Fast_RVM
from methods.Inducing_points import get_inducing_points, rvm_ML, get_inducing_points_regression, rvm_ML_regression, rvm_ML_regression_full, rvm_ML_full
#Check if tensorboardx is installed
try:
    #tensorboard --logdir=./Sparse_DKT_binary_RVM_CUB_log/ --host localhost --port 8090
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

## Training CMD
#ATTENTION: to test each method use exaclty the same command but replace 'train.py' with 'test.py'
# Omniglot->EMNIST without data augmentation
#python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1
#python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5
# CUB + data augmentation
#python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --train_aug
#python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug
IP = namedtuple("inducing_points", "z_values index count alpha gamma x y i_idx j_idx")
class Sparse_DKT_binary_RVM(MetaTemplate):
    def __init__(self, model_func, kernel_type, n_way, n_support, sparse_method='FRVM', add_rvm_mll=False, add_rvm_mll_one=False, lambda_rvm=0.1, 
                        regression=False, rvm_mll_only=False, num_inducing_points=None, normalize=False, 
                        scale=False, config="010", align_threshold=1e-3, gamma=False, dirichlet=False):
        super(Sparse_DKT_binary_RVM, self).__init__(model_func, n_way, n_support)

        self.num_inducing_points = num_inducing_points
        self.sparse_method = sparse_method
        self.add_rvm_mll = add_rvm_mll
        self.add_rvm_mll_one = add_rvm_mll_one
        self.lambda_rvm = lambda_rvm
        self.regression = regression
        self.rvm_mll_only = rvm_mll_only
        self.config = config
        self.align_threshold = align_threshold
        self.gamma = gamma
        self.dirichlet = dirichlet
        self.scale = scale
        self.device ='cuda'
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        if(kernel_type=="cossim"):
            self.normalize=True
        elif(kernel_type=="bncossim"):
            self.normalize=True
            latent_size = np.prod(self.feature_extractor.final_feat_dim)
            self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        else:
            self.normalize=normalize

    def init_summary(self, id, dataset):
        self.id = id
        if(IS_TBX_INSTALLED):
            path = f'./Sparse_DKT_binary_RVM_{dataset}_log'
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir(path):
                os.makedirs(path)
            if dataset in ['miniImagenet', 'CUB']:
                writer_path = path+ '/' + id[33:]
            elif dataset=="omniglot":
                writer_path = path + '/' + id[34:]
            else:
                writer_path = path+ '/' + id  #+'_old'#+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(100, 64).cuda()
        if(train_y is None): train_y=torch.ones(100).cuda()


        if self.dirichlet:
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y.long(), learn_additional_noise=False)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
                
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, dirichlet=self.dirichlet,
                                    kernel=kernel_type, inducing_points=train_x)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def _reset_likelihood(self, debug=False):
        for param in self.likelihood.parameters():
           param.data.normal_(0.0, 0.01)

    def _print_weights(self):
        for k, v in self.feature_extractor.state_dict().items():
            print("Layer {}".format(k))
            print(v)

    def _reset_variational(self):
        mean_init = torch.zeros(128) #num_inducing_points
        covar_init = torch.eye(128, 128) #num_inducing_points
        mean_init = mean_init.repeat(64, 1) #batch_shape
        covar_init = covar_init.repeat(64, 1, 1) #batch_shape
        for idx, param in enumerate(self.gp_layer.variational_parameters()):
            if(idx==0): param.data.copy_(mean_init) #"variational_mean"
            elif(idx==1): param.data.copy_(covar_init) #"chol_variational_covar"
            else: raise ValueError('[ERROR] DKT the variational_parameters at index>1 should not exist!')

    def _reset_parameters(self):
        if(self.leghtscale_list is None):
            self.leghtscale_list = list()
            self.noise_list = list()
            self.outputscale_list = list()
            for idx, single_model in enumerate(self.model.models):
                self.leghtscale_list.append(single_model.base_covar_module.base_kernel.lengthscale.clone().detach())
                self.noise_list.append(single_model.likelihood.noise.clone().detach())
                self.outputscale_list.append(single_model.base_covar_module.outputscale.clone().detach())
        else:
            for idx, single_model in enumerate(self.model.models):
                single_model.base_covar_module.base_kernel.lengthscale=self.leghtscale_list[idx].clone().detach()#.requires_grad_(True)
                single_model.likelihood.noise=self.noise_list[idx].clone().detach()
                single_model.base_covar_module.outputscale=self.outputscale_list[idx].clone().detach()

    def pred_result(self, mean):
        
        max_pred, idx = torch.max(mean, axis=0)
        index = ~idx.to(bool)
        max_pred[index] = -np.inf
        return max_pred

    
    def train_loop(self, epoch, train_loader, optimizer, print_freq=5):
        # if self.dirichlet:
        #     optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
        #                               {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])
        # else:
        #     optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
        #         #                              {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])
        
        l = self.lambda_rvm
        self.frvm_acc = []
        
        for i, (x,_) in enumerate(train_loader):
    
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).cuda()
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).cuda())
            x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
            y_support = np.repeat(range(self.n_way), self.n_support)
            x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
            y_query = np.repeat(range(self.n_way), self.n_query)
            x_train = x_all
            y_train = y_all

            
            samples_per_model = int(len(y_train) / self.n_way) #25 / 5 = 5
            target = torch.ones(len(y_train), dtype=torch.float32) * 1 
            # target = torch.zeros(len(y_train), dtype=torch.float32) 
            start_index = 0
            stop_index = start_index+samples_per_model
            target[start_index:stop_index] = -1.0
            target = target.to(self.device)

            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            # z_train_norm = F.normalize(z_train, p=2, dim=1)
            # train_list = [z_train]*self.n_way
            lenghtscale = 0.0
            noise = 0.0
            outputscale = 0.0

            if self.dirichlet:
                target[target==-1] = 0
                self.model.likelihood.targets = target.long()
                sigma2_labels, transformed_targets, num_classes = self.model.likelihood._prepare_targets(self.model.likelihood.targets, 
                                        alpha_epsilon=self.model.likelihood.alpha_epsilon, dtype=torch.float)
                self.model.likelihood.transformed_targets = transformed_targets.transpose(-2, -1)
                self.model.likelihood.noise.data = sigma2_labels
                self.model.set_train_data(inputs=z_train, targets=self.model.likelihood.transformed_targets, strict=False)
            else: 
                self.model.set_train_data(inputs=z_train, targets=target, strict=False)

            with torch.no_grad():
                if self.sparse_method=="constFRVM":
                    z_train_rvm = self.constant_feature_extractor(x_train)
                    inducing_points, frvm_acc = get_inducing_points(self.constant_model.base_covar_module, #.base_kernel,
                                                            z_train_rvm, target, sparse_method=self.sparse_method, scale=self.scale,
                                                            config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                            num_inducing_points=self.num_inducing_points, verbose=True, device=self.device)
                else:
                    if self.regression:
                        self.config = '0' + self.config
                        inducing_points, frvm_acc = get_inducing_points_regression(self.model.base_covar_module, #.base_kernel,
                                                                z_train, target, sparse_method=self.sparse_method, scale=self.scale, beta=torch.tensor(10.0),
                                                                config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                                num_inducing_points=self.num_inducing_points, verbose=True, task_id=i, device=self.device, classification=True)
                    else:
                        inducing_points, frvm_acc = get_inducing_points(self.model.base_covar_module, #.base_kernel,
                                                                z_train, target, sparse_method=self.sparse_method, scale=self.scale,
                                                                config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                                num_inducing_points=self.num_inducing_points, verbose=True, task_id=i, device=self.device)
                self.frvm_acc.append(frvm_acc)
                
            ip_index = inducing_points.index
            ip_values = z_train[ip_index]
            # ip_values = z_train[inducing_points.index].cuda()
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=True)
   
            
            alpha_m = inducing_points.alpha
            mu_m = inducing_points.mu
            U = inducing_points.U
            scales = inducing_points.scale
            K_m = self.model.base_covar_module(z_train, ip_values).evaluate()
            K_m = K_m.to(torch.float64)
            # scales	= torch.sqrt(torch.sum(K_m**2, axis=0))
            # K = K / scales
            mu_m = mu_m /scales

            if self.rvm_mll_only:
                if self.regression:
                    rvm_mll = rvm_ML_regression_full(K_m, target, alpha_m, mu_m)
                else:
                    rvm_mll = rvm_ML_full(K_m, target, alpha_m, mu_m, U)
            else:
                if self.regression:
                    rvm_mll, _ = rvm_ML_regression(K_m, target, alpha_m, mu_m)
                else:
                    rvm_mll = rvm_ML(K_m, target, alpha_m, mu_m, U)

            if(self.model.covar_module.base_kernel.lengthscale is not None):
                lenghtscale+=self.model.base_covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
            noise+=self.model.likelihood.noise.cpu().detach().numpy().squeeze().mean()
            if(self.model.base_covar_module.outputscale is not None): #Sparse DKT Linear
                outputscale+=self.model.base_covar_module.outputscale.cpu().detach().numpy().squeeze()
            

            ## Optimize
            optimizer.zero_grad()
           
            output = self.model(*self.model.train_inputs)
            if self.dirichlet:
                transformed_targets = self.model.likelihood.transformed_targets
                loss = -self.mll(output, transformed_targets).sum()
            else:
                mll = self.mll(output, self.model.train_targets)
                if self.add_rvm_mll_one:
                    #  
                    loss = -(1-l) * mll  - l * rvm_mll 
                if self.add_rvm_mll:
                    loss = - mll  - l * rvm_mll
                 
                elif self.rvm_mll_only:
                    loss = -rvm_mll
                else:
                    loss = -mll
                    

            loss.backward()
            optimizer.step()

            rvm_mll = rvm_mll.item()
            mll = mll.item()
            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('Loss', loss, self.iteration)
            if(self.writer is not None): self.writer.add_scalar('MLL', -mll, self.iteration)
            if(self.writer is not None): self.writer.add_scalar('RVM MLL', -rvm_mll, self.iteration)

            #Eval on the query (validation set)
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                self.feature_extractor.eval()
                z_support = self.feature_extractor(x_support).detach()
                if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                ## z_support_list = [z_support]*len(y_support)

                # if self.dirichlet:
                #     prediction = self.likelihood(self.model(z_support)) #return 20 MultiGaussian Distributions
                # else:
                #     prediction = self.likelihood(self.model(z_support) #return 20 MultiGaussian Distributions
               
                # if self.dirichlet:
                #     
                #    max_pred = (prediction.mean[0] > prediction.mean[1]).to(int)
                #    y_pred = max_pred.cpu().detach().numpy()
                # else: 
                #    pred = torch.sigmoid(prediction.mean)
                #    y_pred = (pred < 0.5).to(int)
                #    y_pred = y_pred.cpu().detach().numpy()

                # accuracy_support = (np.sum(y_pred==y_support) / float(len(y_support))) * 100.0
                # if(self.writer is not None): self.writer.add_scalar('GP_support_accuracy', accuracy_support, self.iteration)
                z_query = self.feature_extractor.forward(x_query).detach()
                if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                
                
                if self.dirichlet:
                    prediction = self.likelihood(self.model(z_query)) 
                else:
                    prediction = self.likelihood(self.model(z_query))
               
                if self.dirichlet:
                    
                   max_pred = (prediction.mean[0] > prediction.mean[1]).to(int)
                   y_pred = max_pred.cpu().detach().numpy()
                else: 
                   pred = torch.sigmoid(prediction.mean)
                   y_pred = (pred > 0.5).to(int) #0,1 --- 1,-1 ==> change it to right one: 0,1--- -1,1
                   y_pred = y_pred.cpu().detach().numpy()

                accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
                if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query, self.iteration)

            if i % print_freq==0:
                if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                if self.dirichlet:
                    print(Fore.LIGHTRED_EX,'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} || Loss {:f} | MLL {:f} | RVM ML {:f}| Supp. acc {:f} | Query acc {:f}'.format(epoch, i, len(train_loader),
                        outputscale, lenghtscale,  loss.item(), -mll, -rvm_mll,  0, accuracy_query), Fore.RESET) #accuracy_support
                else:
                    print(Fore.LIGHTRED_EX,'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} | Noise {:f} | Loss {:f} | MLL {:f} | RVM ML {:f} | Supp. acc {:f} | Query acc {:f}'.format(epoch, i, len(train_loader),
                        outputscale, lenghtscale, noise, loss.item(), -mll, -rvm_mll,0, accuracy_query), Fore.RESET)

    def get_inducing_points(self, base_covar_module, inputs, targets, verbose=True):

        
        IP_index = np.array([])
        if not self.fast_rvm:
            num_IP = self.num_inducing_points
            
            # self.kmeans_clustering = KMeans(n_clusters=num_IP, init='k-means++',  n_init=10, max_iter=1000).fit(inputs.cpu().numpy())
            # inducing_points = self.kmeans_clustering.cluster_centers_
            # inducing_points = torch.from_numpy(inducing_points).to(torch.float)

            self.kmeans_clustering = Fast_KMeans(n_clusters=num_IP, max_iter=1000)
            self.kmeans_clustering.fit(inputs.cuda())
            inducing_points = self.kmeans_clustering.centroids
            # print(inducing_points.shape[0])

            IP_index = None
            alpha = None
            gamma = None


        else:
            # with sigma and updating sigma converges to more sparse solution
            N   = inputs.shape[0]
            tol = 1e-6
            eps = torch.finfo(torch.float32).eps
            max_itr = 1000
            
            scale = self.scale
            # X = inputs.clone()
            # m = X.mean(axis=0)
            # s = X.std(axis=0)
            # X = (X- m) / s 
            kernel_matrix = base_covar_module(inputs).evaluate()
            # normalize kernel
            scales = torch.ones(kernel_matrix.shape[1]).to(self.device)
            if scale:
                scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
                # print(f'scale: {Scales}')
                scales[scales==0] = 1
                kernel_matrix = kernel_matrix / scales

            kernel_matrix = kernel_matrix.to(torch.float64)
            # targets[targets==-1]= 0
            target = targets.clone().to(torch.float64)
            active, alpha, gamma, beta, mu_m = Fast_RVM(kernel_matrix, target, N, self.config, self.align_threshold, self.gamma,
                                                    eps, tol, max_itr, self.device, verbose)

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
                
                acc = (torch.sum(y_pred==target) / N) * 100 # targets is zero and one (after FRVM)
                if verbose:
                    print(f'FRVM ACC: {(acc):.2f}%')
                
                self.frvm_acc.append(acc.item())

        return IP(inducing_points, IP_index, num_IP, alpha, gamma, None, None, None, None)
  
    def correct(self, x, i=0, N=0, laplace=False):
        self.model.eval()
        self.likelihood.eval()
        self.feature_extractor.eval()
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        ## Laplace approximation of the posterior
        if(laplace):
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF, Matern
            from sklearn.gaussian_process.kernels import ConstantKernel as C
            kernel = 1.0 * RBF(length_scale=0.1 , length_scale_bounds=(0.1, 10.0))
            gp = GaussianProcessClassifier(kernel=kernel, optimizer=None)
            z_support = self.feature_extractor.forward(x_support).detach()
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            gp.fit(z_support.cpu().detach().numpy(), y_support.cpu().detach().numpy())
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            y_pred = gp.predict(z_query.cpu().detach().numpy())
            accuracy = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
            top1_correct = np.sum(y_pred==y_query)
            count_this = len(y_query)
            return float(top1_correct), count_this, 0.0

        x_train = x_support
        y_train = y_support

        samples_per_model = int(len(y_train) / self.n_way) #25 / 5 = 5
        target = torch.ones(len(y_train), dtype=torch.float32) * 1 
        # target = torch.zeros(len(y_train), dtype=torch.float32) 
        start_index = 0
        stop_index = start_index+samples_per_model
        target[start_index:stop_index] = -1.0
        target = target.cuda()

        z_train = self.feature_extractor.forward(x_train).detach() #[340, 64]
        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        # z_train_norm = F.normalize(z_train, p=2, dim=1)
        
        if self.dirichlet:
                target[target==-1] = 0
                self.model.likelihood.targets = target.long()
                sigma2_labels, transformed_targets, num_classes = self.model.likelihood._prepare_targets(self.model.likelihood.targets, 
                                        alpha_epsilon=self.model.likelihood.alpha_epsilon, dtype=torch.float)
                self.model.likelihood.transformed_targets = transformed_targets.transpose(-2, -1)
                self.model.likelihood.noise.data = sigma2_labels
                self.model.set_train_data(inputs=z_train, targets=self.model.likelihood.transformed_targets, strict=False)
        else: 
            self.model.set_train_data(inputs=z_train, targets=target, strict=False)

        with torch.no_grad():
            if self.sparse_method=="constFRVM":
                    z_train_rvm = self.constant_feature_extractor(x_train).detach()
                    inducing_points, frvm_acc = get_inducing_points(self.constant_model.base_covar_module, #.base_kernel,
                                                            z_train_rvm, target, sparse_method=self.sparse_method, scale=self.scale,
                                                            config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                            num_inducing_points=self.num_inducing_points, verbose=False, device=self.device)
            else:
                if self.regression:
                    self.config = '0' + self.config
                    inducing_points, frvm_acc = get_inducing_points_regression(self.model.base_covar_module, #.base_kernel,
                                                            z_train, target, sparse_method=self.sparse_method, scale=self.scale, beta=torch.tensor(10.0), 
                                                            config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                            num_inducing_points=self.num_inducing_points, verbose=False, task_id=i, device=self.device, classification=True)
                else:
                    inducing_points, frvm_acc = get_inducing_points(self.model.base_covar_module, #.base_kernel,
                                                            z_train, target, sparse_method=self.sparse_method, scale=self.scale,
                                                            config=self.config, align_threshold=self.align_threshold, gamma=self.gamma, 
                                                            num_inducing_points=self.num_inducing_points, verbose=False, task_id=i, device=self.device)
            self.frvm_acc.append(frvm_acc) 
            # self.ip_count.append(inducing_points.count) 
            
    
        ip_values = inducing_points.z_values.cuda()
        # ip_values = z_train[inducing_points.index].cuda()
        self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
        self.model.covar_module._clear_cache()

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()

        avg_loss=0.0
        for j in range(0, N):
            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            # z_query_list = [z_query]*len(y_query)
            if self.dirichlet:
                prediction = self.likelihood(self.model(z_query)) #return 2 * 20 MultiGaussian Distributions
            else:
                prediction = self.likelihood(self.model(z_query)) ##return n_way MultiGaussians
            
            if self.dirichlet:
                    
                   max_pred = (prediction.mean[0] > prediction.mean[1]).to(int)
                   y_pred = max_pred.cpu().detach().numpy()
            else: 
                pred = torch.sigmoid(prediction.mean)
                y_pred = (pred > 0.5).to(int)
                y_pred = y_pred.detach().cpu().numpy()

            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
            acc = (top1_correct/ count_this)*100
            #FRVM ACC on query
            K_m = self.model.base_covar_module(z_query, ip_values).evaluate()
            K_m = K_m.to(torch.float64)
            # scales	= torch.sqrt(torch.sum(K_m**2, axis=0))
            scales_m = inducing_points.scale
            mu = inducing_points.mu
            mu_m = mu / scales_m
            y_pred_ = K_m @ mu_m 
            y_pred_r = torch.sigmoid(y_pred_)
            y_pred_r = (y_pred_r > 0.5).to(int)
            y_pred_r = y_pred_r.detach().cpu().numpy()
            top1_correct_r = np.sum(y_pred_r==y_query)
            acc_r = (top1_correct_r / count_this)* 100

            if i%10==0:
                # print(Fore.RED,"-"*25, Fore.RESET)
                print(f'inducing_points count: {inducing_points.count}')
                # print(f'inducing_points alpha: {Fore.LIGHTGREEN_EX}{inducing_points.alpha.cpu().numpy()}',Fore.RESET)
                # print(f'inducing_points gamma: {Fore.LIGHTMAGENTA_EX}{inducing_points.gamma.cpu().numpy()}',Fore.RESET)
                print(Fore.YELLOW, f'itr {i:3}, RVM ACC: {acc_r:.2f}%, ACC: {acc:.2f}%', Fore.RESET)
                print(Fore.RED,"="*50, Fore.RESET)
            if self.show_plot:
                inducing_points = IP(inducing_points.z_values, inducing_points.index, inducing_points.count,
                                inducing_points.alpha, inducing_points.gamma,  
                                x_support[inducing_points.index], y_support[inducing_points.index], None, None)
                self.plot_test(x_query, y_query, y_pred, inducing_points, i)


        return float(top1_correct), count_this, avg_loss/float(N+1e-10), inducing_points.count, top1_correct_r

    def test_loop(self, test_loader, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        acc_all_rvm = []
        num_sv_list = []
        iter_num = len(test_loader)
        self.show_plot = iter_num < 5
        self.frvm_acc = []
        # self.ip_count = []
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this, loss_value, num_sv, correct_this_rvm = self.correct(x, i)
            acc_all.append(correct_this/ count_this*100)
            acc_all_rvm.append(correct_this_rvm/ count_this*100)
            num_sv_list.append(num_sv)
            if(i % 10==0):
                acc_mean = np.mean(np.asarray(acc_all))
                acc_mean_rvm = np.mean(np.asarray(acc_all_rvm))
                print('Test | Batch {:d}/{:d} | Loss {:f} |RVM Acc {:f}| Acc {:f}'.format(i, len(test_loader), loss_value, acc_mean_rvm, acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_mean_rvm = np.mean(np.asarray(acc_all_rvm))
        acc_std  = np.std(acc_all)
        acc_std_rvm  = np.std(acc_all_rvm)
        mean_num_sv = np.mean(num_sv_list)
        print(Fore.LIGHTRED_EX,"\n="*30, Fore.RESET)
        print(Fore.CYAN,f'Avg. FRVM ACC on support set: {np.mean(self.frvm_acc):4.2f}%, Avg. SVs {mean_num_sv:.2f}', Fore.RESET)
        print(Fore.CYAN,f'Avg. FRVM ACC on query set: {acc_mean_rvm:4.2f}%, std: {acc_std_rvm:.2f}', Fore.RESET)
        print(Fore.YELLOW,'%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)), Fore.RESET)
        print(Fore.LIGHTRED_EX,"="*30, Fore.RESET)
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(self.writer is not None): self.writer.add_scalar('Avg. SVs', mean_num_sv, self.iteration)
        if self.rvm_mll_only:
            if(return_std): return acc_mean_rvm, acc_std_rvm
            else: return acc_mean_rvm
        else:
            if(return_std): return acc_mean, acc_std
            else: return acc_mean

    
    def plot_test(self, x_query, y_query, y_pred, inducing_points, k):
        def clear_ax(ax):
            ax.clear()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            return ax
        fig: plt.Figure = plt.figure(1, figsize=(8, 4), tight_layout=True, dpi=125)
        
        r = 3
        c = 5
        i = 1
        if y_query.shape[0] >10:
            x_q       = torch.vstack([x_query[0:5], x_query[10:15]])
            y_q       = np.hstack([y_query[0:5], y_query[10:15]])
            y_pred_   = np.hstack([y_pred[0:5], y_pred[10:15]])
        else:
            x_q     = x_query    
            y_q     = y_query
            y_pred_ = y_pred
        for i in range(10):
            x = self.denormalize(x_q[i])
            y = y_q[i]
            y_p = y_pred_[i]
            ax: plt.Axes = fig.add_subplot(r, c, i+1)
            ax = clear_ax(ax)
            img = transforms.ToPILImage()(x.cpu()).convert("RGB")
            ax.imshow(img)
            ax.set_title(f'pred: {y_p:.0f}, real: {y:.0f}')
        inducing_x, inducing_y = inducing_points.x, inducing_points.y
        j = 5
        if j > inducing_y.shape[0]: j = inducing_y.shape[0]
        for i in range(j):
            x = self.denormalize(inducing_x[i].squeeze())
            y = inducing_y[i]
            ax: plt.Axes = fig.add_subplot(r, c, i+11)
            ax = clear_ax(ax)
            img = transforms.ToPILImage()(x.cpu()).convert("RGB")
            ax.imshow(img)
            ax.set_title(f'{y:.0f}')
            
        os.makedirs('./save_img', exist_ok=True)
        fig.savefig(f'./save_img/test_images_{k}.png')

        

    def denormalize(self, tensor):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        denormalized = tensor.clone()

        for channel, mean, std in zip(denormalized, means, stds):
            channel.mul_(std).add_(mean)

        return denormalized


    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        # Init to dummy values
        x_train = x_support
        y_train = y_support
        target_list = list()
        samples_per_model = int(len(y_train) / self.n_way)
        for way in range(self.n_way):
            target = torch.ones(len(y_train), dtype=torch.float32) * -1.0
            start_index = way * samples_per_model
            stop_index = start_index+samples_per_model
            target[start_index:stop_index] = 1.0
            target_list.append(target.cuda())
        z_train = self.feature_extractor.forward(x_train).detach() #[340, 64]
        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        train_list = [z_train]*self.n_way
        for idx, single_model in enumerate(self.model.models):
            single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)


        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            z_query_list = [z_query]*len(y_query)
            predictions = self.likelihood(*self.model(*z_query_list)) #return n_way MultiGaussians
            predictions_list = list()
            for gaussian in predictions:
                predictions_list.append(gaussian.mean) #.cpu().detach().numpy())
            y_pred = torch.stack(predictions_list, 1)
        return y_pred

class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise (Gaussian)
        base_covar_module.raw_outputscale
        base_covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, dirichlet, inducing_points, kernel='linear'):
        #Set the likelihood noise and enable/disable learning
        if not dirichlet:
            likelihood.noise_covar.raw_noise.requires_grad = False
            likelihood.noise_covar.noise = torch.tensor(0.1)
        super().__init__(train_x, train_y, likelihood)

        if dirichlet:
            self.mean_module  = gpytorch.means.ConstantMean(batch_shape=torch.Size((2,)))
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        ## Linear kernel
        if(kernel=='linear'):
            # if dirichlet:
            #     self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(
            #         batch_shape=torch.Size((2,))
            #     ), batch_shape=torch.Size((2,)),
            #     )
            # else:
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            # self.base_covar_module.outputscale = 0.1
            
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distance
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.base_covar_module.base_kernel.variance = 1.0
            self.base_covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module,
                                         inducing_points=inducing_points, likelihood=likelihood)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

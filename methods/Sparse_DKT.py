## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from fast_pytorch_kmeans import KMeans as Fast_KMeans
from collections import namedtuple
## Our packages
import gpytorch
from time import gmtime, strftime
import random
from colorama import Fore
from configs import kernel_type
from methods.Fast_RVM import Fast_RVM
#Check if tensorboardx is installed
try:
    # tensorboard --logdir=./Sparse_DKT_log/ --host localhost --port 8089
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
IP = namedtuple("inducing_points", "z_values index count x y i_idx j_idx")
class Sparse_DKT(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, config="010", align_threshold=1e-3, gamma=False, dirichlet=False):
        super(Sparse_DKT, self).__init__(model_func, n_way, n_support)
        self.num_inducing_points = 10
        self.fast_rvm = True
        self.config = config
        self.align_threshold = align_threshold
        self.gamma = gamma
        self.dirichlet = dirichlet
        self.device ='cuda'
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        if(kernel_type=="cossim"):
            self.normalize=True
        elif(kernel_type=="bncossim"):
            self.normalize=True
            latent_size = np.prod(self.feature_extractor.final_feat_dim)
            self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        else:
            self.normalize=False

    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir('./Sparse_DKT_log'):
                os.makedirs('./Sparse_DKT_log')
            writer_path = "./Sparse_DKT_log/" + id #+'_'+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x_list=[torch.ones(100, 64).cuda()]*self.n_way
        if(train_y_list is None): train_y_list=[torch.ones(100).cuda()]*self.n_way
        model_list = list()
        likelihood_list = list()
        for train_x, train_y in zip(train_x_list, train_y_list):
            
            if self.dirichlet:
                likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y.long(), learn_additional_noise=False)
            else:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()

            model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, dirichlet=self.dirichlet,
                                 inducing_points=train_x, kernel=kernel_type)
            model_list.append(model)
            likelihood_list.append(model.likelihood)
        self.model = gpytorch.models.IndependentModelList(*model_list).cuda()
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list).cuda()
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model).cuda()

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

    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        if self.dirichlet:
            optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
                                      {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])
        else:
            optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-3},
                                      {'params': self.feature_extractor.parameters(), 'lr': 1e-4}])
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

            target_list = list()
            samples_per_model = int(len(y_train) / self.n_way) #25 / 5 = 5
            for way in range(self.n_way):
                target = torch.ones(len(y_train), dtype=torch.float32) * -1 
                start_index = way * samples_per_model
                stop_index = start_index+samples_per_model
                target[start_index:stop_index] = 1.0
                target_list.append(target.cuda())

            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)

            # train_list = [z_train]*self.n_way
            lenghtscale = 0.0
            noise = 0.0
            outputscale = 0.0
            for idx, single_model in enumerate(self.model.models):

                if self.dirichlet:
                    single_model.likelihood.targets = target_list[idx]
                    sigma2_labels, transformed_targets, num_classes = single_model.likelihood._prepare_targets(single_model.likelihood.targets, 
                                            alpha_epsilon=single_model.likelihood.alpha_epsilon, dtype=torch.float)
                    single_model.likelihood.transformed_targets = transformed_targets.transpose(-2, -1)
                    single_model.likelihood.noise.data = sigma2_labels
                    single_model.set_train_data(inputs=z_train, targets=single_model.likelihood.transformed_targets, strict=False)
                else: 
                    single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)

                with torch.no_grad():
                    inducing_points = self.get_inducing_points(single_model.base_covar_module, #.base_kernel,
                                                            z_train, target_list[idx], verbose=False)
            
                ip_values = inducing_points.z_values.cuda()
                single_model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
                single_model.train()

                if(single_model.covar_module.base_kernel.lengthscale is not None):
                    lenghtscale+=single_model.base_covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
                noise+=single_model.likelihood.noise.cpu().detach().numpy().squeeze().mean()
                if(single_model.base_covar_module.outputscale is not None):
                    outputscale+=single_model.base_covar_module.outputscale.cpu().detach().numpy().squeeze()
            
            if(single_model.base_covar_module.base_kernel.lengthscale is not None): lenghtscale /= float(len(self.model.models))
            noise /= float(len(self.model.models))
            if(single_model.base_covar_module.outputscale is not None): outputscale /= float(len(self.model.models))

            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            if self.dirichlet:
                transformed_targets = [model.likelihood.transformed_targets for model in self.model.models]
                loss = -self.mll(output, transformed_targets).sum()
            else:
                loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)

            #Eval on the query (validation set)
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                self.feature_extractor.eval()
                z_support = self.feature_extractor.forward(x_support).detach()
                if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                # z_support_list = [z_support]*len(y_support)

                # if self.dirichlet:
                #     predictions = self.likelihood(*self.model(*z_support_list)) #return 2 * 20 MultiGaussian Distributions
                # else:
                #     predictions = self.likelihood(*self.model(*z_support_list)) #return 20 MultiGaussian Distributions
                # predictions_list = list()
                
                # if self.dirichlet:
                #     for dirichlet in predictions:

                #         max_pred = self.pred_result(dirichlet.mean)
                #         predictions_list.append(max_pred.cpu().detach().numpy())
                # else:
                #     for gaussian in predictions:
                #         predictions_list.append(torch.sigmoid(gaussian.mean).cpu().detach().numpy())

                # y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
                # accuracy_support = (np.sum(y_pred==y_support) / float(len(y_support))) * 100.0
                # if(self.writer is not None): self.writer.add_scalar('GP_support_accuracy', accuracy_support, self.iteration)
                z_query = self.feature_extractor.forward(x_query).detach()
                if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                z_query_list = [z_query]*len(y_query)
                if self.dirichlet:
                    predictions = self.likelihood(*self.model(*z_query_list)) #return 2 * 20 MultiGaussian Distributions
                else:
                    predictions = self.likelihood(*self.model(*z_query_list)) #return 20 MultiGaussian Distributions
                predictions_list = list()
                if self.dirichlet:
                    for dirichlet in predictions:
          
                        max_pred = self.pred_result(dirichlet.mean)
                        predictions_list.append(max_pred.cpu().detach().numpy())
                else:
                    for gaussian in predictions:
                        predictions_list.append(torch.sigmoid(gaussian.mean).cpu().detach().numpy())

                y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
                accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
                if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query, self.iteration)

            if i % print_freq==0:
                if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                if self.dirichlet:
                    print(Fore.LIGHTRED_EX,'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} || Loss {:f} | Supp. acc {:f} | Query acc {:f}'.format(epoch, i, len(train_loader),
                        outputscale, lenghtscale,  loss.item(), 0, accuracy_query), Fore.RESET) #accuracy_support
                else:
                    print(Fore.LIGHTRED_EX,'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} | Noise {:f} | Loss {:f} | Supp. acc {:f} | Query acc {:f}'.format(epoch, i, len(train_loader),
                        outputscale, lenghtscale, noise, loss.item(), 0, accuracy_query), Fore.RESET)

    def get_inducing_points(self, base_covar_module, inputs, targets, verbose=True):

        
        IP_index = np.array([])
        if not self.fast_rvm:
            num_IP = self.num_induce_points
            
            # self.kmeans_clustering = KMeans(n_clusters=num_IP, init='k-means++',  n_init=10, max_iter=1000).fit(inputs.cpu().numpy())
            # inducing_points = self.kmeans_clustering.cluster_centers_
            # inducing_points = torch.from_numpy(inducing_points).to(torch.float)

            self.kmeans_clustering = Fast_KMeans(n_clusters=num_IP, max_iter=1000)
            self.kmeans_clustering.fit(inputs.cuda())
            inducing_points = self.kmeans_clustering.centroids
            # print(inducing_points.shape[0])


        else:
            # with sigma and updating sigma converges to more sparse solution
            N   = inputs.shape[0]
            tol = 1e-3
            eps = torch.finfo(torch.float32).eps
            max_itr = 1000
            
            scale = True
            # X = inputs.clone()
            # m = X.mean(axis=0)
            # s = X.std(axis=0)
            # X = (X- m) / s
            kernel_matrix = base_covar_module(inputs).evaluate()
            # normalize kernel
            if scale:
                scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
                # print(f'scale: {Scales}')
                scales[scales==0] = 1
                kernel_matrix = kernel_matrix / scales

            kernel_matrix = kernel_matrix.to(torch.float64)
            targets = targets.to(torch.float64)
            active, alpha, Gamma, beta, mu_m = Fast_RVM(kernel_matrix, targets, N, self.config, self.align_threshold, self.gamma,
                                                    eps, tol, max_itr, self.device, verbose)

            index = np.argsort(active)
            active = active[index]
            inducing_points = inputs[active]
            num_IP = active.shape[0]
            IP_index = active
            if True:
                ss = scales[index]
                K = base_covar_module(inputs, inducing_points).evaluate()
                mu_m = mu_m[index] / ss
                mu_m = mu_m.to(torch.float)
                y_pred = K @ mu_m
                targets[targets==-1]= 0
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred > 0.5).to(int)
                acc = torch.sum(y_pred==targets)
                print(f'FRVM ACC: {(acc/N):.1%}')

        return IP(inducing_points, IP_index, num_IP, None, None, None, None)
  
    def correct(self, x, N=0, laplace=False):
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

            if self.dirichlet:
                single_model.likelihood.targets = target_list[idx]
                sigma2_labels, transformed_targets, _ = single_model.likelihood._prepare_targets(single_model.likelihood.targets, 
                                        alpha_epsilon=single_model.likelihood.alpha_epsilon, dtype=torch.float)
                single_model.likelihood.transformed_targets = transformed_targets.transpose(-2, -1)
                single_model.likelihood.noise.data = sigma2_labels
                single_model.set_train_data(inputs=z_train, targets=single_model.likelihood.transformed_targets, strict=False)
            else: 
                single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)
            
            with torch.no_grad():
                inducing_points = self.get_inducing_points(single_model.base_covar_module, #.base_kernel,
                                                            z_train, target_list[idx], verbose=False)
            
            ip_values = inducing_points.z_values.cuda()
            single_model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
            single_model.covar_module._clear_cache()
            # single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.eval()

        avg_loss=0.0
        for i in range(0, N):
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
            z_query_list = [z_query]*len(y_query)
            if self.dirichlet:
                    predictions = self.likelihood(*self.model(*z_query_list)) #return 2 * 20 MultiGaussian Distributions
            else:
                predictions = self.likelihood(*self.model(*z_query_list)) ##return n_way MultiGaussians
            
            predictions_list = list()
            if self.dirichlet:
                    for dirichlet in predictions:
                        max_pred = self.pred_result(dirichlet.mean)
                        predictions_list.append(max_pred.cpu().detach().numpy())
            else:
                for gaussian in predictions:
                    predictions_list.append(torch.sigmoid(gaussian.mean).cpu().detach().numpy())
            
            y_pred = np.vstack(predictions_list).argmax(axis=0) #[model, classes]
            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this, avg_loss/float(N+1e-10)

    def test_loop(self, test_loader, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this, loss_value = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if(i % 10==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Loss {:f} | Acc {:f}'.format(i, len(test_loader), loss_value, acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print(Fore.LIGHTRED_EX,"="*30)
        print(Fore.YELLOW,'\n%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)), Fore.RESET)
        print(Fore.LIGHTRED_EX,"="*30)
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(return_std): return acc_mean, acc_std
        else: return acc_mean

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

## Original packages
# from torch._C import ShortTensor
from numpy.core.defchararray import count
import backbone
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
from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans as Fast_KMeans
from time import gmtime, strftime
import torch.nn.functional as F
import random
## Our packages
import gpytorch
from methods.Fast_RVM_regression import Fast_RVM_regression

from statistics import mean
from data.qmul_loader import get_batch, train_people, val_people, test_people, get_unnormalized_label
from configs import kernel_type
from collections import namedtuple
import torch.optim
#Check if tensorboardx is installed
try:
    #tensorboard --logdir=./QMUL_Loss/ --host localhost --port 8091
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')


IP = namedtuple("inducing_points", "z_values index count alpha gamma  x y i_idx j_idx")
class Sparse_DKT_regression_rvm(nn.Module):
    def __init__(self, backbone, kernel_type='rbf', normalize=False, f_rvm=True, scale=True, config="0000", align_threshold=1e-3, gamma=False, n_inducing_points=12, random=False, 
                    video_path=None, show_plots_pred=False, show_plots_features=False, training=False):
        super(Sparse_DKT_regression_rvm, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.kernel_type = kernel_type
        self.normalize = normalize
        self.num_induce_points = n_inducing_points
        self.config = config
        self.gamma = gamma
        self.align_threshold = align_threshold
        self.f_rvm = f_rvm
        self.random = random
        self.device = 'cuda'
        self.video_path = video_path
        self.best_path = video_path
        self.show_plots_pred = show_plots_pred
        self.show_plots_features = show_plots_features
        if self.show_plots_pred or self.show_plots_features:
            self.initialize_plot(self.video_path, training)
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        
    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(self.num_induce_points, 2916).cuda() #2916: size of feature z
        # if(train_x is None): train_x=torch.rand(19, 3, 100, 100).cuda()
        if(train_y is None): train_y=torch.ones(self.num_induce_points).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.1
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=self.kernel_type, induce_point=train_x)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll
    
    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir('./Sparse_DKT_QMUL_rvm_new_Loss'):
                os.makedirs('./Sparse_DKT_QMUL_rvm_new_Loss')
            writer_path = './Sparse_DKT_QMUL_rvm_new_Loss/' + id #+'_'+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def rvm_ML(self, K, targets, alpha_m, beta, ip_index):
        
        N = targets.shape[0]
        Kt = K.T @ targets
        KK = K.T @ K
        K_m = K[:, ip_index]
        KK_mm = KK[ip_index, :][:, ip_index]
        K_mt = Kt[ip_index]
        A_m = torch.diag(alpha_m)
        H = A_m + beta * KK_mm
        U, info =  torch.linalg.cholesky_ex(H, upper=True)
        # if info>0:
        #     print('pd_err of Hessian')
        U_inv = torch.linalg.inv(U)
        Sigma_m = U_inv @ U_inv.T      
        mu_m = beta * (Sigma_m @ K_mt)
        y_ = K_m @ mu_m  
        e = (targets - y_)
        ED = e.T @ e
        dataLikely	= (N * torch.log(beta) - beta * ED)/2
        logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        
        logML			= dataLikely - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        return logML/N

    def train_loop_fast_rvm(self, epoch, n_support, n_samples, optimizer):
        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        mll_list = []
       
        for itr, (inputs, labels) in enumerate(zip(batch, batch_labels)):


            z = self.feature_extractor(inputs)
            if self.normalize: z = F.normalize(z, p=2, dim=1)
            with torch.no_grad():
                inducing_points = self.get_inducing_points(z, labels, verbose=False)
            ip_index = inducing_points.index
            ip_values = z[ip_index]
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=True)
            self.model.train()
            # NOTE 
            # alpha = inducing_points.alpha.cuda()
            # root_alpha = torch.sqrt(alpha)
            # inv_root_alpha = 1 / root_alpha
            # A = torch.eye(alpha.shape[0]).cuda()  * inv_root_alpha.to(torch.float)
            # self.model.covar_module.A = nn.Parameter(A.cuda(), requires_grad=False)
            sigma = self.model.likelihood.noise[0].clone()
            alpha_m = inducing_points.alpha
            
            K = self.model.base_covar_module(z).evaluate()
            if True:
                scales	= torch.sqrt(torch.sum(K**2, axis=0))
                K = K / scales
        
            rvm_mll = self.rvm_ML(K, labels, alpha_m, 1/sigma, ip_index)
            #  
            # self.model.set_train_data(inputs=ip_values, targets=labels[inducing_points.index], strict=False)
            self.model.set_train_data(inputs=z, targets=labels, strict=False)

            # z = self.feature_extractor(x_query)
            # predictions = self.model(ip_values)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mll_list.append(loss.item())
            # mse = self.mse(predictions.mean, labels[inducing_points.index])
            mse = self.mse(predictions.mean, labels)

            self.iteration = itr+(epoch*len(batch_labels))
            if(self.writer is not None): self.writer.add_scalar('MLL', loss.item(), self.iteration)

            if ((epoch%1==0) & (itr%2==0)):
                print(Fore.LIGHTRED_EX,'[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ),Fore.RESET)
            
            if (self.show_plots_pred or self.show_plots_features) and  self.f_rvm:
                embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
                self.update_plots_train_fast_rvm(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()
        
        return np.mean(mll_list)

    def test_loop_fast_rvm(self, n_support, n_samples, test_person, optimizer=None): # no optimizer needed for GP

        self.model.eval()
        self.likelihood.eval()
        self.feature_extractor.eval()
        inputs, targets = get_batch(test_people, n_samples)

        # support_ind = list(np.random.choice(list(range(n_samples)), replace=False, size=n_support))
        # query_ind   = [i for i in range(n_samples) if i not in support_ind]

        x_all = inputs.cuda()
        y_all = targets.cuda()

        split = np.array([True]*15 + [False]*3)
        # print(split)
        shuffled_split = []
        for _ in range(int(n_support/15)):
            s = split.copy()
            np.random.shuffle(s)
            shuffled_split.extend(s)
        shuffled_split = np.array(shuffled_split)
        support_ind = shuffled_split
        query_ind = ~shuffled_split
        x_support = x_all[test_person, support_ind,:,:,:]
        y_support = y_all[test_person, support_ind]
        x_query   = x_all[test_person, query_ind,:,:,:]
        y_query   = y_all[test_person, query_ind]


        # induce_ind = list(np.random.choice(list(range(n_samples)), replace=False, size=self.num_induce_points))
        # induce_point = self.feature_extractor(x_support[induce_ind, :,:,:])
        z_support = self.feature_extractor(x_support).detach()
        if self.normalize: z_support = F.normalize(z_support, p=2, dim=1)
        with torch.no_grad():
            inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
        
        ip_values = inducing_points.z_values.cuda()
        self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
        self.model.covar_module._clear_cache()

        # alpha = inducing_points.alpha.cuda()
        # root_alpha = torch.sqrt(alpha)
        # inv_root_alpha = 1 / root_alpha
        # A = torch.eye(alpha.shape[0]).cuda()  * inv_root_alpha.to(torch.float)
        # self.model.covar_module.A = nn.Parameter(A.cuda(), requires_grad=False)
        # self.model.set_train_data(inputs=ip_values, targets=y_support[inducing_points.index], strict=False)
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            if self.normalize: z_query = F.normalize(z_query, p=2, dim=1)
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        

        def inducing_max_similar_in_support_x(train_x, inducing_points, train_y):
            y = get_unnormalized_label(train_y.detach().cpu().numpy()) #((train_y.detach().cpu().numpy() + 1) * 60 / 2) + 60
    
            index = inducing_points.index
            x_inducing = train_x[index].detach().cpu().numpy()
            y_inducing = y[index]
            i_idx = []
            j_idx = []
            for r in range(index.shape[0]):
                
                t = y_inducing[r]
                x_t_idx = np.where(y==t)[0]
                x_t = train_x[x_t_idx].detach().cpu().numpy()
                j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
                i = int(t/10-6)
                i_idx.append(i)
                j_idx.append(j)

            return IP(inducing_points.z_values, index, inducing_points.count, inducing_points.alpha.cpu().numpy(),
                                 inducing_points.gamma.cpu().numpy(), 
                                x_inducing, y_inducing, np.array(i_idx), np.array(j_idx))
        
        inducing_points = inducing_max_similar_in_support_x(x_support, inducing_points, y_support)

        #**************************************************************
        mse = self.mse(pred.mean, y_query).item()
        y = get_unnormalized_label(y_query.detach()) #((y_query.detach() + 1) * 60 / 2) + 60
        y_pred = get_unnormalized_label(pred.mean.detach()) # ((pred.mean.detach() + 1) * 60 / 2) + 60
        mse_ = self.mse(y_pred, y).item()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        print(Fore.RED,"="*50, Fore.RESET)
        print(f'inducing_points count: {inducing_points.count}')
        print(f'inducing_points alpha: {Fore.LIGHTGREEN_EX}{inducing_points.alpha}',Fore.RESET)
        print(f'inducing_points gamma: {Fore.LIGHTMAGENTA_EX}{inducing_points.gamma}',Fore.RESET)
        print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
        print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
        print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
        print(Fore.LIGHTRED_EX, f'mse:    {mse_:.4f}, mse (normed):{mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        K = self.model.base_covar_module
        kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
        max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
        y_s = get_unnormalized_label(y_support.detach().cpu().numpy()) #((y_support.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
        
        kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

        # kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        # max_similar_index = np.argmax(kernel_matrix, axis=1)
        # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
        #**************************************************************
        if (self.show_plots_pred or self.show_plots_features) and self.f_rvm:
            embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())
            self.update_plots_test_fast_rvm(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                            z_support.detach(), z_query.detach(), embedded_z_support,
                                            inducing_points, x_query, y_query.detach().cpu().numpy(), pred, 
                                            max_similar_idx_x_s, max_similar_idx_x_ip, None, mse, test_person)
            if self.show_plots_pred:
                self.plots.fig.canvas.draw()
                self.plots.fig.canvas.flush_events()
                self.mw.grab_frame()
            if self.show_plots_features:
                self.plots.fig_feature.canvas.draw()
                self.plots.fig_feature.canvas.flush_events()
                self.mw_feature.grab_frame()

        return mse, mse_

  
    def train(self, stop_epoch, n_support, n_samples, optimizer):

        mll_list = []
        best_mse = 10e5 #stop_epoch//2
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 50, 80], gamma=0.1)
        
        for epoch in range(stop_epoch):
         
            mll = self.train_loop_fast_rvm(epoch, n_support, n_samples, optimizer)
            
            if epoch%2==0:
                print(Fore.GREEN,"-"*30, f'\nValidation:', Fore.RESET)
                mse_list = []
                val_count = 10
                rep = True if val_count > len(val_people) else False
                val_person = np.random.choice(np.arange(len(val_people)), size=val_count, replace=rep)
                for t in range(val_count):
                    mse, mse_ = self.test_loop_fast_rvm(n_support, n_samples, val_person[t],  optimizer)
                    mse_list.append(mse)
                mse = np.mean(mse_list)
                if best_mse >= mse:
                    best_mse = mse
                    model_name = self.best_path + 'best_model.tar'
                    self.save_best_checkpoint(epoch+1, best_mse, model_name)
                    print(Fore.LIGHTRED_EX, f'Best MSE: {best_mse:.4f}', Fore.RESET)
                print(Fore.LIGHTRED_EX, f'\nepoch {epoch+1} => MSE: {mse:.4f}, Best MSE: {best_mse:.4f}', Fore.RESET)
                if(self.writer is not None):
                    self.writer.add_scalar('MSE Val.', mse, epoch)
                print(Fore.GREEN,"-"*30, Fore.RESET)

            mll_list.append(mll)
            if(self.writer is not None): self.writer.add_scalar('MLL per epoch', mll, epoch)
            print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)

            # scheduler.step()
            # if (epoch) in [3]:
            #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            # if (epoch) in [50, 80]:
            #     optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * 0.1


        mll = np.mean(mll_list)

        
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        return mll, mll_list
    
    def test(self, n_support, n_samples, optimizer=None, test_count=None): # no optimizer needed for GP

        mse_list = []
        mse_list_ = []
        # choose a random test person
        rep = True if test_count > len(test_people) else False

        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=rep)
        for t in range(test_count):
            print(f'test #{t}')
            if self.f_rvm:
                mse, mse_ = self.test_loop_fast_rvm(n_support, n_samples, test_person[t],  optimizer)
            elif self.random:
                mse = self.test_loop_random(n_support, n_samples, test_person[t],  optimizer)
            elif not self.f_rvm:
                mse = self.test_loop_kmeans(n_support, n_samples, test_person[t],  optimizer)
            else:
                ValueError()

            mse_list.append(float(mse))
            mse_list_.append(float(mse_))

        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        print(f'MSE (unnormed): {np.mean(mse_list_):.4f}')
        return mse_list
        
    def get_inducing_points(self, inputs, targets, verbose=True):

        
        IP_index = np.array([])
     
        # with sigma and updating sigma converges to more sparse solution
        N   = inputs.shape[0]
        tol = 1e-4
        eps = torch.finfo(torch.float32).eps
        max_itr = 1000
        sigma = self.model.likelihood.noise[0].clone()
        # sigma = torch.tensor([0.0000001])
        # sigma = torch.tensor([torch.var(targets) * 0.1]) #sigma^2
        sigma = sigma.to(self.device)
        beta = 1 /(sigma + eps)
        covar_module = self.model.base_covar_module
        kernel_matrix = covar_module(inputs).evaluate()
        # normalize kernel
        if self.scale:
            scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
            # print(f'scale: {Scales}')
            scales[scales==0] = 1
            kernel_matrix = kernel_matrix / scales

        kernel_matrix = kernel_matrix.to(torch.float64)
        target = targets.clone().to(torch.float64)
        active, alpha, gamma, beta, mu_m = Fast_RVM_regression(kernel_matrix, target, beta, N, self.config, self.align_threshold,
                                                self.gamma, eps, tol, max_itr, self.device, verbose)
        
        index = np.argsort(active)
        active = active[index]
        gamma = gamma[index]
        ss = scales[active]
        alpha = alpha[index] #/ ss**2
        inducing_points = inputs[active]
        num_IP = active.shape[0]
        IP_index = active
        with torch.no_grad():
            if True:
                
                K = covar_module(inputs, inducing_points).evaluate()
                mu_m = (mu_m[index] / ss)
                mu_m = mu_m.to(torch.float)
                y_pred = K @ mu_m
                
                mse = self.mse(y_pred, target)
                print(f'FRVM MSE: {mse:0.4f}')
        

        return IP(inducing_points, IP_index, num_IP, alpha.to(torch.float), gamma, None, None, None, None)
  
    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
    
        ckpt = torch.load(checkpoint)
        if 'best' in checkpoint:
            print(f'\nBest model at epoch {ckpt["epoch"]}, MSE: {ckpt["mse"]}')
        IP = torch.ones(self.num_induce_points, 2916).cuda()
        ckpt['gp']['covar_module.inducing_points'] = IP
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

    def save_best_checkpoint(self, epoch, mse, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 
        'net':nn_state_dict, 'epoch': epoch, 'mse':mse}, checkpoint)


    def initialize_plot(self, video_path, training):
        
        
        if training:
            self.video_path = video_path+'_Train_video'
        else:
            self.video_path = video_path+'_Test_video'

        os.makedirs(self.video_path, exist_ok=True)
        time_now = datetime.now().strftime('%Y-%m-%d--%H-%M')
        self.sparse_method = "FRVM" if self.f_rvm else "KMeans"
        if self.random: self.sparse_method = "random"  
        self.plots = self.prepare_plots()
        # plt.show(block=False)
        # plt.pause(0.0001)
        if self.show_plots_pred:
           
            metadata = dict(title=f'Sparse_DKT_{self.sparse_method}', artist='Matplotlib')
            FFMpegWriter = animation.writers['ffmpeg']
            self.mw = FFMpegWriter(fps=5, metadata=metadata)   
            file = f'{self.video_path}/Sparse_DKT_{self.sparse_method}_{time_now}.mp4'
            self.mw.setup(fig=self.plots.fig, outfile=file, dpi=125)

        if self.show_plots_features:  
            metadata = dict(title=f'Sparse_DKT_{self.sparse_method}', artist='Matplotlib')         
            FFMpegWriter2 = animation.writers['ffmpeg']
            self.mw_feature = FFMpegWriter2(fps=2, metadata=metadata)
            file = f'{self.video_path}/Sparse_DKT_{self.sparse_method}_features_{time_now}.mp4'
            self.mw_feature.setup(fig=self.plots.fig_feature, outfile=file, dpi=150)
    
    def prepare_plots(self):
        Plots = namedtuple("plots", "fig ax fig_feature ax_feature")
        # fig: plt.Figure = plt.figure(1, dpi=200) #, tight_layout=True
        # fig.subplots_adjust(hspace = 0.0001)
        fig, ax = plt.subplots(7, 19, figsize=(16,8), sharex=True, sharey=True, dpi=100) 
        plt.subplots_adjust(wspace=0.1,  
                            hspace=0.8)
        # ax = fig.subplots(7, 19, sharex=True, sharey=True)
          
        # fig.subplots_adjust(hspace=0.4, wspace=0.1)
        fig_feature: plt.Figure = plt.figure(2, figsize=(6, 6), tight_layout=True, dpi=125)
        ax_feature: plt.Axes = fig_feature.add_subplot(1, 1, 1)
        ax_feature.set_ylim(-20, 20)
        ax_feature.set_xlim(-20, 20)

        return Plots(fig, ax, fig_feature, ax_feature)     

    def update_plots_train_fast_rvm(self,plots, train_y, embedded_z, mll, mse, epoch):
        if self.show_plots_features:
            #features
            y = get_unnormalized_label(train_y) #((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}')  

    def update_plots_test_fast_rvm(self, plots, train_x, train_y, train_z, test_z, embedded_z, inducing_points,   
                                    test_x, test_y, test_y_pred, similar_idx_x_s, similar_idx_x_ip, mll, mse, person):
        def clear_ax(plots, i, j):
            plots.ax[i, j].clear()
            plots.ax[i, j].set_xticks([])
            plots.ax[i, j].set_xticklabels([])
            plots.ax[i, j].set_yticks([])
            plots.ax[i, j].set_yticklabels([])
            plots.ax[i, j].set_aspect('equal')
            return plots
        
        def color_ax(plots, i, j, color, lw=0):
            if lw > 0:
                for axis in ['top','bottom','left','right']:
                    plots.ax[i, j].spines[axis].set_linewidth(lw)
            #magenta, orange
            for axis in ['top','bottom','left','right']:
                plots.ax[i, j].spines[axis].set_color(color)

            return plots

        if self.show_plots_pred:

            cluster_colors = ['aqua', 'coral', 'lime', 'gold', 'purple', 'green']
            #train images
            plots.fig.suptitle(f"Sparse DKT ({self.sparse_method}), person {person}, MSE: {mse:.4f}, num IP: {inducing_points.count}")

            y = get_unnormalized_label(train_y)#  ((train_y + 1) * 60 / 2) + 60
            tilt = [60, 70, 80, 90, 100, 110, 120]
            num = 1
            for t in tilt:
                idx = np.where(y==(t))[0]
                if idx.shape[0]==0:
                    i = int(t/10-6)
                    for j in range(0, 19):
                        plots = clear_ax(plots, i, j)
                        plots = color_ax(plots, i, j, 'black', lw=0.5)
                else:    
                    x = train_x[idx]
                    i = int(t/10-6)
                    # z = train_z[idx]
                    for j in range(0, idx.shape[0]): 
                        img = transforms.ToPILImage()(x[j].cpu()).convert("RGB")
                        plots = clear_ax(plots, i, j)
                        plots = color_ax(plots, i, j, 'black', lw=0.5)
                        plots.ax[i, j].imshow(img)
                        plots.ax[i, j].set_title(f'{num}', fontsize=8)
                        num += 1
                    plots.ax[i, 0].set_ylabel(f'{t}',  fontsize=10)
                
        
            # test images
            y = get_unnormalized_label(test_y) #((test_y + 1) * 60 / 2) + 60
            y_mean = test_y_pred.mean.detach().cpu().numpy()
            y_var = test_y_pred.variance.detach().cpu().numpy()
            y_pred = ((y_mean + 1) * 60 / 2) + 60
            y_s = get_unnormalized_label(train_y) #((train_y + 1) * 60 / 2) + 60
            for t in tilt:
                idx = np.where(y==(t))[0]
                if idx.shape[0]==0:
                    continue
                else:
                    x = test_x[idx]
                    sim_x_s_idx = similar_idx_x_s[idx]
                    sim_y_s = y_s[sim_x_s_idx] 
                    sim_x_ip = similar_idx_x_ip[idx]
                    y_p = y_pred[idx]
                    y_v = y_var[idx]
                    i = int(t/10-6)
                    for j in range(idx.shape[0]):
                        
                        img = transforms.ToPILImage()(x[j].cpu()).convert("RGB")
                        ii = 16
                        plots = clear_ax(plots, i, j+ii)
                        plots.ax[i, j+ii].imshow(img)
                        # plots = color_ax(plots, i, j+ii, color=cluster_colors[cluster[j]], lw=2)
                        plots.ax[i, j+ii].set_title(f'{y_p[j]:.1f}', fontsize=8)
                        id_sim_x_s = int(plots.ax[int(sim_y_s[j]/10-6),0].get_title()) +  sim_x_s_idx[j]%15
                        plots.ax[i, j+ii].set_xlabel(f'{id_sim_x_s}|{sim_x_ip[j]+1}', fontsize=10)
                
                    # plots.ax[i, j+16].legend()
            
            for i in range(7):
                plots = clear_ax(plots, i, 15)
                plots = color_ax(plots, i, 15, 'white', lw=0.5)

            # highlight inducing points
            y = get_unnormalized_label(train_y) #((train_y + 1) * 60 / 2) + 60
            if inducing_points.x is not None:
                
                # cluster = self.kmeans_clustering.predict(inducing_points.z_values)
                # cluster = self.kmeans_clustering.predict(z_inducing.detach().cpu().numpy())                
                for r in range(inducing_points.index.shape[0]):
                    
                    # t = inducing_points.y[r]
                    # i = int(t/10-6)
                    plots = color_ax(plots, inducing_points.i_idx[r], inducing_points.j_idx[r], 'black', lw=1) 
                    plots.ax[inducing_points.i_idx[r], inducing_points.j_idx[r]].spines['bottom'].set_color('red')  
                    plots.ax[inducing_points.i_idx[r], inducing_points.j_idx[r]].spines['bottom'].set_linewidth(3) 
                    plots.ax[inducing_points.i_idx[r], inducing_points.j_idx[r]].set_xlabel(r+1, fontsize=10)          

            plots.fig.savefig(f'{self.video_path}/test_person_{person}.png')      
        
        if self.show_plots_features:
            #features
            y = get_unnormalized_label(train_y) #((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')
            plots.fig_feature.suptitle(f"Sparse DKT ({self.sparse_method}), person {person}, MSE: {mse:.4f}, num IP: {inducing_points.count}")
            plots.ax_feature.legend()


from gpytorch.lazy import DiagLazyTensor, LowRankRootAddedDiagLazyTensor, LowRankRootLazyTensor, MatmulLazyTensor, delazify
import copy
import math 
from gpytorch.distributions  import MultivariateNormal
from gpytorch.mlls import InducingPointKernelAddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky

class SparseKernel(gpytorch.kernels.InducingPointKernel):

    def __init__(self, base_kernel, inducing_points, likelihood, alpha, active_dims=None):
        
        super(SparseKernel, self).__init__(base_kernel, inducing_points, likelihood, active_dims=active_dims)
        self.register_parameter(name="A", parameter=torch.nn.Parameter(alpha))

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
        
        self.covar_module = SparseKernel(self.base_covar_module, inducing_points=induce_point , likelihood=likelihood)
    
    def forward(self, x):
        mean_x  = self.mean_module(x)
        # covar_x = self.base_covar_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

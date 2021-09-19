## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
from colorama import Fore
from collections import namedtuple
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
## Our packages
import gpytorch
from time import gmtime, strftime
import random
from statistics import mean
from methods.Fast_RVM_regression import Fast_RVM_regression
from data.qmul_loader import get_batch, train_people, test_people, val_people, get_unnormalized_label
from configs import kernel_type
#Check if tensorboardx is installed
try:
    #tensorboard --logdir=./QMUL_Loss/ --host localhost --port 8091
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

IP = namedtuple("inducing_points", "z_values index count alpha gamma  x y i_idx j_idx") #for test 
class DKT_regression(nn.Module):
    def __init__(self, backbone, video_path=None, show_plots_pred=False, show_plots_features=False, training=False):
        super(DKT_regression, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = 'cuda'
        self.video_path = video_path
        self.best_path = video_path
        self.show_plots_pred = show_plots_pred
        self.show_plots_features = show_plots_features
        if self.show_plots_pred or self.show_plots_features:
            self.initialize_plot(video_path, training)
        
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(19, 2916).cuda()
        if(train_y is None): train_y=torch.ones(19).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.1
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel='rbf')

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir('./QMUL_Loss'):
                os.makedirs('./QMUL_Loss')
            writer_path = './QMUL_Loss/' + id #+'_'+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def train_loop(self, epoch, n_support, n_samples, optimizer):
        
        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        # print(f'{epoch}: {batch_labels[0]}')
        mll_list = []
        for itr, (inputs, labels) in enumerate(zip(batch, batch_labels)):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)

            self.model.set_train_data(inputs=z, targets=labels, strict=False)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)
            mll_list.append(loss.item())
            if ((epoch%2==0) & (itr%5==0)):
                print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
            self.iteration = itr+(epoch*len(batch_labels))
            if(self.writer is not None): self.writer.add_scalar('MLL', loss.item(), self.iteration)

            if (self.show_plots_pred or self.show_plots_features):
                embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
                self.update_plots_train(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()

        return np.mean(mll_list)

    def test_loop(self, n_support, n_samples, test_person, optimizer=None): # no optimizer needed for GP
        inputs, targets = get_batch(test_people, n_samples)

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
        x_all = inputs.cuda()
        y_all = targets.cuda()

        x_support = x_all[test_person,support_ind,:,:,:]
        y_support = y_all[test_person,support_ind]
        x_query   = x_all[test_person,query_ind,:,:,:]
        y_query   = y_all[test_person,query_ind]

    
        z_support = self.feature_extractor(x_support).detach()
        #NOTE for test 
        with torch.no_grad():
            inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)

        ip_values = inducing_points.z_values.cuda()
        self.model.set_train_data(inputs=ip_values, targets=y_support[inducing_points.index], strict=False)
        #****
        # self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query).item()
        #***************************************************
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
        print(Fore.LIGHTRED_EX, f'mse:    {mse_:.4f}, mse (normed): {mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        K = self.model.covar_module
        kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
        max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
        y_s = ((y_support.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.LIGHTGREEN_EX, f'target of most similar in support set: {y_s[max_similar_idx_x_s]}', Fore.RESET)
        #**************************************************

        if (self.show_plots_pred or self.show_plots_features):
            embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

            self.update_plots_test(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                            z_support.detach(), z_query.detach(), embedded_z_support,
                                            x_query, y_query.detach().cpu().numpy(), pred, 
                                            max_similar_idx_x_s, None, mse, test_person)
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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stop_epoch//3, gamma=0.1)
        best_mse = 1e7
        for epoch in range(stop_epoch):
            mll = self.train_loop(epoch, n_support, n_samples, optimizer)

            if epoch%2==0:
                print(Fore.GREEN,"-"*30, f'\nValidation:', Fore.RESET)
                mse_list = []
                val_count = 10
                rep = True if val_count > len(val_people) else False
                val_person = np.random.choice(np.arange(len(val_people)), size=val_count, replace=rep)
                for t in range(val_count):
                    mse, mse_ = self.test_loop(n_support, n_samples, val_person[t],  optimizer)
                    mse_list.append(mse)
                mse = np.mean(mse_list)
                if best_mse >= mse:
                    best_mse = mse
                    model_name = self.best_path + '_best_model.tar'
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
        mll = np.mean(mll_list)
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        return mll, mll_list

    def test(self, n_support, n_samples, optimizer=None, test_count=None):

        mse_list = []
        mse_list_ = []
        # choose a random test person
        rep = True if test_count > len(test_people) else False

        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=rep)
        for t in range(test_count):
            print(f'test #{t}')
            
            mse, mse_ = self.test_loop(n_support, n_samples, test_person[t],  optimizer)
            
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
        scale = True
        covar_module = self.model.covar_module
        # X = inputs.clone()
        # m = X.mean(axis=0)
        # X = (X- m) 
        # X = F.normalize(X, p=2, dim=1)
        kernel_matrix = covar_module(inputs).evaluate()
        # normalize kernel
        if scale:
            scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
            # print(f'scale: {Scales}')
            scales[scales==0] = 1
            kernel_matrix = kernel_matrix / scales

        kernel_matrix = kernel_matrix.to(torch.float64)
        target = targets.clone().to(torch.float64)
        active, alpha, gamma, beta, mu_m = Fast_RVM_regression(kernel_matrix, target, beta, N, "1011",1e-3,
                                                False, eps, tol, max_itr, self.device, verbose)
        
        index = np.argsort(active)
        active = active[index]
        gamma = gamma[index]
        ss = scales[index]
        alpha = alpha[index] / ss
        inducing_points = inputs[active]
        num_IP = active.shape[0]
        IP_index = active
        with torch.no_grad():
            if True:
                
                K = covar_module(inputs, inducing_points).evaluate()
                # K = covar_module(X, X[active]).evaluate()
                mu_m = (mu_m[index] / ss)
                mu_m = mu_m.to(torch.float)
                y_pred = K @ mu_m
                
                mse = self.mse(y_pred, target)
                print(f'FRVM MSE: {mse:0.4f}')
        

        return IP(inducing_points, IP_index, num_IP, alpha, gamma, None, None, None, None)
  
    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
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
         
        self.plots = self.prepare_plots()
        # plt.show(block=False)
        # plt.pause(0.0001)
        if self.show_plots_pred:
           
            metadata = dict(title='DKT', artist='Matplotlib')
            FFMpegWriter = animation.writers['ffmpeg']
            self.mw = FFMpegWriter(fps=5, metadata=metadata)   
            file = f'{self.video_path}/DKT_{time_now}.mp4'
            self.mw.setup(fig=self.plots.fig, outfile=file, dpi=125)

        if self.show_plots_features:  
            metadata = dict(title='DKT', artist='Matplotlib')         
            FFMpegWriter2 = animation.writers['ffmpeg']
            self.mw_feature = FFMpegWriter2(fps=2, metadata=metadata)
            file = f'{self.video_path}/DKT_features_{time_now}.mp4'
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

    def update_plots_train(self,plots, train_y, embedded_z, mll, mse, epoch):
        if self.show_plots_features:
            #features
            y = get_unnormalized_label(train_y)#((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}')

    def update_plots_test(self, plots, train_x, train_y, train_z, test_z, embedded_z,   
                                    test_x, test_y, test_y_pred, similar_idx_x_s, mll, mse, person):
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
            plots.fig.suptitle(f'person {person}, MSE: {mse:.4f}')
            y = get_unnormalized_label(train_y)# ((train_y + 1) * 60 / 2) + 60
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
                    y_p = y_pred[idx]
                    y_v = y_var[idx]
                    i = int(t/10-6)
                    for j in range(idx.shape[0]):
                        
                        img = transforms.ToPILImage()(x[j].cpu()).convert("RGB")
                        ii = 16
                        plots = clear_ax(plots, i, j+ii)
                        plots.ax[i, j+ii].imshow(img)
                        # plots = color_ax(plots, i, j+ii, color=cluster_colors[cluster[j]], lw=2)
                        plots.ax[i, j+ii].set_title(f'{y_p[j]:.1f}', fontsize=10)
                        id_sim_x_s = int(plots.ax[int(sim_y_s[j]/10-6),0].get_title()) +  sim_x_s_idx[j]%15
                        plots.ax[i, j+ii].set_xlabel(f'{int(id_sim_x_s)}', fontsize=10)
                
                    # plots.ax[i, j+16].legend()
            for i in range(7):
                plots = clear_ax(plots, i, 15)
                plots = color_ax(plots, i, 15, 'white', lw=0.5)

            plots.fig.savefig(f'{self.video_path}/test_person_{person}.png') 

        if self.show_plots_features:
            #features
            y = get_unnormalized_label(train_y)#((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()



class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=16, ard_num_dims=2916)
            self.covar_module.initialize_from_data_empspect(train_x, train_y)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

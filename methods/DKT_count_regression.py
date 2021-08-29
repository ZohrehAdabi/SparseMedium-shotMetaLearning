## Original packages
from numpy.core.fromnumeric import around
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
from data.msc44_loader import get_batch, denormalize
from configs import kernel_type

#Check if tensorboardx is installed
try:
    # tensorboard --logdir=./save/checkpoints/MSC44/ResNet50_DKT_Loss/ --host localhost --port 8088
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

class DKT_count_regression(nn.Module):
    def __init__(self, backbone, regressor, base_file=None, val_file=None,
        video_path=None, show_plots_loss=False, show_plots_pred=False, show_plots_features=False, training=False):
        super(DKT_count_regression, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.regressor = regressor
        self.train_file = base_file
        self.val_file = val_file
        self.minmax = False
        self.device = 'cuda'
        self.video_path = video_path
        self.best_path = video_path
        self.show_plots_loss = show_plots_loss
        self.show_plots_pred = show_plots_pred
        self.show_plots_features = show_plots_features
        if self.show_plots_pred or self.show_plots_features:
            self.initialize_plot(video_path, training)
        
        self.get_model_likelihood_mll() #Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if(train_x is None): train_x=torch.ones(50, 2304).cuda()
        if(train_y is None): train_y=torch.ones(50).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # likelihood.noise = 0.1
        likelihood.initialize(noise=0.1)  
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()
        self.mae        = nn.L1Loss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def resize_gt_density(self, z, gt_density, labels):
        gt_density_resized = torch.empty([gt_density.shape[0],1, 1, z.shape[2], z.shape[3]])
        for i in range(z.shape[0]):
            if z[i].shape[1] != gt_density[i].shape[2] or z[i].shape[2] != gt_density[i].shape[3]:
                # print(i, z[i].shape)

                orig_count_i = gt_density[i].sum().detach().item()
                gt_density_resized[i] = F.interpolate(gt_density[i], size=(z[i].shape[1], z[i].shape[2]), mode='bilinear',  align_corners=True)
                new_count_i = gt_density_resized[i].sum().detach().item()
                if new_count_i > 0: 
                    gt_density_resized[i] = gt_density_resized[i] * (orig_count_i / new_count_i)
                    labels[i] = torch.round(gt_density_resized[i].sum())
        return gt_density_resized.cuda(), labels

    def normalize(self, labels, y_mean, y_std):

        if self.minmax:
            return F.normalize(labels, p=2, dim=0)
        else:
            return  (labels - y_mean) / (y_std+1e-10)
    
    def denormalize_y(self, pred, labels,  y_mean, y_std):

        if self.minmax:
            return pred * labels.norm(p=2, dim=0)
        else:
            return  y_mean + pred * y_std

    def train_loop(self, epoch, n_support, n_samples, optimizer):

        # print(f'{epoch}: {batch_labels[0]}')
        validation = True
        mll_list = []
        mse_list = []
        mae_list = []
        for itr, samples in enumerate(get_batch(self.train_file, n_samples)):
            
            self.model.train()
            self.regressor.train()
            self.likelihood.train()
            optimizer.zero_grad()

            inputs = samples['image']
            labels = samples['gt_count']
            gt_density = samples['gt_density']
            with torch.no_grad():
                feature = self.feature_extractor(inputs)
            #predict density map
            feature.requires_grad = True
            z = self.regressor(feature)
            #if image size isn't divisible by 8, gt size is slightly different from output size
            with torch.no_grad():
                gt_density_resized, labels = self.resize_gt_density(z, gt_density, labels)
                y_mean, y_std = labels.mean(), labels.std()
                labels_norm = self.normalize(labels, y_mean, y_std)
            if self.use_mse:
                density_mse = self.mse(z, gt_density_resized.squeeze(1))

            z = z.reshape(z.shape[0], -1)#.to(torch.float64)
            self.model.set_train_data(inputs=z, targets=labels_norm, strict=False)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            if self.use_mse:
                loss = loss + 10 * density_mse
            
            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels_norm)
            mll_list.append(np.around(loss.item(), 4))
            self.iteration = (epoch*31) + itr
            if(self.writer is not None) and self.show_plots_loss: 
                self.writer.add_scalar('MLL_per_itr', loss.item(), self.iteration)
   

            if ((epoch%1==0) & (itr%10==0)):
                print('[%2d/%2d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch+1, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

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
            #*********************************************************
            #validate on train data
            val_freq = 10
            if validation and (epoch%val_freq==0):
                support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
                query_ind   = [i for i in range(n_samples) if i not in support_ind]
                z_support = z[support_ind, :]
                y_s_norm = labels_norm[support_ind]
                # y_support = labels[support_ind]
                z_query   = z[query_ind]
                y_q_norm   = labels_norm[query_ind]
                y_query   = labels[query_ind]

                self.model.set_train_data(inputs=z_support, targets=y_s_norm, strict=False)

                self.model.eval()
                self.regressor.eval()
                self.likelihood.eval()

                with torch.no_grad():
                    pred    = self.likelihood(self.model(z_query))
                    lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

                mse = self.mse(pred.mean, y_q_norm).item()
                mse_list.append(mse)
                y_pred = self.denormalize_y(pred.mean, labels, y_mean, y_std)
                mae = self.mae(y_pred, y_query).item()
                mae_list.append(mae)
                print(Fore.YELLOW, f'epoch {epoch+1}, itr {itr+1}, Val. on Train  MAE:{mae:.2f}, MSE: {mse:.4f}', Fore.RESET)

        if validation and (epoch%val_freq==0):
            print(Fore.LIGHTMAGENTA_EX,"-"*30, f'\n epoch {epoch+1} => Avg. Val. on Train    MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
                                    f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar('MSE Val. on Train', mse, epoch)
                self.writer.add_scalar('MAE Val. on Train', mae, epoch)

        # print(f'epoch {epoch+1} MLL {mll_list}')
        return np.mean(mll_list)

    def test_loop(self, n_support, n_samples, epoch, optimizer=None): # no optimizer needed for GP
        
        
        mse_list = []    
        mae_list = []  
        base_line_mae_list = []
        self.model.eval()
        self.regressor.eval()
        self.likelihood.eval() 
        for itr, samples in enumerate(get_batch(self.val_file, n_samples)):
            
            class_name = samples['class_name']
            print(f'\nclass_name = {class_name}')
            inputs = samples['image']
            targets = samples['gt_count']
            gt_density = samples['gt_density']

            x_all       = inputs.cuda()
            y_all       = targets.cuda()
            support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
            query_ind   = [i for i in range(n_samples) if i not in support_ind]
            x_support    = x_all[support_ind,:,:,:]
            x_query     = x_all[query_ind,:,:,:]
            
            
            with torch.no_grad():
                feature = self.feature_extractor(x_all)
            #predict density map
            z = self.regressor(feature).detach()
            with torch.no_grad():
                gt_density_resized, y_all = self.resize_gt_density(z, gt_density, y_all)
                

            y_support   = y_all[support_ind]
            gt_density_s = gt_density_resized[support_ind, :, :, :, :]
            z_support   = z[support_ind, :, :, :]
            y_query     = y_all[query_ind]
            gt_density_q = gt_density_resized[query_ind, :, :, :, :]
            z_query     = z[query_ind, :, :, :]
            
            with torch.no_grad():
                y_mean, y_std = y_all.mean(), y_all.std()
                y_s_norm = self.normalize(y_support, y_mean, y_std)
                y_q_norm = self.normalize(y_query, y_mean, y_std)

            z_support = z_support.reshape(z_support.shape[0], -1)
            self.model.set_train_data(inputs=z_support, targets=y_s_norm, strict=False)

            with torch.no_grad():
                                
                z_query = z_query.reshape(z_query.shape[0], -1)
                pred    = self.likelihood(self.model(z_query))
                lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

            mse = self.mse(pred.mean, y_q_norm).item()
            mse_list.append(mse)
            y_pred = self.denormalize_y(pred.mean.detach(), y_query, y_mean, y_std)
            mae = self.mae(y_pred, y_query).item()
            mae_list.append(mae)
            #***************************************************
            y = y_query.detach().cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            mean_support_y = y_support.mean()
            base_line_mae = self.mae(mean_support_y.repeat(y_query.shape[0]), y_query).item()
            base_line_mae_list.append(base_line_mae)

            print(Fore.RED,"="*50, Fore.RESET)
            print(f'itr #{itr+1}')
            print(f'mean of support_y {base_line_mae:.2f}')
            print(f'base line MAE: {mean_support_y:.2f}')
            print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
            print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
            print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
            print(Fore.LIGHTRED_EX, f'mae: {mae}, mse:\t{mse:.4f}', Fore.RESET)
            print(Fore.RED,"-"*50, Fore.RESET)

            # K = self.model.covar_module
            # kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
            # max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
            # y_s = y_support.detach().cpu().numpy()
            # print(Fore.LIGHTGREEN_EX, f'target of most similar in support set: {y_s[max_similar_idx_x_s]}', Fore.RESET)
            #**************************************************

            if (self.show_plots_pred or self.show_plots_features):
                embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

                self.update_plots_test(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                                z_support.detach(), z_query.detach(), embedded_z_support,
                                                x_query, y_query.detach().cpu().numpy(), y_pred, pred.variance.detach().cpu().numpy(),
                                                mae, mse, y_support.mean(), itr)
                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()

        print(Fore.CYAN,"-"*30, f'\n epoch {epoch+1} => Avg.   MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
                                    f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
        print(f'Avg. base line MAE: {np.mean(base_line_mae_list):.2f}')
   
        return np.mean(mse_list), np.mean(mae_list), np.sqrt(np.mean(mse_list))

    def train(self, stop_epoch, n_support, n_samples, optimizer, id, use_mse):
        
        self.use_mse = use_mse
        self.feature_extractor.eval()
        best_mae, best_rmse = 10e7, 10e7
        mll_list = []
        for epoch in range(stop_epoch):
            mll = self.train_loop(epoch, n_support, n_samples, optimizer)
            mll_list.append(np.around(mll, 3))
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar('MLL_per_epoch.', mll, epoch)

            print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch+1} => MLL: {mll}\n', "-"*30, Fore.RESET)
            print(Fore.GREEN,"-"*30, f'\nValidation:', Fore.RESET)
            if epoch%2==0:
                val_mse, val_mae, val_rmse = self.test_loop(n_support, n_samples, epoch, optimizer)
                if best_mae >= val_mae:
                    best_mae = val_mae
                    best_rmse = val_rmse
                    model_name = self.best_path + f'_best_mae{best_mae:.2f}_ep{epoch}_{id}.pth'
                    self.save_checkpoint(model_name)
                    print(Fore.LIGHTRED_EX, f'Best MAE: {best_mae:.2f}, RMSE: {best_rmse}', Fore.RESET)
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar('MSE Val.', val_mse, epoch)
                self.writer.add_scalar('MAE Val.', val_mae, epoch)
            print(Fore.GREEN,"-"*30, Fore.RESET)

        mll = np.mean(mll_list)
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        return mll, mll_list

    def test(self, n_support, n_samples, optimizer=None, n_test_epoch=1):
        
        self.feature_extractor.eval()
        mse_list = []
        for e in range(n_test_epoch):
            print(f'test on all test tasks epoch #{e}')
            
            mse, mae, rmse = self.test_loop(n_support, n_samples, e,  optimizer)
            mse_list.append(float(mse))

        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()

        return mse_list


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
    
    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_now = datetime.now().strftime('%Y-%m-%d--%H-%M')
            writer_path = self.video_path+'_Loss' +f"/{id}"
            # os.makedirs(self.video_path, exist_ok=True)
            os.makedirs(writer_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=writer_path)

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
        fig, ax = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True, dpi=150) 
        plt.subplots_adjust(wspace=0.03,  
                    hspace=0.05) 
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
            plots.ax_feature.clear()
            plots.ax_feature.scatter(embedded_z[:, 0], embedded_z[:, 1])

            # plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}, train feature')

    def update_plots_test(self, plots, train_x, train_y, train_z, test_z, embedded_z,   
                                    test_x, test_y, test_y_pred, test_y_var, mae, mse, mean_y_s, itr):
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
            # plots.fig.suptitle(f'DKT, itr {itr}, MSE: {mse:.4f}')
            plots.fig.suptitle(f"DKT, itr {itr}, MAE: {mae:.1f} MSE: {mse:.4f}, mean support_y: {mean_y_s:.2f}")

            # test images
            x_q = test_x
            y_q = test_y 
            y_var = test_y_var
            y_pred = test_y_pred

            k = 0
            r, c = plots.ax.shape
            for i in range(r):
                for j in range(c):
                
                    img = transforms.ToPILImage()(denormalize(x_q[k]).cpu()).convert("RGB")
                    
                    plots = clear_ax(plots, i, j)
                    plots.ax[i, j].imshow(img)
                    plots = color_ax(plots, i, j, color='white')
                    # plots.ax[i, j].set_title(f'prd:{y_pred[k]:.0f}', fontsize=10)
                    plots.ax[i, j].set_xlabel(f'prd:{y_pred[k]:.1f}|gt: {y_q[k]:.1f}', fontsize=10)
                    
                    k += 1

            plots.fig.savefig(f'{self.video_path}/test_{itr}.png') 

        if self.show_plots_features:
            #features
            plots.ax_feature.clear()
            plots.ax_feature.scatter(embedded_z[:, 0], embedded_z[:, 1])
            # plots.ax_feature.legend()



class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.mean_module.register_constraint("constant", gpytorch.constraints.Positive())
        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2304) #
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

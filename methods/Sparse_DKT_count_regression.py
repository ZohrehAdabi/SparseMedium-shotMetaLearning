## Original packages
# from torch._C import ShortTensor
from numpy.core.defchararray import count
from scipy import sparse
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
import torch.nn.functional as F
from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans as Fast_KMeans
from time import gmtime, strftime
import random
## Our packages
import gpytorch
from methods.Fast_RVM_regression import Fast_RVM_regression

from statistics import mean
from data.msc44_loader import get_batch, denormalize
from configs import kernel_type
from collections import namedtuple

#Check if tensorboardx is installed
try:
    # tensorboard --logdir=./save/checkpoints/MSC44/ResNet50_DKT_Loss/ --host localhost --port 8088
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

IP = namedtuple("inducing_points", "z_values index count x y")


class Sparse_DKT_count_regression(nn.Module):
    def __init__(self, backbone, regressor, base_file=None, val_file=None,
                    sparse_method='FRVM', config="1010", align_threshold=1e-3, n_inducing_points=12, 
                    video_path=None, show_plots_loss=False, show_plots_pred=False, show_plots_features=False, training=False):
        super(Sparse_DKT_count_regression, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.regressor = regressor
        self.train_file = base_file
        self.val_file = val_file
        self.sparse_method = sparse_method
        self.num_induce_points = n_inducing_points
        self.config = config
        self.align_threshold = align_threshold
        self.do_normalize = True
        self.minmax = False
        self.device = 'cuda'
        self.video_path = video_path
        self.best_path = video_path
        self.show_plots_loss = show_plots_loss
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
        likelihood.initialize(noise=0.1)  
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type, induce_point=train_x)

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
        return gt_density_resized, labels

    def normalize(self, labels, min, max, y_mean, y_std):

        if self.minmax:
            return (labels - min)/ (max - min) + 0.5
        else:
            return  (labels - y_mean) / (y_std+1e-10)
    
    def denormalize_y(self, pred, min, max, y_mean, y_std):

        if self.minmax:
            return ((pred - 0.5) * (max - min) ) + min
            
        else:
            return  y_mean + pred * y_std
    
    def train_loop(self, epoch, n_support, n_samples, optimizer):

        mll_list = []
        mse_list = []
        mae_list = []
        validation = True
        for itr, samples in enumerate(get_batch(self.train_file, n_samples)):
            
            self.model.train()
            self.regressor.train()
            self.likelihood.train()

            inputs = samples['image']
            labels = samples['gt_count']
            gt_density = samples['gt_density']

            with torch.no_grad():
                feature = self.feature_extractor(inputs)
            #predict density map
            feature.requires_grad = True
            z = self.regressor(feature)

            with torch.no_grad():
                gt_density_resized, labels = self.resize_gt_density(z, gt_density, labels)
                if self.do_normalize:
                    y_mean, y_std = labels.mean(), labels.std()
                    y_min, y_max = labels.min(), labels.max()
                    labels_norm = self.normalize(labels, y_min, y_max, y_mean, y_std)
            
            if self.use_mse:
                density_mse = self.mse(z, gt_density_resized.squeeze(1))

            z = z.reshape(z.shape[0], -1)
            with torch.no_grad():
                inducing_points = self.get_inducing_points(z, labels_norm, n_samples, verbose=False)
            
            def inducing_max_similar_in_support_z(train_z, inducing_points):
 
                kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
                # max_similar_index
                index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
                z_inducing = train_z[index]

                return IP(z_inducing, index, inducing_points.count, 
                                    None, None)
           
            with torch.no_grad():
                if self.sparse_method=='KMeans':
                    inducing_points = inducing_max_similar_in_support_z(z.detach(), inducing_points)

            ip_values = inducing_points.z_values.cuda()
            # with torch.no_grad():
            #     inducing_points = inducing_max_similar_in_support_x(inputs, z.detach(), inducing_points, labels)
            # NOTE set inducing points 
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=True)
            self.model.train()
            if self.do_normalize:
                self.model.set_train_data(inputs=z, targets=labels_norm, strict=False)
            else:
                self.model.set_train_data(inputs=z, targets=labels, strict=False)

            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mll_list.append(np.around(loss.item(), 4))
            if self.do_normalize:
                mse = self.mse(predictions.mean, labels_norm)
            else:
                mse = self.mse(predictions.mean, labels)
            # mse_list.append(mse)
            self.iteration = (epoch*31) + itr
            if(self.writer is not None) and self.show_plots_loss: 
                self.writer.add_scalar(f'MLL_per_itr [Sparse DKT {self.sparse_method}]', loss.item(), self.iteration)
            if ((epoch%2==0) & (itr%5==0)):
                print('[%2d/%2d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr+1, epoch+1, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
            
            if (self.show_plots_pred or self.show_plots_features):
                embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
                self.update_plots_train(self.plots, labels.cpu().numpy(), embedded_z, loss.item(), mse, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()
            #****************************************************
            # validate on train data
            val_freq = 2
            if validation and (epoch%val_freq==0):
                support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
                query_ind   = [i for i in range(n_samples) if i not in support_ind]
                x_support   = inputs[support_ind,:,:,:]
                z_support   = z[support_ind, :]
                y_support = labels[support_ind]
                # x_query   = inputs[query_ind,:,:,:]
                z_query   = z[query_ind]
                y_query   = labels[query_ind]

                if self.do_normalize:
                    y_s_norm = labels_norm[support_ind]
                    y_q_norm   = labels_norm[query_ind]
                


                with torch.no_grad():
                    inducing_points = self.get_inducing_points(z_support, y_s_norm, n_support, verbose=False)
                with torch.no_grad():
                    if self.sparse_method=='KMeans':
                        inducing_points = inducing_max_similar_in_support_z(z_support.detach(), inducing_points)
                
                ip_values = inducing_points.z_values.cuda()
                self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
                self.model.covar_module._clear_cache()
                if self.do_normalize:
                    self.model.set_train_data(inputs=z_support, targets=y_s_norm, strict=False)
                else:
                    self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

                self.model.eval()
                # self.regressor.eval()
                self.likelihood.eval()

                with torch.no_grad():
                    pred    = self.likelihood(self.model(z_query.detach()))
                    lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

                if self.do_normalize:
                    mse = self.mse(pred.mean, y_q_norm).item()
                else: 
                    mse = self.mse(pred.mean, y_query).item()

                mse_list.append(mse)
                if self.do_normalize:
                    y_pred = self.denormalize_y(pred.mean, y_min, y_max, y_mean, y_std)
                else:
                    y_pred = pred.mean
                mae = self.mae(y_pred, y_query).item()
                mae_list.append(mae)
                print(Fore.YELLOW, f'epoch {epoch+1}, itr {itr+1}, Train  MAE:{mae:.2f}, MSE: {mse:.4f}', Fore.RESET)
        
        if validation and (epoch%val_freq==0):
            print(Fore.CYAN,"-"*30, f'\n epoch {epoch+1} => Avg. Val. on Train    MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
                                    f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar(f'MSE Val. on Train [Sparse DKT {self.sparse_method}]', mse, epoch)
                self.writer.add_scalar(f'MAE Val. on Train [Sparse DKT {self.sparse_method}]', mae, epoch)
        return np.mean(mll_list)

    def test_loop(self, n_support, n_samples, epoch, optimizer=None): # no optimizer needed for GP

        mse_list = []
        mae_list = []
        base_line_mae_list = []
        self.model.eval()
        self.regressor.eval()
        self.likelihood.eval()
        for itr, samples in enumerate(get_batch(self.val_file, n_samples)):
 
            inputs = samples['image']
            targets = samples['gt_count']
            gt_density = samples['gt_density']

            x_all = inputs.cuda()
            y_all = targets.cuda()
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
            z_support   = z[support_ind, :, :, :]
            gt_density_s = gt_density_resized[support_ind, :, :, :, :]
            y_query     = y_all[query_ind]
            z_query     = z[query_ind, :, :, :]
            gt_density_q = gt_density_resized[query_ind, :, :, :, :]
            
            with torch.no_grad():
               if self.do_normalize:
                    y_mean, y_std = y_all.mean(), y_all.std()
                    y_min, y_max = y_all.min(), y_all.max()
                    y_s_norm = self.normalize(y_support, y_min, y_max, y_mean, y_std)
                    y_q_norm = self.normalize(y_query, y_min, y_max, y_mean, y_std)

            z_support = z_support.reshape(z_support.shape[0], -1)
            with torch.no_grad():
                inducing_points = self.get_inducing_points(z_support, y_s_norm, n_support, verbose=False)        
            
            def inducing_max_similar_in_support_z(train_z, inducing_points):
 
                kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
                # max_similar_index
                index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
                z_inducing = train_z[index]

                return IP(z_inducing, index, inducing_points.count, 
                                    None, None)
           
            with torch.no_grad():
                if self.sparse_method=='KMeans':
                    inducing_points = inducing_max_similar_in_support_z(z_support, inducing_points)
            
            ip_values = inducing_points.z_values.cuda()
            # inducing_points = inducing_max_similar_in_support_x(x_support, z_support.detach(), inducing_points, y_support)
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)

            if self.do_normalize:
                self.model.set_train_data(inputs=z_support, targets=y_s_norm, strict=False)
            else:
                self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

            with torch.no_grad():
                z_query = z_query.reshape(z_query.shape[0], -1)
                pred    = self.likelihood(self.model(z_query))
                lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

            if self.do_normalize:
                mse = self.mse(pred.mean, y_q_norm).item()
            else:
                mse = self.mse(pred.mean, y_query).item()
            mse_list.append(mse)
            if self.do_normalize:
                y_pred = self.denormalize_y(pred.mean.detach(), y_min, y_max, y_mean, y_std)
            else:
                y_pred = pred.mean
            mae = self.mae(y_pred, y_query).item()
            mae_list.append(mae)
            #**************************************************************
            y = y_query.detach().cpu().numpy() 
            y_pred = y_pred.cpu().numpy() 
            mean_support_y = y_support.mean()
            base_line_mae = self.mae(mean_support_y.repeat(y_query.shape[0]), y_query).item()
            base_line_mae_list.append(base_line_mae)
            print(Fore.RED,"="*50, Fore.RESET)
            print(f'itr #{itr}')
            print(f'mean of support_y {mean_support_y:.2f}')
            print(f'base line MAE: {base_line_mae:.2f}')
            print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
            print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
            print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
            print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
            print(Fore.RED,"-"*50, Fore.RESET)

            # K = self.model.base_covar_module
            # kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
            # max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
            # y_s = y_support.detach().cpu().numpy()
            # print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
            
            # kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
            # max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
            # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

            # kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
            # max_similar_index = np.argmax(kernel_matrix, axis=1)
            # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
            #**************************************************************
            if (self.show_plots_pred or self.show_plots_features):
                embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

                def inducing_points_in_support_x(inducing_points, train_x, train_y):
  
                    index =  inducing_points.index
                    x_inducing = train_x[index]
                    y_inducing = train_y[index].cpu().numpy()

                    return IP(inducing_points.z_values, index, inducing_points.count, 
                                        x_inducing, y_inducing)

                inducing_points = inducing_points_in_support_x(inducing_points, x_support, y_support)

                self.update_plots_test(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                                z_support.detach(), z_query.detach(), embedded_z_support,
                                                inducing_points, x_query, y_query.detach().cpu().numpy(), y_pred, pred.variance.detach().cpu().numpy() 
                                                , mae, mse, itr+1)
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

    
    def train(self, stop_epoch, n_support, n_samples, optimizer,  id, use_mse):

        self.use_mse = use_mse
        self.feature_extractor.eval()
        mll_list = []
        best_mae, best_rmse = 10e7, 10e7
        for epoch in range(stop_epoch):
            
            
            mll = self.train_loop(epoch, n_support, n_samples, optimizer)
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar(f'MLL_per_epoch  [Sparse DKT {self.sparse_method}]', mll, epoch)

            print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)

            print(Fore.GREEN,"-"*30, f'\nValidation:', Fore.RESET)

            val_mse, val_mae, val_rmse = self.test_loop(n_support, n_samples, epoch, optimizer)
            
            mll_list.append(np.around(mll, 4))

            if best_mae >= val_mae:
                best_mae = val_mae
                best_rmse = val_rmse
                model_name = self.best_path + f'_best_mae{best_mae:.2f}_ep{epoch}_{id}.pth'
                self.save_checkpoint(model_name)
                print(Fore.LIGHTRED_EX, f'Best MAE: {best_mae:.2f}, RMSE: {best_rmse}', Fore.RESET)
            if(self.writer is not None) and self.show_plots_loss:
                self.writer.add_scalar(f'MSE Val. [Sparse DKT {self.sparse_method}]', val_mse, epoch)
                self.writer.add_scalar(f'MAE Val. [Sparse DKT {self.sparse_method}]', val_mae, epoch)
            print(Fore.GREEN,"-"*30, Fore.RESET)
            # print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)
        
        mll = np.mean(mll_list)
        
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()

        return mll, mll_list
    
    def test(self, n_support, n_samples, optimizer=None, n_test_epoch=None): # no optimizer needed for GP
        
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


    def get_inducing_points(self, inputs, targets, n_samples, verbose=True):

        
        IP_index = np.array([])

        if self.sparse_method=='FRVM':
            # with sigma and updating sigma converges to more sparse solution
            N   = inputs.shape[0]
            tol = 1e-6
            eps = torch.finfo(torch.float32).eps
            max_itr = 1000
            sigma = self.model.likelihood.noise[0].clone()
            # sigma = torch.tensor([0.0000001])
            # sigma = torch.tensor([torch.var(targets) * 0.1]) #sigma^2
            sigma = sigma.to(self.device)
            beta = 1 /(sigma + eps)
            scale = True
            covar_module = self.model.base_covar_module
            kernel_matrix = covar_module(inputs).evaluate()
            # normalize kernel
            if scale:
                scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
                # print(f'scale: {Scales}')
                scales[scales==0] = 1
                kernel_matrix = kernel_matrix / scales

            kernel_matrix = kernel_matrix.to(torch.float64)
            targets = targets.to(torch.float64)
            active, alpha, Gamma, beta = Fast_RVM_regression(kernel_matrix, targets.view(-1, 1), beta, N, self.config, self.align_threshold,
                                                    eps, tol, max_itr, self.device, verbose)

            active = np.sort(active)
            inducing_points = inputs[active]
            num_IP = active.shape[0]
            IP_index = active

        elif self.sparse_method=='KMeans':
            num_IP = self.num_induce_points
            
            # self.kmeans_clustering = KMeans(n_clusters=num_IP, init='k-means++',  n_init=10, max_iter=1000).fit(inputs.cpu().numpy())
            # inducing_points = self.kmeans_clustering.cluster_centers_
            # inducing_points = torch.from_numpy(inducing_points).to(torch.float)

            self.kmeans_clustering = Fast_KMeans(n_clusters=num_IP, max_iter=1000)
            self.kmeans_clustering.fit(inputs.cuda())
            inducing_points = self.kmeans_clustering.centroids
            # print(inducing_points.shape[0])

        else:
            num_IP = self.num_induce_points
            IP_index = np.random.choice(list(range(n_samples)), replace=False, size=self.num_induce_points)

            z = self.feature_extractor(inputs)

            inducing_points = z[IP_index,:]


        return IP(inducing_points, IP_index, num_IP, None, None)
  

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.regressor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
    
        ckpt = torch.load(checkpoint)
 
        IP = torch.ones(self.num_induce_points, 2916).cuda()
        ckpt['gp']['covar_module.inducing_points'] = IP
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.regressor.load_state_dict(ckpt['net'])

    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_now = datetime.now().strftime('%Y-%m-%d--%H-%M')
            writer_path = self.video_path +'_Loss' +f"/{id}"
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
        fig, ax = plt.subplots(4, 5, figsize=(5,6), sharex=True, sharey=True, dpi=150) 
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

           
            plots.ax_feature.set_title(f'Sparse DKT {self.sparse_method}, epoch {epoch}, MLL: {mll}, MSE:{mse}, train feature Z') 
   
    def update_plots_test(self, plots, train_x, train_y, train_z, test_z, embedded_z, inducing_points,   
                                    test_x, test_y, test_y_pred, test_y_var, mae, mse, itr):
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
            
            plots.fig.suptitle(f"Sparse DKT ({self.sparse_method}), itr {itr}, MAE: {mae:.1f} MSE: {mse:.4f}, num IP: {inducing_points.count}")

            cluster_colors = ['aqua', 'coral', 'lime', 'gold', 'purple', 'green', 'tomato', 
                                'fuchsia', 'chocolate', 'chartreuse', 'orange', 'teal']

            #train images
            # test images
            x_q = test_x
            y_q = test_y 
            y_var = test_y_var
            y_pred = test_y_pred

            k = 0
            r, c = plots.ax.shape
            for i in range(2):
                for j in range(c):
                
                    img = transforms.ToPILImage()(denormalize(x_q[k]).cpu()).convert("RGB")
                    
                    plots = clear_ax(plots, i, j)
                    plots.ax[i, j].imshow(img)
                    plots = color_ax(plots, i, j, color='white')
                    # plots.ax[i, j].set_title(f'prd:{y_pred[k]:.0f}', fontsize=10)
                    plots.ax[i, j].set_xlabel(f'prd:{y_pred[k]:.1f}|gt: {y_q[k]:.1f}', fontsize=10)
                    
                    k += 1
          
            # highlight inducing points
            x_inducing = inducing_points.x
            y_inducing = inducing_points.y
   
            k = 0
            r, c = plots.ax.shape
            for i in range(2, r):
                for j in range(c):
                
                    img = transforms.ToPILImage()(denormalize(x_inducing[k]).cpu()).convert("RGB")
                    
                    plots = clear_ax(plots, i, j)
                    plots.ax[i, j].imshow(img)
                    plots = color_ax(plots, i, j, color='white')
                    # plots.ax[i, j].set_title(f'prd:{y_pred[k]:.0f}', fontsize=10)
                    plots.ax[i, j].set_xlabel(f'gt: {y_inducing[k]:.0f}', fontsize=10)
                    
                    k += 1
        
            plots.fig.savefig(f'{self.video_path}/test_{itr}.png')    

        if self.show_plots_features:
            #features
            plots.ax_feature.clear()             
            plots.ax_feature.scatter(embedded_z[:, 0], embedded_z[:, 1])

            # plots.ax_feature.legend()




######## OLD
    # def train_loop_kmeans(self, epoch, n_support, n_samples, optimizer):

    #     mll_list = []
    #     mse_list = []
    #     mae_list = []
    #     validation = True
    #     for itr, samples in enumerate(get_batch(self.train_file, n_samples)):
            
    #         self.model.train()
    #         self.feature_extractor.train()
    #         self.likelihood.train()

    #         inputs = samples['image']
    #         labels = samples['gt_count']
    #         gt_density = samples['gt_density']

    #         with torch.no_grad():
    #             feature = self.feature_extractor(inputs)
    #         #predict density map
    #         z = self.regressor(feature)

    #         with torch.no_grad():
    #             inducing_points = self.get_inducing_points(z, labels, verbose=False)
            
    #         def inducing_max_similar_in_support_x(train_x, train_z, inducing_points, train_y):
    #             y = train_y.cpu().numpy()
    #             # self.model.covar_module._clear_cache()
    #             # kernel_matrix = self.model.covar_module(inducing_points.z_values, train_z).evaluate()
    #             kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
    #             # max_similar_index
    #             index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
    #             x_inducing = train_x[index].cpu().numpy()
    #             y_inducing = y[index]
    #             z_inducing = train_z[index]
    #             # i_idx = []
    #             # j_idx = []
    #             # for r in range(index.shape[0]):
                    
    #             #     t = y_inducing[r]
    #             #     x_t_idx = np.where(y==t)[0]
    #             #     x_t = train_x[x_t_idx].detach().cpu().numpy()
    #             #     j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
    #             #     i = int(t/10-6)
    #             #     i_idx.append(i)
    #             #     j_idx.append(j)

    #             return IP(z_inducing, index, inducing_points.count, 
    #                                 x_inducing, y_inducing)
            
    #         with torch.no_grad():
    #             inducing_points = inducing_max_similar_in_support_x(inputs, z.detach(), inducing_points, labels)

    #         ip_values = inducing_points.z_values.cuda()
    #         # with torch.no_grad():
    #         #     inducing_points = inducing_max_similar_in_support_x(inputs, z.detach(), inducing_points, labels)
            
    #         self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #         self.model.train()
    #         self.model.set_train_data(inputs=z, targets=labels, strict=False)

    #         # z = self.feature_extractor(x_query)
    #         predictions = self.model(z)
    #         loss = -self.mll(predictions, self.model.train_targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         mll_list.append(loss.item())
    #         mse = self.mse(predictions.mean, labels)
    #         mse_list.append(mse)
    #         if ((epoch%2==0) & (itr%5==0)):
    #             print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
    #                 itr, epoch, loss.item(), mse.item(),
    #                 self.model.likelihood.noise.item()
    #             ))
            
    #         if (self.show_plots_pred or self.show_plots_features) and not self.f_rvm:
    #             embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    #             self.update_plots_train_kmeans(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

    #             if self.show_plots_pred:
    #                 self.plots.fig.canvas.draw()
    #                 self.plots.fig.canvas.flush_events()
    #                 self.mw.grab_frame()
    #             if self.show_plots_features:
    #                 self.plots.fig_feature.canvas.draw()
    #                 self.plots.fig_feature.canvas.flush_events()
    #                 self.mw_feature.grab_frame()
    #         #****************************************************
    #         # validate on train data
            
    #         if validation:
    #             support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #             query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #             z_support = z[support_ind, :]
    #             y_support = labels[support_ind]
    #             z_query   = z[query_ind]
    #             y_query   = labels[query_ind]

    #             with torch.no_grad():
    #                 inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
            
    #             ip_values = inducing_points.z_values.cuda()
    #             self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #             self.model.covar_module._clear_cache()
    #             self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #             self.model.eval()
    #             self.feature_extractor.eval()
    #             self.likelihood.eval()

    #             with torch.no_grad():
    #                 pred    = self.likelihood(self.model(z_query.detach()))
    #                 lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #             mse = self.mse(pred.mean, y_query).item()
    #             mse_list.append(mse)
    #             mae = self.mae(pred.mean, y_query).item()
    #             mae_list.append(mae)
    #             print(Fore.YELLOW, f'epoch {epoch}, itr {itr+1}, Train  MAE:{mae:.2f}, MSE: {mse:.4f}', Fore.RESET)
        
    #     if validation:
    #         print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg. Val. on Train    MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                 f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
    #     return np.mean(mll_list)

    # def train_loop_fast_rvm(self, epoch, n_support, n_samples, optimizer):
            
    #         mll_list = []
    #         mse_list = []
    #         mae_list = []
    #         validation = True
    #         for itr, samples in enumerate(get_batch(self.train_file, n_samples)):
    #             self.model.train()
    #             self.feature_extractor.train()
    #             self.likelihood.train()

    #             inputs = samples['image']
    #             labels = samples['gt_count']
    #             z = self.feature_extractor(inputs)
    #             with torch.no_grad():
    #                 inducing_points = self.get_inducing_points(z, labels, verbose=False)
                
    #             ip_values = inducing_points.z_values.cuda()
    #             self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #             self.model.train()
    #             self.model.set_train_data(inputs=z, targets=labels, strict=False)

    #             # z = self.feature_extractor(x_query)
    #             predictions = self.model(z)
    #             loss = -self.mll(predictions, self.model.train_targets)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             mll_list.append(loss.item())
    #             mse = self.mse(predictions.mean, labels)

    #             if ((epoch%2==0) & (itr%5==0)):
    #                 print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
    #                     itr, epoch, loss.item(), mse.item(),
    #                     self.model.likelihood.noise.item()
    #                 ))
                
    #             if (self.show_plots_pred or self.show_plots_features) and  self.f_rvm:
    #                 embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    #                 self.update_plots_train_fast_rvm(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

    #                 if self.show_plots_pred:
    #                     self.plots.fig.canvas.draw()
    #                     self.plots.fig.canvas.flush_events()
    #                     self.mw.grab_frame()
    #                 if self.show_plots_features:
    #                     self.plots.fig_feature.canvas.draw()
    #                     self.plots.fig_feature.canvas.flush_events()
    #                     self.mw_feature.grab_frame()
    #             #****************************************************
    #             # validate on train data
                
    #             if validation:
    #                 support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #                 query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #                 z_support = z[support_ind, :]
    #                 y_support = labels[support_ind]
    #                 z_query   = z[query_ind]
    #                 y_query   = labels[query_ind]

    #                 with torch.no_grad():
    #                     inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
                
    #                 ip_values = inducing_points.z_values.cuda()
    #                 self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #                 self.model.covar_module._clear_cache()
    #                 self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #                 self.model.eval()
    #                 self.feature_extractor.eval()
    #                 self.likelihood.eval()

    #                 with torch.no_grad():
    #                     pred    = self.likelihood(self.model(z_query.detach()))
    #                     lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #                 mse = self.mse(pred.mean, y_query).item()
    #                 mse_list.append(mse)
    #                 mae = self.mae(pred.mean, y_query).item()
    #                 mae_list.append(mae)
    #                 print(Fore.YELLOW, f'epoch {epoch}, itr {itr+1}, Train  MAE:{mae:.2f}, MSE: {mse:.4f}', Fore.RESET)
            
    #         if validation:
    #             print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg. Val. on Train    MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                     f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
    #         return np.mean(mll_list)

    # def test_loop_kmeans(self, n_support, n_samples, epoch, optimizer=None): # no optimizer needed for GP

    #     mse_list = []
    #     mae_list = []
    #     self.model.eval()
    #     self.regressor.eval()
    #     self.likelihood.eval()
    #     for itr, samples in enumerate(get_batch(self.val_file, n_samples)):
 
    #         inputs = samples['image']
    #         targets = samples['gt_count']

    #         x_all = inputs.cuda()
    #         y_all = targets.cuda()
    #         support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #         query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #         x_support = x_all[support_ind,:,:,:]
    #         y_support = y_all[support_ind]
    #         x_query   = x_all[query_ind,:,:,:]
    #         y_query   = y_all[query_ind]

    #         z_support = self.feature_extractor(x_support).detach()
    #         with torch.no_grad():
    #             inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
            
            
    #         def inducing_max_similar_in_support_x(train_x, train_z, inducing_points, train_y):
    #             y = ((train_y.cpu().numpy() + 1) * 60 / 2) + 60
        
    #             # kernel_matrix = self.model.covar_module(inducing_points.z_values, train_z).evaluate()
    #             kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
    #             # max_similar_index
    #             index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
    #             x_inducing = train_x[index].cpu().numpy()
    #             y_inducing = y[index]
    #             z_inducing = train_z[index]
    #             i_idx = []
    #             j_idx = []
    #             for r in range(index.shape[0]):
                    
    #                 t = y_inducing[r]
    #                 x_t_idx = np.where(y==t)[0]
    #                 x_t = train_x[x_t_idx].detach().cpu().numpy()
    #                 j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
    #                 i = int(t/10-6)
    #                 i_idx.append(i)
    #                 j_idx.append(j)

    #             return IP(z_inducing, index, inducing_points.count, 
    #                                 x_inducing, y_inducing, np.array(i_idx), np.array(j_idx))
            
    #         inducing_points = inducing_max_similar_in_support_x(x_support, z_support.detach(), inducing_points, y_support)
    #         ip_values = inducing_points.z_values.cuda()
    #         # inducing_points = inducing_max_similar_in_support_x(x_support, z_support.detach(), inducing_points, y_support)
    #         self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)

    #         self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #         with torch.no_grad():
    #             z_query = self.feature_extractor(x_query).detach()
    #             pred    = self.likelihood(self.model(z_query))
    #             lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #         mse = self.mse(pred.mean, y_query).item()
    #         mse_list.append(mse)
    #         mae = self.mae(pred.mean, y_query).item()
    #         mae_list.append(mae)
    #         #**************************************************************
    #         y = ((y_query.detach().cpu().numpy() + 1) * 60 / 2) + 60
    #         y_pred = ((pred.mean.detach().cpu().numpy() + 1) * 60 / 2) + 60
    #         print(Fore.RED,"="*50, Fore.RESET)
    #         print(f'itr #{itr}')
    #         print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
    #         print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
    #         print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
    #         print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
    #         print(Fore.RED,"-"*50, Fore.RESET)

    #         K = self.model.base_covar_module
    #         kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
    #         y_s = y_support.detach().cpu().numpy()
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
            
    #         kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

    #         # kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         # max_similar_index = np.argmax(kernel_matrix, axis=1)
    #         # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
    #         #**************************************************************
    #         if (self.show_plots_pred or self.show_plots_features) and not self.f_rvm:
    #             embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

    #             self.update_plots_test_kmeans(self.plots, x_support, y_support.detach().cpu().numpy(), 
    #                                             z_support.detach(), z_query.detach(), embedded_z_support,
    #                                             inducing_points, x_query, y_query.detach().cpu().numpy(), pred, 
    #                                             max_similar_idx_x_s, max_similar_idx_x_ip, None, mse, itr)
    #             if self.show_plots_pred:
    #                 self.plots.fig.canvas.draw()
    #                 self.plots.fig.canvas.flush_events()
    #                 self.mw.grab_frame()
    #             if self.show_plots_features:
    #                 self.plots.fig_feature.canvas.draw()
    #                 self.plots.fig_feature.canvas.flush_events()
    #                 self.mw_feature.grab_frame()
    #     print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg.   MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                 f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
   
    #     return np.mean(mse_list), np.mean(mae_list), np.sqrt(np.mean(mse_list))

    # def test_loop_fast_rvm(self, n_support, n_samples, epoch, optimizer=None): # no optimizer needed for GP
        
    #     mse_list = []
    #     mae_list = []
    #     for itr, samples in enumerate(get_batch(self.val_file, n_samples)):
 
    #         inputs = samples['image']
    #         targets = samples['gt_count']

    #         x_all = inputs.cuda()
    #         y_all = targets.cuda()
    #         support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #         query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #         x_support = x_all[support_ind,:,:,:]
    #         y_support = y_all[support_ind]
    #         x_query   = x_all[query_ind,:,:,:]
    #         y_query   = y_all[query_ind]

    #         z_support = self.feature_extractor(x_support).detach()
    #         with torch.no_grad():
    #             inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
            
    #         ip_values = inducing_points.z_values.cuda()
    #         self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #         self.model.covar_module._clear_cache()
    #         self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #         self.model.eval()
    #         self.feature_extractor.eval()
    #         self.likelihood.eval()

    #         with torch.no_grad():
    #             z_query = self.feature_extractor(x_query).detach()
    #             pred    = self.likelihood(self.model(z_query))
    #             lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #         mse = self.mse(pred.mean, y_query).item()
    #         mse_list.append(mse)
    #         mae = self.mae(pred.mean, y_query).item()
    #         mae_list.append(mae)
    #         def inducing_max_similar_in_support_x(train_x, inducing_points, train_y):
    #             y = train_y.detach().cpu().numpy() 
    #             index = inducing_points.index
    #             x_inducing = train_x[index].detach().cpu().numpy()
    #             y_inducing = y[index]
    #             i_idx = []
    #             j_idx = []
    #             for r in range(index.shape[0]):
                    
    #                 t = y_inducing[r]
    #                 x_t_idx = np.where(y==t)[0]
    #                 x_t = train_x[x_t_idx].detach().cpu().numpy()
    #                 j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
    #                 i = int(t/10-6)
    #                 i_idx.append(i)
    #                 j_idx.append(j)

    #             return IP(inducing_points.z_values, index, inducing_points.count, 
    #                                 x_inducing, y_inducing)
            
    #         index = inducing_points.index
    #         inducing_points = IP(inducing_points.z_values, index, inducing_points.count,
    #                                 x_support[index], y_support[index])

    #         #**************************************************************
    #         y = y_query.detach().cpu().numpy() 
    #         y_pred = pred.mean.detach().cpu().numpy() 
    #         print(Fore.RED,"="*50, Fore.RESET)
    #         print(f'itr #{itr}')
    #         print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
    #         print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
    #         print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
    #         print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
    #         print(Fore.RED,"-"*50, Fore.RESET)

    #         K = self.model.base_covar_module
    #         kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
    #         y_s = y_support.detach().cpu().numpy() 
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
            
    #         kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

    #         # kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         # max_similar_index = np.argmax(kernel_matrix, axis=1)
    #         # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
    #         #**************************************************************
    #         if (self.show_plots_pred or self.show_plots_features) and self.f_rvm:
    #             embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())
    #             self.update_plots_test_fast_rvm(self.plots, x_support, y_support.detach().cpu().numpy(), 
    #                                             z_support.detach(), z_query.detach(), embedded_z_support,
    #                                             inducing_points, x_query, y_query.detach().cpu().numpy(), pred, 
    #                                             max_similar_idx_x_s, max_similar_idx_x_ip, None, mse, itr)
    #             if self.show_plots_pred:
    #                 self.plots.fig.canvas.draw()
    #                 self.plots.fig.canvas.flush_events()
    #                 self.mw.grab_frame()
    #             if self.show_plots_features:
    #                 self.plots.fig_feature.canvas.draw()
    #                 self.plots.fig_feature.canvas.flush_events()
    #                 self.mw_feature.grab_frame()

    #     print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg.   MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                 f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
   
    #     return np.mean(mse_list), np.mean(mae_list), np.sqrt(np.mean(mse_list))

    # def train_loop_random(self, epoch, n_support, n_samples, optimizer):

    #     mll_list = []
    #     mse_list = []
    #     mae_list = []
    #     validation = True
    #     for itr, samples in enumerate(get_batch(self.train_file, n_samples)):
    #         self.model.train()
    #         self.feature_extractor.train()
    #         self.likelihood.train()

    #         inputs = samples['image']
    #         labels = samples['gt_count']

    #         inducing_points_index = list(np.random.choice(list(range(n_samples)), replace=False, size=self.num_induce_points))

    #         z = self.feature_extractor(inputs)

    #         inducing_points_z = z[inducing_points_index,:]
            
    #         ip_values = inducing_points_z.cuda()
    #         self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #         self.model.train()
    #         self.model.set_train_data(inputs=z, targets=labels, strict=False)

    #         # z = self.feature_extractor(x_query)
    #         predictions = self.model(z)
    #         loss = -self.mll(predictions, self.model.train_targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         mll_list.append(loss.item())
    #         mse = self.mse(predictions.mean, labels)
            
    #         if ((epoch%2==0) & (itr%5==0)):
    #             print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
    #                 itr, epoch, loss.item(), mse.item(),
    #                 self.model.likelihood.noise.item()
    #             ))
            
    #         if (self.show_plots_pred or self.show_plots_features) and self.random:
    #             embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    #             self.update_plots_train_kmeans(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

    #             if self.show_plots_pred:
    #                 self.plots.fig.canvas.draw()
    #                 self.plots.fig.canvas.flush_events()
    #                 self.mw.grab_frame()
    #             if self.show_plots_features:
    #                 self.plots.fig_feature.canvas.draw()
    #                 self.plots.fig_feature.canvas.flush_events()
    #                 self.mw_feature.grab_frame()
    #         #****************************************************
    #         # validate on train data
            
    #         if validation:
    #             support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #             query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #             z_support = z[support_ind,:]
    #             y_support = labels[support_ind]
    #             z_query   = z[query_ind]
    #             y_query   = labels[query_ind]

    #             inducing_points_index = list(np.random.choice(list(range(n_support)), replace=False, size=self.num_induce_points))

    #             inducing_points_z = z_support[inducing_points_index,:]
                
    #             ip_values = inducing_points_z.cuda()
    #             self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #             self.model.covar_module._clear_cache()
    #             self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #             self.model.eval()
    #             self.feature_extractor.eval()
    #             self.likelihood.eval()

    #             with torch.no_grad():
    #                 pred    = self.likelihood(self.model(z_query.detach()))
    #                 lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #             mse = self.mse(pred.mean, y_query).item()
    #             mse_list.append(mse)
    #             mae = self.mae(pred.mean, y_query).item()
    #             mae_list.append(mae)
    #             print(Fore.YELLOW, f'epoch {epoch}, itr {itr+1}, Train  MAE:{mae:.2f}, MSE: {mse:.4f}', Fore.RESET)
        
    #     if validation:
    #         print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg. Val. on Train    MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                 f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
    #     return np.mean(mll_list)
    
    # def test_loop_random(self, n_support, n_samples, epoch, optimizer=None): # no optimizer needed for GP
    #     mse_list = []
    #     mae_list = []
    #     for itr, samples in enumerate(get_batch(self.val_file, n_samples)):
 
    #         inputs = samples['image']
    #         targets = samples['gt_count']

    #         x_all = inputs.cuda()
    #         y_all = targets.cuda()
    #         support_ind = np.random.choice(np.arange(n_samples), size=n_support, replace=False)
    #         query_ind   = [i for i in range(n_samples) if i not in support_ind]
    #         x_support = x_all[support_ind,:,:,:]
    #         y_support = y_all[support_ind]
    #         x_query   = x_all[query_ind,:,:,:]
    #         y_query   = y_all[query_ind]


    #         inducing_points_index = np.random.choice(list(range(n_support)), replace=False, size=self.num_induce_points)

    #         z_support = self.feature_extractor(x_support).detach()

    #         inducing_points_z = z_support[inducing_points_index,:]

    #         ip_values = inducing_points_z.cuda()
    #         self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
    #         self.model.covar_module._clear_cache()
    #         self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

    #         self.model.eval()
    #         self.feature_extractor.eval()
    #         self.likelihood.eval()

    #         with torch.no_grad():
    #             z_query = self.feature_extractor(x_query).detach()
    #             pred    = self.likelihood(self.model(z_query))
    #             lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

    #         mse = self.mse(pred.mean, y_query).item()
    #         mse_list.append(mse)
    #         mae = self.mae(pred.mean, y_query).item()
    #         mae_list.append(mae)

    #         def inducing_max_similar_in_support_x(train_x, inducing_points_z, inducing_points_index, train_y):
    #             y = ((train_y.detach().cpu().numpy() + 1) * 60 / 2) + 60
        
    #             index = inducing_points_index
    #             x_inducing = train_x[index].detach().cpu().numpy()
    #             y_inducing = y[index]
    #             i_idx = []
    #             j_idx = []
    #             for r in range(index.shape[0]):
                    
    #                 t = y_inducing[r]
    #                 x_t_idx = np.where(y==t)[0]
    #                 x_t = train_x[x_t_idx].detach().cpu().numpy()
    #                 j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
    #                 i = int(t/10-6)
    #                 i_idx.append(i)
    #                 j_idx.append(j)

    #             return IP(inducing_points_z, index, index.shape, 
    #                                 x_inducing, y_inducing)
            
    #         index = inducing_points_index
    #         inducing_points = IP(inducing_points_z, index, index.shape, x_support[index], y_support[index])
  
    #         #**************************************************************
    #         y = y_query.detach().cpu().numpy() 
    #         y_pred = pred.mean.detach().cpu().numpy() 
    #         print(Fore.RED,"="*50, Fore.RESET)
    #         print(f'itr #{itr}')
    #         print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
    #         print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
    #         print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
    #         print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
    #         print(Fore.RED,"-"*50, Fore.RESET)

    #         K = self.model.base_covar_module
    #         kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
    #         y_s = y_support.detach().cpu().numpy()
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
            
    #         kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
    #         print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

    #         # kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
    #         # max_similar_index = np.argmax(kernel_matrix, axis=1)
    #         # print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
    #         #**************************************************************
    #         if (self.show_plots_pred or self.show_plots_features) and  self.random:
    #             embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())
    #             self.update_plots_test_fast_rvm(self.plots, x_support, y_support.detach().cpu().numpy(), 
    #                                             z_support.detach(), z_query.detach(), embedded_z_support,
    #                                             inducing_points, x_query, y_query.detach().cpu().numpy(), pred, 
    #                                             max_similar_idx_x_s, max_similar_idx_x_ip, None, mse, itr)
    #             if self.show_plots_pred:
    #                 self.plots.fig.canvas.draw()
    #                 self.plots.fig.canvas.flush_events()
    #                 self.mw.grab_frame()
    #             if self.show_plots_features:
    #                 self.plots.fig_feature.canvas.draw()
    #                 self.plots.fig_feature.canvas.flush_events()
    #                 self.mw_feature.grab_frame()

    #     print(Fore.CYAN,"-"*30, f'\n epoch {epoch} => Avg.   MAE: {np.mean(mae_list):.2f}, RMSE: {np.sqrt(np.mean(mse_list)):.2f}'
    #                                 f', MSE: {np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}\n', "-"*30, Fore.RESET)
   
    #     return np.mean(mse_list), np.mean(mae_list), np.sqrt(np.mean(mse_list))

    # def update_plots_test_fast_rvm(self, plots, train_x, train_y, train_z, test_z, embedded_z, inducing_points,   
    #                                 test_x, test_y, test_y_pred, similar_idx_x_s, similar_idx_x_ip, mll, mse, itr):
    #     def clear_ax(plots, i, j):
    #         plots.ax[i, j].clear()
    #         plots.ax[i, j].set_xticks([])
    #         plots.ax[i, j].set_xticklabels([])
    #         plots.ax[i, j].set_yticks([])
    #         plots.ax[i, j].set_yticklabels([])
    #         plots.ax[i, j].set_aspect('equal')
    #         return plots
        
    #     def color_ax(plots, i, j, color, lw=0):
    #         if lw > 0:
    #             for axis in ['top','bottom','left','right']:
    #                 plots.ax[i, j].spines[axis].set_linewidth(lw)
    #         #magenta, orange
    #         for axis in ['top','bottom','left','right']:
    #             plots.ax[i, j].spines[axis].set_color(color)

    #         return plots

    #     if self.show_plots_pred:

    #         plots.fig.suptitle(f"Sparse DKT ({self.sparse_method}), itr {itr}, MSE: {mse:.4f}, num IP: {inducing_points.count}")

    #         cluster_colors = ['aqua', 'coral', 'lime', 'gold', 'purple', 'green', 'tomato', 
    #                             'fuchsia', 'chocolate', 'chartreuse', 'orange', 'teal']

    #         #train images
    #         # test images
    #         x_q = test_x
    #         y_q = test_y 
    #         y_mean = test_y_pred.mean.detach().cpu().numpy()
    #         y_var = test_y_pred.variance.detach().cpu().numpy()
    #         y_pred = y_mean

    #         k = 0
    #         r, c = plots.ax.shape
    #         for i in range(2):
    #             for j in range(c):
                
    #                 img = transforms.ToPILImage()(x_q[k]).convert("RGB")
                    
    #                 plots = clear_ax(plots, i, j)
    #                 plots.ax[i, j].imshow(img)
    #                 plots = color_ax(plots, i, j, color='white')
    #                 # plots.ax[i, j].set_title(f'prd:{y_pred[k]:.0f}', fontsize=10)
    #                 plots.ax[i, j].set_xlabel(f'prd:{y_pred[k]:.0f}|gt: {y_q[k]:.0f}', fontsize=10)
                    
    #                 k += 1
          
    #         # highlight inducing points
    #         x_inducing = inducing_points.x
    #         y_inducing = inducing_points.y
   
    #         k = 0
    #         r, c = plots.ax.shape
    #         for i in range(2, r):
    #             for j in range(c):
                
    #                 img = transforms.ToPILImage()(x_inducing[k]).convert("RGB")
                    
    #                 plots = clear_ax(plots, i, j)
    #                 plots.ax[i, j].imshow(img)
    #                 plots = color_ax(plots, i, j, color='white')
    #                 # plots.ax[i, j].set_title(f'prd:{y_pred[k]:.0f}', fontsize=10)
    #                 plots.ax[i, j].set_xlabel(f'gt: {y_inducing[k]:.0f}', fontsize=10)
                    
    #                 k += 1
        
    #         plots.fig.savefig(f'{self.video_path}/test_{itr}.png')    
        
    #     if self.show_plots_features:
    #         #features
 
    #         plots.ax_feature.scatter(embedded_z[:, 0], embedded_z[:, 1])

    #         plots.ax_feature.legend()


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear', induce_point=None):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=induce_point , likelihood=likelihood)
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



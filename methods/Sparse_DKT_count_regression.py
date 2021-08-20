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
import random
## Our packages
import gpytorch
from methods.Fast_RVM_regression import Fast_RVM_regression

from statistics import mean
from data.msc44_loader import get_batch
from configs import kernel_type
from collections import namedtuple

IP = namedtuple("inducing_points", "z_values index count x y i_idx j_idx")
class Sparse_DKT_count_regression(nn.Module):
    def __init__(self, backbone, regressor, base_file=None, val_file=None,
                    f_rvm=True, config="0000", align_threshold=1e-3, n_inducing_points=12, random=False, 
                    video_path=None, show_plots_pred=False, show_plots_features=False, training=False):
        super(Sparse_DKT_count_regression, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.regressor = regressor
        if training:
            self.train_file = base_file
            self.val_file = val_file
        else:
            self.test_file = val_file
        self.num_induce_points = n_inducing_points
        self.config = config
        self.align_threshold = align_threshold
        self.f_rvm = f_rvm
        self.random = random
        self.device = 'cuda'
        self.video_path = video_path
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
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type, induce_point=train_x)

        self.model      = model.cuda()
        self.likelihood = likelihood.cuda()
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train_loop_kmeans(self, epoch, n_support, n_samples, optimizer):
        
        # batch, batch_labels = get_batch(train_people, n_samples)
        # batch, batch_labels = batch.cuda(), batch_labels.cuda()
        mll_list = []
        for itr, (inputs, labels) in enumerate(get_batch):

            z = self.feature_extractor(inputs)
            with torch.no_grad():
                inducing_points = self.get_inducing_points(z, labels, verbose=False)
            
            def inducing_max_similar_in_support_x(train_x, train_z, inducing_points, train_y):
                y = ((train_y.cpu().numpy() + 1) * 60 / 2) + 60
                # self.model.covar_module._clear_cache()
                # kernel_matrix = self.model.covar_module(inducing_points.z_values, train_z).evaluate()
                kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
                # max_similar_index
                index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
                x_inducing = train_x[index].cpu().numpy()
                y_inducing = y[index]
                z_inducing = train_z[index]
                i_idx = []
                j_idx = []
                # for r in range(index.shape[0]):
                    
                #     t = y_inducing[r]
                #     x_t_idx = np.where(y==t)[0]
                #     x_t = train_x[x_t_idx].detach().cpu().numpy()
                #     j = np.argmin(np.linalg.norm(x_inducing[r].reshape(-1) - x_t.reshape(15, -1), axis=-1))
                #     i = int(t/10-6)
                #     i_idx.append(i)
                #     j_idx.append(j)

                return IP(z_inducing, index, inducing_points.count, 
                                    x_inducing, y_inducing, None, None)
           
            with torch.no_grad():
                inducing_points = inducing_max_similar_in_support_x(inputs, z.detach(), inducing_points, labels)

            ip_values = inducing_points.z_values.cuda()
            # with torch.no_grad():
            #     inducing_points = inducing_max_similar_in_support_x(inputs, z.detach(), inducing_points, labels)
            
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
            self.model.train()
            self.model.set_train_data(inputs=z, targets=labels, strict=False)

            # z = self.feature_extractor(x_query)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mll_list.append(loss.item())
            mse = self.mse(predictions.mean, labels)

            if ((epoch%2==0) & (itr%5==0)):
                print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
            
            if (self.show_plots_pred or self.show_plots_features) and not self.f_rvm:
                embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
                self.update_plots_train_kmeans(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()

        return np.mean(mll_list)

    def train_loop_fast_rvm(self, epoch, n_support, n_samples, optimizer):
        
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        mll_list = []
        for itr, (inputs, labels) in enumerate(zip(batch, batch_labels)):


            z = self.feature_extractor(inputs)
            with torch.no_grad():
                inducing_points = self.get_inducing_points(z, labels, verbose=False)
            
            ip_values = inducing_points.z_values.cuda()
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
            self.model.train()
            self.model.set_train_data(inputs=z, targets=labels, strict=False)

            # z = self.feature_extractor(x_query)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mll_list.append(loss.item())
            mse = self.mse(predictions.mean, labels)

            if ((epoch%2==0) & (itr%5==0)):
                print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
            
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

    def test_loop_kmeans(self, n_support, n_samples, test_person, optimizer=None): # no optimizer needed for GP

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
        with torch.no_grad():
            inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
        
        
        def inducing_max_similar_in_support_x(train_x, train_z, inducing_points, train_y):
            y = ((train_y.cpu().numpy() + 1) * 60 / 2) + 60
    
            # kernel_matrix = self.model.covar_module(inducing_points.z_values, train_z).evaluate()
            kernel_matrix = self.model.base_covar_module(inducing_points.z_values, train_z).evaluate()
            # max_similar_index
            index = torch.argmax(kernel_matrix, axis=1).cpu().numpy()
            x_inducing = train_x[index].cpu().numpy()
            y_inducing = y[index]
            z_inducing = train_z[index]
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

            return IP(z_inducing, index, inducing_points.count, 
                                x_inducing, y_inducing, np.array(i_idx), np.array(j_idx))
        
        inducing_points = inducing_max_similar_in_support_x(x_support, z_support.detach(), inducing_points, y_support)
        ip_values = inducing_points.z_values.cuda()
        # inducing_points = inducing_max_similar_in_support_x(x_support, z_support.detach(), inducing_points, y_support)
        self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)

        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query).item()

        #**************************************************************
        y = ((y_query.detach().cpu().numpy() + 1) * 60 / 2) + 60
        y_pred = ((pred.mean.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.RED,"="*50, Fore.RESET)
        print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
        print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
        print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
        print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        K = self.model.base_covar_module
        kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
        max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
        y_s = ((y_support.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
        
        kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

        kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_index = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
        #**************************************************************
        if (self.show_plots_pred or self.show_plots_features) and not self.f_rvm:
            embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

            self.update_plots_test_kmeans(self.plots, x_support, y_support.detach().cpu().numpy(), 
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

        return mse

    def test_loop_fast_rvm(self, n_support, n_samples, test_person, optimizer=None): # no optimizer needed for GP

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
        with torch.no_grad():
            inducing_points = self.get_inducing_points(z_support, y_support, verbose=False)
        
        ip_values = inducing_points.z_values.cuda()
        self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
        self.model.covar_module._clear_cache()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query).item()

        def inducing_max_similar_in_support_x(train_x, inducing_points, train_y):
            y = ((train_y.detach().cpu().numpy() + 1) * 60 / 2) + 60
    
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

            return IP(inducing_points.z_values, index, inducing_points.count, 
                                x_inducing, y_inducing, np.array(i_idx), np.array(j_idx))
        
        inducing_points = inducing_max_similar_in_support_x(x_support, inducing_points, y_support)

        #**************************************************************
        y = ((y_query.detach().cpu().numpy() + 1) * 60 / 2) + 60
        y_pred = ((pred.mean.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.RED,"="*50, Fore.RESET)
        print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
        print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
        print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
        print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        K = self.model.base_covar_module
        kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
        max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
        y_s = ((y_support.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
        
        kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

        kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_index = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
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

        return mse

    def train_loop_random(self, epoch, n_support, n_samples, optimizer):

        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        mll_list = []
        for itr, (inputs, labels) in enumerate(zip(batch, batch_labels)):

            # random selection of inducing points
            inducing_points_index = list(np.random.choice(list(range(n_samples)), replace=False, size=self.num_induce_points))

            z = self.feature_extractor(inputs)

            inducing_points_z = z[inducing_points_index,:]
            
            ip_values = inducing_points_z.cuda()
            self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
            self.model.train()
            self.model.set_train_data(inputs=z, targets=labels, strict=False)

            # z = self.feature_extractor(x_query)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mll_list.append(loss.item())
            mse = self.mse(predictions.mean, labels)
            
            if ((epoch%2==0) & (itr%5==0)):
                print('[%02d/%02d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    itr, epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
            
            if (self.show_plots_pred or self.show_plots_features) and self.random:
                embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
                self.update_plots_train_kmeans(self.plots, labels.cpu().numpy(), embedded_z, None, mse, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()

        return np.mean(mll_list)
    
    def test_loop_random(self, n_support, n_samples, test_person, optimizer=None): # no optimizer needed for GP

        inputs, targets = get_batch(test_people, n_samples)

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


        inducing_points_index = np.random.choice(list(range(n_support)), replace=False, size=self.num_induce_points)

        z_support = self.feature_extractor(x_support).detach()

        inducing_points_z = z_support[inducing_points_index,:]

        ip_values = inducing_points_z.cuda()
        self.model.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
        self.model.covar_module._clear_cache()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query).item()

        def inducing_max_similar_in_support_x(train_x, inducing_points_z, inducing_points_index, train_y):
            y = ((train_y.detach().cpu().numpy() + 1) * 60 / 2) + 60
    
            index = inducing_points_index
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

            return IP(inducing_points_z, index, index.shape, 
                                x_inducing, y_inducing, np.array(i_idx), np.array(j_idx))
        
        inducing_points = inducing_max_similar_in_support_x(x_support, inducing_points_z, inducing_points_index, y_support)

        #**************************************************************
        y = ((y_query.detach().cpu().numpy() + 1) * 60 / 2) + 60
        y_pred = ((pred.mean.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.RED,"="*50, Fore.RESET)
        print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
        print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
        print(Fore.LIGHTWHITE_EX, f'y_var: {pred.variance.detach().cpu().numpy()}', Fore.RESET)
        print(Fore.LIGHTRED_EX, f'mse:    {mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        K = self.model.base_covar_module
        kernel_matrix = K(z_query, z_support).evaluate().detach().cpu().numpy()
        max_similar_idx_x_s = np.argmax(kernel_matrix, axis=1)
        y_s = ((y_support.detach().cpu().numpy() + 1) * 60 / 2) + 60
        print(Fore.LIGHTGREEN_EX, f'target of most similar in support set:       {y_s[max_similar_idx_x_s]}', Fore.RESET)
        
        kernel_matrix = K(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_idx_x_ip = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (K kernel): {inducing_points.y[max_similar_idx_x_ip]}', Fore.RESET)

        kernel_matrix = self.model.covar_module(z_query, inducing_points.z_values).evaluate().detach().cpu().numpy()
        max_similar_index = np.argmax(kernel_matrix, axis=1)
        print(Fore.LIGHTGREEN_EX, f'target of most similar in IP set (Q kernel): {inducing_points.y[max_similar_index]}', Fore.RESET)
        #**************************************************************
        if (self.show_plots_pred or self.show_plots_features) and  self.random:
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

        return mse

    
    def train(self, stop_epoch, n_support, n_samples, optimizer):

        mll_list = []
        for epoch in range(stop_epoch):
            
            if  self.f_rvm:
                mll = self.train_loop_fast_rvm(epoch, n_support, n_samples, optimizer)
            elif self.random:
                mll = self.train_loop_random(epoch, n_support, n_samples, optimizer)
            elif  not self.f_rvm:
                mll = self.train_loop_kmeans(epoch, n_support, n_samples, optimizer)
            else:
                ValueError("Error")
            mll_list.append(mll)

            print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)
        
        mll = np.mean(mll_list)

        
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        return mll, mll_list
    
    def test(self, n_support, n_samples, optimizer=None, test_count=None): # no optimizer needed for GP

        mse_list = []
        # choose a random test person
        rep = True if test_count > len(test_people) else False

        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=rep)
        for t in range(test_count):
            print(f'test #{t}')
            if self.f_rvm:
                mse = self.test_loop_fast_rvm(n_support, n_samples, test_person[t],  optimizer)
            elif self.random:
                mse = self.test_loop_random(n_support, n_samples, test_person[t],  optimizer)
            elif not self.f_rvm:
                mse = self.test_loop_kmeans(n_support, n_samples, test_person[t],  optimizer)
            else:
                ValueError()

            mse_list.append(float(mse))

        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()

        return mse_list
        
    def get_inducing_points(self, inputs, targets, verbose=True):

        
        IP_index = np.array([])
        if not self.f_rvm:
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
            tol = 1e-4
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

        return IP(inducing_points, IP_index, num_IP, None, None, None, None)
  
    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
    
        ckpt = torch.load(checkpoint)
 
        IP = torch.ones(self.num_induce_points, 2916).cuda()
        ckpt['gp']['covar_module.inducing_points'] = IP
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

    def initialize_plot(self, video_path, training):
        
        
        if training:
            self.video_path = video_path+'_Train_video'
        else:
            self.video_path = video_path+'_Test_video'

        os.makedirs(self.video_path, exist_ok=True)
        time_now = datetime.now().strftime('%Y-%m-%d--%H-%M')
        sparse_method = "f_rvm" if self.f_rvm else "KMeans"
        if self.random: sparse_method = "random"  
        self.plots = self.prepare_plots()
        # plt.show(block=False)
        # plt.pause(0.0001)
        if self.show_plots_pred:
           
            metadata = dict(title=f'Sparse_DKT_{sparse_method}', artist='Matplotlib')
            FFMpegWriter = animation.writers['ffmpeg']
            self.mw = FFMpegWriter(fps=5, metadata=metadata)   
            file = f'{self.video_path}/Sparse_DKT_{sparse_method}_{time_now}.mp4'
            self.mw.setup(fig=self.plots.fig, outfile=file, dpi=125)

        if self.show_plots_features:  
            metadata = dict(title=f'Sparse_DKT_{sparse_method}', artist='Matplotlib')         
            FFMpegWriter2 = animation.writers['ffmpeg']
            self.mw_feature = FFMpegWriter2(fps=2, metadata=metadata)
            file = f'{self.video_path}/Sparse_DKT_{sparse_method}_features_{time_now}.mp4'
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

    def update_plots_train_kmeans(self,plots, train_y, embedded_z, mll, mse, epoch):
        if self.show_plots_features:
            #features
            y = ((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}')
    
    def update_plots_train_fast_rvm(self,plots, train_y, embedded_z, mll, mse, epoch):
        if self.show_plots_features:
            #features
            y = ((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}')
    
    def update_plots_test_kmeans(self, plots, train_x, train_y, train_z, test_z, embedded_z, inducing_points,   
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
            
            plots.fig.suptitle(f"person {person}, MSE: {mse:.4f}")

            cluster_colors = ['aqua', 'coral', 'lime', 'gold', 'purple', 'green', 'tomato', 
                                'fuchsia', 'chocolate', 'chartreuse', 'orange', 'teal']


            #train images
            y = ((train_y + 1) * 60 / 2) + 60
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
                    z = train_z[idx]
                    cluster = self.kmeans_clustering.predict(z)
                    # cluster = self.kmeans_clustering.predict(z.detach().cpu().numpy())
                    for j in range(0, idx.shape[0]): 
                        img = transforms.ToPILImage()(x[j]).convert("RGB")
                        plots = clear_ax(plots, i, j)
                        plots = color_ax(plots, i, j, 'black', lw=2) #cluster_colors[cluster[j]]
                        plots.ax[i, j].imshow(img)
                        plots.ax[i, j].set_title(f'{num}', fontsize=8)
                        num += 1
                    plots.ax[i, 0].set_ylabel(f'{t}',  fontsize=10)
                

            # test images
            y = ((test_y + 1) * 60 / 2) + 60
            y_mean = test_y_pred.mean.detach().cpu().numpy()
            y_var = test_y_pred.variance.detach().cpu().numpy()
            y_pred = ((y_mean + 1) * 60 / 2) + 60
            y_s = ((train_y + 1) * 60 / 2) + 60
            for t in tilt:
                idx = np.where(y==(t))[0]
                if idx.shape[0]==0:
                    continue
                else:
                    x = test_x[idx]
                    z = test_z[idx]
                    sim_x_s_idx = similar_idx_x_s[idx]
                    sim_y_s = y_s[sim_x_s_idx] 
                    sim_x_ip = similar_idx_x_ip[idx]
                    cluster = self.kmeans_clustering.predict(z)
                    y_p = y_pred[idx]
                    y_v = y_var[idx]
                    i = int(t/10-6)
                    for j in range(idx.shape[0]):
                        
                        img = transforms.ToPILImage()(x[j]).convert("RGB")
                        ii = 16
                        plots = clear_ax(plots, i, j+ii)
                        plots.ax[i, j+ii].imshow(img)
                        plots = color_ax(plots, i, j+ii, color='magenta', lw=2) #cluster_colors[cluster[j]]
                        plots.ax[i, j+ii].set_title(f'{y_p[j]:.1f}', fontsize=8)
                        id_sim_x_s = int(plots.ax[int(sim_y_s[j]/10-6),0].get_title()) +  sim_x_s_idx[j]%15
                        plots.ax[i, j+ii].set_xlabel(f'{id_sim_x_s}|{sim_x_ip[j] + 1}', fontsize=10)
 
                    # plots.ax[i, j+16].legend()
            for i in range(7):
                plots = clear_ax(plots, i, 15)
                plots = color_ax(plots, i, 15, 'white', lw=0.5)

            # highlight inducing points
            y = ((train_y + 1) * 60 / 2) + 60
            
            if inducing_points.x is not None:
                
                cluster = self.kmeans_clustering.predict(inducing_points.z_values)
                # cluster = self.kmeans_clustering.predict(z_inducing.detach().cpu().numpy())                
                for r in range(inducing_points.index.shape[0]):
                    
                    # t = inducing_points.y[r]
                    # i = int(t/10-6)
                    plots = color_ax(plots, inducing_points.i_idx[r], inducing_points.j_idx[r], 'black', lw=3) #cluster_colors[cluster[r]]
                    plots.ax[inducing_points.i_idx[r], inducing_points.j_idx[r]].spines['bottom'].set_color('red')   
                    plots.ax[inducing_points.i_idx[r], inducing_points.j_idx[r]].set_xlabel(r+1, fontsize=10)          

            plots.fig.savefig(f'{self.video_path}/test_person_{person}.png')    

        if self.show_plots_features:
            #features
            y = ((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()

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
            plots.fig.suptitle(f'person {person}, MSE: {mse:.4f}')
            y = ((train_y + 1) * 60 / 2) + 60
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
                        img = transforms.ToPILImage()(x[j]).convert("RGB")
                        plots = clear_ax(plots, i, j)
                        plots = color_ax(plots, i, j, 'black', lw=0.5)
                        plots.ax[i, j].imshow(img)
                        plots.ax[i, j].set_title(f'{num}', fontsize=8)
                        num += 1
                    plots.ax[i, 0].set_ylabel(f'{t}',  fontsize=10)
                
        
            # test images
            y = ((test_y + 1) * 60 / 2) + 60
            y_mean = test_y_pred.mean.detach().cpu().numpy()
            y_var = test_y_pred.variance.detach().cpu().numpy()
            y_pred = ((y_mean + 1) * 60 / 2) + 60
            y_s = ((train_y + 1) * 60 / 2) + 60
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
                        
                        img = transforms.ToPILImage()(x[j]).convert("RGB")
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
            y = ((train_y + 1) * 60 / 2) + 60
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
            y = ((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()


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
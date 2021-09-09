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
from data.qmul_loader import get_batch, train_people, test_people
from configs import kernel_type


#Check if tensorboardx is installed
try:
    #tensorboard --logdir=./MAML_Loss/ --host localhost --port 8091
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):

        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out




class MAML_regression(nn.Module):
    def __init__(self, backbone, video_path=None, show_plots_pred=False, show_plots_features=False, training=False):
        super(MAML_regression, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.model = Linear_fw(2916, 1)
        self.device = 'cuda'
        self.video_path = video_path
        self.show_plots_pred = show_plots_pred
        self.show_plots_features = show_plots_features
        if self.show_plots_pred or self.show_plots_features:
            self.initialize_plot(video_path, training)

        self.n_task     = 4
        self.task_update_num = 1
        self.train_lr = 0.01
        self.mse        = nn.MSELoss()
        
    def init_summary(self, id):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir('./MAML_Loss'):
                os.makedirs('./MAML_Loss')
            writer_path = './MAML_Loss/' + id #+'_'+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)


    def set_forward(self, x, is_feature=False):
        z = self.feature_extractor(x)
        pred = self.model(z)
        return pred

    def set_forward_loss(self, x_support, y_support, x_query):
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            y_pred = self.set_forward(x_support)
            set_loss = self.mse(y_pred, y_support) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        y_pred = self.set_forward(x_query)
        return y_pred


    def train_loop(self, epoch, n_support, n_samples, optimizer):
        self.model.train()
        self.feature_extractor.train()
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        # print(f'{epoch}: {batch_labels[0]}')
        mse_list = []
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        split = np.array([True]*15 + [False]*3)
        for itr, (inputs, labels) in enumerate(zip(batch, batch_labels)):
             
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
            y_all = labels.cuda()

            x_support = x_all[support_ind,:,:,:]
            y_support = y_all[support_ind]
            x_query   = x_all[query_ind,:,:,:]
            y_query   = y_all[query_ind]

            y_pred= self.set_forward_loss(x_support, y_support, x_query)
            loss = self.mse(y_pred, y_query) 
            avg_loss = avg_loss+loss.item()#.data[0]
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            
            mse_list.append(loss.item())
            if ((epoch%1==0) & (itr%2==0)):
                print('[%02d/%02d] - Loss: %.3f ' % (
                    itr, epoch, loss.item()
                ))
            self.iteration = itr+(epoch*len(batch_labels))
            if(self.writer is not None): self.writer.add_scalar('MSE', loss.item(), self.iteration)

            if (self.show_plots_pred or self.show_plots_features):
                z =  self.feature_extractor(x_support).detach()
                embedded_z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
                self.update_plots_train(self.plots, labels.cpu().numpy(), embedded_z, None, loss, epoch)

                if self.show_plots_pred:
                    self.plots.fig.canvas.draw()
                    self.plots.fig.canvas.flush_events()
                    self.mw.grab_frame()
                if self.show_plots_features:
                    self.plots.fig_feature.canvas.draw()
                    self.plots.fig_feature.canvas.flush_events()
                    self.mw_feature.grab_frame()

        return np.mean(mse_list)

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

        self.model.eval()
        self.feature_extractor.eval()
        output = self.set_forward_loss(x_support, y_support, x_query)
        mse = self.mse(output, y_query).item()


        #***************************************************
        y = ((output.detach() + 1) * 60 / 2) + 60
        y_pred = ((output + 1) * 60 / 2) + 60
        mse_ = self.mse(y_pred, y).item()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        print(Fore.RED,"="*50, Fore.RESET)
        print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
        print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
        print(Fore.LIGHTRED_EX, f'mse:    {mse_:.4f}, mse:    {mse:.4f}', Fore.RESET)
        print(Fore.RED,"-"*50, Fore.RESET)

        
        #**************************************************

        if (self.show_plots_pred or self.show_plots_features):
            z_support =  self.feature_extractor(x_support).detach()
            embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

            self.update_plots_test(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                            z_support.detach(), None, embedded_z_support,
                                            x_query, y_query.detach().cpu().numpy(), output.squeeze().detach().cpu().numpy(), 
                                            mse, test_person)
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
            mll = self.train_loop(epoch, n_support, n_samples, optimizer)
            mll_list.append(mll)
            if(self.writer is not None): self.writer.add_scalar('MLL per epoch', mll, epoch)
            print(Fore.CYAN,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)
        
        mll = np.mean(mll_list)
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        return mll, mll_list

    def test(self, n_support, n_samples, optimizer=None, test_count=None):

        mse_list = []
        # choose a random test person
        rep = True if test_count > len(test_people) else False

        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=rep)
        for t in range(test_count):
            print(f'test #{t}')
            
            mse = self.test_loop(n_support, n_samples, test_person[t],  optimizer)
            
            mse_list.append(float(mse))

        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()

        return mse_list

    def save_checkpoint(self, checkpoint):
        torch.save({'feature_extractor': self.feature_extractor.state_dict(), 'model':self.model.state_dict()}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.feature_extractor.load_state_dict(ckpt['feature_extractor'])
        self.model.load_state_dict(ckpt['model'])
        
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
            y = ((train_y + 1) * 60 / 2) + 60
            tilt = np.unique(y)
            plots.ax_feature.clear()
            for t in tilt:
                idx = np.where(y==(t))[0]
                z_t = embedded_z[idx]
                
                plots.ax_feature.scatter(z_t[:, 0], z_t[:, 1], label=f'{t}')

            plots.ax_feature.legend()
            plots.ax_feature.set_title(f'epoch {epoch}')

    def update_plots_test(self, plots, train_x, train_y, train_z, test_z, embedded_z,   
                                    test_x, test_y, test_y_pred, mse, person):
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
            y_mean = test_y_pred
            # y_var = test_y_pred.variance.detach().cpu().numpy()
            y_pred = ((y_mean + 1) * 60 / 2) + 60
            for t in tilt:
                idx = np.where(y==(t))[0]
                if idx.shape[0]==0:
                    continue
                else:
                    x = test_x[idx]
                    # sim_x_s_idx = similar_idx_x_s[idx]
                    # sim_y_s = y_s[sim_x_s_idx]
                    y_p = y_pred[idx]
                    # y_v = y_var[idx]
                    i = int(t/10-6)
                    for j in range(idx.shape[0]):
                        
                        img = transforms.ToPILImage()(x[j]).convert("RGB")
                        ii = 16
                        plots = clear_ax(plots, i, j+ii)
                        plots.ax[i, j+ii].imshow(img)
                        # plots = color_ax(plots, i, j+ii, color=cluster_colors[cluster[j]], lw=2)
                        plots.ax[i, j+ii].set_title(f'{y_p[j]:.1f}', fontsize=10)
                        # id_sim_x_s = int(plots.ax[int(sim_y_s[j]/10-6),0].get_title()) +  sim_x_s_idx[j]%15
                        # plots.ax[i, j+ii].set_xlabel(f'{int(id_sim_x_s)}', fontsize=10)
                
                    # plots.ax[i, j+16].legend()
            for i in range(7):
                plots = clear_ax(plots, i, 15)
                plots = color_ax(plots, i, 15, 'white', lw=0.5)

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
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

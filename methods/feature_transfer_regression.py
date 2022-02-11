import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
from colorama import Fore
from collections import namedtuple
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import backbone
from torch.autograd import Variable
from data.qmul_loader import get_batch, train_people, val_people, test_people, get_unnormalized_label

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.layer4 = nn.Linear(2916, 1)

    def return_clones(self):
        layer4_w = self.layer4.weight.data.clone().detach()
        layer4_b = self.layer4.bias.data.clone().detach()

        return (layer4_w, layer4_b)

    def assign_clones(self, weights_list):
        self.layer4.weight.data.copy_(weights_list[0])
        self.layer4.bias.data.copy_(weights_list[1])

    def forward(self, x):
        out = self.layer4(x)
        return out

class FeatureTransfer(nn.Module):
    def __init__(self, backbone,  lr_decay=False, normalize=False, video_path=None, show_plots_pred=False, show_plots_features=False, training=False):
        super(FeatureTransfer, self).__init__()
        regressor = Regressor()
        self.feature_extractor = backbone
        self.model = Regressor()
        self.criterion = nn.MSELoss()
        self.lr_decay = lr_decay
        self.normalize = normalize
        self.training_  = training
        self.video_path = video_path
        self.best_path = video_path
        self.show_plots_pred = show_plots_pred
        self.show_plots_features = show_plots_features
        if self.show_plots_pred or self.show_plots_features:
            self.initialize_plot(video_path, training)

    def train_loop(self, epoch, n_samples, optimizer):
        self.feature_extractor.train()
        self.model.train()
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        mse_list = []
        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)
            if(self.normalize): z = F.normalize(z, p=2, dim=1)
            output = self.model(z)
            loss = self.criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            if(epoch%1==0):
                print('[%02d] - Loss: %.3f' % (
                    epoch, loss.item()
                ))
            mse_list.append(loss.item())
        
        if (self.show_plots_pred or self.show_plots_features):
            embedded_z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
            self.update_plots_train(self.plots, labels.cpu().numpy(), embedded_z, None, loss.item(), epoch)

            if self.show_plots_pred:
                self.plots.fig.canvas.draw()
                self.plots.fig.canvas.flush_events()
                self.mw.grab_frame()
            if self.show_plots_features:
                self.plots.fig_feature.canvas.draw()
                self.plots.fig_feature.canvas.flush_events()
                self.mw_feature.grab_frame()
        return np.mean(mse_list)
   
    def train(self, stop_epoch, n_support, n_samples, optimizer, save_model=False):
        
        mse_list = []
        best_mse = 10e5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self.fine_tune = 3
        for epoch in range(stop_epoch):
            mse = self.train_loop(epoch, n_samples, optimizer)
            mse_list.append(mse)
            if ((epoch>=50) and epoch%1==0) or ((epoch<50) and epoch%10==0):
                print(Fore.GREEN,"-"*30, f'\nValidation:', Fore.RESET)
                val_mse_list = []
                val_count = 80
                rep = True if val_count > len(val_people) else False
                val_person = np.random.choice(np.arange(len(val_people)), size=val_count, replace=rep)
                for t in range(val_count):
                    mse, mse_ = self.test_loop(n_support, n_samples, val_person[t],  optimizer)
                    val_mse_list.append(mse)
                mse = np.mean(val_mse_list)
                if best_mse >= mse:
                    best_mse = mse
                    model_name = self.best_path + '_best_model.tar'
                    self.save_best_checkpoint(epoch+1, best_mse, model_name)
                    print(Fore.LIGHTRED_EX, f'Best MSE: {best_mse:.4f}', Fore.RESET)
                print(Fore.LIGHTRED_EX, f'\nepoch {epoch+1} => Val. MSE: {mse:.4f}, Best MSE: {best_mse:.4f}', Fore.RESET)
                # if(self.writer is not None):
                #     self.writer.add_scalar('MSE Val.', mse, epoch)
                print(Fore.GREEN,"-"*30, Fore.RESET)
            if save_model and epoch>50 and epoch%50==0:
                    model_name = self.best_path + f'_{epoch}'
                    self.save_best_checkpoint(epoch, mse, model_name)    

            if self.lr_decay:
                scheduler.step()
            print(Fore.LIGHTYELLOW_EX,"-"*30, f'\nend of epoch {epoch} => MSE: {mse}\n', "-"*30, Fore.RESET)
        
        mse = np.mean(mse_list)

        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()
        
        return mse, mse_list

    def test_loop(self, n_support, n_samples, test_person, optimizer): # we need optimizer to take one gradient step
        
        if self.training_ : 
            inputs, targets = get_batch(val_people, n_samples)
        else:
            inputs, targets = get_batch(test_people, n_samples)
        # support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        # query_ind   = [i for i in range(19) if i not in support_ind]

        x_all = inputs.cuda()
        y_all = targets.cuda()

        split = np.array([True]*15 + [False]*3)
        # print(split)
        shuffled_split = []
        for _ in range(int(n_support//15)):
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
        #fine-tune
        self.feature_extractor.train()
        self.model.train()
        # NOTE just last layer is fine-tuned
        optimizer.zero_grad()
        z_support = self.feature_extractor(x_support).detach()
        if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
        for k in range(self.fine_tune):
            optimizer.zero_grad()
            output_support = self.model(z_support).squeeze()
            loss = self.criterion(output_support.squeeze(), y_support)
            loss.backward()
            optimizer.step()

        self.feature_extractor.eval()
        self.model.eval()
        mse_support = self.criterion(output_support, y_support).item()
        z_query = self.feature_extractor(x_query).detach()
        if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
        output = self.model(z_query).squeeze()
        mse = self.criterion(output, y_query).item()

        y = get_unnormalized_label(y_query.detach()) #((y_query.detach() + 1) * 60 / 2) + 60
        y_pred = get_unnormalized_label(output.detach()) # ((pred.mean.detach() + 1) * 60 / 2) + 60
        mse_ = self.criterion(y_pred, y).item()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        if self.test_i%2==0:
            print(Fore.RED,"-"*50, Fore.RESET)
            print(Fore.YELLOW, f'y_pred: {y_pred}', Fore.RESET)
            print(Fore.LIGHTCYAN_EX, f'y:      {y}', Fore.RESET)
            print(Fore.LIGHTRED_EX, f'mse:    {mse_:.4f}, mse (normed):{mse:.4f}, support mse: {mse_support:.4f}', Fore.RESET)
            print(Fore.RED,"-"*50, Fore.RESET)


        if (self.show_plots_pred or self.show_plots_features):
            embedded_z_support = TSNE(n_components=2).fit_transform(z_support.detach().cpu().numpy())

            self.update_plots_test(self.plots, x_support, y_support.detach().cpu().numpy(), 
                                            z_support.detach(), z_query.detach(), embedded_z_support,
                                            x_query, y_query.detach().cpu().numpy(), output.squeeze().detach().cpu().numpy(), 
                                            None, mse, test_person)
            if self.show_plots_pred:
                self.plots.fig.canvas.draw()
                self.plots.fig.canvas.flush_events()
                self.mw.grab_frame()
            if self.show_plots_features:
                self.plots.fig_feature.canvas.draw()
                self.plots.fig_feature.canvas.flush_events()
                self.mw_feature.grab_frame()
        return mse, mse_

    def test(self, n_support, n_samples, optimizer, fine_tune=3, test_count=None):

        self.fine_tune = fine_tune
        mse_list = []
        # choose a random test person
        rep = True if test_count > len(test_people) else False

        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=rep)
        for t in range(test_count):
            if t%20==0:print(f'test #{t}')
            self.test_i = t
            weights_1 = self.feature_extractor.return_clones()
            weights_2 = self.model.return_clones()
            mse, mse_ = self.test_loop(n_support, n_samples, test_person[t],  optimizer)
            self.feature_extractor.assign_clones(weights_1)
            self.model.assign_clones(weights_2)
            mse_list.append(float(mse))
        
        if self.show_plots_pred:
            self.mw.finish()
        if self.show_plots_features:
            self.mw_feature.finish()

        # result = {'mse':f'{np.mean(mse_list):.4f}', 'std':f'{np.std(mse_list):.4f}'} #  
        result = {'mse':np.mean(mse_list),  'std':np.std(mse_list)}
        result = {k: np.around(v, 4) for k, v in result.items()}
        result['fine_tune'] = self.fine_tune
        #result = {'mse':np.around(np.mean(mse_list), 3), 'std':np.around(np.std(mse_list),3)}
        return mse_list, result

    def save_checkpoint(self, checkpoint):
        torch.save({'feature_extractor': self.feature_extractor.state_dict(), 'model':self.model.state_dict()}, checkpoint)
    
    def save_best_checkpoint(self, epoch, mse, checkpoint):
        torch.save({'feature_extractor': self.feature_extractor.state_dict(), 
                    'model':self.model.state_dict(), 'epoch': epoch, 'mse': mse}, checkpoint)

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
                                    test_x, test_y, test_y_pred, mll, mse, person):
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
                        img = transforms.ToPILImage()(x[j].cpu()).convert("RGB")
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
                        
                        img = transforms.ToPILImage()(x[j].cpu()).convert("RGB")
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

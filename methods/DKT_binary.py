## Original packages
from gpytorch.kernels import kernel
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from colorama import Fore
## Our packages
import gpytorch
from time import gmtime, strftime
import random
# from configs import kernel_type
#Check if tensorboardx is installed
try:
    # tensorboard --logdir=./log/ --host localhost --port 8090
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

class DKT_binary(MetaTemplate):
    def __init__(self, model_func, kernel_type, n_way, n_support, normalize=False, dirichlet=False):
        super(DKT_binary, self).__init__(model_func, n_way, n_support)
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.dirichlet = dirichlet
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
        self.dataset = dataset
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            writer_path = "./log/" + id #+'_'+ time_string 
            self.writer = SummaryWriter(log_dir=writer_path)

    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x = torch.ones(100, 64).cuda()
        if(train_y_list is None): train_y = torch.ones(100).cuda()

        if self.dirichlet:
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y.long(), learn_additional_noise=False)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, dirichlet=self.dirichlet, kernel=self.kernel_type)

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
                self.leghtscale_list.append(single_model.covar_module.base_kernel.lengthscale.clone().detach())
                self.noise_list.append(single_model.likelihood.noise.clone().detach())
                self.outputscale_list.append(single_model.covar_module.outputscale.clone().detach())
        else:
            for idx, single_model in enumerate(self.model.models):
                single_model.covar_module.base_kernel.lengthscale=self.leghtscale_list[idx].clone().detach()#.requires_grad_(True)
                single_model.likelihood.noise=self.noise_list[idx].clone().detach()
                single_model.covar_module.outputscale=self.outputscale_list[idx].clone().detach()
    
    def pred_result(self, mean):
        
        max_pred, idx = torch.max(mean, axis=0)
        index = ~idx.to(bool)
        max_pred[index] = -np.inf
        return max_pred
        
    def train_loop(self, epoch, train_loader, optimizer, print_freq=5):
        # optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-3},
        #                               {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])
        self.acc_test_list = []
        self.mll_list = []
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
            target = target.cuda()

            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)

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
                # self.model.likelihood.noise.data = sigma2_labels
                self.model.set_train_data(inputs=z_train, targets=self.model.likelihood.transformed_targets, strict=False)
            else: 
                self.model.set_train_data(inputs=z_train, targets=target, strict=False)
            
            if(self.model.covar_module.base_kernel.lengthscale is not None):
                lenghtscale+=self.model.base_covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
            noise+=self.model.likelihood.noise.cpu().detach().numpy().squeeze().mean()
            if(self.model.covar_module.outputscale is not None): #DKT Linear
                outputscale+=self.model.covar_module.outputscale.cpu().detach().numpy().squeeze()

            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            if self.dirichlet:
                transformed_targets = self.model.likelihood.transformed_targets
                loss = -self.mll(output, transformed_targets).sum()
            else:
                loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            self.mll_list.append(loss)
            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)

            #Eval on the query (validation set)
            with torch.no_grad():
                self.model.eval()
                self.likelihood.eval()
                self.feature_extractor.eval()
                z_support = self.feature_extractor.forward(x_support).detach()
                if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                # if self.dirichlet:
                #     prediction = self.likelihood(self.model(z_support)) #return 20 MultiGaussian Distributions
                # else:
                #     prediction = self.likelihood(self.model(z_support) #return 20 MultiGaussian Distributions
               
                # if self.dirichlet:
                #     
                #    max_pred = (prediction.mean[0] < prediction.mean[1]).to(int)
                #    y_pred = max_pred.cpu().detach().numpy()
                # else: 
                #    pred = torch.sigmoid(prediction.mean)
                #    y_pred = (pred < 0.5).to(int)
                #    y_pred = y_pred.cpu().detach().numpy()
                # accuracy_support = (np.sum(y_pred==y_support) / float(len(y_support))) * 100.0
                # if(self.writer is not None): self.writer.add_scalar('GP_support_accuracy', accuracy_support, self.iteration)
                z_query = self.feature_extractor.forward(x_query).detach()
                if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                z_query_list = [z_query]*len(y_query)

                prediction = self.likelihood(self.model(z_query))
               
                if self.dirichlet:
                    
                   max_pred = (prediction.mean[0] > prediction.mean[1]).to(int) #y_query [1, 1, ..., 0, 0, ..., 0]
                   y_pred = max_pred.cpu().detach().numpy()
                else: 
                   pred = torch.sigmoid(prediction.mean)
                   y_pred = (pred > 0.5).to(int)
                   y_pred = y_pred.cpu().detach().numpy()
               
                accuracy_query = (np.sum(y_pred==y_query) / float(len(y_query))) * 100.0
                self.acc_test_list.append(accuracy_query)
                

            if i % print_freq==0:
                if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                print(Fore.LIGHTRED_EX, 'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} | Noise {:f} | Loss {:f} | Supp. {:f} | Query {:f}'.format(epoch, i, len(train_loader), 
                                outputscale, lenghtscale, noise, loss, 0, accuracy_query), Fore.RESET)

        
        if(self.writer is not None): self.writer.add_scalar('Loss', np.mean(self.mll_list), self.iteration)
        if(self.writer is not None): self.writer.add_scalar('GP_query_accuracy', np.mean(self.acc_test_list), self.iteration)

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

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()

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
                y_pred = y_pred.cpu().detach().numpy()

            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
            acc = (top1_correct/ count_this)*100
            K = self.model.covar_module(z_query, z_train).evaluate()
            K = K.detach().cpu().numpy()
            max_similar_idx_q_s = np.argmax(K, axis=1)
            most_sim_y_s = target[max_similar_idx_q_s].detach().cpu().numpy().squeeze()
            most_sim_y_s[most_sim_y_s==-1] = 0
            acc_most_sim = np.sum(most_sim_y_s == y_query)

        if self.show_plot:
            K_idx_sorted = np.argsort(K, axis=1)
            self.plot_test(x_query, y_query, y_pred, acc, x_support, y_support, K, K_idx_sorted, i)

        return float(top1_correct), count_this, avg_loss/float(N+1e-10), acc_most_sim

    def test_loop(self, test_loader, record=None, return_std=False, dataset=None, show_plot=False):
        self.dataset = dataset
        self.show_plot = show_plot
        print_freq = 20
        correct =0
        count = 0
        acc_all = []
        acc_most_sim_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this, loss_value, acc_most_sim = self.correct(x, i)
            acc_all.append(correct_this/ count_this*100)
            acc_most_sim_all.append((acc_most_sim/ count_this)*100)
            if(i % print_freq==0):
                acc_mean = np.mean(np.asarray(acc_all))
                acc_most_sim_mean = np.mean(np.asarray(acc_most_sim_all))
                print(f'ACC based on most similar support: {acc_most_sim_mean:.4f}')
                print('Test | Batch {:d}/{:d} | Loss {:f} | Acc {:f}'.format(i, len(test_loader), loss_value, acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        acc_most_sim_mean = np.mean(np.asarray(acc_most_sim_all))
        print(Fore.CYAN,f'ACC based on most similar support: {acc_most_sim_mean:.2f}', Fore.RESET)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        result = {'acc': acc_mean, 'std': acc_std}
        result = {k: np.around(v, 4) for k, v in result.items()}
        if self.normalize: result['normalize'] = True
        
        if(return_std): return acc_mean, acc_std, result
        else: return acc_mean, result

    def plot_test(self, x_query, y_query, y_pred, acc, x_support, y_support, K, K_idx_sorted, k):
        def clear_ax(ax):
            ax.clear()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0)
            return ax
        
        out_path = f'./save_img/DKT_Bi/{self.dataset}'
        
        if y_query.shape[0] > 30:
            x_q       = torch.vstack([x_query[0:5], x_query[15:20]])
            y_q       = np.hstack([y_query[0:5], y_query[15:20]])
            y_pred_   = np.hstack([y_pred[0:5], y_pred[15:20]])
            
        else:
            x_q     = x_query    
            y_q     = y_query
            y_pred_ = y_pred
         
        r = 2
        c = 10
        fig: plt.Figure = plt.figure(1, figsize=(10, 4), tight_layout=False, dpi=150, frameon=True)
        for i in range(r*c):
            x = self.denormalize(x_q[i])
            y = y_q[i]
            y_p = y_pred_[i]
          
            ax: plt.Axes = fig.add_subplot(r, c, i+1)
            ax = clear_ax(ax)
            # ax.axis('off')
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            img = transforms.ToPILImage()(x.cpu()).convert("RGB")
            ax.imshow(img)
            if i==0: 
                ax.axes.get_yaxis().set_visible(True)
                # ax.spines['left'].set_visible(True)
                ax.set_ylabel('real: 0', fontsize=10)
            if i==10: 
                ax.axes.get_yaxis().set_visible(True)
                # ax.spines['left'].set_visible(True)
                ax.set_ylabel('real: 1', fontsize=10)
                
            ax.set_title(f'pred: {y_p:.0f}', fontsize=10, pad=2)
        
        fig.suptitle(f'ACC: {acc:.2f}%')
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(200, 200, 1500, 600) 
        # plt.show()
        fig.subplots_adjust(
        # top=0.955,
        # bottom=0.445,
        # left=0.055,
        # right=0.965,
        top=0.855,
        bottom=0.250,
        left=0.055,
        right=0.975,
        hspace=0.001,
        wspace=0.001
        )
     
        os.makedirs(f'{out_path}/task_{k}', exist_ok=True)
        fig.savefig(f'{out_path}/task_{k}/query_images.png', bbox_inches='tight')
        # fig.savefig(f'./save_img/task_{k}/query_images.png')
        # plt.show()
        plt.close(1)
        
       

       
        x_s, y_s = x_support, y_support
        d = 3
        top_red = torch.zeros([3, d, 84])
        top_red[0, :, :] = 1
        top_red = transforms.ToPILImage()(top_red).convert("RGB")
        top_grn = torch.zeros([3, d, 84])
        top_grn[1, :, :] = 1
        top_grn = transforms.ToPILImage()(top_grn).convert("RGB")
        rigt_red = torch.zeros([3, 84, d])
        rigt_red[0, :, :] = 1
        rigt_red = transforms.ToPILImage()(rigt_red).convert("RGB")
        rigt_grn = torch.zeros([3, 84, d])
        rigt_grn[1, :, :] = 1
        rigt_grn = transforms.ToPILImage()(rigt_grn).convert("RGB")

        m = 10
        r = y_q.shape[0] // 2
        c = m + 1
        # c = int(np.ceil(m/3)) + 1
      
        fig: plt.Figure = plt.figure(4, figsize=(10, 10), tight_layout=False, dpi=150)
        all_imgs = np.ones(x_s[0].shape[::-1] * np.array([r, c, 1]))
        gap = 5
        x_gap = torch.ones([3, all_imgs.shape[0], gap])
        gap_img = transforms.ToPILImage()(x_gap).convert("RGB")
        all_imgs = np.concatenate([all_imgs, np.ones([all_imgs.shape[0], gap, 3])], axis=1)
        s = x_s[0].shape[2]
        for i in range(r):
            idx_sim = K_idx_sorted[i, :]
            idx_sim = idx_sim[::-1]
            # query_i_sim = indc_x[idx_sim[::-1]]
            for j in range(c):
                if j==0:
                    x = self.denormalize(x_q[i])
                    y = y_q[i]
                    y_p = y_pred_[i]
                    img = transforms.ToPILImage()(x.cpu()).convert("RGB")
                    all_imgs[s*i:s*(i+1), s*j:s*(j+1), :] = img
                    if y==y_p:
                        all_imgs[s*i:s*(i+1), s*(j+1)-d:s*(j+1), :] = rigt_grn
                    else:
                        all_imgs[s*i:s*(i+1), s*(j+1)-d:s*(j+1), :] = rigt_red
                else:
                    jj = idx_sim[j-1]
                    x = self.denormalize(x_s[jj])
                    # y = indc_y_0[i]
                    img = transforms.ToPILImage()(x.cpu()).convert("RGB")
                    all_imgs[s*i:s*(i+1), (s*j) + gap:(s*(j+1))+gap, :] = img
                    if y_s[jj]==y:
                        all_imgs[s*i:s*i+d, (s*j) + gap:(s*(j+1)) + gap, :] = top_grn
                    else:
                        all_imgs[s*i:s*i+d, (s*j) + gap:(s*(j+1)) + gap, :] = top_red
        
        all_imgs[:, s: s + gap, :] = gap_img
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax = clear_ax(ax)
        ax.axis("off")
        ax.imshow(all_imgs.astype('uint8'))
        fig.suptitle(f'Similarity between Query and SV images [{m}/100]', fontsize=8)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50, 200, 800, 500) 
        fig.subplots_adjust(
        top=0.940,
        bottom=0.044,
        left=0.025,
        right=0.975,
        hspace=0.002,
        wspace=0.002
        )
        os.makedirs(f'{out_path}/task_{k}', exist_ok=True)
        fig.savefig(f'{out_path}/task_{k}/Query_Support_sim_1.png', bbox_inches='tight')
        # plt.show()
        plt.close(4)


       
        m = 10
        r = y_q.shape[0] //2
        c = m + 1
        # c = int(np.ceil(m/3)) + 1
      
        fig: plt.Figure = plt.figure(5, figsize=(10, 10), tight_layout=False, dpi=150)
        all_imgs = np.ones(x_s[0].shape[::-1] * np.array([r, c, 1]))
        gap = 5
        x_gap = torch.ones([3, all_imgs.shape[0], gap])
        gap_img = transforms.ToPILImage()(x_gap).convert("RGB")
        all_imgs = np.concatenate([all_imgs, np.ones([all_imgs.shape[0], gap, 3])], axis=1)
        s = x_s[0].shape[2]
        for i in range(r):
            idx_sim = K_idx_sorted[i+r, :]
            idx_sim = idx_sim[::-1]
            # query_i_sim = indc_x[idx_sim[::-1]]
            for j in range(c):
                if j==0:
                    x = self.denormalize(x_q[i+r])
                    # y = indc_y_1[i]
                    y = y_q[i+r]
                    y_p_r = y_pred_[i+r]
                    img = transforms.ToPILImage()(x.cpu()).convert("RGB")
                    all_imgs[s*i:s*(i+1), s*j:s*(j+1), :] = img
                    if y==y_p_r:
                        all_imgs[s*i:s*(i+1), s*(j+1)-d:s*(j+1), :] = rigt_grn
                    else:
                        all_imgs[s*i:s*(i+1), s*(j+1)-d:s*(j+1), :] = rigt_red
                else:
                    jj = idx_sim[j-1]
                    x = self.denormalize(x_s[jj])
                    # y = indc_y_0[i]
                    img = transforms.ToPILImage()(x.cpu()).convert("RGB")
                    all_imgs[s*i:s*(i+1), (s*j) + gap:(s*(j+1))+gap, :] = img
                    if y_s[jj]==y:
                        all_imgs[s*i:s*i+d, (s*j) + gap:(s*(j+1)) + gap, :] = top_grn
                    else:
                        all_imgs[s*i:s*i+d, (s*j) + gap:(s*(j+1)) + gap, :] = top_red
        
        all_imgs[:, s: s + gap, :] = gap_img
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax = clear_ax(ax)
        ax.axis("off")
        ax.imshow(all_imgs.astype('uint8'))
        fig.suptitle(f'Similarity between Query and SV images [{m}/100]', fontsize=8)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50, 200, 800, 500) 
        fig.subplots_adjust(
        top=0.940,
        bottom=0.044,
        left=0.025,
        right=0.975,
        hspace=0.002,
        wspace=0.002
        )
        os.makedirs(f'{out_path}/task_{k}', exist_ok=True)
        fig.savefig(f'{out_path}/task_{k}/Query_Support_sim_2.png', bbox_inches='tight')
        # plt.show()
        plt.close(5)

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

    def denormalize(self, tensor):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        denormalized = tensor.clone()

        for channel, mean, std in zip(denormalized, means, stds):
            channel.mul_(std).add_(mean)

        return denormalized


class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, dirichlet, kernel='linear'):
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
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distance
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

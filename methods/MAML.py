# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from time import gmtime, strftime
import os
#Check if tensorboardx is installed
try:
    #tensorboard --logdir=./MAML_Loss/ --host localhost --port 8091
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False

class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, inner_loop=5, inner_lr=1e-3, first_order=False, normalize=False, mini_batches=False):
        super(MAML, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        self.normalize = normalize
        self.n_task     = 5
        self.task_update_num = inner_loop #5
        self.train_lr = inner_lr
        self.approx = first_order #first order approx.
        self.mini_batch = mini_batches
        self.writer = None       

               
    def init_summary(self, id):
        self.id = id
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            if not os.path.isdir('./MAML_Loss'):
                os.makedirs('./MAML_Loss')
            writer_path = './MAML_Loss/' + id #+'_'+ time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def forward(self,x):
        out  = self.feature.forward(x)
        if(self.normalize): out = F.normalize(out, p=2, dim=1)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self, x_support, x_query, y_support, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x_a_i = x_support.cuda()
        y_a_i = y_support
        x_b_i = x_query.cuda()
        # x_var = x
        # x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        # x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        # y_a_i = torch.tensor( np.repeat(range( self.n_way ), self.n_support ), dtype=torch.long ).cuda() #label for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()
        # self.feature.train()
        # self.classifier.train()
        for task_step in range(self.task_update_num):
            
         
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
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
        #torch.long solved the error of cross_entropy
        # loss = nn.CrossEntropyLoss()
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.empty(3, dtype=torch.long).random_(5)
        # output = loss(input, target)
        # output.backward()
        # # Example of target with class probabilities
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.randn(3, 5).softmax(dim=1)
        # output = loss(input, target)   
        # self.feature.eval()
        # self.classifier.eval()
        scores = self.forward(x_b_i)
        # self.support_scores = self.forward(x_a_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x_support, x_query, y_support, y_query):
        scores = self.set_forward(x_support, x_query, y_support, is_feature = False)
        # y_b_i = torch.tensor( np.repeat(range( self.n_way ), self.n_query), dtype=torch.long ).cuda()
        loss = self.loss_fn(scores, y_query)

        return loss

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        loss_list = []
        optimizer.zero_grad()
        batch_size = 16      
        #train
        for i, (x,_) in enumerate(train_loader):        
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            self.iteration = i+(epoch*len(train_loader))
 
            x_a_i = x[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
            x_b_i = x[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
            y_a_i = torch.tensor( np.repeat(range( self.n_way ), self.n_support ), dtype=torch.long ).cuda() #label for support data
            y_b_i = torch.tensor( np.repeat(range( self.n_way ), self.n_query), dtype=torch.long ).cuda()

            
            if self.mini_batch:
                idx_permuted = torch.randperm(x_a_i.shape[0])
                x_a_i = x_a_i[idx_permuted]
                y_a_i = y_a_i[idx_permuted]
                loss_all = []
                number_of_batches = int(np.ceil(x_a_i.shape[0] / batch_size))
                optimizer.zero_grad()
                for b in range(number_of_batches):
                    x_batch = x_a_i[b*batch_size:(b+1)*batch_size,:, :, :]
                    y_batch = y_a_i[b * batch_size:(b + 1) * batch_size]
                    loss = self.set_forward_loss(x_batch, x_b_i, y_batch, y_b_i)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_all.append(loss.item())
                    avg_loss = avg_loss+loss.item()

                loss_list.append(np.mean(loss_all))
            
                if(self.writer is not None): 
                        self.writer.add_scalar('Loss', np.mean(loss_all), self.iteration)    
            else:
                loss = self.set_forward_loss(x_a_i, x_b_i, y_a_i, y_b_i)
                avg_loss = avg_loss+loss.item()#.data[0]
                loss_all.append(loss)

                task_count += 1

                if task_count == self.n_task: #MAML update several tasks at one time
                    loss_q = torch.stack(loss_all).sum(0)
                    loss_q.backward()
                    optimizer.step()
                    loss_list.append(torch.mean(torch.stack(loss_all)).item())
                    task_count = 0
                    loss_all = []
                    optimizer.zero_grad()

                    if(self.writer is not None): 
                            self.writer.add_scalar('Loss', loss_list[-1], self.iteration)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
        if(self.writer is not None): self.writer.add_scalar('Loss_[mean]', np.mean(loss_list), self.iteration)

    def test_loop(self, test_loader, return_std = False, dataset=None, show_plot=False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )
            if(i % 10==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Acc {:f}'.format(i, len(test_loader), acc_mean))

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('Acc val', acc_mean, self.iteration)
        result = {'acc':acc_mean,  'std':acc_std}
        result = {k: np.around(v, 4) for k, v in result.items()}
        #result = {'mse':np.around(np.mean(mse_list), 3), 'std':np.around(np.std(mse_list),3)}
        result['inner_loop'] = self.task_update_num
        result['inner_lr'] = self.train_lr
        result['first_order'] = self.approx
        if return_std:
            return acc_mean, acc_std, result
        else:
            return acc_mean, result

    def correct(self, x):  
        x_a_i = x[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = torch.tensor( np.repeat(range( self.n_way ), self.n_support ), dtype=torch.long ).cuda() #label for support data     
        scores = self.set_forward(x_support=x_a_i, x_query=x_b_i, y_support=y_a_i)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits


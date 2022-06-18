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

class FeatureTransfer(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, normalize=False):
        super(FeatureTransfer, self).__init__(model_func,  n_way, n_support)

        self.loss_fn = nn.CrossEntropyLoss()
        # self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier = nn.Linear(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        self.normalize = normalize
        self.n_batch = 2
        self.writer = None       

               
    def init_summary(self, id, dataset):
        self.id = id
        self.dataset = dataset
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            writer_path = "./log/" + id #+'_'+ time_string 
            self.writer = SummaryWriter(log_dir=writer_path)

    def forward(self, x, is_feature = False):
        x = x.cuda()
        out  = self.feature.forward(x)
        if(self.normalize): out = F.normalize(out, p=2, dim=1)
        scores  = self.classifier.forward(out)
        return scores


    def set_forward_loss(self, x):
        scores = self.forward(x, is_feature = False)
        # y_b_i = torch.tensor( np.repeat(range( self.n_way ), self.n_query), dtype=torch.long ).cuda()
        y = torch.tensor( np.repeat(range( self.n_way ), (self.n_support + self.n_query) ), dtype=torch.long ).cuda()
        loss = self.loss_fn(scores, y)

        return loss

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 50
        avg_loss=0
        batch_count = 0
        loss_all = []
        optimizer.zero_grad()
             
        #train
        for i, (x,_) in enumerate(train_loader):  
            # x = x.reshape([-1, x.shape[2], x.shape[3], x.shape[4]]).shape      
            
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            x = x.contiguous().view( self.n_way* (self.n_support+ self.n_query), *x.size()[2:])

            loss = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()#.data[0]
            loss_all.append(loss)

            batch_count += 1

            if batch_count == self.n_batch: #Transfer update after several batch (task)
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                batch_count = 0
                loss_all = []
                optimizer.zero_grad()

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('Loss', loss.item(), self.iteration)
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                      
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
        # result['inner_loop'] = self.task_update_num
        # result['inner_lr'] = self.train_lr
        # result['first_order'] = self.approx
        if return_std:
            return acc_mean, acc_std, result
        else:
            return acc_mean, result

    def correct(self, x):       
        scores = self.set_forward_adaptation(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward_adaptation(self, x,is_feature = False):
        # assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x, is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 ).detach()
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ).detach()

        if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
        if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        # y_support = y_support.to(dtype=torch.float64).cuda()
        y_support = y_support.type(torch.LongTensor).cuda()

        # if self.loss_type == 'softmax':
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        # elif self.loss_type == 'dist':        
        #     linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        # set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr = 0.01)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id.to(dtype=torch.long)]
                y_batch = y_support[selected_id.to(dtype=torch.long)] 
                scores = linear_clf(z_batch.to(dtype=torch.float64))
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query.to(dtype=torch.float64))
        return scores

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits = self.set_forward(x)
        return logits

# ==============================================================================================
    # def set_forward(self, x, is_feature = False):
    #     assert is_feature == False, 'MAML do not support fixed feature' 
    #     x = x.cuda()
    #     # x_var = x
    #     # x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
    #     # x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
    #     # y_a_i = torch.tensor( np.repeat(range( self.n_way ), self.n_support ), dtype=torch.long ).cuda() #label for support data

    #     y = torch.tensor( np.repeat(range( self.n_way ), (self.n_support + self.n_query) ), dtype=torch.long ).cuda()
             
    #     scores = self.forward(x)
    #     return scores

    # def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
    #     raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


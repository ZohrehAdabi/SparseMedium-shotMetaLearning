## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from colorama import Fore
from qpth.qp import QPFunction

from time import gmtime, strftime
import random
# from configs import kernel_type
#Check if tensorboardx is installed
try:
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

class MetaOptNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, normalize=False):
        super(MetaOptNet, self).__init__(model_func, n_way, n_support)
        ## GP parameters
       
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.normalize = normalize
        self.SVM = ClassificationHead().cuda()
       

    def init_summary(self, id):
        self.id = id
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M", gmtime())
            writer_path = "./log/" + id #+'_'+ time_string 
            self.writer = SummaryWriter(log_dir=writer_path)


    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass


    def train_loop(self, epoch, train_loader, optimizer, print_freq=5):
        # optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
        #                               {'params': self.feature_extractor.parameters(), 'lr': 1e-3}])

        update = 2
        loss_list = []
     
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).cuda()
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).cuda())
            x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
            y_support = torch.tensor(np.repeat(range(self.n_way), self.n_support)).cuda()
            x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
            y_query = torch.tensor(np.repeat(range(self.n_way), self.n_query)).cuda()
            x_train = x_all
            y_train = y_all

            self.SVM.train()
            self.feature_extractor.train()
            z_support = self.feature_extractor.forward(x_support)
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            z_query = self.feature_extractor.forward(x_query)
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)

            logit_query, num_SV = self.SVM(query=z_query, support=z_support, support_labels=y_support, n_way=self.n_way,  n_shot=self.n_support)

            smoothed_one_hot = one_hot(y_query.reshape(-1), self.n_way)
            eps = 0
            smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (self.n_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, self.n_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            
            loss_list.append(loss)
           
            if update==2:
                ## Optimize
                loss = torch.stack(loss_list).mean()
                loss_list = []
                update = 0
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            update +=1

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss.item(), self.iteration)

            with torch.no_grad():
                accuracy_query = self.count_accuracy(logit_query.reshape(-1, self.n_way), y_query.reshape(-1)).item()
      
            if i % print_freq==0:
                if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                print(Fore.LIGHTRED_EX, 'Epoch [{:d}] [{:d}/{:d}] | Loss {:f} | Supp. {:f} | Query {:f}'.format(epoch, i, len(train_loader), 
                                loss.item(), 0, accuracy_query), Fore.RESET)

    def correct(self, x, N=0, laplace=False):
        self.SVM.eval()
        self.feature_extractor.eval()
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = torch.tensor(np.repeat(range(self.n_way), self.n_query)).cuda()

        with torch.no_grad():
            self.SVM.eval()
            self.feature_extractor.eval()
            z_support = self.feature_extractor.forward(x_support)
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            z_query = self.feature_extractor.forward(x_query)
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)

            logit_query, num_SV = self.SVM(query=z_query, support=z_support, support_labels=y_support, n_way=self.n_way,  n_shot=self.n_support)
            accuracy_query = self.count_accuracy(logit_query.reshape(-1, self.n_way), y_query.reshape(-1)).item()
           
        return accuracy_query, num_SV

    def test_loop(self, test_loader, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        num_SV_list = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            acc_, num_SV = self.correct(x)
            acc_all.append(acc_)
            num_SV_list.append(num_SV)
            if(i % 10==0):
                acc_mean = np.mean(acc_all)
                print('Test | Batch {:d}/{:d} | Acc {:f}'.format(i, len(test_loader), acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        num_SV_mean = np.mean(num_SV_list)

        print(Fore.CYAN,f'Avg. SVs {num_SV_mean:.2f}', Fore.RESET)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if(self.writer is not None): self.writer.add_scalar('Avg. SVs', num_SV_mean, self.iteration)
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)

        result = {'acc': acc_mean, 'std': acc_std, 'SVs':num_SV_mean, 'SVs std':np.std(num_SV_list)}
        result = {k: np.around(v, 4) for k, v in result.items()}
        if self.normalize: result['normalize'] = True
        
        if(return_std): return acc_mean, acc_std, result
        else: return acc_mean, result


    def count_accuracy(self, logits, label):
        pred = torch.argmax(logits, dim=1).view(-1)
        label = label.view(-1)
        accuracy = 100 * pred.eq(label).float().mean()
        return accuracy


def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=20):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    # with torch.no_grad():
    support = support.unsqueeze(0)
    query = query.unsqueeze(0)
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^k_n <= C^k_n \forall k,n
    #where C^k_n = C if k  = y_n,
    #C^k_n = 0 if k != y_n.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_k \alpha^k_n=0 \forall n
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol_ = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    qp_sol_ = qp_sol_.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = qp_sol_ * compatibility
    logits = torch.sum(logits, 1)
    
    idx = qp_sol_> 0.001
    # idx_zero = torch.where((idx==False).all(axis=2))[1]
    idx_nonzero = torch.where((idx==True).any(axis=2))[1]

    return logits, idx_nonzero.shape[0]

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
    A:  a (n_batch, n, d) Tensor.
    B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
    b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)
    
    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
    indices:  a (n_batch, m) Tensor or (m) Tensor.
    depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1])).to(torch.int64)
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    
    return encoded_indicies

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] +\
            list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), 
                matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        
        self.head = MetaOptNetHead_SVM_CS
  
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        if self.enable_scale:
            self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            logit, num_SV = self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return self.scale * logit, num_SV
        else:
            logit, num_SV = self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return logit, num_SV
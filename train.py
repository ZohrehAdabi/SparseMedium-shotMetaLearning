import numpy as np
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from colorama import Fore
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.DKT import DKT
from methods.DKT_binary import DKT_binary
from methods.DKT_binary_new_loss import DKT_binary_new_loss
from methods.Sparse_DKT_Nystrom import Sparse_DKT_Nystrom
from methods.Sparse_DKT_Exact import Sparse_DKT_Exact
from methods.Sparse_DKT_RVM import Sparse_DKT_RVM
from methods.Sparse_DKT_binary_Nystrom import Sparse_DKT_binary_Nystrom
from methods.Sparse_DKT_binary_RVM import Sparse_DKT_binary_RVM
from methods.Sparse_DKT_binary_Nystrom_new_loss import Sparse_DKT_binary_Nystrom_new_loss
from methods.Sparse_DKT_binary_Exact import Sparse_DKT_binary_Exact
from methods.Sparse_DKT_binary_Exact_new_loss import Sparse_DKT_binary_Exact_new_loss
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.MAML import MAML
from methods.MetaOptNet import MetaOptNet
from methods.MetaOptNet_binary import MetaOptNet_binary
from methods.feature_transfer import FeatureTransfer
from io_utils import model_dict, parse_args, get_resume_file
from configs import run_float64

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if run_float64: torch.set_default_dtype(torch.float64)
        if(verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, lr_gp, lr_net, params):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == 'Adam':
        if params.method in ['MAML', 'MetaOptNet', 'MetaOptNet_binary', 'baseline', 'baseline++', 'transfer']:
            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr_net}])
        else:
            optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': lr_gp},
                                      {'params': filter(lambda p: p.requires_grad, model.feature_extractor.parameters()), 'lr': lr_net}])
                                      
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    max_acc = 0
    max_acc_rvm = 0
    print(f'num train task {len(base_loader)}')
    print(f'num val task {len(val_loader)}')

    acc_val_list = []
    tic = time.process_time()
    for epoch in range(start_epoch, stop_epoch):
        #model.eval()
        #acc = model.test_loop(val_loader)
        print(f'Epoch {epoch}')
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        if params.lr_decay:
            scheduler.step()
        # if (epoch) in [50]:
        #         optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        model.eval()
        if params.method not in ['baseline', 'baseline++']:
            if ((epoch+1)%3==0 and (epoch+1) > 50) or ((epoch+1)%10==0 and (epoch+1)<=50)  or (epoch == stop_epoch - 1):
                if not os.path.isdir(params.checkpoint_dir):
                    os.makedirs(params.checkpoint_dir)
                print(Fore.GREEN,"-"*50 ,f'\nValidation {params.method}\n', Fore.RESET)
                acc, result = model.test_loop(val_loader)
                acc_val_list.append(acc)
                if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                    print("--> Best model! save...")
                    max_acc = acc
                    outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                if 'rvm acc' in result.keys():
                    acc_rvm = result['rvm acc']
                    if acc_rvm > max_acc_rvm:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                        print("--> Best RVM model! save...")
                        max_acc_rvm = acc_rvm
                        outfile = os.path.join(params.checkpoint_dir, 'best_model_rvm.tar')
                        torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            
                print(Fore.YELLOW, f'ACC: {acc:4.2f}\n', Fore.RESET)
                print(Fore.YELLOW, f'Avg. Val ACC: {np.mean(acc_val_list):4.2f}\n', Fore.RESET)
                print(Fore.GREEN,"-"*50 ,'\n', Fore.RESET)
        toc = time.process_time()
        eTime = toc - tic 
        print(f'Elapsed time during the whole program in seconds: {eTime}')
        print(f'Elapsed time during the whole program in minutes: {eTime/60}')
        if ((epoch % params.save_freq==0) or (epoch == stop_epoch - 1)):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict(), 'e_time': eTime}, outfile)
    print(f'\n{model.id}\n')
    return model


if __name__ == '__main__':
    params = parse_args('train')
    _set_seed(parse_args('train').seed)
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 100  # default*******************

    if params.method in ['baseline', 'baseline++', 'transfer']:

        if params.method in ['transfer']:
            train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
            base_datamgr = SetDataManager(image_size, **train_few_shot_params, n_query=params.n_query, n_eposide=params.n_task) #n_eposide=100
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

            test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
            val_datamgr = SetDataManager(image_size, **test_few_shot_params, n_query=params.n_query, n_eposide=params.n_task)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

            id_ =f'Transfer_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
            if params.normalize: id_ += '_norm'
            if params.lr_decay: id_ += '_lr_decay'
            if params.train_aug: id_ += '_aug'
            if params.mini_batches: id_ += '_mini_batch'
            model = FeatureTransfer(model_dict[params.model], normalize=params.normalize, mini_batches=params.mini_batches, **train_few_shot_params)
    
            model.init_summary(id=id_, dataset=params.dataset)
        else:
            base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
            val_datamgr = SimpleDataManager(image_size, batch_size=64)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

            if params.dataset == 'omniglot':
                assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
            if params.dataset == 'cross_char':
                assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

            if params.method == 'baseline':
                model = BaselineTrain(model_dict[params.model], params.num_classes, normalize=params.normalize)
            elif params.method == 'baseline++':
                model = BaselineTrain(model_dict[params.model], params.num_classes, normalize=params.normalize, loss_type='dist')
            
            id_ = f'{params.method}_{params.model}_n_class_{params.num_classes}'
            if params.normalize: id_ += '_norm'
            if params.lr_decay: id_ += '_lr_decay'
            if params.train_aug: id_ += '_aug'
            if params.batch_size!=16: id_ += f'_batch_size_{params.batch_size}'
        
            model.init_summary(id=id_, dataset=params.dataset)
        
    elif params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM', 'Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss', 
                            'Sparse_DKT_binary_Exact', 'Sp_DKT_Bin_Exact_NLoss', 
                            'DKT', 'DKT_binary', 'DKT_binary_new_loss', 'protonet', 
                            'matchingnet', 'relationnet', 'relationnet_softmax', 'MAML', 'maml_approx', 'MetaOptNet', 'MetaOptNet_binary']:
        # for fewshot setting
        # n_query = max(1, int(
        #     16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        
        if params.batch_size==16: params.batch_size = 1 #one task per iteration (default value)
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, **train_few_shot_params, n_query=params.n_query, n_eposide=params.n_task) #n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, **test_few_shot_params, n_query=params.n_query, n_eposide=100)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if(params.method == 'Sparse_DKT_Nystrom'):
            model = Sparse_DKT_Nystrom(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method,
                                    add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, 
                                    maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip, 
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, 
                                    gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            
            if params.sparse_method in ['FRVM', 'augmFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)
        
        elif(params.method == 'Sparse_DKT_Exact'):
            model = Sparse_DKT_Exact(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, 
                                    maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            if params.sparse_method in ['FRVM', 'augmFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)
        
        elif(params.method == 'Sparse_DKT_RVM'):
            model = Sparse_DKT_RVM(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method,
                                    add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, 
                                    maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
                                    rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip, 
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, 
                                    gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            
            if params.sparse_method in ['FRVM', 'augmFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.rvm_mll_only: id += f'_rvm_mll_only'
            if params.rvm_ll_only: id += f'_rvm_ll_only'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)      

        elif params.method == 'Sparse_DKT_binary_Nystrom':
            model = Sparse_DKT_binary_Nystrom(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                    maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression,
                                    num_inducing_points=params.num_ip,
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, 
                                    gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            if params.sparse_method in ['FRVM', 'augmFRVM', 'constFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM', 'constFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)
            if params.sparse_method=='constFRVM':
                print(f'\nconstFRVM\n')
                model.load_constant_model()
        
        elif params.method == 'Sparse_DKT_binary_Exact':
            model = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    separate=params.separate, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, 
                                    lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
                                    num_inducing_points=params.num_ip,
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            
            if params.sparse_method in ['FRVM', 'augmFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.separate: id += '_separate'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)   

        elif params.method == 'Sparse_DKT_binary_RVM':
            model = Sparse_DKT_binary_RVM(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                    maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression,
                                    rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip, 
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            if params.sparse_method in ['FRVM', 'augmFRVM', 'constFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}' 
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.rvm_mll_only: id += f'_rvm_mll_only'
            if params.rvm_ll_only: id += f'_rvm_ll_only'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM', 'constFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)
            if params.sparse_method=='constFRVM':
                print(f'\nconstFRVM\n')
                model.load_constant_model()
     

        elif params.method == 'Sp_DKT_Bin_Nyst_NLoss':
            model = Sparse_DKT_binary_Nystrom_new_loss(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    num_inducing_points=params.num_ip,
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            if params.sparse_method in ['FRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            model.init_summary(id=id, dataset=params.dataset)

        elif params.method == 'Sp_DKT_Bin_Exact_NLoss':
            model = Sparse_DKT_binary_Exact_new_loss(model_dict[params.model], params.kernel_type, **train_few_shot_params, sparse_method=params.sparse_method, 
                                    add_rvm_mll=params.rvm_mll, lambda_rvm=params.lambda_rvm, num_inducing_points=params.num_ip,
                                    normalize=params.normalize, scale=params.scale, config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            
            if params.sparse_method in ['FRVM', 'augmFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM']: 
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            model.init_summary(id=id, dataset=params.dataset)

        elif(params.method == 'DKT'):
            model = DKT(model_dict[params.model], params.kernel_type, **train_few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
            if params.dirichlet:
                id=f'DKT_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id=f'DKT_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
           
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            model.init_summary(id=id, dataset=params.dataset)
        
        elif(params.method == 'DKT_binary'):
            model = DKT_binary(model_dict[params.model], params.kernel_type, **train_few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
            if params.dirichlet:
                id=f'DKT_binary_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id=f'DKT_binary_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            model.init_summary(id=id, dataset=params.dataset)
        
        elif(params.method == 'DKT_binary_new_loss'):
            model = DKT_binary_new_loss(model_dict[params.model], params.kernel_type, **train_few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
            if params.dirichlet:
                id=f'DKT_binary_new_loss_{params.model}_{params.dataset}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id=f'DKT_binary_new_loss_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            model.init_summary(id=id)
        
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['MAML', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            id=f'MAML_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}'
         
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            if params.first_order: id += '_first_order'
            if params.mini_batches: id += '_mini_batch'
            model = MAML(model_dict[params.model], inner_loop=params.inner_loop, inner_lr=params.inner_lr, first_order=params.first_order, 
                                        normalize=params.normalize, mini_batches=params.mini_batches, **train_few_shot_params)
            # if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            #     model.n_task = 32
            #     model.task_update_num = 1
            #     model.train_lr = 0.1
            
            model.init_summary(id=id)
        elif params.method in ['MetaOptNet']:
            
            id=f'MetaOptNet_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            model = MetaOptNet(model_dict[params.model], normalize=params.normalize, **train_few_shot_params)
    
            model.init_summary(id=id)
        elif params.method in ['MetaOptNet_binary']: 
            id=f'MetaOptNet_binary_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.train_aug: id += '_aug'
            model = MetaOptNet_binary(model_dict[params.model], normalize=params.normalize, **train_few_shot_params)
    
            model.init_summary(id=id)
        
        print(f'\n{id}\n')
    
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_seed_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.seed)
    # if params.train_aug:
    #     params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        
        if params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM', 'Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 
                                'Sp_DKT_Bin_Nyst_NLoss', 'Sparse_DKT_binary_Exact', 'Sp_DKT_Bin_Exact_NLoss',
                                'transfer']:
            if params.dirichlet:
                id = f'_{params.sparse_method}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'_{params.sparse_method}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           
            if params.sparse_method in ['FRVM', 'augmFRVM', 'constFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            if params.normalize: id += '_norm'
            if params.separate: id += '_separate'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.tol_rvm!=1e-4: id += f'_tol_rvm_{params.tol_rvm}'
            if params.regression: id += f'_regression'
            if params.rvm_mll_only: id += f'_rvm_mll_only'
            if params.rvm_ll_only: id += f'_rvm_ll_only'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.batch_size!=1: id += f'_batch_size_{params.batch_size}'
            if params.sparse_method in ['Random', 'KMeans', 'augmFRVM', 'constFRVM']:  
                if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
            params.checkpoint_dir += id
        elif  params.method in ['DKT', 'DKT_binary']:
            if params.dirichlet:
                id=f'_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
        
            if params.normalize: id += '_norm'
            if params.train_aug: id += '_aug'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.batch_size!=1: id += f'_batch_size_{params.batch_size}'
            params.checkpoint_dir += id
        #MAML, MetaOptNet
        elif params.method in ['MAML']:
            id=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}'
            if params.normalize: id += '_norm'
            if params.train_aug: id += '_aug'
            if params.first_order: id += '_first_order'
            if params.mini_batches: id += '_mini_batch'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.batch_size!=1: id += f'_batch_size_{params.batch_size}'
            params.checkpoint_dir += id
        
        elif params.method in ['MetaOptNet', 'transfer']:
            id=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
            if params.normalize: id += '_norm'
            if params.train_aug: id += '_aug'
            if params.first_order: id += '_first_order'
            if params.warmup:  id += '_warmup'
            if params.freeze: id += '_freeze'
            if params.mini_batches: id += '_mini_batch'
            if params.batch_size!=1: id += f'_batch_size_{params.batch_size}'
            params.checkpoint_dir += id

    else:
        if params.method in ['baseline', 'baseline++']:
            id_ = f'_n_class_{params.num_classes}'
            if params.normalize: id_ += '_norm'
            if params.lr_decay: id_ += '_lr_decay'
            if params.train_aug: id_ += '_aug'
            if params.batch_size!=16: id_ += f'_batch_size_{params.batch_size}'
            params.checkpoint_dir += id_

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    # if params.method == 'MAML' or params.method == 'maml_approx':
    #     stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            # if params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss']:
                
            #     IP = torch.ones(100, 64).cuda()
            #     tmp['state']['model.covar_module.inducing_points'] = IP
            #     tmp['state']['mll.model.covar_module.inducing_points'] = IP
            if params.method in ['Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss']:
                
                IP = torch.ones(100, 64).cuda()
                tmp['state']['model.covar_module.inducing_points'] = IP
                tmp['state']['mll.model.covar_module.inducing_points'] = IP

            if params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_RVM']:
                IP = torch.ones(100, 64).cuda()
                for i in range(params.test_n_way):
                    tmp['state'][f'model.models.{i}.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.mlls.{i}.model.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.model.models.{i}.covar_module.inducing_points'] = IP
                    
            start_epoch = tmp['epoch'] + 1
            stop_epoch = start_epoch + stop_epoch
            model.load_state_dict(tmp['state'])
            print(f'\nResume \n')
    
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper

        
           
        if('DKT_binary' in params.method):
            warmup_model = DKT_binary(model_dict[params.model], **train_few_shot_params, dirichlet=params.dirichlet)
            warmup_model_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, 'DKT_binary')
           
        elif('DKT' in params.method):
            warmup_model = DKT(model_dict[params.model], **train_few_shot_params, dirichlet=params.dirichlet)
            warmup_model_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, 'DKT')
        
        

        warmup_model = warmup_model.cuda()

        if params.dirichlet:
            warmup_id=f'_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
        else:
            warmup_id=f'_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
        
        if params.train_aug: warmup_id += '_aug'
        warmup_model_checkpoint_dir += warmup_id
        # baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        # configs.save_dir, params.dataset, params.model, 'baseline')
        # if params.train_aug:
        #     baseline_checkpoint_dir += '_aug'

        warmup_resume_file = get_resume_file(warmup_model_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:

            warmup_model.load_state_dict(tmp['state'])
            model.feature_extractor.load_state_dict(warmup_model.feature_extractor.state_dict())
            print(f'\nWarmup\n')
            if params.freeze:
                for param in model.feature_extractor.parameters():
                    param.requires_grad = False
                print(f'\nWarmup and Freeze\n')
            # state = tmp['state']
            # state_keys = list(state.keys())
            # for i, key in enumerate(state_keys):
            #     if "feature." in key:
            #         newkey = key.replace("feature.",
            #                              "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            #         state[newkey] = state.pop(key)
            #     else:
            #         state.pop(key)
            # model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params.lr_gp, params.lr_net, params)

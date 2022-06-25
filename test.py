from timeit import repeat
import torch
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import time
from copy import deepcopy
import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.Sparse_DKT_Nystrom import Sparse_DKT_Nystrom
from methods.Sparse_DKT_Exact import Sparse_DKT_Exact
from methods.Sparse_DKT_RVM import Sparse_DKT_RVM
from methods.Sparse_DKT_binary_Nystrom import Sparse_DKT_binary_Nystrom
from methods.Sparse_DKT_binary_RVM import Sparse_DKT_binary_RVM
from methods.Sparse_DKT_binary_Nystrom_new_loss import Sparse_DKT_binary_Nystrom_new_loss
from methods.Sparse_DKT_binary_Exact import Sparse_DKT_binary_Exact
from methods.Sparse_DKT_binary_Exact_new_loss import Sparse_DKT_binary_Exact_new_loss
from methods.DKT import DKT
from methods.DKT_binary import DKT_binary
from methods.DKT_binary_new_loss import DKT_binary_new_loss
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.MAML import MAML
from methods.MetaOptNet import MetaOptNet
from methods.MetaOptNet_binary import MetaOptNet_binary
from methods.feature_transfer import FeatureTransfer
from io_utils import model_dict, get_resume_file, parse_args, get_best_file , get_assigned_file
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

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    # select_class = random.sample(class_list, n_way)
    select_class = list(np.random.choice(list(class_list), n_way, replace=False))
    print(f'selected classes {select_class}')
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc


def single_test(params):
    acc_all = []

    iter_num = 100 #600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'
    if params.method == 'transfer':
        # id_=f'Transfer_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
        # if params.normalize: id_+= '_norm'
        # if params.lr_decay: id_+= '_lr_decay'
        # if params.train_aug: id_+= '_aug'
        # if params.mini_batches: id_+= '_mini_batch'
        model           = FeatureTransfer( model_dict[params.model], normalize=params.normalize, mini_batches=params.mini_batches, **few_shot_params )
        last_model      = FeatureTransfer( model_dict[params.model], normalize=params.normalize, mini_batches=params.mini_batches, **few_shot_params )
        best_model      = FeatureTransfer( model_dict[params.model], normalize=params.normalize, mini_batches=params.mini_batches, **few_shot_params )
    elif params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], normalize=params.normalize, **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], normalize=params.normalize, loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'DKT':
        # model           = DKT(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
        # last_model      = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
        best_model      = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
    elif params.method == 'DKT_binary':
        # model           = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
        # last_model      = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)
        best_model      = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)

    elif params.method == 'DKT_binary_new_loss':
        model           = DKT_binary_new_loss(model_dict[params.model], params.kernel_type, **few_shot_params, normalize=params.normalize, dirichlet=params.dirichlet)   
    elif params.method == 'Sparse_DKT_Nystrom':
        model           = Sparse_DKT_Nystrom(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
                                num_inducing_points=params.num_ip, normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    
    elif params.method == 'Sparse_DKT_Exact':
        model           = Sparse_DKT_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
                                num_inducing_points=params.num_ip, normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    
    elif params.method == 'Sparse_DKT_RVM':
        model           = Sparse_DKT_RVM(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, 
                                tol_rvm=params.tol_rvm, regression=params.regression, 
                                rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip, 
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sparse_DKT_binary_Nystrom':
        model           = Sparse_DKT_binary_Nystrom(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sparse_DKT_binary_Exact':
        model           = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        last_model           = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        best_model           = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        best_model_rvm           = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sparse_DKT_binary_RVM':
        # model           = Sparse_DKT_binary_RVM(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
        #                         add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
        #                         maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
        #                         rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip,
        #                         normalize=params.normalize, scale=params.scale,
        #                         config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        last_model           = Sparse_DKT_binary_RVM(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
                                maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
                                rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        # best_model           = Sparse_DKT_binary_RVM(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
        #                         add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
        #                         maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
        #                         rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip,
        #                         normalize=params.normalize, scale=params.scale,
        #                         config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
        # best_model_rvm           = Sparse_DKT_binary_RVM(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
        #                         add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, lambda_rvm=params.lambda_rvm, 
        #                         maxItr_rvm=params.maxItr_rvm, tol_rvm=params.tol_rvm, regression=params.regression, 
        #                         rvm_mll_only=params.rvm_mll_only, rvm_ll_only=params.rvm_ll_only, num_inducing_points=params.num_ip,
        #                         normalize=params.normalize, scale=params.scale,
        #                         config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)


    elif params.method == 'Sp_DKT_Bin_Nyst_NLoss':
        model           = Sparse_DKT_binary_Nystrom_new_loss(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
   
    elif params.method == 'Sp_DKT_Bin_Exact_NLoss':
        model           = Sparse_DKT_binary_Exact_new_loss(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                add_rvm_mll=params.rvm_mll, lambda_rvm=params.lambda_rvm, num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)

    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    
    elif params.method in ['MAML' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], inner_loop=params.inner_loop, inner_lr=params.inner_lr, first_order=params.first_order, 
                            normalize=params.normalize,  mini_batches=params.mini_batches, **few_shot_params)
        # if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
        #     model.n_task     = 32
        #     model.task_update_num = 1
        #     model.train_lr = 0.1
    elif params.method in ['MetaOptNet']:
            
            id_=f'MetaOptNet_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
            if params.normalize: id_+= '_norm'
            if params.lr_decay: id_+= '_lr_decay'
            if params.train_aug: id_+= '_aug'
            model = MetaOptNet(model_dict[params.model], normalize=params.normalize, **few_shot_params)
            last_model = MetaOptNet(model_dict[params.model], normalize=params.normalize, **few_shot_params)
            best_model = MetaOptNet(model_dict[params.model], normalize=params.normalize, **few_shot_params)
            
    elif params.method in ['MetaOptNet_binary']: 
            id_=f'MetaOptNet_binary_{params.model}_{params.dataset}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
         
            if params.normalize: id_+= '_norm'
            if params.lr_decay: id_+= '_lr_decay'
            if params.train_aug: id_+= '_aug'
            model = MetaOptNet_binary(model_dict[params.model], normalize=params.normalize, **few_shot_params)
            last_model = MetaOptNet_binary(model_dict[params.model], normalize=params.normalize, **few_shot_params)
            best_model = MetaOptNet_binary(model_dict[params.model], normalize=params.normalize, **few_shot_params)
    
    else:
       raise ValueError('Unknown method')

    # model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_seed_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.seed)
    # if params.train_aug:
    #     checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        # checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        if params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM', 'Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss', 
        'Sparse_DKT_binary_Exact', 'Sp_DKT_Bin_Exact_NLoss']:
            if params.dirichlet:
                id_= f'_{params.sparse_method}_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id_= f'_{params.sparse_method}_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           

            if params.sparse_method in ['FRVM', 'augmFRVM', 'constFRVM']: 
                id_+= f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id_+= '_gamma'
                if params.scale: id_+= '_scale'
            
        elif  params.method in ['DKT', 'DKT_binary']:
            if params.dirichlet:
                id_=f'_n_task_{params.n_task}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id_=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
         
         #MAML, MetaOptNet
        elif  params.method in ['MAML']: 
            id_=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}'
        elif  params.method in ['MetaOptNet', 'transfer']: 

            id_=f'_n_task_{params.n_task}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_net}'
        else:
            raise ValueError('Unknown method')

        if params.normalize: id_+= '_norm'
        if params.separate: id_+= '_separate'
        if params.rvm_mll: id_+= f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_ll: id_+= f'_rvm_ll_{params.lambda_rvm}'
        if params.rvm_mll_one: id_+= f'_rvm_mll_one_{params.lambda_rvm}'
        if params.maxItr_rvm!=-1: id_+= f'_maxItr_rvm_{params.maxItr_rvm}'
        if params.tol_rvm!=1e-4: id_+= f'_tol_rvm_{params.tol_rvm}'
        if params.regression: id_+= f'_regression'
        if params.rvm_mll_only: id_+= f'_rvm_mll_only'
        if params.rvm_ll_only: id_+= f'_rvm_ll_only'
        if params.train_aug: id_+= '_aug'
        if params.first_order: id_+= '_first_order'
        if params.warmup:  id_+= '_warmup'
        if params.freeze: id_+= '_freeze'
        if params.mini_batches: id_+= '_mini_batch'
        if params.sparse_method in ['Random', 'KMeans', 'augmFRVM', 'constFRVM']:  
            if params.num_ip is not None:
                    id_+= f'_ip_{params.num_ip}'
        checkpoint_dir += id_
    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] : 
        best = False
        # best = True
        # last = False
        last = True
        best_rvm = False
        if params.method in ['DKT', 'DKT_binary']:
            last = False
            # last = True
            # best = False
            best = True
            best_rvm = False 
        print(f'\n{checkpoint_dir}\n')
        modelfile = None
        if params.save_iter != -1:
            print(f'\nModel at epoch {params.save_iter}\n')
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        
        
        if last:
            if  params.method in ['MetaOptNet_binary', 'MetaOptNet']:
                last_model = last_model.cuda()
            else:
                # last_model = deepcopy(model)
                last_model = last_model.cuda()

            files = os.listdir(checkpoint_dir)
            nums =  [int(f.split('.')[0]) for f in files if 'best' not in f]
            num = max(nums)
            print(f'\nModel at last epoch {num}\n')
            last_modelfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
            print(f'\nlast model {last_modelfile}\n')
        
        if best: #else:
            if  params.method in ['MetaOptNet_binary', 'MetaOptNet']:
                best_model = best_model.cuda()
            else:
                # best_model = deepcopy(model)
                best_model = best_model.cuda()
            best_modelfile   = get_best_file(checkpoint_dir)
            print(f'\nBest model {best_modelfile}\n')

        if best_rvm: #else:
            # best_model_rvm = deepcopy(model)
            best_model_rvm = best_model_rvm.cuda()            
            best_modelfile_rvm   = os.path.join(checkpoint_dir, 'best_model_rvm.tar')
            if not os.path.isfile(best_modelfile_rvm):
                best_modelfile_rvm = None
            print(f'\nBest RVM model {best_modelfile_rvm}\n')

        if modelfile is not None:
            tmp = torch.load(modelfile)
            if params.method in ['Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss']:
                
                IP = torch.ones(100, 64).cuda()
                tmp['state']['model.covar_module.inducing_points'] = IP
                tmp['state']['mll.model.covar_module.inducing_points'] = IP
            if params.method in ['Sparse_DKT_Nystrom']:
                IP = torch.ones(100, 64).cuda()
                for i in range(params.test_n_way):
                    tmp['state'][f'model.models.{i}.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.mlls.{i}.model.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.model.models.{i}.covar_module.inducing_points'] = IP

            model.load_state_dict(tmp['state'])
        
        if last and (last_modelfile is not None):
            tmp = torch.load(last_modelfile)
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
            last_model.load_state_dict(tmp['state'])
            # last_model.feature_extractor.load_state_dict(tmp['state'])

        if best and (best_modelfile is not None):
            tmp = torch.load(best_modelfile)
            best_epoch = tmp['epoch']
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
            
            best_model.load_state_dict(tmp['state'])
            # best_model.feature_extractor.load_state_dict(tmp['state'])
        
        else:
            if best:
                print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))


        if params.method not in ['DKT', 'DKT_binary']:
            if best_rvm and (best_modelfile_rvm is not None):
                tmp = torch.load(best_modelfile_rvm)
                best_epoch_rvm = tmp['epoch']
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
                best_model_rvm.load_state_dict(tmp['state'])

            else:
                if best_rvm:
                    print("[WARNING] Cannot find 'best_file_rvm.tar' in: " + str(checkpoint_dir))

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    if params.method in ['MAML', 'maml_approx', 'MetaOptNet', 'MetaOptNet_binary', 'DKT', 'DKT_binary', 'DKT_binary_new_loss', 'Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM',
                            'Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_RVM', 'Sp_DKT_Bin_Nyst_NLoss', 'Sparse_DKT_binary_Exact', 'Sp_DKT_Bin_Exact_NLoss',
                            'transfer']: #maml do not support testing with feature
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84 
        else:
            image_size = 224

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = params.n_query , **few_shot_params)
        
        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
            else:
                loadfile   = configs.data_dir['CUB'] + split +'.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
            else:
                loadfile  = configs.data_dir['emnist'] + split +'.json' 
        else: # novel
            loadfile    = configs.data_dir[params.dataset] + split + '.json'

        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)

        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.

        if params.save_result:
            info_path = checkpoint_dir
            info_path = info_path.replace('//', '/')
            info_path = info_path.replace('\\', '/')
            info = info_path.split('/')
            info = '_'.join(info[3:]) 
            result_path = f'./record/{params.dataset}/seed_{params.seed}'
            if not os.path.isdir(result_path):
                os.makedirs(result_path)
            file = f'{result_path}/results_{info}.json'
            old_data = None
            if os.path.exists(file):
                if os.stat(file).st_size!=0:
                    f = open(file , "r")
                    old_data = json.load(f)
                    f.close()
            f = open(file , "w")
            if old_data is None:
                f.write('[\n')
            else:
                f = open(file , "w")
                f.write('[\n')
                for data in old_data:
                    json.dump(data, f, indent=2)
                    f.write(',\n')
            timestamp = time.strftime("%Y/%m/%d-%H:%M", time.localtime()) 

        if params.save_iter != -1 and modelfile is not None:    
            model.eval()
            acc_mean, acc_std, result = model.test_loop( novel_loader, return_std = True)
            print("-----------------------------")
            print('Test Acc model at epoch %d = %4.2f%% +- %4.2f%%' %(params.save_iter, acc_mean, acc_std))
            print("-----------------------------") 


        tasks = []
        import torchvision.transforms as transforms
        for i in range(5):

            data_0  = torch.randn([60, 3, 84, 84])
            data_1  = torch.randn([60, 3, 84, 84])
            data = torch.stack([data_0, data_1])
            tasks.append(data)
       
       
        if last:
            print(f'\nModel at last epoch {num}\n')
            last_model.eval()
            # acc_mean, acc_std, result = last_model.test_loop( novel_loader, return_std = True)
            acc_all = []
            acc_most_sim_all = []
            for i, task in enumerate(tasks):
                
                correct_this, count_this, loss_value, acc_most_sim = last_model.correct(task, i)
                acc_all.append(correct_this/ count_this*100)
                acc_most_sim_all.append((acc_most_sim/ count_this)*100)
            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            acc_most_sim_mean = np.mean(np.asarray(acc_most_sim_all))
            if params.save_result:
                f.write('{\n"time": ')
                f.write(f'"{timestamp}",\n')
                f.write('"last model":\n')
                json.dump(result, f, indent=2) #f.write(json.dumps(result))
                f.write(',\n')
            print("-----------------------------")
            print('Test Acc last model = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
            print("-----------------------------") 

        if best:
            print(f'\nBest model epoch {best_epoch}\n')
            best_model.eval()
            # acc_mean, acc_std, result = best_model.test_loop(novel_loader, return_std = True, dataset=params.dataset, show_plot=False)
            acc_all = []
            acc_most_sim_all = []
            for i, task in enumerate(tasks):
                
                correct_this, count_this, loss_value, acc_most_sim = best_model.correct(task, i)
                acc_all.append(correct_this/ count_this*100)
                acc_most_sim_all.append((acc_most_sim/ count_this)*100)
            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            acc_most_sim_mean = np.mean(np.asarray(acc_most_sim_all))
            if params.save_result:
                f.write('"best model":\n')
                json.dump(result, f, indent=2)
                f.write('\n}\n]')
            
            print("-----------------------------")
            print('Test Acc best model = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
            print("-----------------------------") 



        if best_rvm and (best_modelfile_rvm is not None):
            print(f'\nBest RVM model epoch {best_epoch_rvm}\n')
            best_model_rvm.eval()
            acc_mean, acc_std, result = best_model_rvm.test_loop(novel_loader, return_std = True, dataset=params.dataset, show_plot=False)
            if params.save_result:
                f.write('"best rvm model":\n')
                json.dump(result, f, indent=2)
                f.write(',\n')
            print("-----------------------------")
            print('Test Acc GP at best RVM model = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
            print("-----------------------------") 
        
        
        if params.save_result: f.close()
        print(f'\n{id_}\n')
    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = params.n_query, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    if params.save_result:   
       
        with open(f'./record/results_{info}.txt' , 'a') as f:
            # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
            timestamp = time.strftime("%Y/%m/%d-%H:%M", time.localtime()) 
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            f.write( 'Time: %s, Setting: %s, Stop_epoch: %s \n' %(timestamp,exp_setting,acc_str)  )
    return acc_mean

def main():        
    params = parse_args('test')
    seed = params.seed
    repeat = params.repeat
    #repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_list = list()
    for i in range(seed, seed+repeat):
        if(seed!=0): _set_seed(i)
        else: _set_seed(0)
        accuracy_list.append(single_test(parse_args('test')))
    print("-----------------------------")
    print('Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%' %(repeat, np.mean(accuracy_list), np.std(accuracy_list)))
    print("-----------------------------")        
if __name__ == '__main__':
    main()

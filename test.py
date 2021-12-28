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
from methods.Sparse_DKT_binary_Nystrom import Sparse_DKT_binary_Nystrom
from methods.Sparse_DKT_binary_Nystrom_new_loss import Sparse_DKT_binary_Nystrom_new_loss
from methods.Sparse_DKT_binary_Exact import Sparse_DKT_binary_Exact
from methods.DKT import DKT
from methods.DKT_binary import DKT_binary
from methods.DKT_binary_new_loss import DKT_binary_new_loss
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, get_resume_file, parse_args, get_best_file , get_assigned_file

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        if(verbose): print("[INFO] Setting SEED: " + str(seed))   
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
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

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'DKT':
        model           = DKT(model_dict[params.model], params.kernel_type, **few_shot_params, dirichlet=params.dirichlet)
    elif params.method == 'DKT_binary':
        model           = DKT_binary(model_dict[params.model], params.kernel_type, **few_shot_params, dirichlet=params.dirichlet)
    elif params.method == 'DKT_binary_new_loss':
        model           = DKT_binary_new_loss(model_dict[params.model], params.kernel_type, **few_shot_params, dirichlet=params.dirichlet)   
    elif params.method == 'Sparse_DKT_Nystrom':
        model           = Sparse_DKT_Nystrom(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sparse_DKT_Exact':
        model           = Sparse_DKT_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sparse_DKT_binary_Nystrom':
        model           = Sparse_DKT_binary_Nystrom(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
    elif params.method == 'Sp_DKT_Bin_Nyst_NLoss':
        model           = Sparse_DKT_binary_Nystrom_new_loss(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
                                normalize=params.normalize, scale=params.scale,
                                config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)

    elif params.method == 'Sparse_DKT_binary_Exact':
        model           = Sparse_DKT_binary_Exact(model_dict[params.model], params.kernel_type, **few_shot_params, sparse_method=params.sparse_method, 
                                num_inducing_points=params.num_ip,
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
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    # if params.train_aug:
    #     checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        # checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        if params.method in ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_binary_Nystrom', 'Sp_DKT_Bin_Nyst_NLoss', 'Sparse_DKT_binary_Exact']:
            if params.dirichlet:
                id = f'_{params.sparse_method}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id = f'_{params.sparse_method}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'           

            if params.sparse_method in ['FRVM', 'augmFRVM', 'constFRVM']: 
                id += f'_confg_{params.config}_{params.align_thr}'
                if params.gamma: id += '_gamma'
                if params.scale: id += '_scale'
            
        else:
            if params.dirichlet:
                id=f'_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            else:
                id=f'_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}_{params.kernel_type}'
            
           

        if params.normalize: id += '_norm'
        if params.train_aug: id += '_aug'
        if params.warmup:  id += '_warmup'
        if params.freeze: id += '_freeze'
        if params.sparse_method in ['Random', 'KMeans', 'augmFRVM', 'constFRVM']:  
            if params.num_ip is not None:
                    id += f'_ip_{params.num_ip}'
        checkpoint_dir += id
    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] : 
        best = True
        last = True
        print(f'\n{checkpoint_dir}\n')
        modelfile = None
        if params.save_iter != -1:
            print(f'\nModel at epoch {params.save_iter}\n')
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        if last:
            last_model = deepcopy(model)
            files = os.listdir(checkpoint_dir)
            nums =  [int(f.split('.')[0]) for f in files if 'best' not in f]
            num = max(nums)
            print(f'\nModel at last epoch {num}\n')
            last_modelfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
        if best: #else:
            best_model = deepcopy(model)
            print(f'\nBest model\n')
            best_modelfile   = get_best_file(checkpoint_dir)

        if modelfile is not None:
            tmp = torch.load(modelfile)
            if params.method in ['Sparse_DKT_binary_Nystrom', 'Sp_DKT_Bin_Nyst_NLoss']:
                
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

        if last_modelfile is not None:
            tmp = torch.load(last_modelfile)
            if params.method in ['Sparse_DKT_binary_Nystrom', 'Sp_DKT_Bin_Nyst_NLoss']:
                
                IP = torch.ones(100, 64).cuda()
                tmp['state']['model.covar_module.inducing_points'] = IP
                tmp['state']['mll.model.covar_module.inducing_points'] = IP
            if params.method in ['Sparse_DKT_Nystrom']:
                IP = torch.ones(100, 64).cuda()
                for i in range(params.test_n_way):
                    tmp['state'][f'model.models.{i}.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.mlls.{i}.model.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.model.models.{i}.covar_module.inducing_points'] = IP
            last_model.load_state_dict(tmp['state'])

        if best_modelfile is not None:
            tmp = torch.load(best_modelfile)
            best_epoch = tmp['epoch']
            if params.method in ['Sparse_DKT_binary_Nystrom', 'Sp_DKT_Bin_Nyst_NLoss']:
                
                IP = torch.ones(100, 64).cuda()
                tmp['state']['model.covar_module.inducing_points'] = IP
                tmp['state']['mll.model.covar_module.inducing_points'] = IP
            if params.method in ['Sparse_DKT_Nystrom']:
                
                IP = torch.ones(100, 64).cuda()
                for i in range(params.test_n_way):
                    tmp['state'][f'model.models.{i}.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.mlls.{i}.model.covar_module.inducing_points'] = IP
                    tmp['state'][f'mll.model.models.{i}.covar_module.inducing_points'] = IP
            best_model.load_state_dict(tmp['state'])

        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    if params.method in ['maml', 'maml_approx', 'DKT', 'DKT_binary', 'DKT_binary_new_loss', 'Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_binary_Nystrom', 'Sp_DKT_Bin_Nyst_NLoss', 'Sparse_DKT_binary_Exact']: #maml do not support testing with feature
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

        if params.save_iter != -1:    
            model.eval()
            acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)
        if last:
            print(f'\nModel at last epoch {num}\n')
            last_model.eval()
            acc_mean, acc_std = last_model.test_loop( novel_loader, return_std = True)
            print("-----------------------------")
            print('Test Acc last model = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
            print("-----------------------------") 
        if best:
            print(f'\nBest model epoch {best_epoch}\n')
            best_model.eval()
            acc_mean, acc_std = best_model.test_loop( novel_loader, return_std = True)
            print("-----------------------------")
            print('Test Acc best model = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
            print("-----------------------------") 

    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        
    with open(f'./record/results{id}.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
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


import torch
import torch.nn as nn
import torch.optim as optim
import os
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.DKT_regression import DKT_regression
from methods.DKT_regression_New_Loss import DKT_regression_New_Loss
from methods.Sparse_DKT_regression_Exact_new_loss import Sparse_DKT_regression_Exact_new_loss
from methods.Sparse_DKT_regression_Nystrom import Sparse_DKT_regression_Nystrom
from methods.Sparse_DKT_regression_Nystrom_new_loss import Sparse_DKT_regression_Nystrom_new_loss
from methods.Sparse_DKT_regression_Exact import Sparse_DKT_regression_Exact
from methods.Sparse_DKT_regression_RVM import Sparse_DKT_regression_RVM
from methods.MAML_regression import MAML_regression
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np
import time, json
from configs import run_float64

params = parse_args_regression('test_regression')

repeat = params.repeat
seed = params.seed
best_accuracy_list, last_accuracy_list, best_accuracy_list_rvm = list(), list(), list()
for sd in range(seed, seed+repeat):
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if run_float64: torch.set_default_dtype(torch.float64)

    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.dataset=='QMUL':
        bb           = backbone.Conv3().cuda()
    if params.method=='MAML':
        bb      = backbone.Conv3_MAML().cuda()

    if params.method=='DKT':
        id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{sd}'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        params.checkpoint_dir += id
        model = DKT_regression(bb, kernel_type=params.kernel_type, normalize=params.normalize, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
        optimizer = None

    elif params.method=='DKT_New_Loss':
        id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{sd}'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        params.checkpoint_dir += id
        model = DKT_regression_New_Loss(bb, kernel_type=params.kernel_type, normalize=params.normalize, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
        optimizer = None

    elif params.method=='Sparse_DKT_Nystrom':
        print(f'\n{params.sparse_method}\n')
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            id =  f'FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
            if params.gamma: id += '_gamma'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.rvm_ll_one:  id += f'_rvm_ll_one_{params.lambda_rvm}'
            if params.penalty: id += f'_penalty_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.beta: id += f'_beta'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, 
                                add_rvm_mll_one=params.rvm_mll_one, add_rvm_ll_one=params.rvm_ll_one, add_rvm_mse=params.rvm_mse,  add_penalty=params.penalty,
                                lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, beta=params.beta,
                                normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        

        elif params.sparse_method=='KMeans':

            params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
            # print(params.checkpoint_dir)
            model = Sparse_DKT_regression_Nystrom(bb, f_rvm=False, n_inducing_points=params.n_centers, video_path=video_path, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        

        elif params.sparse_method=='random':
            params.checkpoint_dir += '/'
            id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}_seed_{sd}'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.rvm_ll_one:  id += f'_rvm_ll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.beta: id += f'_beta'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir +  id
            model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, 
                                add_rvm_mll_one=params.rvm_mll_one, add_rvm_ll_one=params.rvm_ll_one, add_rvm_mse=params.rvm_mse, 
                                lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, beta=params.beta,
                                normalize=params.normalize, f_rvm=False, random=True,  n_inducing_points=params.n_centers, video_path=params.checkpoint_dir , 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        
        else:
            ValueError('Unrecognised sparse method')

        optimizer = None

    elif params.method=='Sparse_DKT_Nystrom_new_loss':
        print(f'\n{params.sparse_method}\n')
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            id =  f'Nystrom_new_loss_FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
            if params.gamma: id += '_gamma'
            if params.normalize: id += '_norm'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Nystrom_new_loss(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                                normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()

    elif params.method=='Sparse_DKT_Exact':
        print(f'\n{params.sparse_method}\n')
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            id =  f'Exact_FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
            if params.gamma: id += '_gamma'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.beta: id += f'_beta'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Exact(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_ll=params.rvm_ll, 
                                add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, add_rvm_mse=params.rvm_mse, 
                                lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, beta=params.beta,
                                normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        
        elif params.sparse_method=='random':
            params.checkpoint_dir += '/'
            id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}_seed_{sd}'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_ll: id += f'_rvm_ll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.beta: id += f'_beta'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir +  id
            model = Sparse_DKT_regression_Exact(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, 
                                add_rvm_mll_one=params.rvm_mll_one, add_rvm_mse=params.rvm_mse, 
                                lambda_rvm=params.lambda_rvm, beta=params.beta,
                                normalize=params.normalize, f_rvm=False, random=True,  n_inducing_points=params.n_centers, video_path=params.checkpoint_dir , 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()

        optimizer = None

    elif params.method=='Sparse_DKT_Exact_new_loss':
        print(f'\n{params.sparse_method}\n')
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            id =  f'Exact_new_loss_FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
            if params.gamma: id += '_gamma'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Exact_new_loss(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm,
                                normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        


        optimizer = None

    elif params.method=='Sparse_DKT_RVM':
        print(f'\n{params.sparse_method}\n')
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            id =  f'RVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
            if params.gamma: id += '_gamma'
            if params.normalize: id += '_norm'
            if params.init: id += '_init'
            if params.lr_decay: id += '_lr_decay'
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
            if params.maxItr_rvm!=-1: id += f'_maxItr_rvm_{params.maxItr_rvm}'
            if params.rvm_mll_only: id += f'_rvm_mll_only'
            if params.rvm_ll_only: id += f'_rvm_ll_only'
            if params.sparse_kernel: id += f'_sparse_kernel' 
            if params.beta: id += f'_beta'
            if params.beta_trajectory: id += f'_beta_trajectory'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_RVM(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, 
                                add_rvm_mll_one=params.rvm_mll_one, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, rvm_mll_only=params.rvm_mll_only, 
                                rvm_ll_only=params.rvm_ll_only, 
                                sparse_kernel=params.sparse_kernel, maxItr_rvm=params.maxItr_rvm, beta=params.beta, beta_trajectory=params.beta_trajectory,
                                normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        


        optimizer = None


    if params.method=='MAML':
        id = f'_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}_seed_{sd}'
        if params.lr_decay: id += '_lr_decay'
        params.checkpoint_dir += id
        model = MAML_regression(bb, inner_loop=params.inner_loop, inner_lr=params.inner_lr, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
        optimizer = optim.Adam([{'params':model.parameters(),'lr':params.lr_net}])

    elif params.method=='transfer':
        id = f'_{params.lr_net}_seed_{sd}'
        if params.lr_decay: id += '_lr_decay'
        params.checkpoint_dir += id
        model = FeatureTransfer(bb, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
                                
        optimizer = optim.Adam([{'params':model.parameters(),'lr':params.lr_net}])

    else:
        ValueError('Unrecognised method')

    # log test result
    if params.save_result:
        info_path = params.checkpoint_dir
        info_path = info_path.replace('//', '/')
        info_path = info_path.replace('\\', '/')
        info = info_path.split('/')
        info = '_'.join(info[3:])
        result_path = f'./record/{params.dataset}/seed_{sd}'
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

    mse_list, mse_list_best = None, None
    print(f'\n{params.checkpoint_dir}')
  
    if params.save_iter!=-1:
        if os.path.isfile(params.checkpoint_dir+f'_{params.save_iter}'):
            model.load_checkpoint(params.checkpoint_dir+f'_{params.save_iter}')
            print(f'\nMoldel at epoch {params.save_iter}\n{params.checkpoint_dir}_{params.save_iter}')        
            if params.method=='transfer':
                mse_list_, result = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
            else:
                mse_list_, result = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
            print("-------------------")
            print(f"Average MSE model at epoch {params.save_iter}, seed {sd}: " + str(np.mean(mse_list_)) + " +- " + str(np.std(mse_list_)))
            print("-------------------")
        
    if os.path.isfile(params.checkpoint_dir+'_best_model.tar'):
        print(f'\nBest model\n{params.checkpoint_dir}_best_model.tar')
        model.load_checkpoint(params.checkpoint_dir +'_best_model.tar')
        if params.method=='transfer':
            mse_list_best, result = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
        else:
            mse_list_best, result = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
        
        if params.save_result:
            f.write('{\n"time": ')
            f.write(f'"{timestamp}",\n')
            f.write('"best model":\n')
            json.dump(result, f, indent=2) #f.write(json.dumps(result))
            f.write(',\n')

        print("-------------------")
        print(f"Average MSE best model, seed {sd}: " + str(np.mean(mse_list_best)) + " +- " + str(np.std(mse_list_best)))
        print("-------------------")
        best_accuracy_list.append(np.mean(mse_list_best))
    if os.path.isfile(params.checkpoint_dir+'_best_model_rvm.tar'):
        print(f'\nBest RVM model\n{params.checkpoint_dir}_best_model_rvm.tar')
        model.load_checkpoint(params.checkpoint_dir +'_best_model_rvm.tar')
        if params.method=='transfer':
            mse_list_best_rvm, result = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
        else:
            mse_list_best_rvm, result = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
        
        if params.save_result:
            f.write('"best rvm model":\n')
            json.dump(result, f, indent=2) #f.write(json.dumps(result))
            f.write(',\n')

        print("-------------------")
        print(f"Average GP MSE at RVM best model, seed {sd}: " + str(np.mean(mse_list_best_rvm)) + " +- " + str(np.std(mse_list_best_rvm)))
        print("-------------------")
        best_accuracy_list_rvm.append(np.mean(mse_list_best_rvm))
    if os.path.isfile(params.checkpoint_dir):
        model.load_checkpoint(params.checkpoint_dir)

        if params.method=='transfer':
            mse_list, result = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
        else:
            mse_list, result = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)

        print("-------------------")
        print(f"Average MSE, seed {sd}: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
        print("-------------------")
        last_accuracy_list.append(np.mean(mse_list))

        if params.save_result:
            f.write('"last model":\n')
            json.dump(result, f, indent=2)
            f.write('\n}\n]')

    if params.save_result: f.close()
    if mse_list is not None and mse_list_best is not None:
        print(f'\n{id}\n')
        print("-------------------")
        print("Average MSE best model: " + str(np.mean(mse_list_best)) + " +- " + str(np.std(mse_list_best)))
        print("Average MSE last model: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
        print("-------------------")



if len(best_accuracy_list) >0 and len(last_accuracy_list) >0:
    print("===================")
    print(f"Overall Test Acc [best model] [repeat {repeat}]: " + str(np.mean(best_accuracy_list)) + " +- " + str(np.std(best_accuracy_list)))
    if len(best_accuracy_list_rvm) > 0:
        print(f"Overall Test Acc [GP at best RVM model] [repeat {repeat}]: " + str(np.mean(best_accuracy_list_rvm)) + " +- " + str(np.std(best_accuracy_list_rvm)))
    print(f"Overall Test Acc [last model] [repeat {repeat}]: " + str(np.mean(last_accuracy_list)) + " +- " + str(np.std(last_accuracy_list)))
    print("===================")

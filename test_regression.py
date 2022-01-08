
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
from methods.MAML_regression import MAML_regression
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np

params = parse_args_regression('test_regression')

repeat = params.repeat
seed = params.seed
accuracy_list = list()
for sd in range(seed, seed+repeat):
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.dataset=='QMUL':
        bb           = backbone.Conv3().cuda()
    if params.method=='MAML':
        bb      = backbone.Conv3_MAML().cuda()

    if params.method=='DKT':
        id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{sd}'
        if params.normalize: id += '_norm'
        params.checkpoint_dir += id
        model = DKT_regression(bb, kernel_type=params.kernel_type, normalize=params.normalize, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
        optimizer = None

    elif params.method=='DKT_New_Loss':
        id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{sd}'
        params.checkpoint_dir += id
        model = DKT_regression_New_Loss(bb, kernel_type=params.kernel_type, video_path=params.checkpoint_dir, 
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
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
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
            if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
            if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
            params.checkpoint_dir = params.checkpoint_dir +  id
            model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
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
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Nystrom_new_loss(bb, kernel_type=params.kernel_type, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
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
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Exact(bb, kernel_type=params.kernel_type, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
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
            id += f'_{params.kernel_type}_seed_{sd}'
            params.checkpoint_dir = params.checkpoint_dir + id
            model = Sparse_DKT_regression_Exact_new_loss(bb, kernel_type=params.kernel_type, normalize=params.normalize, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
        


        optimizer = None


    if params.method=='MAML':
        id = f'_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}_seed_{sd}'
        params.checkpoint_dir += id
        model = MAML_regression(bb, inner_loop=params.inner_loop, inner_lr=params.inner_lr, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
        optimizer = optim.Adam([{'params':model.parameters(),'lr':params.lr_net}])

    elif params.method=='transfer':
        id = f'_{params.lr_net}_seed_{sd}'
        params.checkpoint_dir += id
        model = FeatureTransfer(bb, video_path=params.checkpoint_dir, 
                                show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
                                
        optimizer = optim.Adam([{'params':model.parameters(),'lr':params.lr_net}])

    else:
        ValueError('Unrecognised method')

    print(f'\n{params.checkpoint_dir}')
    if os.path.isfile(params.checkpoint_dir+'_best_model.tar'):
        print(f'\nBest model\n{params.checkpoint_dir}_best_model.tar')
        model.load_checkpoint(params.checkpoint_dir +'_best_model.tar')
        if params.method=='transfer':
            mse_list_best = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
        else:
            mse_list_best = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
        print("-------------------")
        print(f"Average MSE, seed {sd}: " + str(np.mean(mse_list_best)) + " +- " + str(np.std(mse_list_best)))
        print("-------------------")
        accuracy_list.append(np.mean(mse_list_best))
    if True:
        model.load_checkpoint(params.checkpoint_dir)

        if params.method=='transfer':
            mse_list = model.test(params.n_support, params.n_samples, optimizer, params.fine_tune, params.n_test_epochs)
        else:
            mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)

        print("-------------------")
        print(f"Average MSE, seed {sd}: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
        print("-------------------")

    print(f'\n{id}\n')
    print("-------------------")
    print("Average MSE best model: " + str(np.mean(mse_list_best)) + " +- " + str(np.std(mse_list_best)))
    print("-------------------")

    print("-------------------")
    print("Average MSE last model: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")

print("===================")
print(f"Overall Test Acc [repeat {repeat}]: " + str(np.mean(accuracy_list)) + " +- " + str(np.std(accuracy_list)))
print("===================")

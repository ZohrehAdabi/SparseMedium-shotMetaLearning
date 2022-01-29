import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_regression_Nystrom import Sparse_DKT_regression_Nystrom
from methods.Sparse_DKT_sine_regression_Nystrom import Sparse_DKT_sine_regression_Nystrom
from methods.Sparse_DKT_regression_Nystrom_new_loss import Sparse_DKT_regression_Nystrom_new_loss
from methods.Sparse_DKT_regression_Exact import Sparse_DKT_regression_Exact
from methods.Sparse_DKT_regression_Exact_new_loss import Sparse_DKT_regression_Exact_new_loss
from methods.Sparse_DKT_regression_RVM import Sparse_DKT_regression_RVM
from methods.DKT_regression import DKT_regression
from methods.DKT_regression_New_Loss import DKT_regression_New_Loss
from methods.MAML_regression import     MAML_regression
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import os
import numpy as np
from configs import run_float64
 

params = parse_args_regression('train_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if run_float64: torch.set_default_dtype(torch.float64)

params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

if params.dataset=='QMUL':
    bb           = backbone.Conv3().cuda()
    if params.method=='MAML':
      
        bb           = backbone.Conv3_MAML().cuda()
if params.dataset=='Sine':
    bb           = backbone.SimpleRegressor().cuda()

if params.method=='DKT':
    id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{params.seed}'
    if params.normalize: id += '_norm'
    if params.init: id += '_init'
    if params.lr_decay: id += '_lr_decay'
    params.checkpoint_dir += id
    model = DKT_regression(bb, kernel_type=params.kernel_type, normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    model.init_summary(id=f'DKT_org_{id}')

elif params.method=='DKT_New_Loss':
    id = f'_{params.lr_gp}_{params.lr_net}_{params.kernel_type}_seed_{params.seed}'
    if params.normalize: id += '_norm'
    if params.init: id += '_init'
    if params.lr_decay: id += '_lr_decay'
    params.checkpoint_dir += id
    model = DKT_regression_New_Loss(bb, kernel_type=params.kernel_type, normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    model.init_summary(id=f'DKT_new_loss_{id}')

elif params.method=='Sparse_DKT_Nystrom':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
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
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, 
                            add_rvm_mll_one=params.rvm_mll_one, add_rvm_ll_one=params.rvm_ll_one, add_rvm_mse=params.rvm_mse, add_penalty=params.penalty, 
                            lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, beta=params.beta,
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)


    elif params.sparse_method=='KMeans':
        
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        
        model = Sparse_DKT_regression_Nystrom(bb, f_rvm=False, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    
    elif params.sparse_method=='random':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir +  id
        model = Sparse_DKT_regression_Nystrom(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, 
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=False, random=True,  
                            n_inducing_points=params.n_centers, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)
                            
    else:
       raise  ValueError('Unrecognised sparse method')

elif params.method=='Sparse_DKT_Nystrom_new_loss':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        id =  f'Nystrom_new_loss_FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
        if params.gamma: id += '_gamma'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_regression_Nystrom_new_loss(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)

elif params.method=='Sparse_DKT_Exact':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
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
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_regression_Exact(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_ll=params.rvm_ll, 
                            add_rvm_mll_one=params.rvm_mll_one, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, maxItr_rvm=params.maxItr_rvm, beta=params.beta,
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=True, config=params.config, 
                            align_threshold=params.align_thr, gamma=params.gamma,
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)
    elif params.sparse_method=='random':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir +  id
        model = Sparse_DKT_regression_Exact(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, 
                            normalize=params.normalize, initialize=params.init,  lr_decay=params.lr_decay, f_rvm=False, random=True,  n_inducing_points=params.n_centers, 
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)

elif params.method=='Sparse_DKT_Exact_new_loss':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        id =  f'Exact_new_loss_FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
        if params.gamma: id += '_gamma'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_regression_Exact_new_loss(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                            normalize=params.normalize, initialize=params.init,  lr_decay=params.lr_decay,f_rvm=True, config=params.config, align_threshold=params.align_thr, 
                            gamma=params.gamma, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)

elif params.method=='Sparse_DKT_Sine_Nystrom':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        id =  f'FRVM_{params.config}_{params.align_thr}_{params.lr_gp}_{params.lr_net}'
        if params.gamma: id += '_gamma'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_sine_regression_Nystrom(bb, kernel_type=params.kernel_type, add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=True, config=params.config, align_threshold=params.align_thr, gamma=params.gamma,
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)

    elif params.sparse_method=='random':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}_seed_{params.seed}'
        if params.normalize: id += '_norm'
        if params.lr_decay: id += '_lr_decay'
        if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
        params.checkpoint_dir = params.checkpoint_dir +  id
        model = Sparse_DKT_sine_regression_Nystrom(bb, kernel_type=params.kernel_type,  add_rvm_mll=params.rvm_mll, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                            normalize=params.normalize, lr_decay=params.lr_decay, f_rvm=False, random=True,  n_inducing_points=params.n_centers, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)
                            
    else:
       raise  ValueError('Unrecognised sparse method')

elif params.method=='Sparse_DKT_RVM':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
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
        id += f'_{params.kernel_type}_seed_{params.seed}'
        params.checkpoint_dir = params.checkpoint_dir + id

        model = Sparse_DKT_regression_RVM(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, 
                            add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, 
                            add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, rvm_mll_only=params.rvm_mll_only, 
                            rvm_ll_only=params.rvm_ll_only, sparse_kernel=params.sparse_kernel, 
                            maxItr_rvm=params.maxItr_rvm, beta=params.beta, beta_trajectory=params.beta_trajectory,
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=True, config=params.config, 
                            align_threshold=params.align_thr, gamma=params.gamma,
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)



    elif params.sparse_method=='random':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        id = f'random_{params.lr_gp}_{params.lr_net}_ip_{params.n_centers}_seed_{params.seed}'
        if params.normalize: id += '_norm'
        if params.init: id += '_init'
        if params.lr_decay: id += '_lr_decay'
        if params.rvm_mll: id += f'_rvm_mll_{params.lambda_rvm}'
        if params.rvm_mll_one: id += f'_rvm_mll_one_{params.lambda_rvm}'
        if params.rvm_mse: id += f'_rvm_mse_{params.lambda_rvm}'
        params.checkpoint_dir = params.checkpoint_dir +  id
        model = Sparse_DKT_regression_RVM(bb, kernel_type=params.kernel_type, sparse_method=params.sparse_method, add_rvm_mll=params.rvm_mll, add_rvm_mll_one=params.rvm_mll_one, add_rvm_mse=params.rvm_mse, lambda_rvm=params.lambda_rvm, 
                            normalize=params.normalize, initialize=params.init, lr_decay=params.lr_decay, f_rvm=False, random=True,  
                            n_inducing_points=params.n_centers, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        model.init_summary(id=id)
                            
    else:
       raise  ValueError('Unrecognised sparse method')


elif params.method=='MAML':
    id = f'_{params.lr_net}_loop_{params.inner_loop}_inner_lr_{params.inner_lr}_seed_{params.seed}'
    if params.lr_decay: id += '_lr_decay'
    params.checkpoint_dir += id
    model = MAML_regression(bb, inner_loop=params.inner_loop, inner_lr=params.inner_lr, lr_decay=params.lr_decay, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    model.init_summary(id=f'MAML{id}')

elif params.method=='transfer':
    id = f'_{params.lr_net}_seed_{params.seed}'
    if params.lr_decay: id += '_lr_decay'
    params.checkpoint_dir += id
    model = FeatureTransfer(bb, lr_decay=params.lr_decay, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()

else:
    ValueError('Unrecognised method')


if params.method in ['DKT', 'DKT_New_Loss', 'Sparse_DKT_Nystrom', 'Sparse_DKT_Nystrom_new_loss', 'Sparse_DKT_Exact', 'Sparse_DKT_Exact_new_loss']:

    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': params.lr_gp}, #0.01
                              {'params': model.feature_extractor.parameters(), 'lr': params.lr_net} #0.001
                              ])
    print(f'\n{id}\n')
    mll, _ = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer, params.save_model)

    print(Fore.GREEN,"-"*40, f'\nend of meta-train => MLL: {mll}\n', "-"*40, Fore.RESET)
    print(f'\n{id}\n')
else:
    optimizer = optim.Adam([{'params':model.parameters(),'lr':params.lr_net}])
    mse, _ = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer, params.save_model)

    print(Fore.GREEN,"="*40, f'\nend of meta-train => MSE: {mse}\n', "="*40, Fore.RESET)
    print(f'\n{id}\n')

model.save_checkpoint(params.checkpoint_dir)



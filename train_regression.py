import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_regression import Sparse_DKT_regression
from methods.DKT_regression import DKT
from methods.DKT_regression_New_Loss import DKT_New_Loss
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import os
import numpy as np

 

params = parse_args_regression('train_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

bb           = backbone.Conv3().cuda()

if params.method=='DKT':
    model = DKT(bb, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
elif params.method=='DKT_New_Loss':
    model = DKT_New_Loss(bb, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()

elif params.method=='Sparse_DKT':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    
    
    if params.sparse_method=='KMeans':
        
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        k_means = True
        model = Sparse_DKT_regression(bb, k_means=k_means, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    elif params.sparse_method=='FRVM':

        k_means = False
        model = Sparse_DKT_regression(bb, k_means=k_means, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    elif params.sparse_method=='random':
        k_means = False
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'
        model = Sparse_DKT_regression(bb, k_means=k_means, random=True,  n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    else:
        ValueError('Unrecognised sparse method')

elif params.method=='transfer':
    model = FeatureTransfer(bb, video_path=params.checkpoint_dir, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
else:
    ValueError('Unrecognised method')

optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                              {'params': model.feature_extractor.parameters(), 'lr': 0.001}
                              ])
if params.method=='DKT' or params.method=='Sparse_DKT' or params.method=='DKT_New_Loss':

    mll, _ = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer)

    print(Fore.GREEN,"-"*40, f'\nend of meta-train => MLL: {mll}\n', "-"*40, Fore.RESET)

else:

    mse = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer)

    print(Fore.GREEN,"="*40, f'\nend of meta-train => MSE: {mse}\n', "="*40, Fore.RESET)

model.save_checkpoint(params.checkpoint_dir)


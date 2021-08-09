import torch
import torch.nn as nn
import torch.optim as optim
import os
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.DKT_regression import DKT
from methods.DKT_sparse_regression import Sparse_DKT
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
bb           = backbone.Conv3().cuda()

if params.method=='DKT':
    model = DKT(bb, video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
    optimizer = None
elif params.method=='Sparse_DKT':

    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

    video_path = params.checkpoint_dir
    
    if params.sparse_method=='KMeans':
        
        k_means = True
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        model = Sparse_DKT(bb, k_means=k_means, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    elif params.sparse_method=='FRVM':
        
        k_means = False
        model = Sparse_DKT(bb, k_means=k_means, n_inducing_points=None, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    else:
        pass #ranndom
    model = Sparse_DKT(bb, k_means=k_means, video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
    optimizer = None
elif params.method=='transfer':
    model = FeatureTransfer(bb).cuda()
    optimizer = optim.Adam([{'params':model.parameters(),'lr':0.001}])
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

 
mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")


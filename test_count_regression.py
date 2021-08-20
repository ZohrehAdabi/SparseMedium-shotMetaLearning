import torch
import torch.nn as nn
import torch.optim as optim
import os
import configs
# from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_count_regression import Sparse_DKT_count_regression
from methods.DKT_count_regression import DKT_count_regression
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)


if  params.dataset=='MSC44':
    resnet50_conv, regressor = backbone.ResNet_Regrs()
    novel_file = configs.data_dir[params.dataset] + 'test.json'
else:
    ValueError('Unknown dataset')
if params.method=='DKT':
    model = DKT_count_regression(resnet50_conv, regressor, val_file=novel_file, 
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
    optimizer = None

elif params.method=='Sparse_DKT':
    print(f'\n{params.sparse_method}\n')
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

    video_path = params.checkpoint_dir
    
    if params.sparse_method=='KMeans':

        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        # print(params.checkpoint_dir)
        model = Sparse_DKT_count_regression(resnet50_conv, regressor, val_file=novel_file, 
                            f_rvm=False, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    
    
    elif params.sparse_method=='FRVM':
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'FRVM_{params.config}_{params.align_thr:.6f}'

        model = Sparse_DKT_count_regression(resnet50_conv, regressor, val_file=novel_file, 
                            f_rvm=True, config=params.config, align_threshold=params.align_thr, 
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    
    elif params.sparse_method=='random':

        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'
        model = Sparse_DKT_count_regression(resnet50_conv, regressor, val_file=novel_file, 
                            f_rvm=False, random=True,  n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    else:
        ValueError('Unrecognised sparse method')

    optimizer = None
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

 
mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")

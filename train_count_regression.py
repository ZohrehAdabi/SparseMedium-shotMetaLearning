import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
import configs
# from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_count_regression import Sparse_DKT_count_regression
from methods.DKT_count_regression import DKT_count_regression
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
feat_map = 'map4'
if  params.model=='ResNet50':
    resnet50_conv, regressor = backbone.ResNet_Regrs(feat_map)
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file =  configs.data_dir[params.dataset] + 'val.json'
else:
    ValueError('Unknown model')

if params.method=='DKT':
    model = DKT_count_regression(resnet50_conv, regressor, base_file, val_file,
                            video_path=params.checkpoint_dir, 
                            show_plots_loss=params.show_plots_loss,
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()

elif params.method=='Sparse_DKT':
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir
    params.checkpoint_dir += '/'
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
          
    if params.sparse_method=='FRVM':
        print(f'params.show_plots_loss {params.show_plots_loss}')
        params.checkpoint_dir = params.checkpoint_dir +  f'FRVM_{params.config}_{params.align_thr:.6f}'

        model = Sparse_DKT_count_regression(resnet50_conv, regressor, base_file, val_file,
                            sparse_method = 'FRVM', config=params.config, align_threshold=params.align_thr, 
                            video_path=params.checkpoint_dir, 
                            show_plots_loss=params.show_plots_loss, show_plots_pred=False, 
                            show_plots_features=params.show_plots_features, training=True).cuda()
    
    elif params.sparse_method=='KMeans':
        
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        
        model = Sparse_DKT_count_regression(resnet50_conv, regressor, base_file, val_file, 
                        sparse_method = 'KMeans', n_inducing_points=params.n_centers, video_path=params.checkpoint_dir, 
                        show_plots_loss=params.show_plots_loss, show_plots_pred=False, 
                        show_plots_features=params.show_plots_features, training=True).cuda()
                            
    elif params.sparse_method=='random':
        
        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'
        model = Sparse_DKT_count_regression(resnet50_conv, regressor, base_file, val_file,
                        sparse_method = 'random', n_inducing_points=params.n_centers, video_path=params.checkpoint_dir, 
                        show_plots_loss=params.show_plots_loss, show_plots_pred=False, 
                        show_plots_features=params.show_plots_features, training=True).cuda()
    else:
        ValueError('Unrecognised sparse method')

else:
    ValueError('Unrecognised method')
lr_gp  = 1e-4
lr_reg = 1e-4
mse = False
# mse = True
id = f'g_{lr_gp}_r_{lr_reg}_feat_{feat_map}'
if mse: id = f'g_{lr_gp}_r_{lr_reg}_feat_{feat_map}_mse'
optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr':lr_gp},
                              {'params': model.regressor.parameters(), 'lr': lr_reg}
                              ])
model.init_summary(id)
if params.method=='DKT' or params.method=='Sparse_DKT' :

    mll, mll_list = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer, id=id, use_mse=mse)
    print(f'Avg. MLL hist: {mll_list}')
    print(Fore.GREEN,"-"*40, f'\nend of meta-train => MLL: {mll}\n', "-"*40, Fore.RESET)


model.save_checkpoint(params.checkpoint_dir)


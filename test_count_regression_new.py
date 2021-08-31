import torch
import torch.nn as nn
import torch.optim as optim
import os
import configs
# from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_count_regression import Sparse_DKT_count_regression
from methods.DKT_count_regression_new import DKT_count_regression_new
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np

params = parse_args_regression('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

selective = True
best_models_list = os.listdir(f'./save/checkpoints/{params.dataset}')
if len(best_models_list) > 0 and not selective:
    best_mae = [best.split('_')[3] for best in best_models_list if 'mae' in best]
    best_mae_idx = np.argmin(best_mae)
    best_models_list = [best for best in best_models_list if 'mae' in best]
    best_model = best_models_list[best_mae_idx]


if  params.model=='Conv6':
    Conv6 = backbone.Conv6(flatten=False)
    novel_file = configs.data_dir[params.dataset] + 'novel.json'
else:
    ValueError('Unknown model')
if params.method=='DKT':
    model = DKT_count_regression_new(Conv6, val_file=novel_file, 
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
    optimizer = None

elif params.method=='Sparse_DKT':
    print(f'\n{params.sparse_method}\n')
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

    video_path = params.checkpoint_dir
    params.checkpoint_dir += '/'
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir) 
    
    if params.sparse_method=='FRVM':

        params.checkpoint_dir = params.checkpoint_dir +  f'FRVM_{params.config}_{params.align_thr:.6f}'

        model = Sparse_DKT_count_regression(Conv6, val_file=novel_file, 
                            sparse_method = 'FRVM', config=params.config, align_threshold=params.align_thr, 
                            video_path=params.checkpoint_dir, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    
    elif params.sparse_method=='KMeans':

        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        # print(params.checkpoint_dir)
        model = Sparse_DKT_count_regression(Conv6, val_file=novel_file, 
                            sparse_method = 'KMeans', n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()

    elif params.sparse_method=='random':

        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'
        model = Sparse_DKT_count_regression(Conv6, val_file=novel_file, 
                            sparse_method = 'random',  n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    else:
        ValueError('Unrecognised sparse method')

    optimizer = None
else:
    ValueError('Unrecognised method')

if selective:
    lr_gp  = 1e-3
    lr_reg = 1e-5
    mse = False
    # mse = True
    #'ResNet50_DKT_best_mae37.65_ep440_g_0.001_r_1e-05_feat_map4.'
    id = f'_best_mae{19.85}_ep{240}_g_{lr_gp}_r_{lr_reg}_feat_{feat_map}'
    id = f'_final_mae{28.53}_ep{99}_g_{lr_gp}_r_{lr_reg}_feat_{feat_map}'
    id = f'_final_mae{26.63:.2f}_ep{99}_g_{lr_gp}_r_{lr_reg}_feat_{feat_map}'
    # id = f'_final_mae{27.87:.2f}_ep{49}_g_{lr_gp}_r_{lr_reg}_feat_{feat_map}'
    if mse: id = id + '_mse'
    id = id + '.pth'
    model_path = params.checkpoint_dir + id
    print(f'\n{model_path}')
    model.load_checkpoint(model_path)
else:
    if len(best_models_list) > 0:
        
        model_path = os.path.join(configs.save_dir, 'checkpoints', params.dataset, best_model)
        print(f'\n{model_path}')
        model.load_checkpoint(model_path)
    else:
        model.load_checkpoint(params.checkpoint_dir)
 
mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")


import torch
import torch.nn as nn
import torch.optim as optim
import os
import configs
# from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_count_regression import Sparse_DKT_count_regression
from methods.density_regression import Density_regression
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
feat_map = 'map4'
best_models_list = os.listdir(f'./save/checkpoints/{params.dataset}')
if len(best_models_list) > 0 and not selective:
    best_mae = [best.split('_')[3] for best in best_models_list if 'mae' in best]
    best_mae_idx = np.argmin(best_mae)
    best_models_list = [best for best in best_models_list if 'mae' in best]
    best_model = best_models_list[best_mae_idx]

if  params.model=='ResNet50':
    resnet50_conv, regressor = backbone.ResNet_Regrs(feat_map)
    novel_file = configs.data_dir[params.dataset] + 'novel.json'
else:
    ValueError('Unknown model')

model = Density_regression(resnet50_conv, regressor, val_file=novel_file, 
                        video_path=params.checkpoint_dir, 
                        show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features).cuda()
optimizer = None

if selective:
    lr_reg = params.lr_net
    # mse = True
    #'ResNet50_DKT_best_mae37.65_ep440_g_0.001_r_1e-05_feat_map4.'
    id = f'_best_mae{19.85}_ep{240}_r_{lr_reg}_feat_{feat_map}'
    id = f'_ep{1000}_r_{lr_reg}_feat_{feat_map}'
    # id = f'_final_mae{29.65:.2f}_ep{99}_r_{lr_reg}_feat_{feat_map}'
    # id = f'_final_mae{51.31:.2f}_ep{1499}_r_{lr_reg}_feat_{feat_map}'
    id = id + '_mse'
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


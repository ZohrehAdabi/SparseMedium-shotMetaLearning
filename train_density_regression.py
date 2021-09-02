import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
import configs
# from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_count_regression import Sparse_DKT_count_regression
from methods.density_regression import Density_regression
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

model = Density_regression(resnet50_conv, regressor, base_file, val_file,
                        video_path=params.checkpoint_dir, 
                        show_plots_loss=params.show_plots_loss,
                        show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()



lr_reg = params.lr_net
id = f'r_{lr_reg}_feat_{feat_map}_mse_alpha_{params.alpha}'
optimizer = torch.optim.Adam([{'params': model.regressor.parameters(), 'lr': lr_reg}
                              ])
model.init_summary(id)

mll, mll_list = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer, params.alpha, id=id, use_mse=True)
print(f'Avg. MLL hist: {mll_list}')
print(Fore.GREEN,"-"*40, f'\nend of meta-train => MLL: {mll}\n', "-"*40, Fore.RESET)


model.save_checkpoint(params.checkpoint_dir)


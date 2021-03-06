 



import torch
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_regression import Sparse_DKT_regression
import backbone
import os
import numpy as np
from colorama import Fore
import matplotlib.pyplot as plt

fig_loss: plt.Figure = plt.figure(3, figsize=(16, 8), tight_layout=True, dpi=125)
ax_mll: plt.Axes = fig_loss.add_subplot(2, 1, 1)
ax_mse: plt.Axes = fig_loss.add_subplot(2, 1, 2)
fig_mll_per_num_ip: plt.Figure = plt.figure(4, figsize=(8, 4), tight_layout=True, dpi=150)
ax_mll_per_num_ip: plt.Axes = fig_mll_per_num_ip.add_subplot(1, 1, 1)

mll_hist = []
mse_hist = []
n_centers = np.arange(6, 62, 4) 
for i, n_center in enumerate(n_centers):
    
    params = parse_args_regression('train_regression')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params.method = 'Sparse_DKT'
    params.sparse_method='random'
    params.n_centers = n_center
    params.show_plots_features = True
    print(Fore.CYAN, f'num Inducing points: {params.n_centers}', Fore.RESET)
    params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

    bb           = backbone.Conv3().cuda()


    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir

    print(f'{params.sparse_method}')
    if params.sparse_method=='random':
        
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'

        model = Sparse_DKT_regression(bb, f_rvm=False, random=True, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    elif params.sparse_method=='FRVM':

        model = Sparse_DKT_regression(bb, f_rvm=True, config=params.config, align_threshold=params.align_thr, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    else: ValueError()

    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}
                                ])

    mll, mll_list = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer)
 
    print(Fore.GREEN,"="*40, f'\nend of meta-train  => MLL: {mll}\n', "="*40, Fore.RESET)

    mll_hist.append(mll)

    model.save_checkpoint(params.checkpoint_dir)



    params = parse_args_regression('test_regression')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    params.method = 'Sparse_DKT'
    params.sparse_method='random'
    params.n_centers = n_center
    params.show_plots_features = True

    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    bb           = backbone.Conv3().cuda()


    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

    video_path = params.checkpoint_dir
    print(f'{params.sparse_method}')
    if params.sparse_method=='random':
     
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'random_{str(params.n_centers)}'
        # print(params.checkpoint_dir)
        model = Sparse_DKT_regression(bb, f_rvm=False, random=True, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=True, show_plots_features=params.show_plots_features, training=False).cuda()
    elif params.sparse_method=='FRVM':
        
        model = Sparse_DKT_regression(bb, f_rvm=True, config=params.config, align_threshold=params.align_thr, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    else:
        ValueError('Unrecognised sparse method')

    optimizer = None


    model.load_checkpoint(params.checkpoint_dir)

    
    mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
    mse = np.mean(mse_list)
    print("------------------- ", n_center)
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")

    mse_hist.append(mse)

    ax_mll.clear()
    ax_mse.clear()
    ax_mll.plot(n_centers[:i+1], mll_hist, label='Meta-Train MLL')
    ax_mse.plot(n_centers[:i+1], mse_hist, label='Meta-Test MSE')
    ax_mll.legend()
    ax_mse.legend()
    ax_mll.set_xlabel("number of Inducing points")
    ax_mll.set_ylabel("loss")
    ax_mse.set_ylabel("loss")
    ax_mll.set_title("Sparse DKT with random selection")
    fig_loss.tight_layout()
    fig_loss.savefig(video_path+'/loss.png')

    
    ax_mll_per_num_ip.plot(mll_list, label=f'num IP= {n_center}')
    ax_mll_per_num_ip.set_xlabel("number of epochs")
    ax_mll_per_num_ip.set_ylabel("loss")
    ax_mll_per_num_ip.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=2)
    fig_mll_per_num_ip.tight_layout()
    ax_mll_per_num_ip.set_title("Sparse DKT with random selection (Meta-Train MLL)")
    fig_mll_per_num_ip.savefig(video_path+'/mll_per_num_ip.png')


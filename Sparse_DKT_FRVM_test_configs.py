 



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
fig_mll_per_config: plt.Figure = plt.figure(4, figsize=(8, 4), tight_layout=True, dpi=150)
ax_mll_per_config: plt.Axes = fig_mll_per_config.add_subplot(1, 1, 1)

mll_list_per_config = []
mll_hist = []
mse_hist = []
align_threshold = [1e-2, 1e-3, 1e-4]
                            # update_sugma, del, add, alig_test
              #'0010', '1000', '1010', '1011','1100', '1101'
config_frvm = [  2,       8,      10,    11,    12,     13] 
for align_thr in align_threshold:
    best_mse = 10e7
    for idx, i in enumerate(config_frvm):
        
        params = parse_args_regression('train_regression')
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        params.method = 'Sparse_DKT'
        params.sparse_method='FRVM'
        params.config = f'{i:04b}'
        params.align_thr = align_thr
        params.show_plots_features = True
        params.n_samples = 72
        params.dataset = 'QMUL'

        # print(Fore.CYAN, f'num Inducing points: {params.n_centers}', Fore.RESET)
        params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

        bb           = backbone.Conv3().cuda()


        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                            params.sparse_method)
        video_path = params.checkpoint_dir

        print(f'{params.sparse_method}')
        if params.sparse_method=='KMeans':
            
            params.checkpoint_dir += '/'
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
            model = Sparse_DKT_regression(bb, f_rvm=False, n_inducing_points=params.n_centers, video_path=video_path, 
                                show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()


        elif params.sparse_method=='FRVM':

            params.checkpoint_dir += '/'
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            params.checkpoint_dir = params.checkpoint_dir +  f'FRVM_{params.config}_{params.align_thr:.6f}'

            model = Sparse_DKT_regression(bb, f_rvm=True, config=params.config, align_threshold=params.align_thr, 
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
        else: 
            ValueError('Unrecognised sparse method')
        optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                    {'params': model.feature_extractor.parameters(), 'lr': 0.001}
                                    ])

        mll, mll_list = model.train(params.stop_epoch, params.n_support, params.n_samples, optimizer)
    
        print(Fore.GREEN,"="*40, f'\nend of meta-train  => MLL: {mll}\n', "="*40, Fore.RESET)

        mll_hist.append(mll)
        mll_list_per_config.append(mll_list)

        model.save_checkpoint(params.checkpoint_dir)



        params = parse_args_regression('test_regression')
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        params.method = 'Sparse_DKT'
        params.sparse_method='FRVM'
        params.config = f'{i:04b}'
        params.align_thr = align_thr
        params.show_plots_features = True
        params.n_support = 60 
        params.n_samples = 72
        params.dataset = 'QMUL'
        params.n_test_epochs= 5

        params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
        bb           = backbone.Conv3().cuda()


        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

        video_path = params.checkpoint_dir
        
        if params.sparse_method=='KMeans':
            
            params.checkpoint_dir += '/'
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
            # print(params.checkpoint_dir)
            model = Sparse_DKT_regression(bb, f_rvm=False, n_inducing_points=params.n_centers, video_path=video_path, 
                                show_plots_pred=True, show_plots_features=params.show_plots_features, training=False).cuda()


        elif params.sparse_method=='FRVM':
            params.checkpoint_dir += '/'
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            params.checkpoint_dir = params.checkpoint_dir +  f'FRVM_{params.config}_{params.align_thr:.6f}'
            model = Sparse_DKT_regression(bb, f_rvm=True, config=params.config, align_threshold=params.align_thr, 
                                video_path=params.checkpoint_dir, 
                                show_plots_pred=True, show_plots_features=params.show_plots_features, training=False).cuda()
        else:
            ValueError('Unrecognised sparse method')

        optimizer = None


        model.load_checkpoint(params.checkpoint_dir)

        
        mse_list = model.test(params.n_support, params.n_samples, optimizer, params.n_test_epochs)
        mse = np.mean(mse_list)
        print("------------------- ", params.config,' - ', params.align_thr)
        print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
        print("-------------------")

        mse_hist.append(mse)

        ax_mll.clear()
        ax_mse.clear()
        ax_mll.plot(config_frvm[:idx+1], mll_hist, marker='.', label='Meta-Train MLL')
        ax_mse.plot(config_frvm[:idx+1], mse_hist, marker='.', label='Meta-Test MSE')
        if mse < best_mse:
            best_mse = mse
            ax_mse.scatter(config_frvm[idx], best_mse,  c='r', marker='*', label=f'Best MSE: {best_mse:5f}')
        ax_mll.legend()
        ax_mse.legend()
        ax_mll.hlines(y=0.1, xmin=config_frvm[0], xmax=config_frvm[0], linestyles='dashed')
        ax_mll.hlines(y=0.5, xmin=config_frvm[0], xmax=config_frvm[-1], linestyles='dashed')
        ax_mse.hlines(y=0.01, xmin=config_frvm[0], xmax=config_frvm[-1], linestyles='dashed')
        ax_mse.hlines(y=0.009, xmin=config_frvm[0], xmax=config_frvm[-1], linestyles='dashed')
        ax_mse.set_xlabel("config id")
        ax_mll.set_ylabel("loss")
        ax_mse.set_ylabel("loss")
        ax_mll.set_title("Sparse DKT with FRVM")
        fig_loss.tight_layout()
        fig_loss.savefig(video_path+f'/loss_{align_thr}.png')

    for c in range(len(config_frvm)):
        ax_mll_per_config.plot(mll_list_per_config[c], label=f'c- {config_frvm[c]}')
    ax_mll_per_config.set_xlabel("number of epochs")
    ax_mll_per_config.set_ylabel("loss")
    ax_mll_per_config.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2)
    ax_mll_per_config.hlines(y=0.1, xmin=0, xmax=100, linestyles='dashed')
    fig_mll_per_config.tight_layout()
    ax_mll_per_config.set_title(f"Sparse DKT with KMeans (Meta-Train MLL) [config][align_thr={params.align_thr}]")
    fig_mll_per_config.savefig(video_path+f'/mll_per_config_{align_thr}.png')


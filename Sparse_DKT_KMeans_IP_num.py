 



import torch
import configs
from data.qmul_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.Sparse_DKT_regression import Sparse_DKT
import backbone
import os
import numpy as np
from colorama import Fore
import matplotlib.pyplot as plt

fig_loss: plt.Figure = plt.figure(3, figsize=(6, 3), tight_layout=True, dpi=125)
ax_mll: plt.Axes = fig_loss.add_subplot(2, 1, 1)
ax_mse: plt.Axes = fig_loss.add_subplot(2, 1, 2)
mll_hist = []
mse_hist = []
n_centers = np.arange(6, 48, 4) 
for i, n_center in enumerate(n_centers):
    
    params = parse_args_regression('train_regression')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params.method = 'Sparse_DKT'
    params.sparse_method=='KMeans'
    params.n_centers = n_center
    print(Fore.CYAN, f'num Inducing points: {params.n_centers}', Fore.RESET)
    params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

    bb           = backbone.Conv3().cuda()


    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, 
                                                        params.sparse_method)
    video_path = params.checkpoint_dir


    if params.sparse_method=='KMeans':
        
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        k_means = True
        model = Sparse_DKT(bb, k_means=k_means, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()
    else: #RVM

        k_means = False
        model = Sparse_DKT(bb, k_means=k_means, video_path=video_path, 
                            show_plots_pred=False, show_plots_features=params.show_plots_features, training=True).cuda()

    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}
                                ])

    mll_list = []
    for epoch in range(params.stop_epoch):
        
        print(f"ephoc {epoch}")
        mll = model.train(epoch, params.n_support, params.n_samples, optimizer)
        mll_list.append(mll)

        print(Fore.LIGHTYELLOW_EX,"-"*30, f'\nend of epoch {epoch} => MLL: {mll}\n', "-"*30, Fore.RESET)
    mll = np.mean(mll_list)
    print(Fore.GREEN,"="*40, f'\nend of meta-train  => MLL: {mll}\n', "="*40, Fore.RESET)

    mll_hist.append(mll)

    model.save_checkpoint(params.checkpoint_dir)



    params = parse_args_regression('test_regression')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    params.method = 'Sparse_DKT'
    params.sparse_method=='KMeans'
    params.n_centers = n_center

    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    bb           = backbone.Conv3().cuda()


    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (configs.save_dir, params.dataset, params.model, params.method, params.sparse_method)

    video_path = params.checkpoint_dir
    
    if params.sparse_method=='KMeans':
        
        k_means = True
        params.checkpoint_dir += '/'
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = params.checkpoint_dir +  f'KMeans_{str(params.n_centers)}'
        # print(params.checkpoint_dir)
        model = Sparse_DKT(bb, k_means=k_means, n_inducing_points=params.n_centers, video_path=video_path, 
                            show_plots_pred=True, show_plots_features=params.show_plots_features, training=False).cuda()
    elif params.sparse_method=='FRVM':
        
        k_means = False
        model = Sparse_DKT(bb, k_means=k_means, video_path=video_path, 
                            show_plots_pred=params.show_plots_pred, show_plots_features=params.show_plots_features, training=False).cuda()
    else:
        pass #ranndom

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
    ax_mll.set_xlabel("number of Inducing points/KMeans centers")
    ax_mll.set_ylabel("loss")
    ax_mse.set_ylabel("loss")
    ax_mll.set_title("Sparse DKT with KMeans")
    fig_loss.savefig(video_path+'/loss.png')


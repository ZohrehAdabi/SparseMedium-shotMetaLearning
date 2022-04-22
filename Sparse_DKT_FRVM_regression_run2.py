from subprocess import run

config_list = ['1000', '1001', '1010', '1011']
dataset_list = ['QMUL']
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM']
seed_list = [1, 2, 3]
lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

save_model = True
stop_epoch = 100
lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['1001']
seed_list = [1]
method_list = ['Sparse_DKT_Exact']
sparse_method = 'FRVM' # 'random'
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 1e-3
            if config in ['1000', '1010']:
                align_thr = 0
            for sd in seed_list:
              

                # random
                for method in method_list:
                
                    # just mll of GP
                    L = ['python', f'./train_regression.py', 
                                    '--method', f'{method}', '--sparse_method', f'random',  '--n_samples', '72', '--n_support', '60', '--stop_epoch', f'{stop_epoch}', 
                                #   '--show_plots_features',
                                    '--seed',  f'{sd}',  '--n_centers', '10', 
                                    '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                    '--kernel_type', 'rbf', '--init'
                    ]
                    if save_model: L.append('--save_model')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)

                # MAML
                for in_lr in [3, 5]:
                    L = ['python', f'./train_regression.py', "--method","MAML", "--n_samples", "72",  "--n_support", "60", "--stop_epoch", f'{stop_epoch}', 
                            '--seed',  f'{sd}', 
                            '--lr_net',  f'{lr_net}', "--inner_loop", f"{in_lr}", "--inner_lr", "1e-2"] 
                    if save_model: L.append('--save_model')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)
                
                # transfer learning
                L = ['python', f'./train_regression.py', "--method","transfer", "--n_samples", "72", "--n_support", "60", "--stop_epoch",  f'{stop_epoch}',  
                            '--seed',  f'{sd}', 
                            '--lr_net',  f'{lr_net}'] 
                if save_model: L.append('--save_model')
                print(f'\n{" ".join(L)} \n')
                # run(L)

                    

                      
          



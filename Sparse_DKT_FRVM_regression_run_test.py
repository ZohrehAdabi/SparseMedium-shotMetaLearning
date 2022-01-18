from subprocess import run

config_list = ['1000', '1001', '1010', '1011']
dataset_list = ['QMUL']
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM']
seed_list = [1, 2, 3]
lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

stop_epoch = 150
lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['1011']
seed_list = [1]
method_list = ['Sparse_DKT_Nystrom']
save_result = False
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 1e-3
            if config in ['1000', '1010']:
                align_thr = 0
            for method in method_list:
                for sd in seed_list:
                    lambda_rvm_list = [0.001, 0.1, 0.5, 1.0]
                    # lambda_rvm_list = [0] # for run: just mll of GP
                    for lambda_rvm in lambda_rvm_list:
                        # just mll of GP
                        # run(L)
                        L = ['python', f'./test_regression.py', 
                                        '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', '10', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--beta' 
                                        
                        ]
                        if save_result: L.append('--save_result')
                        print(f'\n{" ".join(L)} \n')
                        run(L)
                        
                        # rvm mll
                        L = ['python', f'./test_regression.py', 
                                        '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', '10', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--lambda_rvm', f'{lambda_rvm}', '--rvm_mll', '--beta'
                        ]
                        if save_result: L.append('--save_result')
                        print(f'\n{" ".join(L)} \n')
                        run(L)

                        # rvm ll
                        L = ['python', f'./test_regression.py', 
                                        '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', '10', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--lambda_rvm', f'{lambda_rvm}', '--rvm_ll', '--beta'
                        ]
                        if save_result: L.append('--save_result')
                        print(f'\n{" ".join(L)} \n')
                        run(L)

          



from subprocess import run

config_list = ['1000', '1001', '1010', '1011']
dataset_list = ['QMUL']
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact', 'Sparse_DKT_RVM']
seed_list = [1, 2, 3]
lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]


lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['1001']
seed_list = [1, 2, 3]
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact']
test_epoch = 100
save_result = True
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 1e-3
            if config in ['1000', '1010']:
                align_thr = 0
            for sd in seed_list:
                L = ['python', f'./test_regression.py', 
                                        '--method', f'DKT',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}',
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--beta' , '--init'
                                        
                        ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                # run(L)
              
                for method in method_list:
                     # just mll of GP
                       
                    L = ['python', f'./test_regression.py', 
                                    '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                                #   '--show_plots_pred',
                                    '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                    '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                    '--kernel_type', 'rbf', '--beta' , '--init'
                                    
                    ]
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)

                    lambda_rvm_list = [0.1]
                    for lambda_rvm in lambda_rvm_list:
                       
                        
                        # rvm mll
                        L = ['python', f'./test_regression.py', 
                                        '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--lambda_rvm', f'{lambda_rvm}', '--rvm_mll', '--beta', '--init'
                        ]
                        if save_result: L.append('--save_result')
                        print(f'\n{" ".join(L)} \n')
                        # run(L)

                        # rvm ll
                        L = ['python', f'./test_regression.py', 
                                        '--method', f'{method}', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                                    #   '--show_plots_pred',
                                        '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                        '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                        '--kernel_type', 'rbf', '--lambda_rvm', f'{lambda_rvm}', '--rvm_ll', '--beta' , '--init'
                        ]
                        if save_result: L.append('--save_result')
                        print(f'\n{" ".join(L)} \n')
                        # run(L)

            
              
                # rvm mll 
                L = ['python', f'./test_regression.py', 
                                '--method', f'Sparse_DKT_RVM', '--sparse_method', 'FRVM',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                            #   '--show_plots_pred',
                                '--seed',  f'{sd}', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                                '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                '--kernel_type', 'rbf', '--rvm_mll_only',  '--init'
                ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                # run(L)


                for method in method_list:
                
                    # just mll of GP + random
                    L = ['python', f'./test_regression.py', 
                                    '--method', f'{method}', '--sparse_method', f'random',  '--n_samples', '72', '--n_support', '60', '--n_test_epoch', f'{test_epoch}', 
                                #   '--show_plots_features',
                                    '--seed',  f'{sd}',  '--n_centers', '10', 
                                    '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}',
                                    '--kernel_type', 'rbf', '--init'
                    ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                # run(L)

                for in_lr in [3, 5]:
                    L = ['python', f'./test_regression.py', "--method","MAML", "--n_samples", "72",  "--n_support", "60",'--n_test_epoch', f'{test_epoch}', 
                            '--seed',  f'{sd}', 
                            '--lr_net',  f'{lr_net}', "--inner_loop", f"{in_lr}", "--inner_lr", "1e-2"] 
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    run(L)
            
                for fine_tune in [10, 16, 32]:
                    L = ['python', f'./test_regression.py', "--method","transfer", "--n_samples", "72", "--n_support", "60", '--n_test_epoch', f'{test_epoch}',   
                                '--seed',  f'{sd}', 
                                '--lr_net',  f'{lr_net}', '--fine_tune', f'{fine_tune}'] 
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)
            


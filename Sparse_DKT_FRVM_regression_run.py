from subprocess import run

config_list = ['1000', '1001', '1010', '1011']
dataset_list = ['QMUL']
method_list = ['Sparse_DKT_regression_Nystrom', 'Sparse_DKT_regression_Exact']

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

lr_gp_list = [0.1, 0.01]
lr_net_list = [0.001]
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 1e-3
            if config in ['1000', '1010']:
                align_thr = 0

            run(['python', f'./train.py', 
                         '--method', 'Sparse_DKT_Nystrom', '--sparse_method', 'FRVM',  '--n_samples', '72', '--stop_epoch', '100' 
                        #   '--show_plots_features',
                            '--seed',  '2', '--config', f'{config}', '--align_thr', f'{align_thr}',  
                          '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}'
            ])

          

# config_list = ['000', '001', '010', '011']
# lr_gp_list = [ 0.01, 0.001]
# lr_net_list = [ 0.001, 0.0001]
# for config in config_list:
#     for lr_gp in lr_gp_list:
#         for lr_net in lr_net_list:
#             align_thr = 1e-3
#             if config in ['000', '010']:
#                 align_thr = 0

            # run(['python', f'./train.py', 
            #              '--method', 'Sparse_DKT_Exact', '--sparse_method', 'FRVM',  '--n_samples', '72', '--stop_epoch', '100' 
            #               #'--show_plots_features', 
            #                '--seed',  '2', '--config', f'{config]', '--align_thr', f'{align_thr}',  
            #               '--lr_gp',  f'{lr_gp}', '--lr_net',  f'{lr_net}'
            # ])

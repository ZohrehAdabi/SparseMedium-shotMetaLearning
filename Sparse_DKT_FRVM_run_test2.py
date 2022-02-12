from subprocess import run

config_list = ['000', '001', '010', '011']
dataset_list = ['omniglot', 'CUB', 'miniImagenet']
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact'] # , 'DKT_binary'

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['001']
sd = 1
save_result = True
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 0.0015
            if config in ['000', '010']:
                align_thr = 0
            L = ['python', f'./test.py', 
                        "--method","DKT", "--dataset", "omniglot", 
                        "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                            "--seed",  f"{sd}",  
                            "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}",
                            '--kernel_type', 'linear', "--n_task", "30"
                         
            ]
            if save_result: L.append('--save_result')
            print(f'\n{" ".join(L)} \n')
            # run(L)
            
            for method in method_list:
                L = ['python', f'./test.py', 
                            "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                '--kernel_type', 'linear', "--scale",  "--n_task", "30"
                                , "--regression"
                            
                ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                run(L)
            lambda_rvm_list = [1.0]
            for lambda_rvm in lambda_rvm_list:
                for method in method_list:
                    L = ['python', f'./test.py', 
                                "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                 "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}",
                                    '--kernel_type', 'linear', "--scale", "--n_task", "30",
                                    "--regression", 
                                    "--rvm_mll", "--lambda_rvm", f"{lambda_rvm}"
                                  
                    ]
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    run(L)
                
            lambda_rvm_list = [1.0]
            for lambda_rvm in lambda_rvm_list:
                for method in method_list:
                    L = ['python', f'./test.py', 
                                "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                 "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                    '--kernel_type', 'linear', "--scale",  "--n_task", "30",
                                    "--regression", 
                                    "--rvm_ll", "--lambda_rvm", f"{lambda_rvm}"
                                
                    ]
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)

            L = ['python', f'./test.py', 
                                "--method", f"Sparse_DKT_RVM", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                 "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                    '--kernel_type', 'linear', "--scale",  "--n_task", "30",
                                    "--regression", 
                                    "--rvm_mll_only"
                                
                    ]
            if save_result: L.append('--save_result')
            print(f'\n{" ".join(L)} \n')
            run(L)
            # run(['python', f'./train.py', 
            #             "--method","Sparse_DKT_binary_Nystrom", "--sparse_method", "FRVM", "--dataset", "omniglot", 
            #             "--train_n_way", "2", "--test_n_way", "2", "--n_shot", "15", "--n_query", "5",
            #                 "--seed","1", "--config", f"{config}", "--align_thr", "1e-3" , 
            #                 "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--gamma" 
            #                 #,"--train_aug"
            # ])
# config_list = ['000', '001', '010', '011']
# lr_gp_list = [ 0.01, 0.001]
# lr_net_list = [ 0.001, 0.0001]
# for config in config_list:
#     for lr_gp in lr_gp_list:
#         for lr_net in lr_net_list:
#             align_thr = 1e-3
#             if config in ['000', '010']:
#                 align_thr = 0
            
#             run(['python', f'./train.py', 
#                         "--method","Sparse_DKT_binary_Nystrom", "--sparse_method", "FRVM", "--dataset", "CUB", 
#                         "--train_n_way", "2", "--test_n_way", "2", "--n_shot", "50", "--n_query", "10",
#                             "--seed","1", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
#                             "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}"
#                             #,"--train_aug"
#             ])
from subprocess import run

config_list = ['000', '001', '010', '011']
dataset_list = ['omniglot', 'CUB', 'miniImagenet']
method_list = ['Sparse_DKT_binary_Nystrom', 'Sparse_DKT_binary_Exact'] # , 'DKT_binary'

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]
dataset =  'CUB' #'miniImagenet' # 'CUB
lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['001']
n_task = 20
n_shot = 150 # 50
n_query = 25 # 10

tol_rvm = 1e-4
max_itr = -1
sd = 1
save_result = True
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 0.03
            if config in ['000', '010']:
                align_thr = 0
            L = ['python', f'./test.py', 
                        "--method","DKT_binary", "--dataset", f"{dataset}", 
                        "--train_n_way", "2", "--test_n_way", "2", "--n_shot", f"{n_shot}", "--n_query", f"{n_query}",
                            "--seed",  f"{sd}",  
                            "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                            '--kernel_type', 'linear', "--normalize", "--n_task",  f"{n_task}",
                         
                            "--train_aug"
            ]
            if save_result: L.append('--save_result')
            print(f'\n{" ".join(L)} \n')
            run(L)

            for method in method_list:
                L = ['python', f'./test.py', 
                            "--method",f"{method}", "--sparse_method", "FRVM", "--dataset", f"{dataset}", 
                            "--train_n_way", "2", "--test_n_way", "2", "--n_shot", f"{n_shot}", "--n_query", f"{n_query}",
                                "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                '--kernel_type', 'linear', "--scale", "--normalize", "--n_task",  f"{n_task}",
                                "--maxItr_rvm", f"{max_itr}", "--tol_rvm", f"{tol_rvm}", "--regression",
                                "--train_aug"
                ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                run(L)

            lambda_rvm_list = [2.0]
            for lambda_rvm in lambda_rvm_list:
                for method in method_list:
                    L = ['python', f'./test.py', 
                                "--method",f"{method}", "--sparse_method", "FRVM", "--dataset", f"{dataset}", 
                                "--train_n_way", "2", "--test_n_way", "2", "--n_shot", f"{n_shot}", "--n_query", f"{n_query}",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                    '--kernel_type', 'linear', "--scale", "--normalize",  "--n_task",  f"{n_task}",
                                    "--regression", 
                                    "--rvm_mll", "--lambda_rvm", f"{lambda_rvm}", "--maxItr_rvm", f"{max_itr}", "--tol_rvm", f"{tol_rvm}",
                                    "--train_aug"
                    ]
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    run(L)


            for lambda_rvm in lambda_rvm_list:
                for method in method_list:
                    L = ['python', f'./test.py', 
                                "--method",f"{method}", "--sparse_method", "FRVM", "--dataset", f"{dataset}", 
                                "--train_n_way", "2", "--test_n_way", "2", "--n_shot", f"{n_shot}", "--n_query", f"{n_query}",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                                    '--kernel_type', 'linear', "--scale", "--normalize",  "--n_task", f"{n_task}",
                                    "--regression", 
                                    "--rvm_ll", "--lambda_rvm", f"{lambda_rvm}", "--maxItr_rvm", f"{max_itr}", "--tol_rvm", f"{tol_rvm}",
                                    "--train_aug"
                    ]
                    if save_result: L.append('--save_result')
                    print(f'\n{" ".join(L)} \n')
                    # run(L)
             
                
            L = ['python', f'./test.py', 
                        "--method", "Sparse_DKT_binary_RVM", "--sparse_method", "FRVM", "--dataset", f"{dataset}", 
                        "--train_n_way", "2", "--test_n_way", "2", "--n_shot", f"{n_shot}", "--n_query", f"{n_query}",
                            "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                            "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", 
                            '--kernel_type', 'linear', "--scale", "--normalize", "--n_task",  f"{n_task}",
                            "--regression", 
                            "--rvm_mll_only", "--maxItr_rvm", f"{max_itr}", "--tol_rvm", f"{tol_rvm}",
                            "--train_aug"
            ]
            if save_result: L.append('--save_result')
            print(f'\n{" ".join(L)} \n')
            run(L)

     
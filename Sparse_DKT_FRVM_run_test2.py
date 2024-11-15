from subprocess import run

config_list = ['000', '001', '010', '011']
dataset_list = ['omniglot', 'CUB', 'miniImagenet']
method_list = ['Sparse_DKT_Nystrom', 'Sparse_DKT_Exact'] # , 'DKT_binary'

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]


lr_gp_list = [0.0001]
lr_net_list = [0.001]
config_list = ['001']
n_task = 200

seed_list = [1, 2, 3]
method_list = ['Sparse_DKT_Exact']
save_result = True
best = True
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 0.0
            if config in ['000', '010']:
                align_thr = 0

            for sd in seed_list:

                L = ['python', f'./test.py', 
                            "--method","DKT", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}",
                                '--kernel_type', 'linear', "--n_task", f"{n_task}",
                            
                ]
                if save_result: L.append('--save_result')
                if best: L.append('--best')
                print(f'\n{" ".join(L)} \n')
                run(L)
                #Sparse DKT Exact
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
                    # run(L)


                lambda_rvm_list = [1.0]
                method_list = ['Sparse_DKT_Exact']
                lambda_rvm_list = [1.0]
                for align_thr in [ 0.0]:
                    for lambda_rvm in lambda_rvm_list:
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
                        # run(L)
                    
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
                                        '--kernel_type', 'linear', "--scale",  "--n_task", f"{n_task}",
                                        "--regression", 
                                        "--rvm_mll_only"
                                    
                        ]
                if save_result: L.append('--save_result')
                if best: L.append('--best')
                print(f'\n{" ".join(L)} \n')
                run(L)

                inner_lr = 0.01
                L = ['python', f'./test.py', 
                            "--method","MAML", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_net", f"{lr_net}", "--inner_lr", f"{inner_lr}", '--inner_loop', '10', '--first_order',
                                
                                "--n_task",  f"{n_task}",
                            
                ]
                if save_result: L.append('--save_result')
                if best: L.append('--best')
                print(f'\n{" ".join(L)} \n')
                run(L)

                L = ['python', f'./train.py', 
                            "--method","baseline", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_net", f"{lr_net}", 
                            "--n_task", "30"
                            
                ]
                if save_result: L.append('--save_result')
                print(f'\n{" ".join(L)} \n')
                # run(L)

                L = ['python', f'./test.py', 
                                "--method","MetaOptNet_binary", "--dataset", "omniglot", 
                                "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                    "--seed",  f"{sd}",  
                                    "--lr_net", f"{lr_net}", "--n_task",  f"{n_task}",
                                
                    ]
                if save_result: L.append('--save_result')
                if best: L.append('--best')
                print(f'\n{" ".join(L)} \n')
                run(L)
     
from subprocess import run

config_list = ['000', '001', '010', '011']
dataset_list = ['omniglot', 'CUB', 'miniImagenet']
method_list = ['Sparse_DKT_Exact', 'Sparse_DKT_Nystrom'] # , 'DKT_binary'

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

lr_gp_list = [0.001]
lr_net_list = [0.001]
config_list = ['001']
seed_list = [1, 2]
# method_list = ['Sparse_DKT_Nystrom']
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 0.0
            if config in ['000', '010']:
                align_thr = 0

            for sd in seed_list:
                L = ['python', f'./train.py', 
                            "--method","DKT", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--stop_epoch", "100",
                                '--kernel_type', 'linear', "--save_model", "--n_task", "30"
                            
                ]
                print(f'\n{" ".join(L)} \n')
                # run(L)
                
                method_list = ['Sparse_DKT_Exact']
                lambda_rvm_list = [1.0]
                for align_thr in [ 0.0]:
                    for lambda_rvm in lambda_rvm_list:
                        for method in method_list:
                            L = ['python', f'./train.py', 
                                        "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                        "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                            "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                            "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--stop_epoch", "100",
                                            '--kernel_type', 'linear', "--scale", "--save_model", "--n_task", "30",
                                            "--regression", 
                                            "--rvm_mll", "--lambda_rvm", f"{lambda_rvm}"
                                        
                            ]
                            print(f'\n{" ".join(L)} \n')
                            # run(L)  


                for method in method_list:
                    L = ['python', f'./train.py', 
                                "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                    "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                    "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--stop_epoch", "100",
                                    '--kernel_type', 'linear', "--scale", "--save_model", "--n_task", "30"
                                    , "--regression"
                                
                    ]
                    print(f'\n{" ".join(L)} \n')
                    # run(L) 
                    
                # rvm ll
                lambda_rvm_list = [1.0]
                for lambda_rvm in lambda_rvm_list:
                    for method in method_list:
                        L = ['python', f'./train.py', 
                                    "--method", f"{method}", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                    "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                        "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                        "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--stop_epoch", "100",
                                        '--kernel_type', 'linear', "--scale", "--save_model", "--n_task", "30",
                                        "--regression", 
                                        "--rvm_ll", "--lambda_rvm", f"{lambda_rvm}"
                                    
                        ]
                        print(f'\n{" ".join(L)} \n')
                        # run(L)

                L = ['python', f'./train.py', 
                                    "--method", f"Sparse_DKT_RVM", "--sparse_method", "FRVM", "--dataset", "omniglot", 
                                    "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                        "--seed",  f"{sd}", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                                        "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--stop_epoch", "100",
                                        '--kernel_type', 'linear', "--scale", "--save_model", "--n_task", "30",
                                        "--regression", 
                                        "--rvm_mll_only"
                                    
                        ]
                print(f'\n{" ".join(L)} \n')
                # run(L)

               

                inner_lr = 0.01
                L = ['python', f'./train.py', 
                            "--method","MAML", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_net", f"{lr_net}", "--inner_lr", f"{inner_lr}", '--inner_loop', '5',
                                "--stop_epoch", "100",
                                "--save_model", "--n_task", "30"
                            
                ]
                print(f'\n{" ".join(L)} \n')
                run(L)

                L = ['python', f'./train.py', 
                            "--method","baseline", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_net", f"{lr_net}", 
                                "--stop_epoch", "100",
                                "--save_model", "--n_task", "30"
                            
                ]
                print(f'\n{" ".join(L)} \n')
                # run(L)

                L = ['python', f'./train.py', 
                            "--method","MetaOptNet_binary", "--dataset", "omniglot", 
                            "--train_n_way", "5", "--test_n_way", "5", "--n_shot", "15", "--n_query", "5",
                                "--seed",  f"{sd}",  
                                "--lr_net", f"{lr_net}", 
                                "--stop_epoch", "100",
                                "--save_model", "--n_task", "30"
                            
                ]
                print(f'\n{" ".join(L)} \n')
                run(L)
            
             
            
            L = ['python', f'./Sparse_DKT_FRVM_run_test2.py'
                    ]
            print(f'\n{" ".join(L)} \n')
            run(L)

     

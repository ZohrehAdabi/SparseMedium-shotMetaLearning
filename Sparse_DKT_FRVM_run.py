from subprocess import run

config_list = ['001'] #['000', '001', '010', '011']
dataset_list = ['omniglot', 'CUB', 'miniImagenet']
method_list = ['Sparse_DKT_binary', 'DKT_binary']

lr_gp_list = [0.1, 0.01, 0.001, 0.0001]
lr_net_list = [0.01, 0.001, 0.0001]

# for config in config_list:
#     for lr_gp in lr_gp_list:
#         for lr_net in lr_net_list:
    

#             run(['python', f'./train.py', 
#                         "--method","Sparse_DKT_binary", "--sparse_method", "FRVM", "--dataset", "omniglot", 
#                         "--train_n_way", "2", "--test_n_way", "2", "--n_shot", "15", "--n_query", "5",
#                             "--seed","1", "--config", f"{config}", "--align_thr", "1e-3" , 
#                             "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}"
#                             #,"--train_aug"
#             ])

            # run(['python', f'./train.py', 
            #             "--method","Sparse_DKT_binary", "--sparse_method", "FRVM", "--dataset", "omniglot", 
            #             "--train_n_way", "2", "--test_n_way", "2", "--n_shot", "15", "--n_query", "5",
            #                 "--seed","1", "--config", f"{config}", "--align_thr", "1e-3" , 
            #                 "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}", "--gamma" 
            #                 #,"--train_aug"
            # ])
config_list = ['000', '001', '010', '011']
lr_gp_list = [ 0.01, 0.001]
lr_net_list = [ 0.001, 0.0001]
for config in config_list:
    for lr_gp in lr_gp_list:
        for lr_net in lr_net_list:
            align_thr = 1e-3
            if config in ['000', '010']:
                align_thr = 0
            
            run(['python', f'./train.py', 
                        "--method","Sparse_DKT_binary", "--sparse_method", "FRVM", "--dataset", "CUB", 
                        "--train_n_way", "2", "--test_n_way", "2", "--n_shot", "50", "--n_query", "10",
                            "--seed","1", "--config", f"{config}", "--align_thr", f"{align_thr}" , 
                            "--lr_gp", f"{lr_gp}", "--lr_net", f"{lr_net}"
                            #,"--train_aug"
            ])
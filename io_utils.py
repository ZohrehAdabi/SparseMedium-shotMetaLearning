import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='DKT',   help='DKT/baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_query'      , default=2, type=int,  help='number of test labeled data in each class') 
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--config', default='010', type=str, help='config for Fast RVM = {delete_priority|add_priority|align_test}')
    parser.add_argument('--align_thr', default=1e-3, type=float, help='1e-3, larger value leads to more rejection and sparseness')
    parser.add_argument('--sparse_method', default='FRVM', type=str, help='FRVM|KMeans|random')
    parser.add_argument('--dirichlet', action='store_true',  help='perform dirichlet classification')
    parser.add_argument('--gamma', action='store_true', help='Delete data with low Gamma in FRVM algorithm') 
    parser.add_argument('--lr_gp', default=1e-3, type=float, help='learning rate for [GP] model')
    parser.add_argument('--lr_net', default=1e-3, type=float, help='learning rate for feature extractor')
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--repeat', default=1, type=int, help ='Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]')
    else:
       raise ValueError('Unknown script')


    return parser.parse_args()

def parse_args_regression(script):
        parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
        parser.add_argument('--seed' , default=2, type=int,  help='Seed for Numpy and pyTorch. Default: 0 (None)')
        parser.add_argument('--model'       , default='Conv3',   help='model: Conv{3} / ResNet{50}')
        parser.add_argument('--method'      , default='DKT',   help='DKT / Sparse_DKT / transfer')
        parser.add_argument('--sparse_method', default='KMeans', type=str, help='KMeans / FRVM / random')
        parser.add_argument('--dataset'     , default='QMUL',    help='QMUL / MSC44')
        parser.add_argument('--spectral', action='store_true', help='Use a spectral covariance kernel function')
        parser.add_argument('--n_samples', default=72, type=int, help='Number of points on trajectory') #at most 19 
        parser.add_argument('--show_plots_loss', action='store_true', help='Show plots') 
        parser.add_argument('--show_plots_features', action='store_true', help='Show plots') 
        parser.add_argument('--n_centers', default=24, type=int, help='Number of Inducing points/ KMeans centers in Kmeans sparsifying')
        parser.add_argument('--config', default='0000', type=str, help='config for Fast RVM = {update_sigma|delete_priority|add_priority|align_test} recom = {"0010", "1000", "1010", "1011","1100", "1101"}')
        parser.add_argument('--align_thr', default=1e-3, type=float, help='1e-3, larger value leads to more rejection and sparseness')
        parser.add_argument('--lr_gp', default=1e-3, type=float, help='Learning rate for GP and Neural Network')
        parser.add_argument('--lr_net', default=1e-3, type=float, help='Learning rate for Neural Network')
        parser.add_argument('--gamma', action='store_true', help='Delete data with low Gamma in FRVM algorithm') 
        # parser.add_argument('--alpha', default=1e3, type=float, help='coefficient for mse loss')
        if script == 'train_regression':
            parser.add_argument('--start_epoch' , default=0, type=int, help ='Starting epoch')
            parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
            parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
            parser.add_argument('--n_support', default=60, type=int, help='Number of points on trajectory to be given as support points')
        elif script == 'test_regression':
            parser.add_argument('--n_support', default=60, type=int, help='Number of points on trajectory to be given as support points')
            parser.add_argument('--n_test_epochs', default=1, type=int, help='{QMUL:How many test people? def=5| MSC44:How manytimes test on all test tasks def=1')
            parser.add_argument('--show_plots_pred', action='store_true', help='Show plots')
        return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

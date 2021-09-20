import numpy as np
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from colorama import Fore
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.DKT import DKT
from methods.DKT_binary import DKT_binary
from methods.Sparse_DKT import Sparse_DKT
from methods.Sparse_DKT_binary import Sparse_DKT_binary
from methods.Sparse_DKT_binary_rvm import Sparse_DKT_binary_rvm
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if(verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, lr_gp, lr_net, params):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': lr_gp},
                                      {'params': model.feature_extractor.parameters(), 'lr': lr_net}])
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    print(f'num train task {len(base_loader)}')
    print(f'num val task {len(val_loader)}')
    for epoch in range(start_epoch, stop_epoch):
        #model.eval()
        #acc = model.test_loop(val_loader)
        print(f'Epoch {epoch}')
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        model.eval()
        if epoch%5==0:
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            print(Fore.GREEN,"-"*50 ,f'\nValidation \n', Fore.RESET)
            acc = model.test_loop(val_loader)
            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("--> Best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            print(Fore.YELLOW, f'ACC: {acc:4.2f}\n', Fore.RESET)
            print(Fore.GREEN,"-"*50 ,'\n', Fore.RESET)
    return model


if __name__ == '__main__':
    params = parse_args('train')
    _set_seed(parse_args('train').seed)
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 100  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['Sparse_DKT', 'Sparse_DKT_binary', 'Sparse_DKT_binary_rvm', 'DKT', 'DKT_binary', 'protonet', 
                        'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        # for fewshot setting
        # n_query = max(1, int(
        #     16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=params.n_query, **train_few_shot_params) #n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if(params.method == 'Sparse_DKT'):
            model = Sparse_DKT(model_dict[params.model], **train_few_shot_params, 
                                    config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'           
            if params.gamma: id += '_gamma'
            model.init_summary(id=id, dataset=params.dataset)
        elif params.method == 'Sparse_DKT_binary':
            model = Sparse_DKT_binary(model_dict[params.model], **train_few_shot_params, 
                                    config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'           
            if params.gamma: id += '_gamma'
            model.init_summary(id=id, dataset=params.dataset)

        elif params.method == 'Sparse_DKT_binary_rvm':
            model = Sparse_DKT_binary_rvm(model_dict[params.model], **train_few_shot_params, 
                                    config=params.config, align_threshold=params.align_thr, gamma=params.gamma, dirichlet=params.dirichlet)
            if params.dirichlet:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id = f'{params.method}_{params.sparse_method}_{params.model}_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'           
            if params.gamma: id += '_gamma'
            model.init_summary(id=id, dataset=params.dataset)

        elif(params.method == 'DKT'):
            model = DKT(model_dict[params.model], **train_few_shot_params, dirichlet=params.dirichlet)
            if params.dirichlet:
                id=f'DKT_{params.model}_{params.dataset}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id=f'DKT_{params.model}_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
            model.init_summary(id=id)
        elif(params.method == 'DKT_binary'):
            model = DKT_binary(model_dict[params.model], **train_few_shot_params, dirichlet=params.dirichlet)
            if params.dirichlet:
                id=f'DKT_binary_{params.model}_{params.dataset}_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id=f'DKT_binary_{params.model}_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
            model.init_summary(id=id)
        
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        
        if params.method=='Sparse_DKT' or params.method=='Sparse_DKT_binary' or params.method=='Sparse_DKT_binary_rvm':
            if params.dirichlet:
                id = f'_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id = f'_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_{params.config}_{params.align_thr}_lr_{params.lr_gp}_{params.lr_net}'           
            if params.gamma : id += '_gamma'
            params.checkpoint_dir += id
        else:
            if params.dirichlet:
                id=f'_dirichlet_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
            else:
                id=f'_{params.dataset}_way_{params.train_n_way}_shot_{params.n_shot}_query_{params.n_query}_lr_{params.lr_gp}_{params.lr_net}'
        
            params.checkpoint_dir += id

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params.lr_gp, params.lr_net, params)

"""Main training/test program for RULSTM"""
#python main.py --gpu_ids 0 --batch_size 128 --wd 1e-5 --lr 0.1 --reinforce_verb_weight 0.01 --reinforce_noun_weight 0.01 --revision_weight 0.8 --mode train --modality rgb --hidden 1024 --feat_in 1024
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import socket
from os.path import join
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import os

from args import parse_args
from train import train, train_attm
from model import get_model
from validation import validation
from fusion_validation import fusion_validation
from fusion_validation import fusion_test
from dataloaders.EPIC import SequenceDataset
from utils import Logger, topk_accuracy
pd.options.display.float_format = '{:05.2f}'.format
# def join(str1,str2):
#     return str1+'/'+str2;
# def join(str1,str2,str3,str4):
#     return str1+'/'+str2+'/'+str3+'/'+str4;

def get_loader(args, mode, override_modality = None):
    if args.modality != 'fusion':
        path_to_lmdb = os.path.join(args.path_to_lmdb, args.modality)
    else:
        path_to_lmdb = [join(args.path_to_lmdb, m) for m in ['rgb', 'obj']]
        #path_to_lmdb = [join(args.path_to_lmdb, m) for m in ['rgb', 'flow']]

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant if args.task == 'early_recognition' else None,
        'past_features': args.task == 'anticipation',
        'sequence_length': args.S_enc + args.S_ant + args.S_Rad,
        'Rand_length': args.S_Rad,
        'label_type': ['verb', 'noun', 'action'],
        'revision_threshold': args.revision_threshold,
    }
    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')
def load_checkpoint(args, model, best=False):
    if best:
        chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, 'results_sensors', args.best_model_name, args.exp_name + '_best.pth.tar'), weights_only = False)
    else:
        chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, 'results_sensors', args.resume_timestamp, args.exp_name + '.pth.tar'), weights_only = False)

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf
def load_checkpoint_FGM(args, model, best=False):


    """
    只加载参数名称和形状都匹配的预训练权重，忽略形状不匹配的参数。

    Args:
        model: 目标模型
        pretrained_dict: 预训练权重字典
        strict: 是否严格匹配参数名称（默认False，允许缺失或多余的参数）

    Returns:
        model: 加载部分权重后的模型
        missing_keys: 缺失的参数（模型有但权重没有）
        unexpected_keys: 多余的参数（权重有但模型没有）
        shape_mismatch_keys: 形状不匹配的参数
    """
    pretrained_dict = torch.load('./premodel/R62620/bese_rgb.tar',weights_only=False)
    model_dict = model.state_dict()

    # 1. 过滤出名称匹配的参数
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items()
                                if k in model_dict and v.shape == model_dict[k].shape}
    # 3. 加载过滤后的权重
    model.load_state_dict(pretrained_dict_filtered, strict=False)



def main():
    print("begin", datetime.now())
    ## set parameters
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    print(torch.cuda.device_count())

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:",args.device)
    if args.dataset == 'EPIC':
        args.num_class = 2513
        args.verb_num_class = 125
        args.noun_num_class = 352
    if args.dataset == 'Gaze':
        args.num_class = 106
        args.verb_num_class = 19
        args.noun_num_class = 51
    args.exp_name = f"{args.model_name}-{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}_{args.modality}"

    if args.mode == 'train':
        args.timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        args.save_path = join(args.path_to_results, args.dataset, args.model_name, args.timestamp)
        os.makedirs(args.save_path)

        trainval_logger = Logger(join(args.save_path, 'trainval.log'), ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])

        # save parameters
        with open(os.path.join(args.save_path, 'args.json'), 'w') as args_file:
            json.dump(vars(args), args_file)

        # Logging Tensorboard
        log_dir = os.path.join(args.save_path, 'tensorboard', 'runs', socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir, comment='-params')


    model = get_model(args)

    if type(model) == list:
        model = [m.to(args.device) for m in model]
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        model.to(args.device)

    if args.mode == 'train':
        loaders = {m: get_loader(args, m) for m in ['training', 'validation']}
        print("loaders", datetime.now())
        if args.pre_train is False and args.modality != 'fusion':
            load_checkpoint_FGM(args, model)
        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(args, model)
        else:
            start_epoch = 0
            start_best_perf = 0
        print("load_checkpoint", datetime.now())
        
        if args.modality == 'fusion':
            train_attm(args, model, loaders, optimizer, args.epochs, start_epoch, start_best_perf, trainval_logger, writer)
        else:
            train(args, model, loaders, optimizer, args.epochs, start_epoch, start_best_perf, trainval_logger, writer)

        writer.close()
    elif args.mode == 'validate':
        if args.modality == 'fusion1':
            print('Fuison validation')
            loader = get_loader(args, 'validation')
            fusion_validation(args, model, loader)
        else:
            epoch, perf, _ = load_checkpoint(args, model, best=True)
            print(f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")
            loader = get_loader(args, 'validation')
            validation(args, model, loader)
    elif args.mode == 'test':#only fusion
        print('Fuison test')
        mm = ['seen', 'unseen']
        for m in mm:
            loader = get_loader(args, f"test_{m}")

            fusion_test(args, model, loader, m)
    # elif args.mode == 'pre_train':
    #


if __name__ == '__main__':
    main()

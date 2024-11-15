#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
        ：clic -> fine_tuning.py
@IDE    ：PyCharm
@Author ：liu shipeng
@Date   ：2024/11/12
@info   ：modified from https://github.com/tinglyfeng/IC9600
=================================================='''
import os
import sys
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from scipy.stats import pearsonr, spearmanr

from clic.loader import ic_dataset
from clic.icnet import ICNet_ft


parser = argparse.ArgumentParser()
parser.add_argument('--warm', type=int, default=1, help='num of epoch for warming up')
parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay of optimizer')
parser.add_argument('--lr_decay_rate',type = float,default=0.2, help='lr decay of optimizer')
parser.add_argument('--milestone',type = list,default=[10,20], help = 'perform lr decay in each milestone')
parser.add_argument('--batch_size',type=int, default=64, help='batch size for dataloader')
parser.add_argument('--num_workers',type=int,default=8, help = 'num of worker for dataloader')
parser.add_argument('--epoch',type=int,default=30, help='total epoch of training')
parser.add_argument('--image_size',type = int ,default= 512, help = 'input size of model')
parser.add_argument('--gpu_id',type = int,default=0, help = 'gpu id')
parser.add_argument('--ck_save_dir',type = str, default='cks', help ='directory to save checkpoints')
parser.add_argument('--clic_ckpt', type=str, default='checkpoint_0199.pth.tar', help='clic pretrained checkpoints')

# logging setting
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def evaInfo(score, label):
    score = np.array(score)
    label = np.array(label)

    Pearson = pearsonr(label, score)[0]
    Spearmanr = spearmanr(label, score)[0]

    info = 'Pearsonr : {:.4f} ,   Spearmanr : {:.4f}'.format(Pearson, Spearmanr)

    return info


def train(epoch):
    model.train()
    for batch_index, (image,label,_) in enumerate(trainDataLoader):        
        image = image.to(device)
        label = label.to(device)       
        Opmimizer.zero_grad()
        score1, cly_map = model(image)
        score2 = cly_map.mean(axis = (1,2,3))
        loss1 = loss_function(score1,label)
        loss2 = loss_function(score2,label)
        loss = 0.9*loss1 + 0.1*loss2
        loss.backward()
        Opmimizer.step()
        if epoch <= args.warm:
            Warmup_scheduler.step()

        if (batch_index+1) % (len(trainDataLoader) // 3) == 0:
            logging.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                Opmimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(image),
                total_samples=len(trainDataLoader.dataset)
            ))


def evaluation():
    model.eval()
    all_scores = []
    all_labels = []
    for (image, label, _) in testDataLoader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            score, _= model(image)
            all_scores += score.tolist()
            all_labels += label.tolist()
    info = evaInfo(score=all_scores, label=all_labels)
    logging.info(info + '\n')




if __name__ == "__main__":
    args = parser.parse_args()
    
    trainTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    testTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
  
    trainDataset = ic_dataset(
        txt_path ="../IC9600/train.txt",
        img_path = "../IC9600/images/",
        transform = trainTransform
    )

    
    trainDataLoader = DataLoader(trainDataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=True
                             )

    testDataset = ic_dataset(
        txt_path ="../IC9600/test.txt",
        img_path = "../IC9600/images/",
        transform=testTransform
    )
    
    testDataLoader = DataLoader(testDataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False
                            )
    if not os.path.exists(args.ck_save_dir):
        os.mkdir(args.ck_save_dir)
    
    model = ICNet_ft()

    # load checkpoint
    if os.path.isfile(args.clic_ckpt):
        checkpoint = torch.load(args.clic_ckpt)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info("=> Loaded checkpoint '{}'".format(args.clic_ckpt))
    else:
        logger.error("=> No checkpoint found at '{}'".format(args.clic_ckpt))
        sys.exit(-1)
    
    device = torch.device("cuda:{}".format(args.gpu_id))
    model.to(device)

    loss_function = nn.MSELoss()
    
    # optimize
    params = model.parameters()
    Opmimizer = optim.SGD(params, lr =args.lr,momentum=0.9,weight_decay=args.weight_decay)
    Scheduler = optim.lr_scheduler.MultiStepLR(Opmimizer,milestones=args.milestone,gamma = args.lr_decay_rate)
    iter_per_epoch = len(trainDataLoader)
    if args.warm > 0:
        Warmup_scheduler = WarmUpLR(Opmimizer,iter_per_epoch*args.warm)
    
    # running
    for epoch in range(1, args.epoch+1):
        train(epoch)
        if epoch > args.warm:
            Scheduler.step(epoch)
        evaluation()
        torch.save(model.state_dict(), os.path.join(args.ck_save_dir,'ck_{}.pth'.format(epoch)))

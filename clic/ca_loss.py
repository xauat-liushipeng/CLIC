#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
        ：clic -> clic -> ca_loss.py
@IDE    ：PyCharm
@Author ：liu shipeng
@Date   ：2024/11/12
@info   ：compute complexity aware loss
=================================================='''

import torch
from scipy.stats import entropy
import torch.nn.functional as F

def cal_batch_ge(batch_imgs):
    """ compute global entropy (ge) of per image in mini-batch """

    batch_ge = []
    for img in batch_imgs:
        im_q_gray = img.mean(dim=0)  # rgb to gray
        hist = torch.histc(im_q_gray, bins=256, min=0, max=1)
        hist /= hist.sum()  # norm
        batch_ge.append(entropy(hist.cpu().numpy()))
    return torch.tensor(batch_ge).cuda()


def cal_stage_fae(feature_map):
    """
    calculate feature activation energy of every stage
    - Input: feature_map: (n, c, h, w)
    - Output: stage_fae: (n,)
    """
    batch_size, channels, width, height = feature_map.shape
    stage_fae = feature_map.sum(dim=(1, 2, 3)) / (width * height * channels)
    return stage_fae


def ge_fae_error(batch_ge, stage_maps):
    """ complexity aware loss: compute ge and fae by MSE loss """
    batch_fae = torch.zeros_like(batch_ge).cuda()
    for stage in stage_maps:
        batch_fae += cal_stage_fae(stage_maps[stage])

    # generate a mask, filter nan or inf
    mask = torch.isfinite(batch_fae) & torch.isfinite(batch_ge)
    filtered_fae = batch_fae[mask]
    filtered_ge = batch_ge[mask]

    # MSE
    ca_loss = F.mse_loss(filtered_fae, filtered_ge)
    return ca_loss

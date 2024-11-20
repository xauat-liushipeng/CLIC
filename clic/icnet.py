#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
        ：clic -> clic -> icnet.py
@IDE    ：PyCharm
@Author ：liu shipeng
@Date   ：2024/11/12
@info   ：ICNet official impl: https://github.com/tinglyfeng/IC9600
=================================================='''
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class slam(nn.Module):
    def __init__(self, spatial_dim):
        super(slam,self).__init__()
        self.spatial_dim = spatial_dim
        self.linear = nn.Sequential(
             nn.Linear(spatial_dim**2,512),
             nn.ReLU(),
             nn.Linear(512,1),
             nn.Sigmoid()
        )

    def forward(self, feature):
        n,c,h,w = feature.shape
        if (h != self.spatial_dim):
            x = F.interpolate(feature,size=(self.spatial_dim,self.spatial_dim),mode= "bilinear", align_corners=True)
        else:
            x = feature


        x = x.view(n,c,-1)
        x = self.linear(x)
        x = x.unsqueeze(dim =3)
        out = x.expand_as(feature)*feature

        return out
        

class to_map(nn.Module):
    def __init__(self,channels):
        super(to_map,self).__init__()
        self.to_map = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=1, kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        
    def forward(self,feature):
        return self.to_map(feature)
    
        
class conv_bn_relu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, padding= padding, stride = stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

        

class up_conv_bn_relu(nn.Module):
    def __init__(self,up_size, in_channels, out_channels = 64, kernal_size = 1, padding =0, stride = 1):
        super(up_conv_bn_relu,self).__init__()
        self.upSample = nn.Upsample(size = (up_size,up_size),mode="bilinear",align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size = kernal_size, stride = stride, padding= padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class ICNet(nn.Module):
    def __init__(self, is_pretrain = True, size1 = 512, size2 = 256):
        super(ICNet,self).__init__()
        resnet18Pretrained1 = torchvision.models.resnet18(pretrained= is_pretrain)
        resnet18Pretrained2 = torchvision.models.resnet18(pretrained= is_pretrain)
        
        self.size1 = size1
        self.size2 = size2
        
        ## detail branch
        self.b1_1 = nn.Sequential(*list(resnet18Pretrained1.children())[:5])   
        self.b1_1_slam = slam(32)
    
        self.b1_2 = list(resnet18Pretrained1.children())[5]   
        self.b1_2_slam = slam(32)

        ## context branch
        self.b2_1 = nn.Sequential(*list(resnet18Pretrained2.children())[:5])
        self.b2_1_slam = slam(32)
        
        self.b2_2 = list(resnet18Pretrained2.children())[5]
        self.b2_2_slam = slam(32)
    
        self.b2_3 = list(resnet18Pretrained2.children())[6]
        self.b2_3_slam = slam(16)
        
        self.b2_4 = list(resnet18Pretrained2.children())[7]
        self.b2_4_slam = slam(8)

        ## upsample
        self.upsize = size1 // 8
        self.up1 = up_conv_bn_relu(up_size = self.upsize, in_channels = 128, out_channels = 256)
        self.up2 = up_conv_bn_relu(up_size = self.upsize, in_channels = 512, out_channels = 256)
        
        ## map prediction head
        self.to_map_f = conv_bn_relu(256*2,256*2)
        self.to_map_f_slam = slam(32)
        self.to_map = to_map(256*2)
        
        ## score prediction head
        self.to_score_f = conv_bn_relu(256*2,256*2)
        self.to_score_f_slam = slam(32)
        self.head = nn.Sequential(
            nn.Linear(256*2,512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        
    def forward(self,x1):
        assert(x1.shape[2] == x1.shape[3] == self.size1)
        x2 = F.interpolate(x1, size= (self.size2,self.size2), mode = "bilinear", align_corners= True)

        x1 = self.b1_2_slam(self.b1_2(self.b1_1_slam(self.b1_1(x1))))
        x2 = self.b2_2_slam(self.b2_2(self.b2_1_slam(self.b2_1(x2))))
        x2 = self.b2_4_slam(self.b2_4(self.b2_3_slam(self.b2_3(x2))))

        
        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x_cat = torch.cat((x1,x2),dim = 1)
        
        cly_map = self.to_map(self.to_map_f_slam(self.to_map_f(x_cat)))
        
        score_feature = self.to_score_f_slam(self.to_score_f(x_cat))
        score_feature = self.avgpool(score_feature)
        score_feature = score_feature.squeeze()
        score = self.head(score_feature)
        score = score.squeeze()
    
        return score,cly_map


# fine-tuning with CLIC q encoder
class ICNet_ft(nn.Module):
    def __init__(self, size=512, encoder_name='resnet50'):
        super(ICNet_ft, self).__init__()
        self.size1 = size
        self.encoder_name = encoder_name

        # base_encoders map
        models_name = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "swin_b": models.swin_b,
        }

        # q encoder
        base_encoder = models_name[encoder_name]()
        # remove avg-pool fc
        self.encoder_q = nn.Sequential(OrderedDict(list(base_encoder.named_children())[:-2]))

        # freeze q params
        for param in self.encoder_q.parameters():
            param.requires_grad = False

        ## upsample
        self.upsize = size // 8

        if encoder_name == 'resnet18':
            in_channels = 512
        elif encoder_name.startswith('swin'):
            in_channels = 1024
        else:
            in_channels = 2048

        self.up = up_conv_bn_relu(up_size=self.upsize, in_channels=in_channels, out_channels=512)

        ## map prediction head
        self.to_map_f = conv_bn_relu(256 * 2, 256 * 2)
        self.to_map_f_slam = slam(32)
        self.to_map = to_map(256 * 2)

        ## score prediction head
        self.to_score_f = conv_bn_relu(256 * 2, 256 * 2)
        self.to_score_f_slam = slam(32)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.encoder_q(x)

        # swin transformer need permute
        if self.encoder_name.startswith("swin"):
            x = x.permute(0, 3, 1, 2)

        x = self.up(x)

        cly_map = self.to_map(self.to_map_f_slam(self.to_map_f(x)))

        score_feature = self.to_score_f_slam(self.to_score_f(x))
        score_feature = self.avgpool(score_feature)
        score_feature = score_feature.squeeze()
        score = self.head(score_feature)
        score = score.squeeze()

        return score, cly_map

if __name__ == '__main__':
    input = torch.randn(8, 3, 512, 512)

    icnet_ft = ICNet_ft(encoder_name='swin_b')
    print(icnet_ft)
    print(icnet_ft(input))

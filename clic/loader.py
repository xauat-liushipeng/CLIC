#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
        ：clic -> clic -> loader.py
@IDE    ：PyCharm
@Author ：liu shipeng
@Date   ：2024/11/12
@info   ：datasets, transforms
=================================================='''
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GaussianNoise:
    def __init__(self, mean=0, stddev=25, p=0.5):
        self.mean = mean
        self.stddev = stddev
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            np_img = np.array(img)
            noise = np.random.normal(self.mean, self.stddev, np_img.shape).astype(np.int16)
            noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return img


class RandomRotation:
    def __init__(self, max_angle=30, crop_size=224):
        self.max_angle = max_angle
        self.crop_size = crop_size

    def __call__(self, img):
        if random.random() < 0.5:
            angle = random.uniform(-self.max_angle, self.max_angle)
            img_rotated = img.rotate(angle, expand=True)
            left = max((img_rotated.width - self.crop_size) // 2, 0)
            top = max((img_rotated.height - self.crop_size) // 2, 0)
            right = left + self.crop_size
            bottom = top + self.crop_size
            return img_rotated.crop((left, top, right, bottom))
        return img


class ImageDataset(Dataset):
    def __init__(self, root_dir, label_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.label_file = label_file
        self.labels = {}

        if label_file is not None:
            # 读取标签文件，将图像文件名和ic分数存储在字典中
            with open(label_file, 'r') as f:
                for line in f:
                    img_name, ic_score = line.split(" ")
                    self.labels[img_name] = float(ic_score)

            # 获取图像路径，确保路径中只包含有对应标签的图像
            self.image_paths = [
                os.path.join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG')) and fname in self.labels
            ]
        else:
            self.image_paths = [
                os.path.join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # 应用transform

        if self.label_file is not None:
            img_name = os.path.basename(img_path)  # 获取图像文件名
            label = self.labels[img_name]
            return image, label

        return image, 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

q_transform = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
]

k_transform = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]

"""
RandomResizedCrop(224, scale=(0.2, 1.0))
RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
RandomApply([GaussianBlur((5, 9), (0.1, 2.0))], p=0.5)
RandomHorizontalFlip()
"""
# k_transform = [
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomApply([transforms.GaussianBlur((5, 9), (0.1, 2.0))], p=0.5),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     normalize
# ]

"""
RandomResizedCrop(224, scale=(0.2, 1.0))
RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
RandomApply([GaussianBlur((5, 9), (0.1, 2.0))], p=0.5)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(max_angle=30)
GaussianNoise(mean=0, stddev=25, p=0.5)
"""
# k_transform_more = [
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomApply([transforms.GaussianBlur((5, 9), (0.1, 2.0))], p=0.5),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     RandomRotation(max_angle=30),
#     GaussianNoise(mean=0, stddev=25, p=0.5),
#     transforms.ToTensor(),
#     normalize
# ]

class CropTransform:
    """
    Take one random crop of one image as the key.
    image as query
    """
    def __init__(self):
        self.q_transform = transforms.Compose(q_transform)
        self.k_transform = transforms.Compose(k_transform)

    def __call__(self, x):
        q = self.q_transform(x)
        k = self.k_transform(x)
        return [q, k]


class ic_dataset(Dataset):
    """ ic_dataset in IC9600: https://github.com/tinglyfeng/IC9600 """
    def __init__(self, txt_path, img_path, transform=None):
        super(ic_dataset, self).__init__()
        self.txt_lines = self.readlines(txt_path)
        self.img_path = img_path
        self.transform = transform
        self.img_info_list = self.parse_lines(self.txt_lines)

    def parse_lines(self, lines):
        image_info_list = []
        for line in lines:
            line_split = line.strip().split("  ")
            img_name = line_split[0]
            img_label = line_split[1]
            image_info_list.append((img_name, img_label))
        return image_info_list

    def readlines(self, txt_path):
        f = open(txt_path, 'r')
        lines = f.readlines()
        f.close()
        return lines

    def __getitem__(self, index):
        imgName, imgLabel = self.img_info_list[index]
        oriImgPath = os.path.join(self.img_path, imgName)
        img = Image.open(oriImgPath).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(float(imgLabel))
        return img, label, imgName

    def __len__(self):
        return len(self.img_info_list)
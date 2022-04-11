# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:messidor
    author: 12718
    time: 2022/4/9 13:51
    tool: PyCharm
"""
import cv2
import os
import glob
from albumentations import Compose, RandomCrop
from albumentations import GaussianBlur, ColorJitter
from albumentations import HorizontalFlip, VerticalFlip
from albumentations import ShiftScaleRotate, Resize
from albumentations import Normalize
from torch.utils.data import Dataset
import numpy as np

class MessidorSR(Dataset):
    def __init__(self, img_paths, hr_size:list=[744, 1280], upscale_rate=4):
        self.img_paths = img_paths
        self.upscale_rate = upscale_rate
        self.hr_height, self.hr_width = hr_size
        self.length = len(img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_height = self.hr_height // self.upscale_rate
        out_width = self.hr_width // self.upscale_rate
        crop_height = out_height*int(self.upscale_rate)
        crop_width = out_width*int(self.upscale_rate)
        hr_aug_set = Compose(
            [
                RandomCrop(height=crop_height, width=crop_width),
                ColorJitter(),
                HorizontalFlip(),
                VerticalFlip(),
                ShiftScaleRotate()
            ]
        )
        resize_set = Compose(
            [
                Resize(height=out_height, width=out_width),
                GaussianBlur(always_apply=True)
            ]
        )
        hr = hr_aug_set(image=img)["image"]
        lr = resize_set(image=hr)["image"]
        normalization = Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        lr = normalization(image=lr)["image"]
        hr = normalization(image=hr)["image"]
        lr = np.transpose(lr, axes=[2, 0, 1])
        hr = np.transpose(hr, axes=[2, 0, 1])
        return lr, hr
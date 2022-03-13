# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cityscape
    author: 12718
    time: 2022/1/23 18:53
    tool: PyCharm
"""
import torch
from torch.utils.data import Dataset
import os
import glob
from typing import Union,List
from albumentations import Compose, ColorJitter, HorizontalFlip
from albumentations import VerticalFlip, ChannelDropout, GaussianBlur
from albumentations import Normalize, Resize, ShiftScaleRotate
import cv2
import numpy as np

from comm.helper import to_tuple


def get_paths(root, split="train"):
    img_data_dir = os.path.join(root, "leftImg8bit" ,split)
    mask_data_dir = os.path.join(root, "gtFine", split)
    img_paths = glob.glob(os.path.join(img_data_dir,"*","*.png"))
    mask_paths = []
    for path in img_paths:
        filename = os.path.basename(path)
        mask_filename = filename.replace("leftImg8bit", "gtFine_labelTrainIds")
        assert os.path.exists(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename)), \
            "mask file {} is not exists".format(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename))
        mask_paths.append(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename))
    return img_paths, mask_paths

class CityScapeDataset(Dataset):
    def __init__(self, image_paths:List[str], mask_paths:List[str],
                 augmentation:bool=False,
                 output_size:Union[int,tuple,list]=512,
                 super_reso:bool=False, upscale_rate:int=2,
                 inter=cv2.INTER_CUBIC):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        assert len(self.image_paths) == len(self.mask_paths), "Length for list of image paths must be" \
                                                              "equal to length for list mask paths, " \
                                                              "except {} but got {} and {}".format(len(image_paths),
                                                                                                   len(mask_paths),
                                                                                                   len(mask_paths))
        self.augmentation = augmentation
        self.output_size = to_tuple(output_size, 2)
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.inter = inter

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, ext = os.path.splitext(mask_path)
        mask = cv2.imread(mask_path, 0)
        h, w = image.shape[:2]
        if self.augmentation:
            aug_task = [
                ColorJitter(),
                ShiftScaleRotate(interpolation=self.inter, scale_limit=0.5),
                HorizontalFlip(),
                VerticalFlip(),
                ChannelDropout(),
                GaussianBlur(),
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        out_h, out_w = self.output_size
        hr = None
        if self.super_reso:
            if out_h * self.upscale_rate != h or out_w * self.upscale_rate != w:
                hr_re = Resize(out_h*self.upscale_rate, out_w*self.upscale_rate,
                               interpolation=self.inter)(image=image, mask=mask)
                hr = hr_re["image"]
                mask = hr_re["mask"]
            else:
                hr = image.copy()
        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image, mask=mask)
        image = re_data["image"]
        if hr is None:
            mask = re_data["mask"]
        if image.ndim > 2:
            channel = 3
        else:
            channel = 1
        normalize = Normalize(mean=[0.5]*channel, std=[0.5]*channel)
        # normalize = Normalize(mean=[0.39068785, 0.40521392, 0.41434407],
        #                       std=[0.29652068, 0.30514979, 0.30080369])
        normalize_data = normalize(image=image, mask=mask)
        image = normalize_data["image"]
        mask = normalize_data["mask"]
        if hr is not None:
            hr = normalize(image=hr)["image"]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            if hr is not None:
                hr = np.expand_dims(hr, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
            if hr is not None:
                hr = np.transpose(hr, axes=[2, 0, 1])
        if hr is None:
            return torch.from_numpy(image), torch.from_numpy(mask)
        return torch.from_numpy(image), torch.from_numpy(hr), torch.from_numpy(mask)
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentation
    author: 12718
    time: 2022/1/15 15:22
    tool: PyCharm
"""
import torch
from torch.utils.data import Dataset
import os
import glob
from typing import Union,List
from albumentations import Compose, ColorJitter, HorizontalFlip
from albumentations import ChannelDropout, GaussianBlur, VerticalFlip
from albumentations import Normalize, Resize, ShiftScaleRotate
import cv2
from PIL import Image
import numpy as np

from comm.helper import to_tuple


def get_paths(image_dir:str, mask_dir:str, image_suffix:str=".png", mask_suffix:str=".png"):
    """
    Get the image file paths and mask file paths,
    only used by filename of image file and mask file is the same
    Args:
        image_dir(str): Path for the directory that contain image files.
        mask_dir(str: Path for the directory that contain mask files.
        image_suffix(str): Suffix of image file
        mask_suffix(str): Suffix of mask file

    Returns:
        Tuple[List[str]:List of image file paths, List[str]:List of mask file paths]
    """
    image_paths = glob.glob(os.path.join(image_dir, "*{}".format(image_suffix)))
    mask_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        mask_filename = name + mask_suffix
        mask_path = os.path.join(mask_dir, mask_filename)
        mask_paths.append(mask_path)
    return image_paths, mask_paths


class SegPathDataset(Dataset):
    def __init__(self, image_paths:List[str], mask_paths:List[str],
                 augmentation:bool=False,
                 output_size:Union[int,tuple,list]=512,
                 super_reso:bool=False, upscale_rate:int=2,
                 divide=False, inter=cv2.INTER_CUBIC):
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
        self.divide = divide
        self.inter = inter

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path)
        _, ext = os.path.splitext(mask_path)
        if ext == ".gif":
            mask = Image.open(mask_path)
        else:
            mask = cv2.imread(mask_path, 0)
        if self.divide:
            mask = mask // 255
        h, w = image.shape[:2]
        if self.augmentation:
            aug_task = [
                ChannelDropout(),
                ColorJitter(),
                HorizontalFlip(),
                VerticalFlip(),
                ShiftScaleRotate(),
                GaussianBlur()
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
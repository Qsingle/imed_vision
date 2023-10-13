# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:crossmoda
    author: 12718
    time: 2022/5/16 12:32
    tool: PyCharm
"""
import torch
from typing import Union
from torch.utils.data import Dataset
from imed_vision.comm.helper import _pair
from albumentations import HorizontalFlip, VerticalFlip
from albumentations import Compose, Resize, MotionBlur
from albumentations import GaussianBlur
import numpy as np
import cv2


class CrossMoDA(Dataset):
    def __init__(self, image_paths, mask_paths, augmentation=False, output_size:Union[int, tuple, list]=512,
                 super_reso:bool=False, upscale_rate:int=2, inter=cv2.INTER_CUBIC):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.inter = inter
        self.output_size = _pair(output_size)
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.load(image_path)
        mask = cv2.imread(self.mask_paths[index], 0)
        out_h, out_w = self.output_size
        # image, mask = crop_image_from_gray(image, mask_img=mask)
        if self.augmentation:
            aug_task = [
                # ChannelDropout(),
                # ColorJitter(contrast=0.4, hue=0.1, brightness=0.4, saturation=0.4),
                HorizontalFlip(),
                VerticalFlip(),
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
            # crop = RandomCrop(height=int(out_h * self.upscale_rate), width=int(out_w * self.upscale_rate))
            # crop_data = crop(image=image, mask=mask)
            # image = crop_data["image"]
            # mask = crop_data["mask"]
        h, w = image.shape[:2]

        hr = None
        if self.super_reso:
            if out_h * self.upscale_rate != h or out_w * self.upscale_rate != w:
                hr_re = Resize(out_h * self.upscale_rate, out_w * self.upscale_rate,
                               interpolation=cv2.INTER_CUBIC)(image=image, mask=mask)
                hr = hr_re["image"]
                mask = hr_re["mask"]
            else:
                hr = image.copy()
        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image, mask=mask)
        image = re_data["image"]
        if self.super_reso:
            blur = Compose([
                GaussianBlur(),
                MotionBlur()
            ])
            blur_data = blur(image=image)
            image = blur_data["image"]
        if hr is None:
            mask = re_data["mask"]

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

class CrossMoDATarget(Dataset):
    def __init__(self, image_paths, augmentation=False, output_size:Union[int, tuple, list]=512,
                 super_reso:bool=False, upscale_rate:int=2, inter=cv2.INTER_CUBIC):
        self.image_paths = image_paths
        self.augmentation = augmentation
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.inter = inter
        self.output_size = _pair(output_size)
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.load(image_path)
        out_h, out_w = self.output_size
        # image, mask = crop_image_from_gray(image, mask_img=mask)
        if self.augmentation:
            aug_task = [
                # ChannelDropout(),
                # ColorJitter(contrast=0.4, hue=0.1, brightness=0.4, saturation=0.4),
                HorizontalFlip(),
                VerticalFlip(),
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=image)
            image = aug_data["image"]
            mask = aug_data["mask"]
            # crop = RandomCrop(height=int(out_h * self.upscale_rate), width=int(out_w * self.upscale_rate))
            # crop_data = crop(image=image, mask=mask)
            # image = crop_data["image"]
            # mask = crop_data["mask"]
        h, w = image.shape[:2]

        hr = None
        if self.super_reso:
            if out_h * self.upscale_rate != h or out_w * self.upscale_rate != w:
                hr_re = Resize(out_h * self.upscale_rate, out_w * self.upscale_rate,
                               interpolation=cv2.INTER_CUBIC)(image=image)
                hr = hr_re["image"]
                mask = hr_re["mask"]
            else:
                hr = image.copy()
        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image)
        image = re_data["image"]
        if self.super_reso:
            blur = Compose([
                GaussianBlur(),
                MotionBlur()
            ])
            blur_data = blur(image=image)
            image = blur_data["image"]
        if hr is None:
            mask = re_data["mask"]

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            if hr is not None:
                hr = np.expand_dims(hr, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
            if hr is not None:
                hr = np.transpose(hr, axes=[2, 0, 1])

        if hr is None:
            return torch.from_numpy(image)
        return torch.from_numpy(image), torch.from_numpy(hr)
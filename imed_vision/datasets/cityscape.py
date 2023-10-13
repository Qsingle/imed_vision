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
from albumentations import VerticalFlip, RandomCrop, MotionBlur
from albumentations import GaussianBlur, GaussNoise
from albumentations import Normalize, Resize, ShiftScaleRotate, OneOf, JpegCompression
import cv2
import numpy as np

from imed_vision.comm.helper import to_tuple


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
                 inter=cv2.INTER_CUBIC, crop=False):
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
        self.crop = crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, ext = os.path.splitext(mask_path)
        mask = cv2.imread(mask_path, 0)

        if self.augmentation:
            aug_task = [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ShiftScaleRotate(rotate_limit=45, scale_limit=(0.35, 1.0)),
                HorizontalFlip(),
                VerticalFlip(),
            ]
            if self.crop:
                if self.super_reso:
                    aug_task.append(
                        RandomCrop(self.output_size[0]*self.upscale_rate, self.output_size[1]*self.upscale_rate)
                    )
                else:
                    aug_task.append(
                        RandomCrop(self.output_size[0], self.output_size[1])
                    )
            aug = Compose(aug_task)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        h, w = image.shape[:2]
        out_h, out_w = self.output_size
        hr = None
        if self.super_reso:
            if out_h * self.upscale_rate != h or out_w * self.upscale_rate != w:
                hr_re = Resize(out_h*self.upscale_rate, out_w*self.upscale_rate, always_apply=True)(image=image, mask=mask)
                hr = hr_re["image"]
                mask = hr_re["mask"]
                image = hr.copy()
            else:
                hr = image.copy()
        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image, mask=mask)
        image = re_data["image"]
        if self.augmentation:
            noise = OneOf([
                GaussianBlur(),
                GaussNoise(),
                JpegCompression(quality_lower=90),
                MotionBlur()
            ])
            image = noise(image=image)["image"]
        if hr is None:
            mask = re_data["mask"]
        if image.ndim > 2:
            channel = 3
        else:
            channel = 1
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        # normalize = Normalize(mean=[0.39068785, 0.40521392, 0.41434407],
        #                       std=[0.29652068, 0.30514979, 0.30080369])
        normalize_data = normalize(image=image, mask=mask)
        image = normalize_data["image"]
        # mask = normalize_data["mask"]
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

class CityScapeDepthSegDataset(Dataset):
    def __init__(self, image_paths:List[str], mask_paths:List[str],
                 augmentation:bool=False,
                 output_size:Union[int,tuple,list]=512,
                 super_reso:bool=False, upscale_rate:int=2,
                 inter=cv2.INTER_CUBIC):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        depth_paths = []
        for path in self.image_paths:
            filename = os.path.basename(path).replace("leftImg8bit", "disparity")[:-4] + ".png"
            depth_file = os.path.join(os.path.dirname(path).replace("leftImg8bit", "disparity"), filename)
            depth_paths.append(depth_file)
        self.depth_paths = depth_paths
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
        depth = np.load(self.depth_paths[index])
        # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = depth / 255
        out_h, out_w = self.output_size
        if self.augmentation:
            aug_task = [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # RandomCrop(out_h, out_w),
                # ShiftScaleRotate(rotate_limit=45, scale_limit=(0.35, 1.0)),
                HorizontalFlip(),
                VerticalFlip(),
                GaussNoise()
            ]
            aug = Compose(aug_task, additional_targets={"depth": "mask"})
            aug_data = aug(image=image, mask=mask, depth=depth)
            image = aug_data["image"]
            mask = aug_data["mask"]
            depth = aug_data["depth"]

        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image, mask=mask)
        depth = cv2.resize(depth.astype(np.float32), (out_w, out_h),cv2.INTER_NEAREST)
        image = re_data["image"]
        # if self.augmentation:
        #     noise = GaussianBlur()
        #     image = noise(image=image)["image"]
        mask = re_data["mask"]
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        # normalize = Normalize(mean=[0.39068785, 0.40521392, 0.41434407],
        #                       std=[0.29652068, 0.30514979, 0.30080369])
        normalize_data = normalize(image=image, mask=mask)
        image = normalize_data["image"]
        mask = normalize_data["mask"]
        depth = np.expand_dims(depth, axis=0)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])

        return torch.from_numpy(image), torch.from_numpy(depth), torch.from_numpy(mask)
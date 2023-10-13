# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pfseg_dataset
    author: Zhongxi Qiu
    time: 2022/2/9 17:46
    tool: Visual Studio Code
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from torch.utils.data import Dataset
from albumentations import Compose
from albumentations import ColorJitter, ShiftScaleRotate
from albumentations import GaussianBlur
from albumentations import HorizontalFlip, VerticalFlip, HueSaturationValue
from albumentations import Normalize
from albumentations import Resize
from PIL import Image
import numpy as np
import glob
import cv2
import os


from torch.nn.modules.utils import _pair

def get_paths(image_dir, mask_dir, image_suffix, mask_suffix):
    image_paths = glob.glob(os.path.join(image_dir, "*{}".format(image_suffix)))
    mask_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_dir, name+mask_suffix)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

def drive_get_paths(image_dir, mask_dir, image_suffix=".tif", mask_suffix=".gif"):
    image_paths = glob.glob(os.path.join(image_dir, "*{}".format(image_suffix)))
    mask_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        id = name.split("_")[0]
        mask_name = "{}_manual1{}".format(id, mask_suffix)
        mask_path = os.path.join(mask_dir, mask_name)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

class PFSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, output_size, augmentation=False,
                green_channel=False, divide=False, interpolation=cv2.INTER_LINEAR,
                 upscale_rate=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), origin=False):
        assert len(image_paths) == len(mask_paths), "Length of the image path lists must be equal," \
                                                    "but got len(image_paths)={} and len(mask_paths)={}".format(
            len(image_paths), len(mask_paths)
        )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.green_channel = green_channel
        self.mean = mean
        self.std = std
        if self.green_channel:
            assert len(mean) == 1 and len(std) == 1, "If use the green channel of the image, " \
                                                     "please use the mean and std for the green channel," \
                                                     "except length of mean and std to 1 but got {} and {}".format(len(mean), len(std))
        self.interpolation = interpolation
        self.upscale_rate = upscale_rate
        self.output_size = _pair(output_size)
        self.augmentation = augmentation
        self.length = len(image_paths)
        self.divide = divide
        self.origin_output = origin

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(image_path), "The image file {} is not exists.".format(image_path)
        assert os.path.exists(mask_path), "The mask file {} is not exists.".format(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None
        basename = os.path.basename(mask_path)
        name, ext = os.path.splitext(basename)
        if ext == ".gif":
            mask = Image.open(mask_path)
            mask = np.array(mask)
        else:
            mask = cv2.imread(mask_path, 0)
        assert mask is not None
        if self.divide:
            mask = mask // 255
        height, width = image.shape[:2]
        if self.augmentation:
            aug_tasks = [
                ColorJitter(),
                # HueSaturationValue(),
                ShiftScaleRotate(rotate_limit=45),
                GaussianBlur(),
                HorizontalFlip(),
                VerticalFlip()
            ]
            aug_func = Compose(aug_tasks)
            aug_data = aug_func(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        normalize = Normalize(mean=self.mean, std=self.std)
        out_height = self.output_size[0]
        out_width = self.output_size[1]
        crop_width = out_width // 2
        crop_height = out_height // 2
        if self.origin_output:
            hr = image.copy()
        elif (out_width*self.upscale_rate != width
                                or out_height * self.upscale_rate != height):
            resize = Resize(height=out_height*self.upscale_rate, width=out_width*self.upscale_rate,
                            interpolation=cv2.INTER_CUBIC)
            re_data = resize(image=image, mask=mask)
            mask = re_data["mask"]
            hr = re_data["image"]
        else:
            hr = image.copy()
        resize =  Resize(height=out_height, width=out_width,
                         interpolation=self.interpolation)
        re_data = resize(image=image, mask=mask)
        image = re_data["image"]
        nor_data = normalize(image=image)
        image = nor_data["image"]
        h, w = hr.shape[:2]
        c_x = w // 2
        c_y = h // 2
        hr = normalize(image=hr)["image"]
        guidance = hr[c_y - crop_height // 2:c_y + crop_height//2, c_x - crop_width//2:c_x + crop_width//2, :]
        ps_mask = np.zeros(shape=(h, w), dtype="uint8")
        ps_mask[c_y - crop_height//2:c_y + crop_height//2, c_x - crop_width//2:c_x + crop_width//2] = 1
        ps_mask = np.expand_dims(ps_mask, axis=0)
        if self.green_channel:
            image = image[..., 1]
            hr = hr[..., 1]
            guidance=guidance[..., 1]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            hr = np.expand_dims(image, axis=0)
            guidance = np.expand_dims(guidance, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
            hr = np.transpose(hr, axes=[2, 0, 1])
            guidance = np.transpose(guidance, axes=[2, 0, 1])
        return image, hr, mask, guidance, ps_mask
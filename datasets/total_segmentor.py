#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   total_segmentor.py
    @Time    :   2023/09/03 16:30:52
    @Author  :   12718 
    @Version :   1.0
'''
import os
import json

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from typing import Tuple

from comm.transform import random_crop3d as random_crop, random_flip_3d as random_flip, random_rotate_3d as random_rotate, random_shift_3d as random_shift
from comm.transform import gaussian_noise_3d, poisson_noise_3d

class TotalSegmentor(Dataset):
    def __init__(self, data_root, data_dir="preprocess", file_suffix=".npy",mode="train", augmentation=False, 
                 img_size=[64, 128, 128], super_reso=False,
                 upscale_rate=2) -> None:
        super(TotalSegmentor, self).__init__()
        self.data_root = data_root
        self.data_dir = data_dir
        self.mode = mode
        self.augmentation = augmentation
        self.img_size = img_size
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.file_suffix = file_suffix
        self._get_paths()
    
    def _get_paths(self):
        with open(os.path.join(self.data_root, "dataset.json")) as f:
            dataset_info = json.load(f) 
            mode_infos = dataset_info[self.mode]
            self.mean = dataset_info['mean']
            self.std = dataset_info['std']
            self.img_paths = []
            self.mask_paths = []
            for mode_info in mode_infos:
                image_id = mode_info["image_id"]
                image_path = os.path.join(self.data_root, self.data_dir, image_id, "ct{}".format(self.file_suffix))
                mask_path = os.path.join(self.data_root, self.data_dir, image_id, "seg{}".format(self.file_suffix))
                self.img_paths.append(image_path)
                self.mask_paths.append(mask_path)
            self.length = len(self.img_paths)
    
    def __len__(self):
        return self.length

    def normal(self, img, mean, std):
        percentage_0_5 = np.percentile(img, 0.5)
        percentage_99_5 = np.percentile(img, 99.5)
        img = np.clip(img, percentage_0_5, percentage_99_5)
        img = (img - mean) / np.clip(std, a_min=1e-8, a_max=None)
        return img

    def __getitem__(self, index) -> Tuple:
        vox_path = self.img_paths[index]
        label_path = self.mask_paths[index]
        if self.file_suffix == ".nii.gz":
            vox = np.transpose(nib.load(vox_path).get_fdata(), [2, 0, 1])
            label = np.transpose(nib.load(label_path).get_fdata(), axes=[2, 0, 1])
        elif self.file_suffix == ".npy" or self.file_suffix == ".npz":
            vox = np.transpose(np.load(vox_path), [2, 0, 1])
            label = np.transpose(np.load(label_path), axes=[2, 0, 1])
        else:
            raise ValueError("Unsupported file suffix:{}".format(self.file_suffix))
        if self.augmentation:
            if self.super_reso:
                sr_size = [self.upscale_rate*i  for i in self.img_size]
                if np.random.random() < 0.5:
                    vox, label = random_crop(vox, label, img_size=vox.shape, crop_size=sr_size)
                else:
                    zoom_size = [s/o for s,o in zip(sr_size, vox.shape)]
                    vox = zoom(vox, zoom_size, order=1)
                    label = zoom(label, zoom_size, order=0)
            else:
                if np.random.random() < 0.5:
                    vox, label = random_crop(vox, label, crop_size=self.img_size, img_size=self.img_size)
                else:
                    zoom_size = [s/o for s,o in zip(self.img_size, vox.shape)]
                    vox = zoom(vox, zoom_size, order=1)
                    label = zoom(label, zoom_size, order=0)
            vox, label = random_flip(vox, label)
            vox, label = random_shift(vox, label)
            vox, label = random_rotate(vox, label)
        else:
            if self.super_reso:
                sr_size = [self.upscale_rate*i  for i in self.img_size]
                zoom_size = [s/o for s,o in zip(sr_size, vox.shape)]
                vox = zoom(vox, zoom_size, order=1)
                if self.mode != "test":
                    label = zoom(label, zoom_size, order=0)
            else:
                zoom_size = [s/o for s,o in zip(self.img_size, vox.shape)]
                vox = zoom(vox, zoom_size, order=1)
                if self.mode != "test":
                    label = zoom(label, zoom_size, order=0)
        if self.super_reso:
            sr = vox
            size = [s/o for s, o in zip(self.img_size, vox.shape)]
            vox = zoom(vox, size, order=1)
            # if self.augmentation:
            #     if np.random.randint(0, 1) < 1:
            #         vox = gaussian_noise_3d(vox)
            #     else:
            #         vox = poisson_noise_3d(vox)
            vox = self.normal(vox, mean=self.mean, std=self.std)
            sr = self.normal(sr, mean=self.mean, std=self.std)
            return torch.from_numpy(vox).unsqueeze(0).to(torch.float32), torch.from_numpy(sr).unsqueeze(0).to(torch.float32), torch.from_numpy(label).to(torch.long)
        vox = self.normal(vox, mean=self.mean, std=self.std)
        return torch.from_numpy(vox).unsqueeze(0).to(torch.float32), torch.from_numpy(label).to(torch.long)
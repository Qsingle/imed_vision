#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   sppin.py
    @Time    :   2023/08/30 11:09:15
    @Author  :   12718 
    @Version :   1.0
'''

import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from typing import List, Union
from pathlib import Path
import random
import cv2
import glob
import tqdm

from imed_vision.comm.transform import random_crop3d as random_crop, random_flip_3d as random_flip, random_rotate_3d as random_rotate, random_shift_3d as random_shift

class SPPIN(Dataset):
    def __init__(self, data_dir:Union[str, Path], 
                 img_size:List=[64, 96, 96], crop_size=[64, 96, 96],
                 guide:bool=False, augmentation:bool=False, super_reso=False,
                 upscale_rate=2, mode="train"):
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.img_size = img_size
        self.crop_size = crop_size
        self.guide = guide
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.mode = mode
        self.get_path()
        if self.guide:
            self.guide_img_size = [s // 2 for s in self.crop_size]

    def __len__(self):
        return self.length

    def get_path(self):
        patient_dirs = glob.glob(os.path.join(self.data_dir, "PT_*"))
        train_dirs, val_dirs = train_test_split(patient_dirs, random_state=66,test_size=0.1)
        data_map = {
            "train": train_dirs,
            "val": val_dirs
        }
        self.images = []
        self.labels = []
        bar = tqdm.tqdm(data_map[self.mode])
        mode = self.mode
        for dirname in bar:
            bar.set_description("Processing sample {}".format(os.path.basename(dirname)))
            pt_id = int(os.path.basename(dirname).split("_")[-1])
            t1_paths = glob.glob(os.path.join(dirname, "*", "PT_{:02d}_T1*.nii".format(pt_id)))
            for t1 in t1_paths:
                pre_dir = os.path.basename(os.path.dirname(t1))
                m_path = glob.glob(os.path.join(dirname, pre_dir, "*NB*"))[0]
                vox = sitk.GetArrayFromImage(sitk.ReadImage(t1))
                label = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)
                vox_shape = vox.shape
                img_size = [s / v for s, v in zip(self.img_size, vox_shape)]
                if self.super_reso:
                    vox = self.resize(vox, [self.img_size[0]*self.upscale_rate/vox_shape[0], self.img_size[1]*self.upscale_rate/vox_shape[1], self.img_size[1]*self.upscale_rate/vox_shape[2]])
                    if mode != "test":
                        label = self.resize(label, [self.img_size[0]*self.upscale_rate/vox_shape[0],  self.img_size[1]*self.upscale_rate/vox_shape[1], self.img_size[1]*self.upscale_rate/vox_shape[2]], order=0)
                else:
                    vox = self.resize(vox, img_size)
                    if mode != "test":
                        label = self.resize(label, img_size, order=0)
                label[label >= 0.9] = 1
                label[label < 0.9] = 0
                self.images.append(self.normal(vox))
                self.labels.append(label)
        self.length = len(self.images)

    def crop_guidance(self, label_sr):
        D, M, N = label_sr.shape
        D = int(D / 2)
        M = int(M / 2)
        N = int(N / 2)

        mask = np.zeros(label_sr.shape).astype(np.float64)
        mask[D - self.guide_img_size[0] // 2:D + self.guide_img_size[0] // 2,
        M - self.guide_img_size[1] // 2:M + self.guide_img_size[1] // 2,
        N - self.guide_img_size[2] // 2:N + self.guide_img_size[2] // 2] = 1
        return label_sr[D - self.guide_img_size[0] // 2:D + self.guide_img_size[0] // 2,
               M - self.guide_img_size[1] // 2:M + self.guide_img_size[1] // 2,
               N - self.guide_img_size[2] // 2:N + self.guide_img_size[2] // 2], mask

    def resize(self, img, img_size, order=1):
        return zoom(img, img_size, order=order)

    def normal(self, img):
        percentage_0_5 = np.percentile(img, 0.5)
        percentage_99_5 = np.percentile(img, 99.5)
        img = np.clip(img, percentage_0_5, percentage_99_5)
        min_value = img.min()
        max_value = img.max()
        img = (img - min_value) / (max_value - min_value + 1e-9)
        return img
    
    def __getitem__(self, index):
        vox = self.images[index]
        label = self.labels[index]
        # vox = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[index]))
        # label = sitk.GetArrayFromImage(sitk.ReadImage(self.mask_paths[index]))
        # vox = self.normal(vox)
        vox_sr = None
        if self.augmentation:
            if self.super_reso:
                vox, label = random_crop(vox, label, vox.shape, self.crop_size)
            vox, label = random_flip(vox, label)
            vox, label = random_shift(vox, label)
            vox, label = random_rotate(vox, label)
            # vox, label = random_crop(vox, label, vox.shape, self.crop_size)
        if self.super_reso:
            vox_sr = vox
            if self.mode == "train":
                _img_size = [1/self.upscale_rate, 1 / self.upscale_rate, 1/self.upscale_rate]
            else:
                _img_size = [s/v for s,v in zip(self.img_size, vox.shape)]
            vox = self.resize(vox, _img_size)
        
        label[label >= 0.9] = 1
        label[label < 0.9] = 0
        if self.super_reso:
            vox = np.expand_dims(vox, axis=0).astype(float)
            label = np.expand_dims(label, axis=0).astype(np.int32)
            vox_sr = np.expand_dims(vox_sr, axis=0).astype(float)
            if self.guide:
                guide, guide_mask = self.crop_guidance(vox_sr)
                guide = np.expand_dims(guide, axis=0)
                guide_mask = np.expand_dims(guide_mask, axis=0)
                return vox, vox_sr, label, guide, guide_mask
            return vox, vox_sr, label
        vox = np.expand_dims(vox, axis=0)
        label = np.expand_dims(label, axis=0)
        return vox, label
# -*- coding:utf-8 -*-
"""
    FileName: atm
    Author: 12718
    Create Time: 2023-06-16 09:10
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
from typing import List, Union
from pathlib import Path
import random
import cv2

from comm.transform import random_crop3d as random_crop, random_flip_3d as random_flip, random_rotate_3d as random_rotate, random_shift_3d as random_shift

class ATM(Dataset):
    def __init__(self, data_dir:Union[str, Path], txt_path:Union[str, Path],
                 img_size:List=[64, 96, 96], crop_size=[64, 96, 96],
                 guide:bool=False, augmentation:bool=False, super_reso=False,
                 upscale_rate=2):
        self.data_dir = data_dir
        self.txt_path = txt_path
        self.augmentation = augmentation
        self.img_size = img_size
        self.crop_size = crop_size
        self.guide = guide
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.get_path()
        if self.guide:
            self.guide_img_size = [s // 2 for s in self.crop_size]

    def __len__(self):
        return len(self.paths)
    
    def reset_spacing(self, itkimage, resamplemethod=sitk.sitkBSpline, newSpacing=[1.5, 1.5, 1.5]):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # original spacing
        originSpacing = itkimage.GetSpacing()
        newSpacing = np.array(newSpacing, float)
        factor = originSpacing / newSpacing
        newSize = originSize * factor
        newSize = newSize.astype(np.int)  # spacing is integer
        resampler.SetReferenceImage(itkimage)  # the target image
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # resampled image
        return itkimgResampled

    def get_path(self):
        with open(self.txt_path, "r") as f:
            lines = f.readlines()
            self.paths = [os.path.join(self.data_dir, "imagesTr", n.strip()) for n in lines]
            self.mask_paths = [os.path.join(self.data_dir, "labelsTr", n.strip()) for n in lines]
        self.images = []
        self.labels = []
        # if self.super_reso:
        #     self.vox_sr = []
        mode = os.path.splitext(os.path.basename(self.txt_path))[0]
        self.mode = mode
        for path, label_path in zip(self.paths, self.mask_paths):
            print("Adding {} sample {}".format(mode, path))
            data_path = path
            vox = sitk.ReadImage(data_path)
            vox = sitk.GetArrayFromImage(vox)
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label).astype(np.uint8)
            vox_shape = vox.shape
            img_size = [s / v for s, v in zip(self.img_size, vox_shape)]
            if self.super_reso:
                vox = self.resize(vox, [self.img_size[0]*self.upscale_rate/vox_shape[0], self.img_size[1]*self.upscale_rate/vox_shape[1], self.img_size[1]*self.upscale_rate/vox_shape[2]])
        #         vox_sr = self.normal(vox_sr)
        #         self.vox_sr.append(vox_sr)
        #         vox = self.resize(vox, img_size)
                if mode != "test":
                    label = self.resize(label, [self.img_size[0]*self.upscale_rate/vox_shape[0],  self.img_size[1]*self.upscale_rate/vox_shape[1], self.img_size[1]*self.upscale_rate/vox_shape[2]], order=0)
            else:
                vox = self.resize(vox, img_size)
                if mode != "test":
                    label = self.resize(label, img_size, order=0)
        #     print(label.shape)
            vox = self.normal(vox)
            # self.images.append(vox)
            label[label >= 0.9] = 1
            label[label < 0.9] = 0
            self.images.append(vox)
            self.labels.append(label)

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

    def normal(self, img, wc=-1000, ww=500):
        img[img > ww] = ww
        img[img < wc] = wc
        img = img + 1000
        img = img / 1500
        # img = (img - min_value) / (max_value - min_value + 1e-9)
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

if __name__ == "__main__":
    img, label = random_shift(np.random.randn(64, 128, 128), np.random.randn(256, 512, 512), upscale=4)
    print(label.shape)
# -*- coding:utf-8 -*-
"""
    FileName: brats
    Author: 12718
    Create Time: 2023-06-25 16:20
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
import glob

def random_flip(img, label=None, p=0.5, sr_img=None):
    if random.random() < p:
        if random.random() < p:
            for i in range(img.shape[0]):
                img[i, :, :] = cv2.flip(img[i, :, :], 0)
                if not (label is None):
                    label[i, :, :] = cv2.flip(label[i, :, :], 0)
                if not (sr_img is None):
                    sr_img[i, :, :] = cv2.flip(sr_img[i, :, :], 0)
        else:
            for i in range(img.shape[0]):
                img[i, :, :] = cv2.flip(img[i, :, :], 1)
                if not (label is None):
                    label[i, :, :] = cv2.flip(label[i, :, :], 1)
                if not (sr_img is None):
                    sr_img[i, :, :] = cv2.flip(sr_img[i, :, :], 0)
    if sr_img is not None:
        return img, label, sr_img
    return img, label

def random_shift(img, label, p=0.5, sr_img=None):
    if random.random()< p:  #Shift
        vertical = np.random.randint(-img.shape[1] // 8, img.shape[1] // 8)
        horizon = np.random.randint(-img.shape[1] // 8, img.shape[1] // 8)
        M_img = np.float32([[0, 1, horizon], [1, 0, vertical]])
        M_label = np.float32([[0, 1, 2 * horizon], [1, 0, 2 * vertical]])
        for i in range(img.shape[0]):
            img[i, :, :] = cv2.warpAffine(img[i, :, :], M_img, (img.shape[1], img.shape[2]))
        for i in range(label.shape[0]):
            label[i, :, :] = cv2.warpAffine(label[i, :, :], M_label, (label.shape[1], label.shape[2]))
        if sr_img is not None:
            sr_img[i, :, :] = cv2.warpAffine(sr_img[i, :, :], M_label, (label.shape[1], label.shape[2]))
    if sr_img is not None:
        return img, label, sr_img
    return img, label

def random_rotate(img, label, p=0.5, sr_img=None):
    if random.random() < p:  #Shift
        degree=np.random.randint(0,360)
        M_img = cv2.getRotationMatrix2D(((img.shape[1]-1)/2.0,(img.shape[2]-1)/2.0),degree,1)
        M_label=cv2.getRotationMatrix2D(((label.shape[1]-1)/2.0,(label.shape[2]-1)/2.0),degree,1)
        for i in range(img.shape[0]):
            img[i,:,:]=cv2.warpAffine(img[i,:,:], M_img, (img.shape[1],img.shape[2]))
        for i in range(label.shape[0]):
            label[i, :, :] = cv2.warpAffine(label[i, :, :], M_label, (label.shape[1], label.shape[2]))
            if sr_img is not None:
                sr_img[i, :, :] = cv2.warpAffine(sr_img[i, :, :], M_label, (label.shape[1], label.shape[2]))
    if sr_img is not None:
        return img, label, sr_img
    return img, label

def random_crop(img, label, img_size, crop_size, upscale_rate=1, sr_img=None):
    start_z = random.randint(0, img_size[0] - crop_size[0])
    start_x = random.randint(0, img_size[1] - crop_size[1])
    start_y = random.randint(0, img_size[2] - crop_size[2])
    crop_img = img[start_z:(start_z + crop_size[0]), start_x:(start_x + crop_size[1]), start_y:(start_y + crop_size[2])]
    crop_label = label[upscale_rate*start_z:upscale_rate*(start_z+crop_size[0]),
                 upscale_rate*start_x:upscale_rate*(start_x+crop_size[1]),upscale_rate*start_y:upscale_rate*(start_y+crop_size[2])]
    if sr_img is not None:
        crop_sr = sr_img[upscale_rate*start_z:upscale_rate*(start_z+crop_size[0]),
                 upscale_rate*start_x:upscale_rate*(start_x+crop_size[1]),upscale_rate*start_y:upscale_rate*(start_y+crop_size[2])]
        return crop_img, crop_label, crop_sr
    return crop_img, crop_label

class Brats20(Dataset):
    def __init__(self, data_dir:Union[str, Path], mode="train",
                 img_size:List=[64, 96, 96], crop_size=[64, 96, 96],
                 guide:bool=False, augmentation:bool=False, super_reso=False,
                 upscale_rate=2):
        self.data_dir = data_dir
        self.mode = mode
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
        return self.length
    def reset_spacing(self, itkimage, resamplemethod=sitk.sitkBSpline, newSpacing=[1.5, 1.5, 1.5]):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSpacing = np.array(newSpacing, float)
        factor = originSpacing / newSpacing
        newSize = originSize * factor
        newSize = newSize.astype(np.int)  # spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return itkimgResampled

    def get_path(self):
        images = sorted(glob.glob(os.path.join(self.data_dir, "*/*/*_t2.nii")))
        labels = sorted(glob.glob(os.path.join(self.data_dir, "*/*/*_seg.nii")))
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        n_train = int(train_frac * len(images)) + 1
        n_val = int(val_frac * len(images)) + 1
        n_test = min(len(images) - n_train - n_val, int(test_frac * len(images)))
        self.images = []
        self.labels = []
        if self.super_reso:
            self.img_sr = []
        self.length = 0
        if self.mode == "train":
            images = images[:n_train]
            labels = labels[:n_train]
            self.length = len(images)
            print("{} training sample".format(len(images)))
            for i in range(len(images)):
                print('Adding train sample:', images[i])
                image = sitk.ReadImage(images[i])
                image_arr = sitk.GetArrayFromImage(image)
                lesion = sitk.ReadImage(labels[i])
                lesion_arr = sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr > 1] = 1  # 只做WT分割
                image_arr, lesion_arr = self.center_crop(image_arr, lesion_arr)
                if self.super_reso:
                    img_sr = self.normal(self.resize(image_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1]))
                    image_arr = self.normal(self.resize(image_arr, [self.img_size[0] / 155, 1 / self.upscale_rate, 1 / self.upscale_rate]))
                    lesion_arr = self.resize(lesion_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1])
                    self.img_sr.append(img_sr)
                else:
                    image_arr = self.normal(self.resize(image_arr, [self.img_size[0] / 155, 1, 1]))
                    lesion_arr = self.resize(lesion_arr, [self.img_size[0] / 155, 1, 1], order=0)
                lesion_arr[lesion_arr < 0.5] = 0
                lesion_arr[lesion_arr >= 0.5] = 1
                self.images.append(image_arr)
                self.labels.append(lesion_arr)
        elif self.mode == "val":
            images = images[n_train:n_train+n_val]
            labels = labels[n_train:n_train+n_val]
            self.length = len(images)
            print("{} validation sample".format(n_val))
            for i in range(len(images)):
                print('Adding validation sample:', images[i])
                image = sitk.ReadImage(images[i])
                image_arr = sitk.GetArrayFromImage(image)
                lesion = sitk.ReadImage(labels[i])
                lesion_arr = sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr > 1] = 1  # 只做WT分割
                image_arr, lesion_arr = self.center_crop(image_arr, lesion_arr)
                if self.super_reso:
                    img_sr = self.normal(self.resize(image_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1]))
                    image_arr = self.normal(self.resize(image_arr, [self.img_size[0] / 155, 1/self.upscale_rate, 1 / self.upscale_rate]))
                    lesion_arr = self.resize(lesion_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1], order=0)
                    self.img_sr.append(img_sr)
                else:
                    image_arr = self.normal(self.resize(image_arr, [self.img_size[0] / 155, 1, 1]))
                    lesion_arr = self.resize(lesion_arr, [self.img_size[0] / 155, 1, 1], order=0)
                lesion_arr[lesion_arr < 0.5] = 0
                lesion_arr[lesion_arr >= 0.5] = 1
                self.images.append(image_arr)
                self.labels.append(lesion_arr)
        elif self.mode == "test":
            images = images[n_train + n_val:n_train + n_val + n_test]
            labels = labels[n_train + n_val:n_train + n_val + n_test]
            self.length = len(images) - 1
            self.original_shape = []
            self.paths = []
            print("{} test sample".format(n_test))
            for i in range(len(images)):
                print('Adding test sample:', images[i])
                self.paths.append(images[i])
                image = sitk.ReadImage(images[i])
                image_arr = sitk.GetArrayFromImage(image)
                lesion = sitk.ReadImage(labels[i])
                lesion_arr = sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr > 1] = 1  # 只做WT分割
                image_arr, lesion_arr = self.center_crop(image_arr, lesion_arr)
                self.original_shape.append(lesion_arr.shape)
                if self.super_reso:
                    img_sr = self.normal(self.resize(image_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1]))
                    self.img_sr.append(img_sr)
                    image_arr = self.normal(
                        self.resize(image_arr, [self.img_size[0] / 155, 1 / self.upscale_rate, 1 / self.upscale_rate]))
                    # lesion_arr = self.resize(lesion_arr, [self.upscale_rate * self.img_size[0] / 155, 1, 1], order=0)
                else:
                    image_arr = self.normal(self.resize(image_arr, [self.img_size[0] / 155, 1, 1]))
                    # lesion_arr = self.resize(lesion_arr, [self.img_size[0] / 155, 1, 1], order=0)
                # lesion_arr[lesion_arr < 0.5] = 0
                # lesion_arr[lesion_arr >= 0.5] = 1
                self.images.append(image_arr)
                self.labels.append(lesion_arr)

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

    def center_crop(self, img, mask):
        img = img[:, 24:-24, 24:-24]
        mask = mask[:, 24:-24, 24:-24]
        return img, mask

    def normal(self, img):
        min_value = img.min()
        max_value = img.max()
        img = 1.0*(img - min_value) / (max_value - min_value + 1e-9)
        return img
    
    def __getitem__(self, index):
        vox = self.images[index]
        label = self.labels[index]
        vox_sr = None
        if self.super_reso:
            vox_sr = self.img_sr[index]
        if self.augmentation:
            upscale_rate = self.upscale_rate if self.super_reso else 1
            if self.super_reso:
                vox, label, vox_sr = random_flip(vox, label, sr_img=vox_sr)
                vox, label, vox_sr = random_shift(vox, label, sr_img=vox_sr)
                vox, label, vox_sr = random_rotate(vox, label, sr_img=vox_sr)
                vox, label, vox_sr = random_crop(vox, label, self.img_size, self.crop_size, upscale_rate, sr_img=vox_sr)
            else:
                vox, label = random_flip(vox, label)
                vox, label = random_shift(vox, label)
                vox, label = random_rotate(vox, label)
                vox, label = random_crop(vox, label, self.img_size, self.crop_size, upscale_rate)
        label[label < 0.5] = 0
        label[label >= 0.5] = 1
        if self.super_reso:
            vox = np.expand_dims(vox, axis=0).astype(float)
            label = np.expand_dims(label, axis=0).astype(np.uint8)
            vox_sr = np.expand_dims(vox_sr, axis=0).astype(float)
            if self.guide:
                guide, guide_mask = self.crop_guidance(self.img_sr[index])
                guide = np.expand_dims(guide, axis=0)
                guide_mask = np.expand_dims(guide_mask, axis=0)
                return vox, vox_sr, label, guide, guide_mask
            return vox, vox_sr, label
        vox = np.expand_dims(vox, axis=0).astype(float)
        label = np.expand_dims(label, axis=0).astype(np.uint8)
        return vox, label
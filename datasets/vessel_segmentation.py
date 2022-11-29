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
from albumentations import GaussianBlur, VerticalFlip, RandomCrop, MotionBlur
from albumentations import Normalize, Resize, ShiftScaleRotate, CLAHE
import cv2
from PIL import Image
import numpy as np

from comm.helper import to_tuple


def crop_image_from_gray(img, tol=7, mask_img=None):
    """
    References:https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy/notebook
    Args:
        img (ndarray): image
        tol (floa): threshold

    Returns:
        image after croped
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("gray", gray_img)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            if mask_img is not None:
                mask_img = mask_img[np.ix_(mask.any(1), mask.any(0))]
                return img, mask_img
        return img

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
                 crop=False, divide=False, inter=cv2.INTER_CUBIC,
                 mask_value=None
                 ):
        """

        Args:
            image_paths:
            mask_paths:
            augmentation:
            output_size:
            super_reso:
            upscale_rate:
            crop:
            divide:
            inter:
        """
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
        self.crop = crop
        self.mask_value = mask_value


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path)
        img_ext = os.path.splitext(image_path)[1]
        normalize = True
        if img_ext == ".npy":
            image = np.load(image_path)
            image = np.float32(image)
            normalize = False
        _, ext = os.path.splitext(mask_path)
        if ext == ".gif":
            mask = Image.open(mask_path)
        else:
            mask = cv2.imread(mask_path, 0)
        if self.divide:
            mask = mask // 255
        if self.mask_value is not None:
            mask[mask == self.mask_value] = 0
        out_h, out_w = self.output_size
        #image, mask = crop_image_from_gray(image, mask_img=mask)
        if self.augmentation:
            aug_task = [
                # CLAHE(),
                ColorJitter(contrast=0.4, hue=0.1, brightness=0.4, saturation=0.4),
                # ColorJitter(),
                HorizontalFlip(),
                VerticalFlip(),
                ShiftScaleRotate(rotate_limit=60, scale_limit=0, shift_limit=0.1),
            ]
            aug = Compose(aug_task)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
            if self.crop:
                crop = RandomCrop(height=int(out_h * self.upscale_rate), width=int(out_w * self.upscale_rate))
                crop_data = crop(image=image, mask=mask)
                image = crop_data["image"]
                mask = crop_data["mask"]
        h, w = image.shape[:2]
        hr = None
        if self.super_reso:
            if out_h * self.upscale_rate != h or out_w * self.upscale_rate != w:
                hr_re = Resize(out_h*self.upscale_rate, out_w*self.upscale_rate,
                               interpolation=cv2.INTER_CUBIC)(image=image, mask=mask)
                hr = hr_re["image"]
                mask = hr_re["mask"]
            else:
                hr = image.copy()

        re = Resize(out_h, out_w, interpolation=self.inter)
        re_data = re(image=image, mask=mask)
        image = re_data["image"]
        # gaussian_data = gaussian(image=image)
        # image = gaussian_data["image"]
        # if self.super_reso:
        #     blur = Compose([
        #         GaussianBlur(),
        #         MotionBlur(),
        #         CLAHE()
        #     ])
        #     blur_data = blur(image=image)
        #     image = blur_data["image"]
        if hr is None:
            mask = re_data["mask"]
        if image.ndim > 2:
            channel = 3
        else:
            channel = 1
        if normalize:
            normalize = Normalize(mean=[0.5] * channel, std=[0.5] * channel)
            normalize_data = normalize(image=image, mask=mask)
            image = normalize_data["image"]
            mask = normalize_data["mask"]
        if hr is not None and normalize:
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

if __name__ == "__main__":
    image = cv2.imread(r"D:\workspace\datasets\segmentation\DDR dataset\DDR-dataset\lesion_segmentation\test\image\007-2808-100.jpg")
    croped = crop_image_from_gray(image, tol=18)
    cv2.imshow("test", croped)
    cv2.waitKey(0)
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:kits
    author: 12718
    time: 2022/6/15 15:47
    tool: PyCharm
"""
import os
import glob

import torch
from torch.utils.data import Dataset
import numpy as np


def get_kits19(data_range, data_dir):
    """
    Get the paths of the data and mask
    Args:
        data_range (list): list of data id
        data_dir (str): path of the data directory

    Returns:
        list,list
    """
    image_paths = []
    mask_paths = []
    for did in data_range:
        local_paths = glob.glob(os.path.join(data_dir, "case_{:05d}".format(did), "*.npy"))
        local_mask_paths = [os.path.join(os.path.dirname(path), "mask", os.path.basename(path)[:-4]+".png")
                            for path in local_paths]
        image_paths += local_paths
        mask_paths += local_mask_paths
    return image_paths, mask_paths

def flip_3d(data, mask=None, axes=[0, 1, 2]):
    """
    Random flip
    Args:
        data (ndarray): the data array
        mask (ndarray): the mask data
        axes (Union[tuple, list]): the axes to flip

    Returns:

    """
    assert len(data.shape) == 3, "Invalid " \
                                                         "dimension for data." \
                                                         "The dimension for data" \
                                                         "should be either [x,y,z] or" \
                                                         " [channel, x, y, z]"
    if data.ndim == 4 and min(axes) == 0 and len(axes)==3 and max(axes) < 4:
        axes = [i+1 for i in axes]
    elif data.ndim == 4 and min(axes) == 0 and max(axes) < 4:
        axes = [i+1 for i in axes if i == 0] + [i for i in axes if i != 0]
        axes = np.unique(axes)
    elif max(axes) > data.ndim:
        raise ValueError("The axis greater than dimension is not supported")

    if data.ndim == 4:
        if 1 in axes and np.random.uniform() < 0.5:
            data = data[:, ::-1, :, :]
            if mask is not None:
                mask = mask[:, ::-1, :, :]
        if 2 in axes and np.random.uniform() < 0.5:
            data = data[:, :, ::-1, :]
            if mask is not None:
                mask = mask[:, :, ::-1, :]
        if 3 in axes and np.random.uniform() < 0.5:
            data = data[:, :, :, ::-1]
            if mask is not None:
                mask = mask[:, :, :, ::-1]
    elif data.ndim == 3:
        if 0 in axes and np.random.uniform() < 0.5:
            data = data[::-1, :, :]
            if mask is not None:
                mask = mask[::-1, :, :]
        if 1 in axes and np.random.uniform() < 0.5:
            data = data[:, ::-1, :]
            if mask is not None:
                mask = mask[:, ::-1, :]
        if 2 in axes and np.random.uniform() < 0.5:
            data = data[:, :, ::-1]
            if mask is not None:
                mask = mask[:, :, ::-1]
    data = np.ascontiguousarray(data)
    if mask is not None:
        mask = np.ascontiguousarray(mask)
    return data, mask

class Kits3D(Dataset):
    def __init__(self, data_dir, data_idxs, augmentation=False, crop_size=[64, 64, 64]):
        """
        3D sample dataset.
        Args:
            data_dir (str): Path for the data directory
            data_idxs (list): indexes of the data
            augmentation (bool): whether use augmentation, default: False
            crop_size (Union[tuple,list]): crop size
        """
        self.data_dir = data_dir
        self.data_lists = self._getfile_list(self.data_dir, data_idxs)
        self.length = len(self.data_lists)
        self.crop_size = crop_size
        self.augmentation = augmentation

    def _getfile_list(self, data_dir, data_idxs):
        data_lists = []
        for idx in data_idxs:
            image_path = os.path.join(data_dir, "case_{:05d}".format(idx), "image.npy")
            mask_path = os.path.join(data_dir, "case_{:05d}".format(idx), "mask.npy")
            data_lists.append((image_path, mask_path))
        return data_lists

    def __getitem__(self, index):
        image_path, label_path = self.data_lists[index]
        data = np.load(image_path)
        mask = np.load(label_path)
        if self.augmentation:
            if self.crop_size[0] > data.shape[0]:
                size = data.shape[0]
                patch_z = self.crop_size[0]
                need_pad = patch_z-size
                data = np.pad(data, [(need_pad//2+1, need_pad//2+1), (0, 0), (0, 0)])
                mask = np.pad(mask, [(need_pad//2+1, need_pad//2+1), (0, 0), (0, 0)])
            lb_z = np.random.randint(0, data.shape[0])
            if lb_z + self.crop_size[0] > data.shape[0]:
                lb_z = lb_z - (lb_z + self.crop_size[0] - data.shape[0])
            ub_z = lb_z + self.crop_size[0]
            lb_y = np.random.randint(0, data.shape[1])
            if lb_y + self.crop_size[1] > data.shape[1]:
                lb_y = lb_y - (lb_y + self.crop_size[1] - data.shape[1])
            ub_y = lb_y + self.crop_size[1]
            lb_x = np.random.randint(0, data.shape[2])
            if lb_x + self.crop_size[2] > data.shape[2]:
                lb_x = lb_x - (lb_x + self.crop_size[2] - data.shape[2])
            ub_x = lb_x + self.crop_size[2]
            data = data[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x]
            mask = mask[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x]
            # print(data.shape)
            data, mask = flip_3d(data, mask)
            # print(data.shape)
        data = np.expand_dims(data, axis=0)
        return torch.from_numpy(data), torch.from_numpy(mask)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    img_paths, mask_paths = get_kits19([1, 2, 9], "D:/workspace/datasets/segmentation/kits19/preprocess")
    print(img_paths)
    print(mask_paths)

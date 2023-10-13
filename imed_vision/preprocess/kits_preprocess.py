# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:kits_preprocess
    author: 12718
    time: 2022/6/15 11:05
    tool: PyCharm
"""
import nibabel as nib
import os
import glob
import argparse
import tqdm
import numpy as np
import cv2

parser = argparse.ArgumentParser("CrossMDA dataset preprocess")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the dataset")
parser.add_argument("--output_dir", default=None,
                    help="Path to store the preprocessed data, default: data_dir/preprocessed")
parser.add_argument("--mode", type=str, default="3d")

def list_files(data_dir):
    """
    List the paths of the nii.gz files
    Args:
        data_dir (str): Path to the dataset
    Returns:
        List: paths
    """
    dir_names = glob.glob(os.path.join(data_dir, "case_*[0-9]"))
    nii_paths = []
    mask_paths = []
    for dir_name in dir_names:
        nii_paths.append(os.path.join(dir_name, "imaging.nii.gz"))
        mask_paths.append(os.path.join(dir_name, "segmentation.nii.gz"))
    return nii_paths, mask_paths

def nii_split(image, output_dir, label=None):
    """
    Split the 3D file to 2d images
    Args:
        image (ndarray): 3d image data, shape[x, y, z]
        output_dir (str): path to store the data
        label (ndarray): optional, 3d label data, shape [x, y, z]

    Returns:
        None
    """
    for i in range(image.shape[0]):
        img = image[i, ...]
        np.save(os.path.join(output_dir, "{}.npy".format(i)), img)
        if label is not None:
            mask = label[i, ...]
            cv2.imwrite(os.path.join(output_dir, "mask", "{}.png".format(i)), mask.astype(np.uint8))

def _get_voxels_in_foreground(voxel, mask_data):
    mask_voxel_coords = np.where(mask_data != 0)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    croped = voxel[minzidx:maxzidx, minyidx:maxyidx, minxidx:maxxidx]
    return croped

def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_mode = args.mode
    if output_dir is None:
        output_dir = os.path.join(data_dir, "preprocessed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_files, mask_files = list_files(data_dir)
    bar = tqdm.tqdm(range(len(image_files)))
    zs = []
    for i in bar:
        image_filepath = image_files[i]
        filename = os.path.basename(os.path.dirname(image_filepath))
        sub_output_dir = os.path.join(output_dir, filename)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
            if output_mode == "2d":
                os.makedirs(os.path.join(sub_output_dir, "mask"))
        image_data = nib.load(image_filepath).get_fdata()
        zs.append(image_data.shape[0])
        percentage_0_5 = np.percentile(image_data, 0.5)
        percentage_99_5 = np.percentile(image_data, 99.5)

        image_data = np.clip(image_data, percentage_0_5, percentage_99_5)
        image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-9)
        mask_data = None
        if len(mask_files) > 0:
            mask_data = nib.load(mask_files[i]).get_fdata()
            # labels, labels_num = label(mask_data, return_num=True)
        if output_mode == "2d":
            nii_split(image_data, sub_output_dir, label=mask_data)
        elif output_mode == "3d":
            np.save(os.path.join(sub_output_dir, "image.npy"), image_data)
            if mask_data is not None:
                np.save(os.path.join(sub_output_dir, "mask.npy"), mask_data)
    print(max(zs), " ", min(zs))

if __name__ == "__main__":
    main()
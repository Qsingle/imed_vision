# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:crossmda_preprocess
    author: 12718
    time: 2022/5/16 9:56
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
parser.add_argument("--data_type", type=str, default="source",
                    help="type of the preprocess data, source data or target data")
parser.add_argument("--output_dir", default=None,
                    help="Path to store the preprocessed data, default: data_dir/preprocessed")

def list_files(data_dir, data_type="source"):
    """
    List the paths of the nii.gz files
    Args:
        data_dir (str): Path to the dataset
        data_type (str): type of the data, Choices [source, target]. Default:source

    Returns:
        List: paths
    """
    if data_type == "source":
        dtype = "T1"
    elif data_type == "target":
        dtype = "T2"
    else:
        raise ValueError("Unknown data type:{}, please make sure the data type is target or source")
    image_paths = glob.glob(os.path.join(data_dir, "training_{}".format(data_type),
                                         "*{}.nii.gz".format(dtype)))
    if data_type == "source":
        label_paths = [p.replace("ceT1", "Label") for p in image_paths]
        return image_paths, label_paths
    return image_paths

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
    for i in range(image.shape[-1]):
        img = image[:, :, i]
        np.save(os.path.join(output_dir, "{}.npy".format(i)), img)
        if label is not None:
            mask = label[:, :, i]
            cv2.imwrite(os.path.join(output_dir, "mask", "{}.png".format(i)), mask.astype(np.uint8))

def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    data_type = args.data_type
    if output_dir is None:
        output_dir = os.path.join(data_dir, "preprocessed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = list_files(data_dir, data_type)
    mask_files = None
    if data_type == "source":
        image_files, mask_files = files
    elif data_type == "target":
        image_files = files
    else:
        raise ValueError("Unknown data type:{}, please make sure the data type is target or source")
    bar = tqdm.tqdm(range(len(image_files)))
    for i in bar:
        image_filepath = image_files[i]
        filename = os.path.basename(os.path.splitext(image_filepath)[0].split(".")[0])
        sub_output_dir = os.path.join(output_dir, data_type, filename)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
            if data_type == "source":
                os.makedirs(os.path.join(sub_output_dir, "mask"))
        image_data = nib.load(image_filepath).get_fdata()
        image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-6)
        mask_data = None
        if mask_files is not None:
            mask_data = nib.load(mask_files[i]).get_fdata()
        nii_split(image_data, sub_output_dir, label=mask_data)

if __name__ == "__main__":
    main()
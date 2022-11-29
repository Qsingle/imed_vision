# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:crop_fp
    author: 12718
    time: 2022/7/18 11:53
    tool: PyCharm
"""
import os
import cv2

data_dir = "D:/workspace/MachineLearning/MegVision/data/prime-fp20/train"
image_dir = os.path.join(data_dir, "images")
mask_dir = os.path.join(data_dir, "labels")
image_crop_dir = os.path.join(data_dir, "image_crop")
mask_crop_dir = os.path.join(data_dir, "label_crop")
patch_size = 512
if not os.path.join(image_crop_dir):
    os.makedirs(image_crop_dir)

if not os.path.join(mask_crop_dir):
    os.makedirs(mask_crop_dir)

file_lists = os.listdir(image_dir)
for filename in file_lists:
    fid = filename[5:7]
    image_crop_sub = os.path.join(image_crop_dir, fid)
    if not os.path.exists(image_crop_sub):
        os.makedirs(image_crop_sub)
    mask_crop_sub = os.path.join(mask_crop_dir, fid)
    if not os.path.exists(mask_crop_sub):
        os.makedirs(mask_crop_sub)
    image = cv2.imread(os.path.join(image_dir, filename))
    mask = cv2.imread(os.path.join(mask_dir, filename), 0)
    start_y = 0
    while start_y < image.shape[0]:
        end_y = start_y + patch_size
        if end_y > image.shape[0]:
            end_y = image.shape[0]
            start_y = image.shape[0] - patch_size
        start_x = 0
        while start_x < image.shape[1]:
            end_x = start_x + patch_size
            if end_x > image.shape[1]:
                end_x = image.shape[1]
                start_x = image.shape[1] - patch_size
            image_crop = image[start_y:end_y, start_x:end_x]
            mask_crop = mask[start_y:end_y, start_x:end_x]
            cv2.imwrite(os.path.join(image_crop_sub, "{}_{}_{}_{}_{}.png".format(filename[:-4],
                                                        start_x, end_x, start_y, end_y)), image_crop)
            cv2.imwrite(os.path.join(mask_crop_sub, "{}_{}_{}_{}_{}.png".format(filename[:-4],
                                                                                 start_x, end_x, start_y, end_y)),
                        mask_crop)
            print(os.path.join(mask_crop_sub, "{}_{}_{}_{}_{}.png".format(filename[:-4],
                                                                                 start_x, end_x, start_y, end_y)))
            start_x = end_x
        start_y = end_y
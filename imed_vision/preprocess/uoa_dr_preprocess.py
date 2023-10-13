# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:uoa_dr_preprocess
    author: 12718
    time: 2022/4/15 14:31
    tool: PyCharm
"""
import cv2
import os
import glob
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, required=True,
#                     help="path to the image directory")
parser.add_argument("--output_dir", type=str, default=None,
                    help="path to output")
args = parser.parse_args()
args.data_dir = "D:/workspace/datasets/segmentation/UoA-DR/data"
data_dir = args.data_dir
ouput_dir = os.path.join(data_dir, "preprocess")
if args.output_dir is not None:
    output_dir = args.output_dir
if not os.path.exists(ouput_dir):
    os.makedirs(ouput_dir)
vessel_output_dir = os.path.join(ouput_dir, "vessel")
if not os.path.exists(vessel_output_dir):
    os.makedirs(vessel_output_dir)
disc_output_dir = os.path.join(ouput_dir, "disc")
if not os.path.exists(disc_output_dir):
    os.makedirs(disc_output_dir)
cup_output_dir = os.path.join(ouput_dir, "cup")
if not os.path.exists(cup_output_dir):
    os.makedirs(cup_output_dir)
img_output_dir = os.path.join(ouput_dir, "image")
if not os.path.exists(img_output_dir):
    os.makedirs(img_output_dir)
for i in tqdm.tqdm(range(1, 201)):
    img_paths = glob.glob(os.path.join(data_dir, str(i), "*.jpg"))
    for path in img_paths:
        filename = os.path.basename(path)
        splits = filename.split(".")
        iid = splits[1]
        len_split = len(splits)
        img = cv2.imread(path)
        img[2002:2048, 0:470, :] = (0, 0, 0)
        if i == 102:
            img[1405:1443, 0:320, :] = (0, 0, 0)
            if len_split > 2:
                label = np.zeros(img.shape[:2], dtype=np.uint8)
                label[np.all(img > (200, 200, 200), axis=2)] = 255
                if iid == "1" or (iid == "2" and i == 72):
                    if i == 186:
                        label = np.zeros(img.shape[:2], dtype=np.uint8)
                        label[np.all(img > (100, 150, 150), axis=2)] = 255
                        cv2.imwrite(os.path.join(vessel_output_dir, "{}.png".format(i)), label)
                elif iid == "2" or (iid == "1" and i == 72):
                    cv2.imwrite(os.path.join(disc_output_dir, "{}.png".format(i)), label)
                elif iid == "3":
                    cv2.imwrite(os.path.join(cup_output_dir, "{}.png".format(i)), label)
            else:
                cv2.imwrite(os.path.join(img_output_dir, "{}.png".format(i)), img)
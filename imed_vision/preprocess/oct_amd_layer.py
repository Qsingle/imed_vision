# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:oct_amd_layer
    author: 12718
    time: 2022/6/9 16:39
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import glob
from scipy.io import loadmat
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path of the directory", required=True)
parser.add_argument("--output_dir", type=str, default=None, help="path to store the output files")
args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir

_types = ["AMD", "Control"]
if output_dir is None:
    output_dir = os.path.join(data_dir, "preprocess")
for _type in _types:
    paths = glob.glob(os.path.join(data_dir, _type, "*.mat"))
    cnt = 0
    sub_output_dir = os.path.join(output_dir, _type)
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir)
    for path in paths:
        sub_sub_output_dir = os.path.join(sub_output_dir, str(cnt))
        mask_output_dir = os.path.join(sub_sub_output_dir, "mask")
        if not os.path.exists(sub_sub_output_dir):
            os.makedirs(sub_sub_output_dir)
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)
        data = loadmat(path)
        vol = data["images"]
        lay = data["layerMaps"]
        for i in range(vol.shape[2]):
            img = vol[:, :, i]
            layers = lay[i, ...]
            layer = np.zeros(shape=img.shape)
            for j in range(layers.shape[0]):
                for k in range(layers.shape[1]-1):
                    y = layers[j, k]
                    t = layers[j, k+1]
                    if not np.isnan(y):
                        v = k+1
                        layer[int(y):int(t), int(j)] = v

            cv2.imwrite(os.path.join(sub_sub_output_dir, "{}.png".format(i)), img)
            cv2.imwrite(os.path.join(mask_output_dir, "{}.png".format(i)), layer)
        cnt += 1
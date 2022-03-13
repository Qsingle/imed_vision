# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:av_preprocess
    author: 12718
    time: 2022/2/23 19:09
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import os
import glob
import numpy as np

data_dir = r"D:\workspace\datasets\segmentation\LES-AV\arteries-and-veins"
# os.makedirs(os.path.join(data_dir, "preprocessed"))
output_dir = os.path.join(data_dir, "preprocessed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#for drive and les-av, the red is artery, green is overlay, blue is vein and white is uncertain
color2id={ 1:(255, 0, 0), 2:(0, 0, 255), 3:(0, 255, 0)}

image_list = glob.glob(os.path.join(data_dir, "*.png"))
for image_path in image_list:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[np.all(img==(255,255,255), axis=-1)] = (0, 0, 0)
    for i in range(1, 4):
        img[np.all(img==color2id[i], axis=-1)] = (i, i, i)
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
    mask = img[..., 0]
    mask = np.where(mask > 2, 1, mask)
    print(np.unique(mask))
    filename=os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, filename), mask)
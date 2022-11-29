# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:sy_preprocess
    author: 12718
    time: 2022/7/7 18:17
    tool: PyCharm
"""
import pydicom
import numpy as np
import cv2

file_path = r"D:\workspace\Unsaved\I730"
data = pydicom.dcmread(file_path)
img = data.pixel_array
window_level = data.WindowCenter[0]
window_width = data.WindowWidth[0]
rescale_slope = data.RescaleSlope
rescale_intercept = data.RescaleIntercept
img = img*rescale_slope + rescale_intercept

def apply_window(img, window_level, window_width):
    grayscale = (img - window_level + window_width / 2) / window_width
    grayscale[grayscale < 0] = 0
    grayscale[grayscale > 1] = 1
    grayscale = grayscale*255
    return grayscale

grayscale = apply_window(img, window_level, window_width)

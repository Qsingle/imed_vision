# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:amgad2019_preprocess
    author: 12718
    time: 2022/2/13 16:42
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import os
import glob
import numpy as np
import tqdm

data_dir = "D:/workspace/datasets/segmentation/CamVID"
image_folders = ["train", "val", "test"]

"""
64 128 64	Animal
192 0 128	Archway
0 128 192	Bicyclist
0 128 64	Bridge
128 0 0		Building
64 0 128	Car
64 0 192	CartLuggagePram
192 128 64	Child
192 192 128	Column_Pole
64 64 128	Fence
128 0 192	LaneMkgsDriv
192 0 64	LaneMkgsNonDriv
128 128 64	Misc_Text
192 0 192	MotorcycleScooter
128 64 64	OtherMoving
64 192 128	ParkingBlock
64 64 0		Pedestrian
128 64 128	Road
128 128 192	RoadShoulder
0 0 192		Sidewalk
192 128 128	SignSymbol
128 128 128	Sky
64 128 192	SUVPickupTruck
0 0 64		TrafficCone
0 64 64		TrafficLight
192 64 128	Train
128 128 0	Tree
192 128 192	Truck_Bus
64 0 64		Tunnel
192 192 0	VegetationMisc
0 0 0		Void
64 192 0	Wall
"""
color_id_list = [(64, 128, 64),
  (192, 0, 128),
  (0, 128, 192),
  (0, 128, 64),
  (128, 0, 0),
  (64, 0, 128),
  (64, 0, 192),
  (192, 128, 64),
  (192, 192, 128),
  (64, 64, 128),
  (128, 0, 192),
  (192, 0, 64),
  (128, 128, 64),
  (192, 0, 192),
  (128, 64, 64),
  (64, 192, 128),
  (64, 64, 0),
  (128, 64, 128),
  (128, 128, 192),
  (0, 0, 192),
  (192, 128, 128),
  (128, 128, 128),
  (64, 128, 192),
  (0, 0, 64),
  (0, 64, 64),
  (192, 64, 128),
  (128, 128, 0),
  (192, 128, 192),
  (64, 0, 64),
  (192, 192, 0),
  (0, 0, 0),
  (64, 192, 0)]

name_list = ['Animal',
  'Archway',
  'Bicyclist',
  'Bridge',
  'Building',
  'Car',
  'CartLuggagePram',
  'Child',
  'Column_Pole',
  'Fence',
  'LaneMkgsDriv',
  'LaneMkgsNonDriv',
  'Misc_Text',
  'MotorcycleScooter',
  'OtherMoving',
  'ParkingBlock',
  'Pedestrian',
  'Road',
  'RoadShoulder',
  'Sidewalk',
  'SignSymbol',
  'Sky',
  'SUVPickupTruck',
  'TrafficCone',
  'TrafficLight',
  'Train',
  'Tree',
  'Truck_Bus',
  'Tunnel',
  'VegetationMisc',
  'Void',
  'Wall'
]

def get_paths(image_folder, mask_foder):
    global data_dir
    image_dir = os.path.join(data_dir, image_folder)
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    mask_paths = []
    for path in image_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        filename = filename + "_L.png"
        mask_path = os.path.join(data_dir, mask_foder, filename)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

id2code = {k:v for k,v in enumerate(color_id_list)}

output_dir = os.path.join(data_dir, "preprocessed")
for image_folder in image_folders:
    print("process folder:{}".format(image_folder))
    sub_output_image_folder = os.path.join(output_dir, image_folder)
    if not os.path.exists(sub_output_image_folder):
        os.makedirs(sub_output_image_folder)
    mask_folder = image_folder+"_labels"
    sub_output_mask_folder = os.path.join(output_dir, mask_folder)
    if not os.path.exists(sub_output_mask_folder):
        os.makedirs(sub_output_mask_folder)
    image_paths, mask_paths = get_paths(image_folder, mask_folder)
    bar = tqdm.tqdm(zip(image_paths, mask_paths))
    for image_path, mask_path in bar:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(image_path)
        num_classes = len(id2code)
        shape = mask.shape[:2] + (num_classes,)
        encoded_image = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(id2code):
            encoded_image[:, :, i] = np.all(mask.reshape((-1, 3)) == id2code[i], axis=1).reshape(shape[:2])
        encoded_image = np.argmax(encoded_image, axis=-1)
        cv2.imwrite(os.path.join(sub_output_image_folder, filename), image)
        cv2.imwrite(os.path.join(sub_output_mask_folder, filename), encoded_image)
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pd
    author: 12718
    time: 2022/8/17 20:26
    tool: PyCharm
"""
import glob
import os
import numpy as np
import json

import pandas as pd
import tqdm
from PIL import Image
from PIL import ImageDraw
import cv2
import sys

data_dir = r"D:\datas\zhou\AMD+PCV"
sub_dir = ["1_wet_AMD", "2_PCV"]
# color_map = {
#         "1": [128, 0, 0],
#         "2": [0, 128, 0],
#         "3": [0, 0, 128],
#         "4": [128, 0, 128],
#         "5": [0, 128, 128],
#         "6": [128, 128, 128],
#         "7": [64, 0, 0],
#         "8": [0, 64, 0],
#         "9": [0, 0, 64],
#         "10": [192, 0, 0],
#         "11": [0, 192, 0]
# }
image_names = []
diseases = []
thicknesses = {}
for i in range(1, 6):
    thicknesses[str(i)] = {"mean": [], "median": [], "min": [], "max": [], "std":[]}
types = []
for sub_dir in sub_dir:
    dirs = os.listdir(os.path.join(data_dir, sub_dir))
    disease = sub_dir[1:]
    for subdir in dirs:
        json_files = glob.glob(os.path.join(data_dir, sub_dir, subdir, "*.json"))
        for json_file in tqdm.tqdm(json_files):
            with open(json_file, "rb") as f:
                data = json.load(f)
            output_dir = os.path.join(data_dir, "output", sub_dir, subdir)
            t = data["imagePath"][:-4].split("_")[1][0]
            types.append(t)
            sub_out = os.path.join(output_dir, t)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            image = Image.open(os.path.join(os.path.dirname(json_file), data["imagePath"]))
            image_arr = np.array(image)
            mask = np.zeros(image_arr.shape[:2], dtype=np.uint8)
            mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask)
            for label in data["shapes"]:
                id = label["label"]
                points = label["points"]
                xy = [tuple(point) for point in points]
                draw.line(xy, fill=int(id), width=1)

            # mask.show(title="test")
            mask = np.asarray(mask)
            save_mask = np.zeros(mask.shape, dtype=np.uint8)
            thickness = {}
            for i in range(1, 6):
                thickness[str(i)] = []
            for w in range(mask.shape[1]):
                for l in np.unique(mask[:, w]):
                    if l > 0:
                        index = np.where(mask[:, w] == l)[0][0]
                        if l != 6:
                            index1 = np.where(mask[:, w] == (l + 1))[0]
                            if len(index1) > 0:
                                index1 = index1[0]
                                save_mask[index:index1, w] = l
                                thickness[str(l)].append(index1 - index)
            for i in range(1, 6):
                if len(thickness[str(i)]) > 0:
                    thicknesses[str(i)]["mean"].append(np.mean(np.asarray(thickness[str(i)])))
                    thicknesses[str(i)]["median"].append(np.median(thickness[str(i)]))
                    thicknesses[str(i)]["min"].append(np.min(thickness[str(i)]))
                    thicknesses[str(i)]["max"].append(np.max(thickness[str(i)]))
                    thicknesses[str(i)]["std"].append(np.std(thickness[str(i)]))
                else:
                    thicknesses[str(i)]["mean"].append(-1.)
                    thicknesses[str(i)]["median"].append(-1.)
                    thicknesses[str(i)]["min"].append(-1.)
                    thicknesses[str(i)]["max"].append(-1.)
                    thicknesses[str(i)]["std"].append(-1.)
            image_names.append(data["imagePath"])
            diseases.append(disease)
            if not os.path.exists(sub_out):
                os.makedirs(sub_out)
            # Image.fromarray(save_mask).save(os.path.join(sub_out, "_".join(data["imagePath"][:-4].split()) + ".png"))
            # image.save(os.path.join(sub_out, "_".join(data["imagePath"][:-4].split()) + ".jpg"))
for i in range(1, 6):
    out_dict = {"image_name": image_names, "disease": diseases, "type": types}
    for k, v in thicknesses[str(i)].items():
        out_dict[k] = v
    data = pd.DataFrame(out_dict)
    print(data.head(5))
    data.to_excel(os.path.join(data_dir, "thickness_statistic_layer_{}.xlsx".format(i)))
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:gamma
    author: 12718
    time: 2022/11/15 20:31
    tool: PyCharm
"""
import glob

import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class GammaDataset(Dataset):
    def __init__(self, data_dir, csv_path, fundus_transform=None, oct_transform=None, oct_index=127):
        self.fundus_transform = fundus_transform
        self.oct_transform = oct_transform
        self.oct_index = oct_index
        self._get_path(data_dir, csv_path)
        self.length = len(self.labels)

    def _get_path(self, data_dir, csv_path):
        data = pd.read_csv(csv_path)
        labels = data["label"].values
        ids = data.iloc[:, 0]
        self.labels = labels
        # self.labels = np.argmax(labels, axis=-1)
        self.fundus_paths = []
        self.oct_paths = []
        for pid in ids:
            sid = "{:04d}".format(pid)
            fundus_path = os.path.join(data_dir, sid , "{}.jpg".format(sid))
            self.fundus_paths.append(fundus_path)
            oct_paths = glob.glob(os.path.join(data_dir, sid, sid, "*.jpg"))
            oct_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
            self.oct_paths.append(oct_paths)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        fundus = Image.open(self.fundus_paths[index]).convert("RGB")
        if self.oct_index is not None:
            assert self.oct_index in list(range(0, 256))
            oct = Image.open(self.oct_paths[index][self.oct_index]).convert("RGB")
        else:
            raise ValueError("Unsupported operation for the index of OCT is None, expect it from 0 to 255")
        label = self.labels[index]
        if self.fundus_transform is not None:
            fundus = self.fundus_transform(fundus)
        if self.oct_transform is not None:
            oct = self.oct_transform(oct)
        return fundus, oct, torch.from_numpy(np.asarray(label))
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:atlas
    author: 12718
    time: 2022/7/4 16:58
    tool: PyCharm
"""
import os
import re
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--atlas_dir", type=str, required=True,
                    help="Path to the atlas")
parser.add_argument("--output_dir", default=None,
                    help="Path to store the preprocessed data, default: atlas_dir/preprocessed")


def main():
    args = parser.parse_args()
    data_root = args.atlas_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(data_root, "preprocessed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(data_root):
        if "Training" in root:
            if len(dirs) == 0:
                for file in files:
                    if re.match(".*\.nii\.gz", file) is not None:
                        sid = file.split("_")[0]
                        sub_dir = os.path.join(output_dir, "train", sid)
                        if not os.path.exists(sub_dir):
                            os.makedirs(sub_dir)
                        print("Copy file from {} to {}".format(os.path.join(root, file), os.path.join(sub_dir, file)))
                        copyfile(os.path.join(root, file), os.path.join(sub_dir, file))
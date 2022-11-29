# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:paip
    author: 12718
    time: 2022/9/15 15:02
    tool: PyCharm
"""
import os
from xml.etree import ElementTree as ET
import glob

import cv2
import numpy as np
import openslide
import skimage.io as io
import argparse
import zipfile
import tifffile
import tqdm

parser = argparse.ArgumentParser("PAIP-2019 preprocess")
parser.add_argument("--data_dir", "-d", required=True, type=str,
                    help="The path of the directory that store the challenge data")
parser.add_argument("--mode", "-m", default="xml2mask", type=str,
                    help="the run mode, default xml2mask")
parser.add_argument("--output_dir", "-o", default=None, type=str,
                    help="the path of the directory to store the results")
parser.add_argument("--level", "-l", default=0, type=int,
                    help="The level of the dimension, default:0")
parser.add_argument("--crop_size", type=str, default="2048,2048",
                    help="The crop size.\nformat:width,height, default:2048,2048")

def xml2mask(xml_data, slide, level=0):
    """
        Annotations(root):
          -Annotation
           -Regions
             -RegionAttributeHeaders
               -AttributeHeader
             -Region
               -Vertices
                 -Vertex label points(x,y)
    Args:
        xml_data (ElementTree): xml source data
        slide (OpenSlide): Origin Image data
        level (int): Level of the output

    Returns:
        ndarray:the output mask
    """
    src_width, src_height = slide.level_dimensions[0]
    dest_w, dest_h = slide.level_dimensions[level]
    w_ratio = src_width / dest_w
    h_ratio = src_height / dest_h
    annotations = xml_data.getroot()
    mask = np.zeros((dest_h, dest_w), dtype=np.uint8)
    for annotation in annotations:
        cntr_pts = []
        bbox_pts = []
        regions = annotation.findall("Regions")[0]
        label = 0
        for region in regions.findall("Region"):
            nega_roa = region.get("NegativeROA")
            pts = []
            if nega_roa != "1":
                re_id = region.get("Id")
                if re_id in ["2"]:
                    # print(re_id)
                    label = int(re_id) - 1
                    vertices = region.findall("Vertices")[0]
                    for vertic in vertices.findall("Vertex"):
                        x = round(float(vertic.get("X")))
                        y = round(float(vertic.get("Y")))
                        # Match target level coordinates
                        x = np.clip(round(x / w_ratio), 0, dest_w)
                        y = np.clip(round(y / h_ratio), 0, dest_h)
                        pts.append([x, y])
                    if len(pts) == 4:
                        bbox_pts += [pts]
                    else:
                        cntr_pts += [pts]
        for pts in bbox_pts:
            pts = [np.array(pts, dtype=np.int32)]
            mask = cv2.drawContours(mask, pts, -1, label, -1)

        for pts in cntr_pts:
            pts = [np.array(pts, dtype=np.int32)]
            mask = cv2.drawContours(mask, pts, -1, label, -1)
            # mask = cv2.polylines(mask, pts, isClosed=True, color=label, thickness=1)

    return mask

def unzip(file, output_dir):
    """
        Uncompress one zip file
    Args:
        file (str): path to the zip file
        output_dir (str): path to the output directory

    Returns:
            None
    """
    zip = zipfile.ZipFile(file, "r")
    zip.extractall(output_dir)

def crop(svs, mask, level, crop_size, output_dir):
    start_x = 0
    total_x, total_y = svs.level_dimensions[level]
    crop_x, crop_y = crop_size
    assert crop_y < total_y and crop_x < total_x, "The crop size {} must less than the data size{}".format((crop_x,
                                                                                        crop_y), (total_x, total_y))
    while start_x < total_x:
        end_x = start_x + crop_x
        actual_start_x = start_x
        actual_end_x = end_x
        if end_x > total_x:
            actual_start_x = total_x - crop_x
            actual_end_x = total_x
        start_y = 0
        while start_y < total_y:
            end_y = start_y + crop_y
            actual_start_y = start_y
            actual_end_y = end_y
            if end_y > total_y:
                actual_start_y = total_y - crop_y
                actual_end_y = total_y
            crop_img = svs.read_region((actual_start_x, actual_start_y), level, (crop_x, crop_y))
            crop_mask = mask[actual_start_y:actual_end_y, actual_start_x:actual_end_x]
            crop_img = np.asarray(crop_img)
            # crop_mask = mask.read_region((actual_start_x, actual_start_y), level, (crop_x, crop_y))
            io.imsave(os.path.join(output_dir, f"{actual_start_x}_{actual_start_y}_{actual_end_x}_{actual_end_y}.tif"),
                      crop_img.astype(np.uint8))
            io.imsave(os.path.join(output_dir, f"{actual_start_x}_{actual_start_y}_{actual_end_x}_{actual_end_y}.png"),
                      crop_mask.astype(np.uint8))
            start_y = end_y
        start_x = end_x

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    mode = args.mode
    output_dir = args.output_dir
    level = args.level
    crop_size = [int(s) for s in args.crop_size.split(",")]
    if output_dir is None:
        output_dir = data_dir
    if mode == "xml2mask":
        svs_files = glob.glob(os.path.join(data_dir, "*.svs"))
        total = len(svs_files)
        cnt = 1
        for svs_file in svs_files:
            filename = os.path.basename(svs_file)[:-4]
            xml_file = os.path.join(data_dir, filename + ".xml")
            print("processing:{:.2f}%".format(cnt/total*100), end='', flush=True)
            svs = openslide.OpenSlide(svs_file)
            xml = ET.parse(xml_file)
            mask = xml2mask(xml, svs, level=level)
            io.imsave(os.path.join(output_dir, filename+".tif"), mask, compress=9)
            cnt += 1
            svs.close()
    elif mode == "unzip":
        zipfiles = glob.glob(os.path.join(data_dir, "*.zip"))
        for cnt, file in enumerate(zipfiles):
            print("processing:{:.2f}%".format(cnt / len(zipfiles)*100), end='', flush=True)
            filename = os.path.basename(file)[:-4]
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            unzip(file, output_path)
    elif mode == "crop":
        print(data_dir)
        svs_files = glob.glob(data_dir+"/**/*.svs", recursive=True)
        bar = tqdm.tqdm(svs_files)
        for svs_file in bar:
            file_id = os.path.basename(svs_file)[:-4]
            bar.set_description("Processing {}".format(file_id))
            data = openslide.OpenSlide(svs_file)
            mask = tifffile.imread(svs_file[:-4] + "_whole.tif")
            viable = tifffile.imread(svs_file[:-4] + "_viable.tif")
            mask[viable > 0] = 2
            # mask = openslide.OpenSlide(svs_file[:-4] + "_viable.tif")
            output_path = os.path.join(output_dir, "croped", f"{level}", file_id)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            crop(data, mask, level, crop_size, output_path)
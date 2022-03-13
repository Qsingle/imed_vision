# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cityscape_predict
    author: 12718
    time: 2022/1/23 19:44
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn.functional as F
import glob
from cityscapesscripts.helpers.labels import labels
from skimage import io
from albumentations import Normalize, Resize
import os
import cv2
import argparse
import numpy as np
import tqdm
from comm.metrics.metric import Metric

from models.segmentation.deeplab import DeeplabV3Plus, DeeplabV3

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

def camvid_get_paths(image_folder, mask_foder):
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))
    mask_paths = []
    for path in image_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        filename = filename + ".png"
        mask_path = os.path.join(mask_foder, filename)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

def get_paths(root, split="train"):
    img_data_dir = os.path.join(root, "leftImg8bit" ,split)
    mask_data_dir = os.path.join(root, "gtFine", split)
    img_paths = glob.glob(os.path.join(img_data_dir,"*","*.png"))
    mask_paths = []
    for path in img_paths:
        filename = os.path.basename(path)
        mask_filename = filename.replace("leftImg8bit", "gtFine_labelTrainIds")
        assert os.path.exists(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename)), os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename)
        mask_paths.append(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename))
    return img_paths, mask_paths

def img2color(img, map_dict=None):
    if map_dict is None:
        map_dict = { label.trainId : label.color for label in labels}
    color = np.zeros(shape=img.shape+(3,))
    for v in np.unique(img):
        color[np.equal(img, v)] = np.array(map_dict[v])
    # for col in range(img.shape[0]):
    #     for row in range(img.shape[1]):
    #         label = img[col,row]
    #         color[col, row] = np.array(map_dict[label])
    return color

def to_id(img):
    map_dict = { label.trainId : label.id for label in labels}
    ids = np.zeros(shape=img.shape)
    for v in np.unique(img):
        ids[np.equal(img, v)] = np.array(map_dict[v])
    # for col in range(img.shape[0]):
    #     for row in range(img.shape[1]):
    #         ids[col][row] = np.array(map_dict[img[col,row]])
    return ids

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    channels = args.channels
    num_classes = args.num_classes
    model_name = args.model_name
    backbone = args.backbone
    root_dir = args.root
    output_size = [int(k) for k in args.image_size.split(",")]
    super_reso = args.super_reso
    upscale_rate = args.upscale_rate
    fusion = args.fusion
    output_dir = args.output_dir
    dataset = args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "deeplabv3":
        model = DeeplabV3(in_ch=channels, num_classes=num_classes, backbone=backbone)
    elif model_name == "deeplabv3plus":
        model = DeeplabV3Plus(in_ch=channels, num_classes=num_classes, backbone=backbone,
                              super_reso=super_reso, upscale_rate=upscale_rate, sr_seg_fusion=fusion)
    else:
        raise ValueError("unknown model name")

    image_paths, mask_paths = get_paths(root_dir, split=args.split)
    if dataset == "camvid":
        image_dir = os.path.join(root_dir, args.split)
        mask_dir = os.path.join(root_dir, args.split+"_labels")
        image_paths, mask_paths = camvid_get_paths(image_dir, mask_dir)
    normalize = Normalize(mean=[0.5]*3, std=[0.5]*3)
    resize = Resize(height=output_size[0], width=output_size[1])
    map_dict = None
    if dataset == "camvid":
        map_dict = {i:color for i, color in enumerate(color_id_list)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights
    assert os.path.exists(weights_path), "The weights file {} is not found".format(weights_path)
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights["model"])
    model.to(device)
    model.eval()
    metric = Metric(num_classes)
    i = 0
    with torch.no_grad():
        bar = tqdm.tqdm(image_paths)
        for path in bar:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(path)
            bar.set_description("process:{}".format(filename))
            h, w = img.shape[:2]
            input = resize(image=img)["image"]
            input = normalize(image=input)["image"]
            if input.ndim == 3:
                input = np.transpose(input, axes=[2, 0, 1])
            input = torch.from_numpy(input)
            input = input.unsqueeze(0).to(device, dtype=torch.float32)
            pred = model(input)
            pred = torch.softmax(pred, dim=1)
            pred = torch.max(pred, dim=1)[1]
            mask = cv2.imread(mask_paths[i], 0)
            mask = torch.from_numpy(mask).to(device, dtype=torch.long)
            i += 1
            metric.update(pred, mask)
            pred = pred.squeeze().detach().cpu().numpy()
            if not super_reso:
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_color = img2color(pred, map_dict=map_dict)
            if dataset == "cityscapes":
                pred_lid = to_id(pred)
            if dataset == "camvid":
                io.imsave(os.path.join(output_dir, os.path.splitext(filename)[0]+"_predict.png"), np.uint8(pred))
                io.imsave(os.path.join(output_dir, os.path.splitext(filename)[0]+"color.png"), np.uint8(pred_color))
            else:
                io.imsave(os.path.join(output_dir, filename.replace("leftImg8bit", "predict")), pred.astype(np.uint8))
                io.imsave(os.path.join(output_dir, filename.replace("leftImg8bit", "predict_color")), pred_color.astype(np.uint8))
                io.imsave(os.path.join(output_dir, filename.replace("leftImg8bit", "predict_labelids")), pred_lid.astype(np.uint8))
    result = metric.evalutate()
    show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
    result_text = ""
    for metric in show_metric:
        if num_classes <= 2:
            result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
        else:
            result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].cpu().numpy()))
    print(result_text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root directory of cityscapes dataset")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--weights", type=str, default="./ckpt/best.pt")
    parser.add_argument("--fusion", action="store_true", default=False)
    parser.add_argument("--super_reso", action="store_true", default=False)
    parser.add_argument("--upscale_rate", default=4, type=int)
    parser.add_argument("--model_name", type=str,
                        choices=["deeplabv3", "deeplabv3plus"],
                        help="name of model")
    parser.add_argument("--backbone", type=str,
                        choices=["resnet50", "resnet101", "resnest50", "resnest101", "seresnet50"],
                        default="resnet50",
                        help="name of backbone for deeplab and encnet")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of channels for input")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes")
    parser.add_argument("--block_name", type=str, choices=["res", "splat", "novel", "rrblock"],
                        default="splat", help="type of block used in UNet")
    parser.add_argument("--avd", action="store_true",
                        help="whether use avd layer")
    parser.add_argument("--avd_first", action="store_true",
                        help="whether use avd layer before splat conv")
    parser.add_argument("--reduction", type=int, default=4,
                        help="reduction rate")
    parser.add_argument("--drop_prob", type=float, default=0.0,
                        help="dropout rate")
    parser.add_argument("--layer_attention", action="store_true",
                        help="whether use layer attention")
    parser.add_argument("--expansion", type=float, default=1.0,
                        help="expansion rate for hidden channel")
    parser.add_argument("--gpu_index", type=str, default="0")
    parser.add_argument("--image_size", type=str, default="256,512")
    parser.add_argument("--split", type=str, default="val",
                        help="the split type of the data to use, default is val")
    parser.add_argument("--dataset", type=str, default="cityscapes", help="name of the dataset")

    args = parser.parse_args()
    main()
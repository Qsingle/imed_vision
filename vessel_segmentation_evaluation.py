# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentation_evaluation
    author: 12718
    time: 2022/1/16 15:14
    tool: PyCharm
"""

import torch
import os
import argparse
import json
from albumentations import Normalize, Resize
import numpy as np
import tqdm
import cv2

from datasets.vessel_segmentation import get_paths
from models.segmentation import Unet, SAUnet, NestedUNet
from models.segmentation.pfseg import PFSeg
from layers.unet_blocks import *
from comm.helper import to_tuple
from comm.metrics import Metric


def main(config):
    image_dir = config["image_dir"]
    image_suffix = config["image_suffix"]
    mask_dir = config["mask_dir"]
    mask_suffix = config["mask_suffix"]
    image_size = config["image_size"]
    super_reso = config["super_reso"]
    fusion = config["fusion"]
    num_classes = config["num_classes"]
    divide = config["divide"]
    model_name = config["model_name"]
    block_name = config["block_name"]
    gpu_index = config["gpu_index"]
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    weights = config["weights"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]
    output_dir = config["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "unet":
        if block_name == "origin":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=DoubleConv,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion)
        elif block_name == "resblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=ResBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion)
        elif block_name == "splatblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=SplAtBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion)
        elif block_name == "rrblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=RRBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion)
        else:
            raise ValueError("Unknown block name {}".format(block_name))
        model_name = model_name + "_" +block_name
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes=num_classes, input_channels=channel,
                           deep_supervision=True)
    elif model_name == "saunet":
        model = SAUnet(in_ch=channel, num_classes=num_classes)
    elif model_name == "pfseg":
        model = PFSeg(channel, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
    print("Find {} pairs".format(len(image_paths)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_size = to_tuple(image_size, 2)
    assert os.path.exists(weights), "Please sure the weights file {} is exists".format(weights)
    state = torch.load(weights, map_location="cpu")
    model_state = state["model"]
    model.load_state_dict(model_state)
    bar = tqdm.tqdm(zip(image_paths, mask_paths))
    model.to(device)
    out_h, out_w = image_size
    resize = Resize(height=out_h, width=out_w, interpolation=cv2.INTER_CUBIC)
    normalize = Normalize([0.5]*channel, [0.5]*channel)
    metric = Metric(num_classes=num_classes)
    for image_path, mask_path in bar:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        if divide:
            mask = mask // 255
        h, w = image.shape[:2]
        x = resize(image=image)["image"]
        if out_w * upscale_rate != w or out_h*upscale_rate != h:
            hr = Resize(out_h*upscale_rate, width=out_w*upscale_rate)(image=image)["image"]
        else:
            hr = image.copy()
        x = normalize(image=x)["image"]
        hr = normalize(image=hr)["image"]
        guidance = None
        if model_name == "pfseg":
            crop_w = out_w // 2
            crop_h = out_h // 2
            hr_h, hr_w = hr.shape[:2]
            c_y, c_x = hr_h // 2, hr_w // 2
            guidance = hr[c_y-crop_h//2:c_y+crop_h//2, c_x-crop_w//2:c_x+crop_w//2, :]
            guidance = np.transpose(guidance, axes=[2, 0, 1])
            guidance = torch.from_numpy(guidance)
            guidance = guidance.to(device, dtype=torch.float32).unsqueeze(0)
        x = np.transpose(x, axes=[2, 0, 1])
        if super_reso:
            hr = np.transpose(hr, axes=[2, 0, 1])
        x = torch.from_numpy(x).to(device, dtype=torch.float32).unsqueeze(0)
        mask = torch.from_numpy(mask).to(device)
        if model_name == "pfseg":
            pred, sr = model(x, guidance)
        else:
            pred = model(x)
        if num_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        if out_w != w or out_h != h:
            # pred = F.interpolate(pred, size=(h,w), mode="nearest")
            pred = cv2.resize(pred.cpu().squeeze().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            pred = torch.from_numpy(pred).to(device, dtype=torch.long)
        metric.update(pred, mask.unsqueeze(0))
        cv2.imwrite(os.path.join(output_dir, filename), pred.cpu().numpy()*255)

    result = metric.evalutate()
    show_metric = ["precision", "acc", "dice", "specifity", "iou", "recall", "mcc", "bm"]
    result_text = ""
    for met in show_metric:
        if num_classes <=2:
            result_text += "{}:{} ".format(met, result[met][1].item())
        else:
            result_text += "{}:{} ".format(met, np.mean(result[met].cpu().numpy()))
    print(result_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation evaluation")
    parser.add_argument("--config_path", type=str, default="configs/vessel_segmentation_test.json",
                        help="path to the config file")
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    main(config)
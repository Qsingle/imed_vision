# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentation_evaluation
    author: 12718
    time: 2022/1/16 15:14
    tool: PyCharm
"""
import glob

import torch
import os
import argparse
import json
from albumentations import Normalize, Resize
import numpy as np
import tqdm
import cv2
import pandas as pd
from torcheval.metrics.classification.precision_recall_curve import BinaryPrecisionRecallCurve
from sklearn.metrics import precision_recall_curve, auc
from torch.nn.functional import one_hot

from datasets.vessel_segmentation import get_paths
from models.segmentation import Unet, SAUnet, NestedUNet, MiniUnet
from models.segmentation.pfseg import PFSeg
from layers.unet_blocks import *
from comm.helper import to_tuple
from comm.metrics import Metric
from models.segmentation.segformer import *
from models.segmentation import bisenetv2, bisenetv2_l
from models.segmentation import STDCNetSeg
from models.segmentation import DeeplabV3Plus
from models.segmentation.scsnet import SCSNet
from models.segmentation.denseunet import Dense_Unet
from models.segmentation.dedcgcnee import DEDCGCNEE
from models.super_resolution import ESPCN
from models.segmentation.dpt import DPT


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
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    weights = config["weights"]
    dataset = config["dataset"]
    output_dir = config["output_dir"]
    sr_preprocess = config["sr_preprocess"]
    image_size = to_tuple(image_size, 2)
    color_map = config["color_map"]
    before = config["before"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "unet":
        if block_name == "origin":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=DoubleConv,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5, before=before)
        elif block_name == "resblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=ResBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5, before=before)
        elif block_name == "splatblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=SplAtBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5, before=before)
        elif block_name == "rrblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=RRBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5, before=before)
        else:
            raise ValueError("Unknown block name {}".format(block_name))
        model_name = model_name + "_" +block_name
    elif model_name.lower() == "miniunet":
        model = MiniUnet(in_ch=channel, out_ch=num_classes, convblock=DoubleConv,
                     super_reso=super_reso, upscale_rate=upscale_rate,
                     sr_seg_fusion=fusion, sr_layer=3, base_ch=32)
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes=num_classes, input_channels=channel,
                           deep_supervision=True)
    elif model_name == "saunet":
        model = SAUnet(in_ch=channel, num_classes=num_classes)
    elif model_name == "pfseg":
        model = PFSeg(channel, num_classes=num_classes)
    elif model_name == "segformer":
        model = segformer_b5(img_size=image_size[0], num_classes=num_classes)
    elif model_name.lower() == "deeplabv3plus":
        model = DeeplabV3Plus(channel, num_classes, backbone=block_name, super_reso=super_reso,
                              upscale_rate=upscale_rate, sr_seg_fusion=fusion, output_stride=8)
    elif model_name.lower() == "bisenetv2":
        model = bisenetv2(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "bisenetv2_l":
        model = bisenetv2_l(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "scsnet":
        model = SCSNet(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "stdcnet":
        model = STDCNetSeg(in_ch=channel, num_classes=num_classes,
                           backbone=block_name, boost=True)
    elif model_name.lower() == "denseunet":
        model = Dense_Unet(channel, num_classes, 64)
    elif model_name.lower() == "dedcgcnee":
        model = DEDCGCNEE(channel, num_classes, img_size=image_size)
    elif model_name.lower() == "supervessel":
        model = Unet(channel, num_classes, sr_layer=5, sr_seg_fusion=True,
                     super_reso=True, upscale_rate=upscale_rate, fim=True)
    elif model_name.lower() == "cogseg":
        model = Unet(channel, num_classes, super_reso=True, sr_seg_fusion=False,
                     before=False, l1=True, sr_layer=5)
    elif model_name.lower() == "ss_maf":
        model = Unet(channel, num_classes, sr_layer=5, before=True, sr_seg_fusion=True, ss_maf=True,
                     upscale_rate=upscale_rate, super_reso=True)
        super_reso = True
    elif model_name == "dpt":
        img_size = real = 1024
        if "dinov2" in block_name:
            img_size = 518
            p_size = 1024 // 14
            real = p_size*14
        print("Eval at size {}x{}".format(real, real))
        image_size = [real, real]
        model = DPT(arch=block_name, img_size=img_size, real_img_size=real, task="seg", out_dim=num_classes, checkpoint=None)
        model_name = model_name + "_" + block_name
    else:
        raise ValueError("Unknown model name {}".format(model_name))
    if sr_preprocess:
        sr_model = ESPCN(channel, upscale_factor=upscale_rate)
        sr_checkpoint = config["sr_checkpoint"]
        sr_model.load_state_dict(torch.load(sr_checkpoint, map_location="cpu")["model"])
        sr_model.eval()
        print("Enable super-resolution preprocess")
    image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
    if dataset == "oct":
        image_paths = glob.glob(image_dir+"/**/*.{}".format(image_suffix), recursive=True)
        mask_paths = [os.path.dirname(p)+"/mask_11/{}".format(os.path.basename(p)) for p in image_paths]
    print("Find {} pairs".format(len(image_paths)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    assert os.path.exists(weights), "Please sure the weights file {} is exists".format(weights)
    state = torch.load(weights, map_location="cpu")
    model_state = state["model"]
    try:
        model.load_state_dict(model_state)
    except:
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in model_state.items():
            new_state[k[7:]] = model_state[k]
        model.load_state_dict(new_state)
    bar = tqdm.tqdm(zip(image_paths, mask_paths))
    model.to(device)
    if sr_preprocess:
        sr_model.to(device)
    out_h, out_w = image_size
    resize = Resize(height=out_h, width=out_w, interpolation=cv2.INTER_CUBIC)
    if sr_preprocess:
        resize = Resize(height=out_h//upscale_rate, width=out_w//upscale_rate, interpolation=cv2.INTER_CUBIC)
    normalize = Normalize([0.5]*channel, [0.5]*channel)
    metric = Metric(num_classes=num_classes)
    if dataset == "idrid" or dataset == "ddr":
        prcs = [BinaryPrecisionRecallCurve() for _ in range(1, num_classes)]
    # file_map = {"MA": 1, "HE": 2, "EX": 3, "SE": 4}
    preds = []
    masks = []
    with torch.no_grad():
        for image_path, mask_path in bar:
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)
            # mask = mask.convert("L")
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
            if sr_preprocess:
                x = sr_model(x)
            mask = torch.from_numpy(mask).to(device)
            if model_name == "pfseg":
                pred, sr = model(x, guidance)
            else:
                pred = model(x)
                if isinstance(model, STDCNetSeg):
                    pred = pred[0]
                elif isinstance(model, NestedUNet):
                    pred = pred[-1]
            if num_classes == 1:
                pred = torch.sigmoid(pred)
                pred = torch.where(pred >= 0.5, 1, 0)
            else:
                pred = torch.softmax(pred, dim=1)
                if dataset == "ddr" or dataset == "idrid":
                    preds.append(pred.cpu())
                    masks.append(torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=pred.size()[2:]).cpu())
                if out_w != w or out_h != h:
                    pred = torch.nn.functional.interpolate(pred, size=(h, w))
                pred = torch.max(pred, dim=1)[1]
            metric.update(pred, mask.unsqueeze(0))
            pred = pred.detach().squeeze().cpu().numpy()
            if color_map is not None:
                color = np.zeros(shape=pred.shape+(3,))
                for k in color_map.keys():
                    color[np.equal(pred, int(k))] = np.array(color_map[k])
                pred = color
                pred = np.uint8(pred)
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            else:
                pred = pred * 255
            cv2.imwrite(os.path.join(output_dir, filename), pred.astype(np.uint8))

    result = metric.evaluate()
    show_metric = ["precision", "acc", "dice", "specifity", "iou", "recall", "mcc", "bm"]
    result_text = ""
    torch.save(result, os.path.join(output_dir, "results.pkl"))
    for met in show_metric:
        if num_classes <=2:
            result_text += "{}:{} ".format(met, result[met][1].item())
        else:
            result_text += "{}:{} ".format(met, np.nanmean(result[met][1:].cpu().numpy()))
    print(result_text)
    result_text = ""
    for met in show_metric:
        if num_classes <=2:
            result_text += "{:f} ".format(result[met][1].item())
        else:
            result_text += "{:f} ".format(np.nanmean(result[met][1:].cpu().numpy()))
    print("Our implementation:")
    print(result_text)
    if dataset == "idrid" or dataset == "ddr":
        aucs = []
        masks = torch.cat(masks, dim=0)
        preds = torch.cat(preds, dim=0)
        for i in range(1, num_classes):
            m = torch.where(masks == i, 1, 0).flatten()
            p = preds[:, i, :, :].flatten()
            prcs[i - 1].update(p, m)
            precision, recall, _ = prcs[i-1].compute()
            # precision, recall, thresholds = precision_recall_curve(m.detach().cpu().numpy(), p.detach().cpu().numpy())
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)
        print("auc:", aucs)
        print("mean auc:", np.nanmean(aucs))
        result['aucs'] = aucs
    return result
    # all_results = {}
    # for k in result.keys():
    #     met = result[k]
    #     all_results[k] = {}
    #     for c in range(len(met)):
    #         all_results[k][str(c)] = met[c].item()
    # pd.DataFrame(all_results).to_excel(os.path.join(output_dir, "results.xlsx"), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation evaluation")
    parser.add_argument("--config_path", type=str, default="configs/vessel_segmentation_test.json",
                        help="path to the config file")
    parser.add_argument("--focal", action="store_true", default=False,
                        help="whether use focal loss")
    parser.add_argument("--cbce", action="store_true", default=False,
                        help="whether use cbce loss")
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    image_size = config["image_size"]
    upscale_rate = config["upscale_rate"]
    super_reso = config["super_reso"]
    model_name = config["model_name"]
    fusion = config["fusion"]
    need_results = {
        "precision":[],
        "acc": [],
        "dice": [],
        "specifity": [],
        "iou": [],
        "recall": [],
        "mcc": [],
        "bm": []
    }
    for i in range(5):
        config["image_size"] = [image_size[0] // upscale_rate, image_size[1] // upscale_rate]
        if super_reso:
            mid = "_sr_{}".format(upscale_rate)
            if fusion:
                mid = "_sr_fusion_{}".format(upscale_rate)
            if config["before"]:
                mid += "before"
            else:
                mid += "after"
        else:
            mid = ""
        if args.focal:
            mid += "_focal"
        elif args.cbce:
            mid += "_cbce"
        sub_name = model_name if model_name != "unet" and model_name.lower() != "dpt" else model_name+"_"+config["block_name"]
        config["weights"] = os.path.join("ckpt", config["dataset"],
                                          model_name + "_" + "_".join([str(s) for s in config["image_size"]]) + mid,
                                          str(i), sub_name,
                                         config["dataset"], sub_name+"_"+config["dataset"]+"_best.pth")
        config["output_dir"] = os.path.join("output", config["dataset"], model_name+"_"+"_".join([str(s) for s in config["image_size"]]) + mid,
                                            str(i))
        print(config["weights"])
        result = main(config)
        for key in need_results.keys():
            if not ("auc" in key):
                need_results[key].append(
                    np.nanmean(result[key][1:].detach().cpu().numpy())
                )
        if config['dataset'] == "ddr" or config["dataset"] == "idrid":
            for i in range(1, config['num_classes']):
                k = "auc_{}".format(i)
                if k in need_results:
                    need_results[k].append(result['aucs'][i - 1])
                else:
                    need_results[k] = [result['aucs'][i-1]]

    outpu_dir = os.path.join("output", config["dataset"], model_name+"_"+"_".join([str(s) for s in config["image_size"]]) + mid)
    pd.DataFrame(need_results).to_excel(os.path.join(outpu_dir, "results.xlsx"), index=False)
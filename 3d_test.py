# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:3d_test
    author: 12718
    time: 2022/6/29 9:41
    tool: PyCharm
"""
import torch
from torch.utils.data import DataLoader
import SimpleITK as sitk
import argparse
import json
import numpy as np
import os
import tqdm
from medpy.metric import hd95
from medpy.metric import jc, dc

from models.segmentation.unet import UNet3D
from models.segmentation.pfseg import PFSeg3D
from models.segmentation.vnet import VNet
from models.segmentation.resunet import ResUNet3D
from comm.metrics import Metric
from datasets.atm import ATM
from datasets.brats import Brats20
from comm.metrics.atm_metric import branch_detected_calculation, precision_calculation, tree_length_calculation

def dice_coeff(output, pred):
    inter = torch.sum(output & pred)
    union = torch.sum(output | pred)
    iou = inter / union
    dice = 2*inter/union
    return iou, dice

def main(config):
    data_dir = config["data_dir"]
    crop_size = config["crop_size"]
    super_reso = config["super_reso"]
    img_size = config['img_size']
    fusion = config["fusion"]
    num_classes = config["num_classes"]
    divide = config["divide"]
    model_name = config["model_name"]
    gpu_index = config["gpu_index"]
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    ckpt_dir = config["ckpt_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]
    # weights = os.path.join(ckpt_dir, model_name, dataset, "{}_{}_best.pth".format(model_name, dataset))
    weights = config["weights"]
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    guide = False
    if model_name == "unet3d":
        model = UNet3D(channel, num_classes, super_reso=super_reso, fusion=fusion, upscale_rate=upscale_rate)
    elif model_name == "pfseg":
        model = PFSeg3D(channel, out_channels=num_classes)
        guide = True
        super_reso = True
    elif model_name == "resunet":
        model = ResUNet3D(channel, num_classes, super_reso=super_reso, fusion=fusion, upscale_rate=upscale_rate)
    elif model_name == "vnet":
        model = VNet(in_channels=channel, classes=num_classes)
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    if dataset == "atm":
        test_dataset = ATM(data_dir, txt_path=os.path.join(data_dir, "test.txt"), img_size=img_size, crop_size=crop_size,
                           super_reso=super_reso, upscale_rate=upscale_rate, augmentation=False, guide=guide)
    elif dataset == "brats20":
        test_dataset = Brats20(data_dir, mode="test", img_size=img_size, crop_size=crop_size, guide=guide, augmentation=False,
                               super_reso=super_reso, upscale_rate=upscale_rate)
    else:
        raise ValueError("Unsupported dataset {}".format(dataset))
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False,
                             pin_memory=True)
    # optimizer
    # optimizer = opt.SGD([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr,
    #                     momentum=momentum, nesterov=True, weight_decay=weight_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    # dice_loss = DiceLoss()
    test_metric = Metric(num_classes)
    print("Evaluation on {} samples".format(len(test_dataset)))

    dices = []
    ious = []
    hd50s = []
    if dataset == "atm":
        bds = []
        precisions = []
        tlcs = []
    cnt = 0
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            path =test_dataset.paths[cnt]
            img_name = os.path.basename(path)
            if super_reso:
                if guide:
                    x, hr, mask, guidance, guidance_mask = data
                    guidance = guidance.to(device, dtype=torch.float32)
                else:
                    x, hr, mask = data
                hr = hr.to(device)
            else:
                x, mask = data
            x = x.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            if divide:
                mask = mask // 255
            if model_name == "pfseg":
                pred, _ = model(x, guidance)
            else:
                pred = model(x)
            if num_classes < 2:
                pred = torch.sigmoid(pred)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
            else:
                pred = torch.softmax(pred, dim=1)
                pred = torch.max(pred, dim=1)[1].unsqueeze(1).to(torch.float32)
            shape = mask.size()[2:]
            if mask.ndim == 4:
                shape = mask.size()[1:]
            pred = torch.nn.functional.interpolate(pred, size=shape).to(torch.long)
            test_metric.update(pred.flatten(), mask.to(dtype=torch.long).flatten())
            pred, mask = pred.squeeze().detach().cpu().numpy().astype(np.uint8), mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            dice = dc(pred, mask)
            iou = jc(pred, mask)
            dices.append(dice)
            ious.append(iou)
            hd50s.append(hd95(pred, mask))
            if dataset == "atm":
                label_parse = sitk.ReadImage(os.path.join(data_dir, "tree_parse_valid", img_name))
                label_parse = sitk.GetArrayFromImage(label_parse).transpose(1, 2, 0)
                label_sk = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir, "skeleton", img_name))).transpose(1, 2, 0)
                pred = pred.transpose(1, 2, 0)
                mask = mask.transpose(1, 2, 0)
                bds.append(branch_detected_calculation(pred, label_parsing=label_parse, label_skeleton=label_sk)[-1])
                precisions.append(precision_calculation(pred, mask))
                tlcs.append(tree_length_calculation(pred, label_sk))
            cnt+=1
            nii_img = sitk.GetImageFromArray(pred.astype(np.uint8))
            img_name = os.path.splitext(img_name)[0]
            if dataset == "brats20":
                img_name = "{}_{}.nii".format(img_name, dice)
            elif dataset == "atm":
                img_name = "{}_{}.nii.gz".format(img_name.split(".")[0], dice)
            sitk.WriteImage(nii_img, os.path.join(output_dir, img_name))
        show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
        result = test_metric.evaluate()
        result_text = ""
        for metric in show_metric:
            if num_classes <= 2:
                result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
            else:
                result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].detach().cpu().numpy()[1:]))
        test_metric.reset()
        print("{}".format(result_text))
        print("Average Dice:{} std:{}".format(np.mean(dices), np.std(dices)))
        print("Average IoU:{} std:{}".format(np.mean(ious), np.std(ious)))
        print("Average hausdorff:{} std:{}".format(np.mean(hd50s), np.std(hd50s)))
        if dataset == "atm":
            print("Average bd:{} std:{}".format(np.mean(bds), np.std(bds)))
            print("Average precision:{} std:{}".format(np.mean(precisions), np.std(precisions)))
            print("Average tlc:{} std:{}".format(np.mean(tlcs), np.std(tlcs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation training sample")
    parser.add_argument("--json_path", type=str, help="path to config json file",
                        default="configs/vessel_segmentation.json")
    args = parser.parse_args()
    with open(args.json_path) as f:
        config = json.load(f)
    print(config)
    main(config)
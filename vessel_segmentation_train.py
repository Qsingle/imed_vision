# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentaion_train
    author: 12718
    time: 2022/1/15 15:18
    tool: PyCharm
"""
import glob

import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
import argparse
import json
import tqdm

from comm.scheduler.poly import PolyLRScheduler
from datasets.vessel_segmentation import get_paths, SegPathDataset
from models.segmentation import Unet, SAUnet, NestedUNet
from models.unsupervised.unsr import UnUNetV1
from layers.unet_blocks import *
from models.segmentation import ConvNeXtUNet
from comm.metrics import Metric
from loss import DiceLoss
from loss import FALoss

def get_ddr_paths(
        root_dir,
        split="train",
        ltype="EX",
        image_suffix=".jpg",
        mask_suffix=".tif"
     ):
    image_paths = glob.glob(os.path.join(root_dir, split, "image", "*{}".format(image_suffix)))
    mask_paths = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0]
        mask_path = os.path.join(root_dir, split, "label", ltype, filename+mask_suffix)
        mask_paths.append(mask_path)
    return image_paths, mask_paths


def main(config):
    init_lr = config["init_lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    image_dir = config["image_dir"]
    image_suffix = config["image_suffix"]
    mask_dir = config["mask_dir"]
    mask_suffix = config["mask_suffix"]
    epochs = config["epochs"]
    lr_sche = config["lr_sche"]
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
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    ckpt_dir = config["ckpt_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]
    crop = config["crop"]
    distance = config["distance"]

    if model_name == "unet":
        if block_name == "origin":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=DoubleConv,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5)
        elif block_name == "resblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=ResBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5)
        elif block_name == "splatblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=SplAtBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5)
        elif block_name == "rrblock":
            model = Unet(in_ch=channel, out_ch=num_classes, convblock=RRBlock,
                         super_reso=super_reso, upscale_rate=upscale_rate,
                         sr_seg_fusion=fusion, sr_layer=5)
        else:
            raise ValueError("Unknown block name {}".format(block_name))
        model_name = model_name + "_" +block_name
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes=num_classes, input_channels=channel,
                           deep_supervision=True)
    elif model_name == "saunet":
        model = SAUnet(in_ch=channel, num_classes=num_classes)
    elif model_name == "ununetv1":
        model = UnUNetV1(channel, num_classes=num_classes, upscale_rate=upscale_rate, finetune=True)
        assert config["pretrained"] != "", "Unsupervised pretrain weights must provide"
        state = model.state_dict()
        state.update(torch.load(config["pretrained"], map_location="cpu")["model"])
        model.load_state_dict(state)
        model.init_seg_decoder()
    elif model_name.lower() == "convnextunet":
        model = ConvNeXtUNet(channel, num_classes)
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    if crop and dataset == "hrf":
        train_range, test_range = train_test_split(list(range(1, 11)), random_state=0, test_size=0.2)
        train_paths = []
        for i in train_range:
            train_paths += glob.glob(os.path.join(image_dir, "{:02d}_*{}".format(i, image_suffix)))
        train_mask_paths = [os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0]) + mask_suffix for p in train_paths]
        test_paths = []
        for i in test_range:
            test_paths += glob.glob(os.path.join(image_dir, "{:02d}_*{}".format(i, image_suffix)))
        test_mask_paths = [os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0]) + mask_suffix for p in test_paths]
    elif dataset == "ddr":
        train_paths, train_mask_paths = get_ddr_paths(image_dir, "train", ltype=config["ltype"],
                                                      image_suffix=image_suffix, mask_suffix=mask_suffix)
        test_paths, test_mask_paths = get_ddr_paths(image_dir, "valid", ltype=config["ltype"],
                                                    image_suffix=image_suffix, mask_suffix=mask_suffix)
    else:
        image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
        train_paths, test_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths,
                                                                                      random_state=0, test_size=0.2)
    train_dataset = SegPathDataset(train_paths, train_mask_paths, augmentation=True,
                                   output_size=image_size, super_reso=super_reso,
                                   upscale_rate=upscale_rate, divide=divide, distance=distance)
    test_dataset = SegPathDataset(test_paths, test_mask_paths, augmentation=False,
                                  output_size=image_size, super_reso=super_reso,
                                  upscale_rate=upscale_rate, divide=divide, distance=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    optimizer = opt.SGD([{"params":model.parameters(), "initial_lr":init_lr}], lr=init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if lr_sche == "poly":
        sche = PolyLRScheduler(optimizer, len(train_dataset), batch_size=batch_size,
                               epochs=epochs)
    else:
        raise ValueError("Unknown learning rate scheduler")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)
        with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=4)
    else:
        if not os.path.exists(os.path.join(ckpt_dir, model_name, dataset)):
            os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)

    writer = SummaryWriter(comment=model_name+"_"+dataset)

    if super_reso:
        sr_loss = nn.MSELoss()
    # if fusion:
    #     fusion_loss = FALoss()
    seg_loss = nn.CrossEntropyLoss()
    # dice_loss = DiceLoss()
    train_metric = Metric(num_classes)
    test_metric = Metric(num_classes)
    global_step = 0
    best_iou = 0.0
    print("train on {} samples, evaluation on {} samples".format(len(train_dataset), len(test_dataset)))
    for epoch in range(epochs):
        bar = tqdm.tqdm(train_loader)
        losses = 0.0
        total = 0
        iteration = 0
        model.train()
        for data in bar:
            distance_mask = None
            if super_reso:
                if distance:
                    x, hr, mask, distance_mask = data
                    distance_mask = distance_mask.to(device)
                else:
                    x, hr, mask, distance_mask = data
                hr = hr.to(device)
            else:
                if distance:
                    x, mask, distance_mask = data
                    distance_mask = distance_mask.to(device)
                else:
                    x, mask = data
            x = x.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long if isinstance(seg_loss, nn.CrossEntropyLoss) else torch.float32)
            optimizer.zero_grad()
            pred = model(x)
            bs = x.size(0)
            total += bs
            fusion_seg = None
            fusion_sr = None
            sr = None
            if isinstance(pred, tuple):
                if len(pred) == 2:
                    pred, sr = pred
                elif len(pred) == 4:
                    pred, sr, fusion_seg, fusion_sr = pred
            # print(pred.shape)
            if distance:
                pred = distance_mask * pred
            loss = seg_loss(pred, mask)
            if super_reso:
                if sr is not None:
                    loss += sr_loss(sr, hr)
                if fusion_sr is not None:
                    loss += sr_loss(fusion_sr, hr)
                if fusion_seg is not None:
                    loss += seg_loss(fusion_seg+pred, mask)
                    # loss += dice_loss(fusion_seg, mask)
                # if fusion_sr is not None and fusion_seg is not None:
                #     loss += fusion_loss(fusion_seg, fusion_sr)
            loss.backward()
            optimizer.step()
            sche.step()
            losses += loss.item()
            pred = torch.softmax(pred, dim=1)
            train_metric.update(pred, mask)
            result = train_metric.evalutate()
            show_metric = ["acc", "recall", "iou"]
            # if num_classes == 2:
            result_text = ""
            for metric in show_metric:
                writer.add_scalar("train/{}".format(metric), result[metric][1], global_step=global_step)
                result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
            bar.set_description("[{}:{}:{}] loss:{:.4f} {}".format(epochs, epoch+1, iteration+1, losses / total, result_text))
            iteration += 1
            global_step += 1
        model.eval()
        train_metric.reset()

        with torch.no_grad():
            for data in test_loader:
                if super_reso:
                    x, hr, mask = data
                    hr = hr.to(device)
                else:
                    x, mask = data
                x = x.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.long if isinstance(seg_loss, nn.CrossEntropyLoss) else torch.float32)
                pred = model(x)
                pred = torch.softmax(pred, dim=1)
                test_metric.update(pred, mask)
            show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
            result = test_metric.evalutate()
            result_text = ""
            iou = result["iou"][1]
            for metric in show_metric:
                writer.add_scalar("train/{}".format(metric), result[metric][1], global_step=epoch)
                result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
            test_metric.reset()
            print("[{}:{}] {}".format(epochs, epoch+1, result_text))
            if iou > best_iou:
                best_iou = iou
                save_obj = {
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "model": model.state_dict(),
                    "lr_sche": sche.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(
                    save_obj, os.path.join(ckpt_dir, "{}_{}_best.pth".format(model_name, dataset))
                )
            save_obj = {
                "epoch": epoch,
                "best_iou": best_iou,
                "model": model.state_dict(),
                "lr_sche": sche.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(
                save_obj, os.path.join(ckpt_dir, "{}_{}_last.pth".format(model_name, dataset))
            )

    print("best iou:", best_iou)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation training sample")
    parser.add_argument("--json_path", type=str, help="path to config json file",
                        default="configs/vessel_segmentation.json")
    args = parser.parse_args()
    with open(args.json_path) as f:
        config = json.load(f)
    print(config)
    main(config)
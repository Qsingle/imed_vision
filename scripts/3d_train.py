# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:3d_train
    author: 12718
    time: 2022/6/29 9:41
    tool: PyCharm
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import argparse
import json
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import tqdm

from imed_vision.models.segmentation.unet import UNet3D
from imed_vision.models.segmentation.pfseg import PFSeg3D
from imed_vision.models.segmentation.resunet import ResUNet3D
from imed_vision.models.segmentation.vnet import VNet
from imed_vision.comm.scheduler.poly import PolyLRScheduler
from imed_vision.comm.metrics import Metric
from imed_vision.datasets.atm import ATM
from imed_vision.datasets.brats import Brats20

from imed_vision.loss import CBCE, DiceLoss, FocalLoss
from imed_vision.loss.task_fusion import TaskFusionLoss

def dice_coeff(output, pred):
    inter = torch.sum(output & pred)
    union = torch.sum(output | pred)
    iou = inter / union
    dice = 2*inter/union
    return iou, dice

def main(config):
    init_lr = config["init_lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    data_dir = config["data_dir"]
    epochs = config["epochs"]
    lr_sche = config["lr_sche"]
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
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    ckpt_dir = config["ckpt_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]
    drop_last = False
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
        train_dataset = ATM(data_dir, txt_path=os.path.join(data_dir, "train.txt"), img_size=img_size, super_reso=super_reso,
                            upscale_rate=upscale_rate, crop_size=crop_size, augmentation=True, guide=guide)
        test_dataset = ATM(data_dir, txt_path=os.path.join(data_dir, "val.txt"), img_size=img_size, crop_size=crop_size,
                           super_reso=super_reso, upscale_rate=upscale_rate, augmentation=False, guide=guide)
    elif dataset == "brats20":
        train_dataset = Brats20(data_dir, mode="train", img_size=img_size, crop_size=crop_size, guide=guide, augmentation=True,
                                super_reso=super_reso, upscale_rate=upscale_rate)
        test_dataset = Brats20(data_dir, mode="val", img_size=img_size, crop_size=crop_size, guide=guide, augmentation=False,
                               super_reso=super_reso, upscale_rate=upscale_rate)
    else:
        raise ValueError("Unsupported dataset {}".format(dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              pin_memory=True,
                              drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
                             pin_memory=True)
    # optimizer
    optimizer = opt.SGD([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr,
                        momentum=momentum, nesterov=True, weight_decay=weight_decay)
    # optimizer = opt.Adam([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr)
    # optimizer = opt.AdamW([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr)
    if lr_sche == "poly":
        sche = PolyLRScheduler(optimizer, len(train_dataset), batch_size=batch_size,
                               epochs=epochs)
    elif lr_sche == "reducelr":
        sche = ReduceLROnPlateau(optimizer, mode="max", patience=20)
    else:
        raise ValueError("Unknown learning rate scheduler")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)
    else:
        if not os.path.exists(os.path.join(ckpt_dir, model_name, dataset)):
            os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)
    with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    writer = SummaryWriter(comment=model_name + "_" + dataset)

    if super_reso:
        sr_loss = nn.MSELoss()
    # if fusion:
    seg_loss = nn.CrossEntropyLoss()
    # dice_loss = DiceLoss()
    if num_classes == 1:
        seg_loss = nn.BCELoss()
    # rmi_loss = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    # if dataset == "atm":
    #     seg_loss = FocalLoss()
    #     if num_classes == 1:
    #         seg_loss = FocalLoss(cross_entropy=F.binary_cross_entropy)
    dice_loss = DiceLoss(smooth=1, gdice=False, p=1)
    task_fusion_loss = None
    if model_name == "pfseg":
        task_fusion_loss = TaskFusionLoss()

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
            guidance = None
            if super_reso:
                if guide:
                    x, hr, mask, guidance, guidance_mask = data
                    guidance = guidance.to(device, dtype=torch.float32)
                    guidance_mask = guidance_mask.to(device, dtype=torch.float32)
                else:
                    x, hr, mask = data
                hr = hr.to(device, dtype=torch.float32)
            else:
                x, mask = data
            if num_classes > 1:
                mask = mask.squeeze(1)
            x = x.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long if isinstance(seg_loss, nn.CrossEntropyLoss) or isinstance(seg_loss,
                                                                                                               CBCE) else torch.float32)
            if divide:
                mask = mask // 255
            optimizer.zero_grad()
            if model_name == "pfseg":
                pred = model(x, guidance)
            else:
                pred = model(x)
            bs = x.size(0)
            total += bs
            fusion_seg = None
            fusion_sr = None
            sr = None
            if super_reso:
                if model_name == "pfseg":
                    pred, sr = pred
                else:
                    pred, sr, fusion_sr, fusion_seg = pred
            if num_classes == 1:
                pred = torch.sigmoid(pred)
            print(pred.shape, mask.shape)
            loss = seg_loss(pred, mask)
            loss += dice_loss(pred, mask)
            if super_reso:
                w_sr = 1
                if model_name == "pf_seg":
                    w_sr = 0.5
                if sr is not None:
                    loss += w_sr*sr_loss(sr, hr)
                    # loss += sr_loss(sr, hr)

                if fusion_sr is not None:
                    loss += w_sr*sr_loss(fusion_sr, hr)

                if fusion_seg is not None:
                    # loss += rmi_loss(fusion_seg, mask)
                    if num_classes == 1:
                        fusion_seg = torch.sigmoid(fusion_seg)
                    loss += dice_loss(fusion_seg, mask)

                if task_fusion_loss is not None:
                    loss += 0.5*task_fusion_loss(pred, sr, mask*hr)
                if model_name == "pfseg":
                    loss += 0.5*sr_loss(guidance_mask*sr, guidance_mask*hr)
            loss.backward()
            optimizer.step()
            if isinstance(sche, PolyLRScheduler):
                sche.step()
            losses += loss.item()

            if num_classes > 1:
                pred = torch.softmax(pred, dim=1)
                pred = torch.max(pred, dim=1)[1]
            else:
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
            train_metric.update(pred.flatten(), mask.to(dtype=torch.long).flatten())
            result = train_metric.evaluate()
            show_metric = ["acc", "recall", "iou"]
            # if num_classes == 2:
            result_text = ""
            for metric in show_metric:
                if num_classes <= 2:
                    writer.add_scalar("train/{}".format(metric), result[metric][1], global_step=epoch)
                    result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                else:
                    writer.add_scalar("train/{}".format(metric), np.nanmean(result[metric].detach().cpu().numpy()[1:]),
                                      global_step=epoch)
                    result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].detach().cpu().numpy()[1:]))
            bar.set_description(
                "[{}:{}:{}] loss:{:.4f} {}".format(epochs, epoch + 1, iteration + 1, losses / total, result_text))
            iteration += 1
            global_step += 1
        dice = result['dice']
        if num_classes <= 2:
            dice = dice[1]
        else:
            dice = torch.nanmean(dice[1:])
        if isinstance(sche, ReduceLROnPlateau):
            sche.step(dice)
        model.eval()
        train_metric.reset()

        with torch.no_grad():
            for data in test_loader:
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
                mask = mask.to(device, dtype=torch.long if isinstance(seg_loss, nn.CrossEntropyLoss) else torch.float32)
                if num_classes > 1:
                    mask = mask.squeeze(1)
                if divide:
                    mask = mask // 255
                if model_name == "pfseg":
                    pred, _ = model(x, guidance)
                else:
                    pred = model(x)
                if num_classes < 2:
                    pred = torch.sigmoid(pred)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0
                else:
                    pred = torch.softmax(pred, dim=1)
                    pred = torch.max(pred, dim=1)[1]
                test_metric.update(pred.flatten(), mask.to(dtype=torch.long).flatten())
            show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
            result = test_metric.evaluate()
            result_text = ""
            if num_classes <= 2:
                iou = result["iou"][1]
            else:
                iou = np.nanmean(result["iou"].detach().cpu().numpy()[1:])
            for metric in show_metric:
                if num_classes <= 2:
                    writer.add_scalar("val/{}".format(metric), result[metric][1], global_step=epoch)
                    result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                else:
                    writer.add_scalar("val/{}".format(metric), np.nanmean(result[metric].detach().cpu().numpy()[1:]),
                                      global_step=epoch)
                    result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].detach().cpu().numpy()[1:]))
            test_metric.reset()
            print("[{}:{}] {}".format(epochs, epoch + 1, result_text))
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
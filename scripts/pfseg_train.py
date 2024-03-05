# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentaion_train
    author: 12718
    time: 2022/1/15 15:18
    tool: PyCharm
"""
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

from imed_vision.comm.scheduler.poly import PolyLRScheduler
from imed_vision.loss.tfm import TFM
from imed_vision.datasets.pfseg_dataset import get_paths, PFSegDataset
from imed_vision.models.segmentation import PFSeg
from imed_vision.comm.metrics import Metric


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
    gpu_index = config["gpu_index"]
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    ckpt_dir = config["ckpt_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]


    if model_name == "pfseg":
        model = PFSeg(channel, num_classes)
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
    train_paths, test_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths, random_state=0, test_size=0.2)
    train_dataset = PFSegDataset(train_paths, train_mask_paths, augmentation=True,
                                   output_size=image_size, super_reso=super_reso,
                                   upscale_rate=upscale_rate, divide=divide)
    test_dataset = PFSegDataset(test_paths, test_mask_paths, augmentation=False,
                                  output_size=image_size, super_reso=super_reso,
                                  upscale_rate=upscale_rate, divide=divide)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    optimizer = opt.SGD([{"params":model.parameters(), "initial_lr":init_lr}], lr=init_lr, momentum=momentum, 
                        nesterov=True, weight_decay=weight_decay)
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
            json.dump(config, f)
    else:
        if not os.path.exists(os.path.join(ckpt_dir, model_name, dataset)):
            os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)

    writer = SummaryWriter(comment=model_name+"_"+dataset)

    sr_loss = nn.MSELoss()
    seg_loss = nn.BCELoss()
    task_fusion_loss = TFM()
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
            x, hr, mask, guidance, ps_mask = data
            x = x.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)
            mask = mask.unsqueeze(1)
            guidance = guidance.to(device, dtype=torch.float32)
            ps_mask = ps_mask.to(device, dtype=torch.long)
            hr = hr.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = model(x, guidance)
            bs = x.size(0)
            total += bs
            pred, sr = pred
            pred = torch.sigmoid(pred)
            loss = seg_loss(pred, mask) + sr_loss(sr, hr) + task_fusion_loss(sr, pred, mask*hr)
            loss += sr_loss(sr*ps_mask, hr*ps_mask)
            loss.backward()
            optimizer.step()
            sche.step()
            losses += loss.item()
            train_metric.update(pred, mask)
            result = train_metric.evalutate()
            show_metric = ["acc", "recall", "iou"]
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
                x, hr, mask, guidance, ps_mask = data
                x = x.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)
                guidance = guidance.to(device, dtype=torch.float32)
                ps_mask = ps_mask.to(device, dtype=torch.long)
                hr = hr.to(device, dtype=torch.float32)
                mask = mask.unsqueeze(1)
                optimizer.zero_grad()
                pred = model(x, guidance)
                bs = x.size(0)
                total += bs
                pred, sr = pred
                pred = torch.sigmoid(pred)
                loss = seg_loss(pred, mask) + sr_loss(sr, hr) + task_fusion_loss(sr, torch.sigmoid(pred), mask*hr)
                loss += sr_loss(sr*ps_mask, hr*ps_mask)
                test_metric.update(pred, mask)
            show_metric = ["acc", "recall", "iou", "recall", "precision", "specificity", "dice"]
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
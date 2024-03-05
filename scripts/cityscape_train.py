# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cityscape_train.py
    author: 12718
    time: 2022/1/23 18:29
    tool: PyCharm
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import json
import tqdm
import glob
import warnings
import torch.backends.cudnn as cudnn
import random


from imed_vision.comm.scheduler.poly import PolyLRScheduler
from imed_vision.datasets.cityscape import get_paths, CityScapeDataset
from imed_vision.models.segmentation.deeplab import DeeplabV3Plus
from imed_vision.models.segmentation.espnets import ESPNetV2_Seg
from imed_vision.comm.metrics import Metric

def get_camvid_paths(root_dir, split):
    image_paths = glob.glob(os.path.join(root_dir, split, "*.png"))
    mask_paths = []
    for path in image_paths:
        mask_path = os.path.join(root_dir, split+"_labels", os.path.basename(path))
        mask_paths.append(mask_path)
    return image_paths, mask_paths

def main(config):
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    if args.distribution:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    args.gpu = gpu
    init_lr = config["init_lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    root_dir = config["root"]
    backbone = config["backbone"]
    epochs = config["epochs"]
    lr_sche = config["lr_sche"]
    image_size = config["image_size"]
    super_reso = config["super_reso"]
    fusion = config["fusion"]
    num_classes = config["num_classes"]
    model_name = config["model_name"]
    gpu_index = config["gpu_index"]
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    ckpt_dir = config["ckpt_dir"]
    if gpu_index != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    dataset = config["dataset"]

    if args.distribution:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distribution:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        batch_size = batch_size // ngpus_per_node
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)
    if model_name == "deeplabv3plus":
        model = DeeplabV3Plus(channel, num_classes, backbone=backbone, output_stride=16, super_reso=super_reso,
                              sr_seg_fusion=fusion, pretrained=True, upscale_rate=upscale_rate)
        model_name = model_name + "_" + backbone
    elif model_name == "espnetv2":
        model = ESPNetV2_Seg(channel, num_classes=num_classes, backbone=backbone,
                             super_reso=super_reso, upscale_rate=upscale_rate)
        model_name = model_name + "_" + backbone
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    train_paths, train_mask_paths = get_paths(root_dir, "train")
    test_paths, test_mask_paths = get_paths(root_dir, "val")
    if dataset == "camvid":
        train_paths, train_mask_paths = get_camvid_paths(root_dir, "train")
        test_paths, test_mask_paths = get_camvid_paths(root_dir, "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.distribution:
        device = torch.device("cuda:{}".format(gpu))
    model.to(device)
    if args.distribution:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[gpu])
    train_dataset = CityScapeDataset(train_paths, train_mask_paths, augmentation=True,
                                   output_size=image_size, super_reso=super_reso,
                                   upscale_rate=upscale_rate)
    test_dataset = CityScapeDataset(test_paths, test_mask_paths, augmentation=False,
                                  output_size=image_size, super_reso=super_reso,
                                  upscale_rate=upscale_rate)
    if args.distribution:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=True)

    optimizer = opt.SGD([{"params":model.parameters(), "initial_lr":init_lr}], lr=init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if lr_sche == "poly":
        sche = PolyLRScheduler(optimizer, len(train_dataset), batch_size=batch_size,
                               epochs=epochs, drop_last=True)
    else:
        raise ValueError("Unknown learning rate scheduler")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)
        # with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
        #     json.dump(config, f)
    else:
        if not os.path.exists(os.path.join(ckpt_dir, model_name, dataset)):
            os.makedirs(os.path.join(ckpt_dir, model_name, dataset))
        ckpt_dir = os.path.join(ckpt_dir, model_name, dataset)

    writer = SummaryWriter(comment=model_name+"_"+dataset)
    with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    if super_reso:
        sr_loss = nn.MSELoss()
    seg_loss = nn.CrossEntropyLoss(ignore_index=255)
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
            if super_reso:
                x, hr, mask = data
                hr = hr.to(device)
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
            if len(pred) == 2:
                if isinstance(model, ESPNetV2_Seg) and not super_reso:
                    pred, pred_l3 = pred
                elif super_reso:
                    pred, sr = pred
            elif len(pred) == 4:
                if isinstance(model, ESPNetV2_Seg):
                    pred, pred_l3, sr, fusion_seg = pred
                else:
                    pred, sr, fusion_seg, fusion_sr = pred
            loss = seg_loss(pred, mask)
            if isinstance(model, ESPNetV2_Seg):
                loss += seg_loss(pred_l3, mask)
            if super_reso:
                if sr is not None:
                    loss += sr_loss(sr, hr)
                if fusion_sr is not None:
                    loss += sr_loss(fusion_sr, hr)
                if fusion_seg is not None:
                    loss += seg_loss(fusion_seg, mask)
            loss.backward()
            optimizer.step()
            sche.step()
            losses += loss.item()
            pred = torch.softmax(pred, dim=1)
            train_metric.update(pred, mask)
            result = train_metric.evalutate()
            show_metric = ["acc", "recall", "iou"]
            result_text = ""
            for metric in show_metric:
                if num_classes <= 2:
                    writer.add_scalar("train/{}".format(metric), result[metric][1], global_step=epoch)
                    result_text += " {}:{:.4f}  ".format(metric, result[metric][1].cpu().item())
                else:
                    writer.add_scalar("train/mean_{}".format(metric), np.nanmean(result[metric].cpu().numpy()),
                                      global_step=epoch)
                    result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].cpu().numpy()))
            bar.set_description("[{}:{}:{}] loss:{:.4f} {} ".format(epochs, epoch+1, iteration+1, losses / total, result_text))
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

            if not args.distribution or (args.distribution and args.rank==0):
                show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
                result = test_metric.evalutate()
                result_text = ""
                if num_classes <= 2:
                    iou = result["iou"][1]
                else:
                    iou = np.nanmean(result["iou"].detach().cpu().numpy())
                for metric in show_metric:
                    if num_classes <= 2:
                        writer.add_scalar("validation/{}".format(metric), result[metric][1], global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                    else:
                        writer.add_scalar("validation/mean_{}".format(metric), np.nanmean(result[metric].cpu().numpy()),
                                          global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].cpu().numpy()))
                test_metric.reset()
                print("Validation [{}:{}] {}".format(epochs, epoch + 1, result_text))
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
    if not args.distribution or (args.distribution and args.rank==0):
        print("best iou:", best_iou)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation training sample")
    parser.add_argument("--json_path", type=str, help="path to config json file",
                        default="configs/cityscape_train.json")
    parser.add_argument("--rank", type=int, default=-1,
                        help="rank of the distribution program")
    parser.add_argument("--world-size", default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--distribution", action="store_true",
                        help="whether use distribution training")
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='ID of the gpu ')
    args = parser.parse_args()
    with open(args.json_path) as f:
        config = json.load(f)
    print(config)
    main(config)
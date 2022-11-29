# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:vessel_segmentaion_train
    author: 12718
    time: 2022/1/15 15:18
    tool: PyCharm
"""
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
import argparse
import json
import tqdm
import cv2
import warnings
import torch.backends.cudnn as cudnn
import random

from comm.scheduler.poly import PolyLRScheduler
from datasets.vessel_segmentation import get_paths, SegPathDataset
from datasets.crossmoda import CrossMoDA
from datasets.kits import get_kits19
from models.segmentation import Unet, SAUnet, NestedUNet, MiniUnet
from models.segmentation import DeeplabV3Plus
from layers.unet_blocks import *
from models.segmentation import ConvNeXtUNet
from models.segmentation.segformer import *
from models.segmentation import bisenetv2, bisenetv2_l, BiseNetV2
from models.segmentation import STDCNetSeg
from models.segmentation.scsnet import SCSNet
from comm.metrics import Metric
from models.segmentation.denseunet import Dense_Unet
from models.segmentation.dedcgcnee import DEDCGCNEE
from models.segmentation import DualLearning
from loss import RMILoss
from loss import SSIMLoss
from loss import CBCE
from loss import DetailLoss
from loss import DiceLoss
from loss.distance_loss import DisPenalizedCE


def get_ddr_paths(
        root_dir,
        split="train",
        ltype="EX",
        image_suffix=".jpg",
        mask_suffix=".tif",
        skip_path=False
     ):
    image_paths = glob.glob(os.path.join(root_dir, split, "image", "*{}".format(image_suffix)))
    final_image_paths = []
    mask_paths = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        filename = os.path.splitext(filename)[0]
        mask_path = os.path.join(root_dir, split, "label", ltype, filename+mask_suffix)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if skip_path:
            if mask.max() > 0:
                final_image_paths.append(image_path)
                mask_paths.append(mask_path)
        else:
            final_image_paths.append(image_path)
            mask_paths.append(mask_path)
        # mask_paths.append(mask_path)
    return final_image_paths, mask_paths

def main(config):
    args = parser.parse_args()
    gpu_index = config["gpu_index"]
    if gpu_index != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
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
    print(ngpus_per_node)
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
    channel = config["channel"]
    upscale_rate = config["upscale_rate"]
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    ckpt_dir = config["ckpt_dir"]

    dataset = config["dataset"]
    crop = config["crop"]
    distance = config["distance"]
    before = config["before"]
    drop_last = False
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
    elif model_name == "nestedunet":
        model = NestedUNet(num_classes=num_classes, input_channels=channel,
                           deep_supervision=True)
    elif model_name.lower() == "miniunet":
        model = MiniUnet(in_ch=channel, out_ch=num_classes, convblock=DoubleConv,
                     super_reso=super_reso, upscale_rate=upscale_rate,
                     sr_seg_fusion=fusion, sr_layer=3, base_ch=32)
    elif model_name == "saunet":
        model = SAUnet(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "convnextunet":
        model = ConvNeXtUNet(channel, num_classes)
    elif model_name.lower() == "deeplabv3plus":
        model = DeeplabV3Plus(channel, num_classes, backbone=block_name, super_reso=super_reso,
                              upscale_rate=upscale_rate, sr_seg_fusion=fusion, output_stride=8,
                              pretrained=config["pretrain"])
        drop_last = True
    elif model_name.lower() == "bisenetv2":
        model = bisenetv2(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "bisenetv2_l":
        model = bisenetv2_l(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "scsnet":
        model = SCSNet(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "stdcnet":
        pretrain = config["pretrain"]
        pretrained_model = config["pretrained"]
        model = STDCNetSeg(in_ch=channel, num_classes=num_classes,
                           backbone=block_name, pretrained=pretrain,
                           checkpoint=pretrained_model, boost=True, super_reso=super_reso,
                           upscale_rate=upscale_rate, fusion=fusion)
        drop_last = True
    elif model_name.lower() == "segformer":
        pretrain = config["pretrain"]
        pretrained_model = config["pretrained"]
        model = segformer_b5(img_size=image_size[0], num_classes=num_classes,
                             pretrain=pretrain, pretrained_model=pretrained_model)
    elif model_name.lower() == "denseunet":
        model = Dense_Unet(channel, num_classes, 64)
    elif model_name.lower() == "dedcgcnee":
        model = DEDCGCNEE(channel, num_classes, img_size=image_size)
    elif model_name.lower() == "supervessel":
        model = Unet(channel, num_classes, sr_layer=5, sr_seg_fusion=True,
                     super_reso=True, upscale_rate=upscale_rate, fim=True)
    elif model_name.lower() == "cogseg":
        model = Unet(channel, num_classes, sr_layer=5, before=False, sr_seg_fusion=False,
                     super_reso=True, l1=True, upscale_rate=upscale_rate)
        super_reso = True
    elif model_name.lower() == "dlnet":
        model = DualLearning(channel, num_classes, arch=config["block_name"],
                             upscale_rate=upscale_rate, pretrained=config["pretrain"])
        super_reso = True
    elif model_name.lower() == "ss_maf":
        model = Unet(channel, num_classes, sr_layer=5, before=True, sr_seg_fusion=True, ss_maf=True,
                     upscale_rate=upscale_rate, super_reso=True)
        super_reso = True
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
    elif dataset == "prime-fp20" and crop:
        fid_range = list(range(10))
        train_fids, test_fids = train_test_split(fid_range, random_state=0, test_size=0.2)
        train_paths = []
        train_mask_paths = []
        test_paths = []
        test_mask_paths = []
        for train_fid in train_fids:
            local_images = glob.glob(os.path.join(image_dir, "{:02d}".format(train_fid), "*{}".format(image_suffix)))
            local_masks = glob.glob(os.path.join(mask_dir, "{:02d}".format(train_fid), "*{}".format(mask_suffix)))
            train_paths += local_images
            train_mask_paths += local_masks
        for fid in test_fids:
            local_images = glob.glob(os.path.join(image_dir, "{:02d}".format(fid), "*{}".format(image_suffix)))
            local_masks = glob.glob(os.path.join(mask_dir, "{:02d}".format(fid), "*{}".format(mask_suffix)))
            test_paths += local_images
            test_mask_paths += local_masks

    elif dataset == "ddr":
        if num_classes <= 2:
            train_paths, train_mask_paths = get_ddr_paths(image_dir, "train", ltype=config["ltype"],
                                                          image_suffix=image_suffix, mask_suffix=mask_suffix)
            test_paths, test_mask_paths = get_ddr_paths(image_dir, "valid", ltype=config["ltype"],
                                                        image_suffix=image_suffix, mask_suffix=mask_suffix)
        else:
            train_paths, train_mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
            test_paths, test_mask_paths = get_paths(image_dir.replace("train", "valid"),
                                                          mask_dir.replace("train", "valid"), image_suffix,
                                                          mask_suffix)

    elif dataset == "crossmoda":
        image_dirs = glob.glob(os.path.join(image_dir, "*"))
        train_dirs, val_dirs = train_test_split(image_dirs, test_size=0.3)
        train_paths = []
        train_mask_paths = []
        for dirname in train_dirs:
            local_paths = glob.glob(os.path.join(dirname, "*.npy"))
            train_paths += local_paths
            train_mask_paths += [os.path.join(dirname, "mask", os.path.splitext(os.path.basename(p))[0]+".png") for p
                                 in local_paths]
        test_paths = []
        test_mask_paths = []
        for dirname in val_dirs:
            local_paths = glob.glob(os.path.join(dirname, "*.npy"))
            test_paths += local_paths
            test_mask_paths += [os.path.join(dirname, "mask", os.path.splitext(os.path.basename(p))[0] + ".png") for p
                                 in local_paths]
    elif dataset == "kits19":
        data_range = [61, 24, 30, 60, 56, 153, 89, 19, 157, 130, 54, 168, 51, 182, 143, 106, 139, 111, 183, 97,
                      150, 14, 27, 112, 189, 20, 46, 173, 205, 144, 62, 2, 59, 196, 204, 43, 10, 164, 73, 201, 203,
                      186, 152, 98, 3, 93, 123, 162, 200, 50, 113, 0, 94, 137, 95, 64, 146, 41, 69, 49, 48, 85, 207,
                      13, 155, 23, 78, 100, 131, 169, 208, 6, 68, 84, 121, 159, 178, 160, 91, 179, 11, 119, 102, 35, 57,
                      65, 1, 120, 209, 42, 105, 132, 181, 17, 38, 133, 53, 161, 128, 34, 28, 114, 163, 151, 31, 171,
                      127, 184, 32, 167, 142, 180, 147, 29, 177, 99, 82, 175, 79, 115, 148, 202, 72, 77, 25, 165, 81, 197,
                      174, 199, 39, 193, 58, 140, 88, 70, 87, 36, 21, 9, 103, 195, 67, 192, 117, 47, 172]
        train_range, test_range = train_test_split(data_range, test_size=0.3, random_state=0)
        train_paths, train_mask_paths = get_kits19(train_range, data_dir=image_dir)
        test_paths, test_mask_paths = get_kits19(test_range, data_dir=image_dir)
    elif dataset == "oct":
        data_dirs = glob.glob(os.path.join(image_dir, "*.fds"))
        train_dirs, test_dirs = train_test_split(data_dirs, test_size=0.3, random_state=0)
        train_paths = []
        train_mask_paths =[]
        for sub_dir in train_dirs:
            sub_paths = glob.glob(os.path.join(sub_dir, "*.png"))
            train_paths += sub_paths
            for path in sub_paths:
                train_mask_paths.append(os.path.join(os.path.dirname(path), "mask_11", os.path.basename(path)))
        test_paths = []
        test_mask_paths = []
        for sub_dir in test_dirs:
            sub_paths = glob.glob(os.path.join(sub_dir, "*.png"))
            test_paths += sub_paths
            for path in sub_paths:
                test_mask_paths.append(os.path.join(os.path.dirname(path), "mask_11", os.path.basename(path)))
        divide = False
    elif dataset == "fives":
        image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
        label_map = {"N": 0, "A": 1, "G": 2, "D": 3}
        class_labels = [label_map[os.path.splitext(path)[0].split("_")[1]] for path in image_paths]
        train_paths, test_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths,
                                                                                      random_state=0, test_size=0.2,
                                                                                      stratify=class_labels)
    elif dataset == "paip2019":
        with open(os.path.join(image_dir, "data.json"), "rb") as f:
            train_info = json.load(f)
        train_ids = train_info["train_ids"]
        test_ids = train_info["test_ids"]
        train_paths = []
        train_mask_paths = []
        for fid in train_ids:
            sub_paths = glob.glob(os.path.join(image_dir, fid, "*{}".format(image_suffix)))
            train_paths += sub_paths
            for path in sub_paths:
                train_mask_paths.append(os.path.join(os.path.dirname(path), os.path.basename(path)[:-4]+f"{mask_suffix}"))
        test_paths = []
        test_mask_paths = []
        for fid in test_ids:
            sub_paths = glob.glob(os.path.join(image_dir, fid, "*{}".format(image_suffix)))
            test_paths += sub_paths
            for path in sub_paths:
                test_mask_paths.append(
                    os.path.join(os.path.dirname(path), os.path.basename(path)[:-4] + f"{mask_suffix}"))
    elif dataset == "idrid":
        train_paths, train_mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
        test_paths, test_mask_paths = get_paths(image_dir.replace("train", "test"), mask_dir.replace("train", "test"),
                                                image_suffix, mask_suffix)
    else:
        test_size = 0.2
        image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
        train_paths, test_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths,
                                                                                      random_state=0, test_size=test_size)
    train_dataset = SegPathDataset(train_paths, train_mask_paths, augmentation=True,
                                   output_size=image_size, super_reso=super_reso,
                                   upscale_rate=upscale_rate, divide=divide, crop=False)
    test_dataset = SegPathDataset(test_paths, test_mask_paths, augmentation=False,
                                  output_size=image_size, super_reso=super_reso,
                                  upscale_rate=upscale_rate, divide=divide, crop=False)
    if dataset == "crossmoda":
        train_dataset = CrossMoDA(train_paths, train_mask_paths, augmentation=True, output_size=image_size,
                                  upscale_rate=upscale_rate, super_reso=super_reso)
        test_dataset = CrossMoDA(test_paths, test_mask_paths, augmentation=False,
                                  output_size=image_size, super_reso=super_reso,
                                  upscale_rate=upscale_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.distribution:
        device = torch.device("cuda:{}".format(gpu))
    model.to(device)
    if args.distribution:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[gpu])

    if args.distribution:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=True)
    #optimizer
    optimizer = opt.SGD([{"params":model.parameters(), "initial_lr":init_lr}], lr=init_lr,
                        momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if lr_sche == "poly":
        sche = PolyLRScheduler(optimizer, len(train_dataset), batch_size=batch_size,
                               epochs=epochs)
    else:
        raise ValueError("Unknown learning rate scheduler")

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
    writer = SummaryWriter(comment=model_name+"_"+dataset)

    if super_reso:
        sr_loss = nn.MSELoss()
        decouple_sr_loss = SSIMLoss(data_range=1)
    # if fusion:
    # rmi_loss = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    seg_loss = nn.CrossEntropyLoss()
    if num_classes == 1:
        seg_loss = nn.BCEWithLogitsLoss()
    if dataset == "ddr" and model_name.lower() == "ss_maf":
        seg_loss = CBCE()
    # rmi_loss = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    decouple_loss_seg = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    if model_name.lower() == "supervessel" or model_name.lower() == "ss_maf":
        decouple_loss_sr = nn.MSELoss()
    if model_name.lower() == "supervessel":
        decouple_loss_seg = nn.CrossEntropyLoss()
    # decouple_loss = DiceLoss(smooth=0, gdice=False)
    # seg_loss = DiceLoss(smooth=0, gdice=False)
    if isinstance(model, STDCNetSeg):
        detail_loss = DetailLoss()
        optimizer.add_param_group({"params" : detail_loss.parameters(), "initial_lr":init_lr })
        detail_loss.to(device)
    if distance:
        seg_loss = DisPenalizedCE()
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
            if super_reso:
                x, hr, mask = data
                hr = hr.to(device, dtype=torch.float32)
            else:
                x, mask = data
            x = x.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long if isinstance(seg_loss, nn.CrossEntropyLoss) or isinstance(seg_loss, CBCE) else torch.float32)
            if num_classes == 1:
                mask = mask.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            bs = x.size(0)
            total += bs
            fusion_seg = None
            fusion_sr = None
            sr = None
            if isinstance(pred, tuple):
                if isinstance(model, STDCNetSeg):
                    if super_reso:
                        if fusion:
                            pred, sr, fusion_seg, detail, out_s4, out_s5 = pred
                        else:
                            pred, sr, detail, out_s4, out_s5 = pred
                    else:
                        pred, detail, out_s4, out_s5 = pred
                elif isinstance(model, BiseNetV2):
                    pred, out_s1, out_s3, out_s4, out_s5 = pred
                else:
                    if model_name == "cogseg":
                        pred, sr, l1 = pred
                    else:
                        if len(pred) == 2:
                            pred, sr = pred
                        elif len(pred) == 4:
                            pred, sr, fusion_seg, fusion_sr = pred
            # print(pred.shape)

            if isinstance(model, NestedUNet):
                loss = seg_loss(pred[0], mask)
                for i in range(1, 4):
                    loss += seg_loss(pred[i], mask)
                pred = pred[-1]
            else:
                loss = seg_loss(pred, mask)
            if model_name == "cogseg":
                loss += l1
            if isinstance(model, STDCNetSeg):
                loss += detail_loss(detail, mask.float())
                loss += seg_loss(out_s4, mask)
                loss += seg_loss(out_s5, mask)
            elif isinstance(model, BiseNetV2):
                loss += seg_loss(out_s1, mask)
                loss += seg_loss(out_s3, mask)
                loss += seg_loss(out_s4, mask)
                loss += seg_loss(out_s5, mask)
            if super_reso:
                if sr is not None:
                    loss += sr_loss(sr, hr)
                    # loss += sr_loss(sr, hr)

                if fusion_sr is not None:
                    # loss += sr_loss(fusion_sr, hr)
                    loss += decouple_loss_sr(fusion_sr, hr)

                if fusion_seg is not None:
                    # loss += seg_loss(fusion_seg, mask)
                    # loss += rmi_loss(fusion_seg, mask)
                    loss += decouple_loss_seg(fusion_seg, mask)
                    # loss += dice(fusion_seg, mask)
                    # loss += seg_loss(fusion_seg, mask)
                    # loss += dice(fusion_seg, mask)
                    # loss += rmi_loss(fusion_seg, mask)
            loss.backward()
            optimizer.step()
            sche.step()
            losses += loss.item()
            if num_classes > 1:
                pred = torch.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)
            train_metric.update(pred, mask.to(dtype=torch.long))
            if not args.distribution or (args.distribution and args.rank == 0):
                result = train_metric.evaluate()
                show_metric = ["acc", "recall", "iou"]
                # if num_classes == 2:
                result_text = ""
                for metric in show_metric:
                    if num_classes <= 2:
                        writer.add_scalar("train/{}".format(metric), result[metric][1], global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                    else:
                        writer.add_scalar("train/{}".format(metric), np.nanmean(result[metric][1:].detach().cpu().numpy()[1:]),
                                          global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric][1:].detach().cpu().numpy()[1:]))
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
                if isinstance(model, STDCNetSeg):
                    pred = pred[0]
                elif isinstance(model, NestedUNet):
                    pred = pred[-1]
                if num_classes == 1:
                    pred = torch.sigmoid(pred)
                else:
                    pred = torch.softmax(pred, dim=1)
                test_metric.update(pred, mask.to(dtype=torch.long))
            if not args.distribution or (args.distribution and args.rank == 0):
                show_metric = ["acc", "recall", "iou", "recall", "precision", "specifity", "dice"]
                result = test_metric.evaluate()
                result_text = ""
                if num_classes <= 2:
                    iou = result["iou"][1]
                else:
                    iou = np.nanmean(result["iou"][1:].detach().cpu().numpy())
                for metric in show_metric:
                    if num_classes <= 2:
                        writer.add_scalar("validation/{}".format(metric), result[metric][1], global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                    else:
                        writer.add_scalar("validation/mean_{}".format(metric),
                                          np.nanmean(result[metric].cpu().numpy()),
                                          global_step=epoch)
                        result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric][1:].cpu().numpy()))
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
    if not args.distribution or (args.distribution and args.rank == 0):
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
    parser.add_argument('--dist-url', default='tcp://localhost:8888', type=str,
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
    image_size = config["image_size"]
    upscale_rate = config["upscale_rate"]
    super_reso = config["super_reso"]
    model_name = config["model_name"]
    fusion = config["fusion"]
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
        config["ckpt_dir"] = os.path.join("ckpt", config["dataset"],
                                          model_name+"_" + "_".join([str(s) for s in config["image_size"]])+mid, str(i))
        print(config["ckpt_dir"])
        main(config)
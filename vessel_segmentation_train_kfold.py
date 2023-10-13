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
from torch.utils.data import Subset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import os
import argparse
import json
import tqdm
import cv2
import warnings
import torch.backends.cudnn as cudnn
import random

from imed_vision.comm.scheduler.poly import PolyLRScheduler
from imed_vision.datasets.vessel_segmentation import get_paths, SegPathDataset
from imed_vision.datasets.crossmoda import CrossMoDA
from imed_vision.models.segmentation import Unet, SAUnet, NestedUNet, MiniUnet
from imed_vision.models.segmentation import DeeplabV3Plus
from imed_vision.layers.unet_blocks import *
from imed_vision.models.segmentation import ConvNeXtUNet
from imed_vision.models.segmentation.segformer import *
from imed_vision.models.segmentation import bisenetv2, bisenetv2_l, BiseNetV2
from imed_vision.models.segmentation import STDCNetSeg
from imed_vision.models.segmentation.scsnet import SCSNet
from imed_vision.comm.metrics import Metric
from imed_vision.models.segmentation.denseunet import Dense_Unet
from imed_vision.models.segmentation.dedcgcnee import DEDCGCNEE
from imed_vision.loss import RMILoss
from imed_vision.loss import SSIMLoss
from imed_vision.loss import CBCE
from imed_vision.loss import DetailLoss
from imed_vision.loss.distance_loss import DisPenalizedCE
from imed_vision.models.segmentation.pctunet import PCTUnet

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    for c in layer.children():
        reset_weights(c)
    layer.reset_parameters()

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
    elif model_name.lower() == "saunet":
        model = SAUnet(in_ch=channel, num_classes=num_classes)
    elif model_name.lower() == "pctunet":
       model = PCTUnet(img_size=image_size, in_ch=channel, out_ch=num_classes)
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
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    if crop and dataset == "hrf":
        train_range = list(range(15))
        train_paths = []
        for i in train_range:
            train_paths += glob.glob(os.path.join(image_dir, "{:02d}_*{}".format(i, image_suffix)))
        train_mask_paths = [os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0]) + mask_suffix for p in train_paths]

    elif dataset == "prime-fp20" and crop:
        fid_range = list(range(15))
        train_paths = []
        train_mask_paths = []
        for train_fid in fid_range:
            local_images = glob.glob(os.path.join(image_dir, "{:02d}".format(train_fid), "*{}".format(image_suffix)))
            local_masks = glob.glob(os.path.join(mask_dir, "{:02d}".format(train_fid), "*{}".format(mask_suffix)))
            train_paths += local_images
            train_mask_paths += local_masks
    else:
        train_paths, train_mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
    train_paths.sort(key=lambda x: os.path.basename(x))
    train_mask_paths.sort(key=lambda  x: os.path.basename(x))
    train_dataset = SegPathDataset(train_paths, train_mask_paths, augmentation=True,
                                   output_size=image_size, super_reso=super_reso,
                                   upscale_rate=upscale_rate, divide=divide, crop=False)

    if dataset == "crossmoda":
        train_dataset = CrossMoDA(train_paths, train_mask_paths, augmentation=True, output_size=image_size,
                                  upscale_rate=upscale_rate, super_reso=super_reso)

    kfold = KFold(n_splits=5, shuffle=True, random_state=66)
    if args.distribution:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.distribution:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[gpu])

    if args.distribution:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

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
        ssim_loss = SSIMLoss(data_range=1)
    # if fusion:
    # rmi_loss = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    seg_loss = nn.CrossEntropyLoss()
    if num_classes == 1:
        seg_loss = nn.BCEWithLogitsLoss()
    if dataset == "ddr":
        seg_loss = CBCE()
    rmi_loss = RMILoss(num_classes=num_classes, rmi_pool_way=1)
    # seg_loss = DiceLoss(smooth=0, gdice=False)
    fold_best_ious = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        train_sub_dataset = Subset(train_dataset, train_ids)
        test_sub_dataset = Subset(train_dataset, test_ids)
        test_sub_dataset.dataset.augmentation = False
        train_loader = DataLoader(train_sub_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                                  sampler=train_sampler)

        test_loader = DataLoader(test_sub_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=False, pin_memory=True, drop_last=False,
                                  sampler=None)
        model.apply(reset_weights)
        # optimizer
        optimizer = opt.SGD([{"params": model.parameters(), "initial_lr": init_lr}], lr=init_lr,
                            momentum=momentum, nesterov=True, weight_decay=weight_decay)
        if lr_sche == "poly":
            sche = PolyLRScheduler(optimizer, len(train_dataset), batch_size=batch_size,
                                   epochs=epochs)
        else:
            raise ValueError("Unknown learning rate scheduler")
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
        print("fold: {}".format(fold))
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
                        loss += ssim_loss(fusion_sr, hr)

                    if fusion_seg is not None:
                        # l
                        if model_name == "supervessel":
                            loss += seg_loss(fusion_seg, mask)
                        else:
                            loss += rmi_loss(fusion_seg, mask)
                        # loss += dice(fusion_seg, mask)
                        # loss += seg_loss(fusion_seg, mask)
                        # loss += dice(fusion_seg, mask)
                        # loss += rmi_loss(fusion_seg, mask)
                    # if distance:
                    #     pass
                        # loss += dice_loss(fusion_seg, mask)
                    # if fusion_sr is not None and fusion_seg is not None:
                    #     loss += fusion_loss(fusion_seg, fusion_sr)
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
                            writer.add_scalar("train/{}".format(metric), np.nanmean(result[metric].detach().cpu().numpy()[1:]),
                                              global_step=epoch)
                            result_text += "{}:{:.4f} ".format(metric, np.nanmean(result[metric].detach().cpu().numpy()[1:]))
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
                        iou = np.nanmean(result["iou"].detach().cpu().numpy())
                    for metric in show_metric:
                        if num_classes <= 2:
                            writer.add_scalar("validation/{}".format(metric), result[metric][1], global_step=epoch)
                            result_text += "{}:{:.4f} ".format(metric, result[metric][1].item())
                        else:
                            writer.add_scalar("validation/mean_{}".format(metric),
                                              np.nanmean(result[metric].cpu().numpy()),
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
                            "optimizer": optimizer.state_dict(),
                            "results": result
                        }
                        torch.save(
                            save_obj, os.path.join(ckpt_dir, "{}_{}_{}_best.pth".format(fold, model_name, dataset))
                        )
                    save_obj = {
                        "epoch": epoch,
                        "best_iou": best_iou,
                        "model": model.state_dict(),
                        "lr_sche": sche.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "results": result
                    }
                    torch.save(
                        save_obj, os.path.join(ckpt_dir, "{}_{}_{}_last.pth".format(fold, model_name, dataset))
                    )
        fold_best_ious["{}".format(fold)] = best_iou
    if not args.distribution or (args.distribution and args.rank == 0):
        print("best iou:", fold_best_ious.items())



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
    upscale_rate = config["upscale_rate"]
    config["image_size"] = [img_size // upscale_rate  for img_size in config["image_size"]]
    upscale_rate = config["upscale_rate"]
    super_reso = config["super_reso"]
    model_name = config["model_name"]
    fusion = config["fusion"]
    if super_reso:
        mid = "_sr_{}".format(upscale_rate)
        if fusion:
            mid = "_sr_fusion_{}".format(upscale_rate)
        if config["before"]:
            mid += "_before"
        else:
            mid += "_after"
    else:
        mid = ""
    config["ckpt_dir"] = os.path.join("ckpt", config["dataset"],
                                      model_name + "_" + "_".join([str(s) for s in config["image_size"]]) + mid + "_5fold")
    print(config["ckpt_dir"])
    main(config)
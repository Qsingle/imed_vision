# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/18 13:23
    @filename: segmentation.py
    @software: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch


class SegmentationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_inter = torch.zeros(self.num_classes)
        self.total_union = torch.zeros(self.num_classes)
        self.total_correct = 0
        self.total_labels = 0

    def update(self, pred, target):
        corrects, labels = batch_pixel_accuracy(pred, target)
        self.total_correct += corrects
        self.total_labels += labels
        area_inter, area_union = batch_inter_union(pred, target, self.num_classes)
        if self.total_union.device != area_union.device:
            self.total_union.to(area_union.device)
            self.total_union.to(area_inter.device)
        self.total_union += area_union
        self.total_inter += area_inter

    def evaluate(self):
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_labels)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        meanIoU = IoU.mean().item()
        dice = 1.0 * (2*self.total_inter) / (2.220446049250313e-16 + self.total_union)
        mean_dice = dice.mean().item()
        return pixAcc,meanIoU, mean_dice


def batch_pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy.
    Args:
        pred (Tensor): 4d or 3d tensor, if 4d (bs, nclass, h, w)
        target: 3d tensor, (bs, h, w)

    Returns:
        tuple(number of corrects pixel, number of labels)
    """
    if pred.dim() == 4:
        pred = torch.max(pred, dim=1)[1]
    pred = pred.long() + 1
    target = target.long() + 1
    total_labels = torch.sum(target > 0).item()
    corrects = torch.sum((pred == target)*(target > 0)).item()
    assert corrects < total_labels, "Number of corrects must small than number of labels"
    return corrects, total_labels

def batch_inter_union(pred, target, num_classes):
    if pred.dim() == 4:
        pred = torch.max(pred, dim=1)[1]
    mini = 1
    maxi = num_classes
    nbins = num_classes
    pred = pred.long() + 1
    target = target.long() + 1
    pred = pred * (target > 0)
    target = target * (target > 0)
    intersection = pred * (pred == target)
    area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
    area_out = torch.histc(pred, bins=nbins, min=mini, max=maxi)
    area_target = torch.histc(target, bins=nbins, min=mini, max=maxi)
    area_union = area_out + area_target - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.cpu(), area_union.cpu()
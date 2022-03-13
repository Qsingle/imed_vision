# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/13 11:09
    @filename: metric.py
    @software: PyCharm
"""

import torch
import numpy as np

from .confusion_matrix import confusion_matrix

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        self.matrix = torch.zeros((num_classes, num_classes))

    def update(self, output, target):
        if output.dim() == 4 and self.num_classes != 1:
            output = torch.max(output, dim=1)[1]
        if self.num_classes == 1:
            output = torch.where(output >= 0.5, 1, 0)
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        matrix = confusion_matrix(output, target, num_classes)
        if self.matrix.device != matrix.device:
            self.matrix = self.matrix.to(matrix.device)
        self.matrix += matrix.detach()

    def evalutate(self):
        result = dict()
        FP = self.matrix.sum(0) - torch.diag(self.matrix)
        FN = self.matrix.sum(1) - torch.diag(self.matrix)
        TP = torch.diag(self.matrix)
        TN = self.matrix.sum() - (FP + FN + TP)
        precision = TP / (TP + FP)
        acc = (TP + TN) / (TP+FP+FN+TN)
        recall = TP / (TP + FN)
        npv = TN/(TN+FN)
        fnr = FN / (TP+FN)
        fpr = FP / (FP+TN)
        mcc = (TP*TN-FP*FN) / torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # f1 =  2 * (precision * recall) / (precision + recall)
        specficity = TN / (TN + FP)
        iou = TP / (TP + FN +FP)
        dice = (2*TP) / (2*TP + FN + FP)
        result["FP"] = FP
        result["FN"] = FN
        result["TP"] = TP
        result["TN"] = TN
        result["precision"] = precision
        result["acc"] = acc
        result["dice"] = dice
        result["specifity"] = specficity
        result["iou"] = iou
        result["recall"] = recall
        result["mk"] = precision + npv - 1
        result["npv"] = npv
        result["mcc"] = mcc
        result["bm"] = (recall+specficity - 1)
        result["fnr"] = fnr
        result["fpr"] = fpr
        result["tpr"] = recall
        result["tnr"] = specficity
        return result
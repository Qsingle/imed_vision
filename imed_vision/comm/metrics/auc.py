# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:auc
    author: 12718
    time: 2022/10/14 17:39
    tool: PyCharm
"""
import torch

def fast_auc_calculate(labels, preds, n_bins=100):
    labels = labels.to(preds.device)
    postive_len = torch.sum(labels)
    negative_len = torch.sum(torch.where(labels==0,1,0))
    total_case = postive_len * negative_len

    bin_width = 1.0 / n_bins

    pos_bin = (preds[labels==1]/bin_width).int()
    neg_bin = (preds[labels==0]/bin_width).int()

    pos_histogram = torch.bincount(pos_bin).float()

    if len(pos_histogram) < n_bins:
        pos_histogram = torch.cat([pos_histogram,torch.zeros(n_bins - len(pos_histogram)).to(preds.device)],dim=0)

    neg_histogram = torch.bincount(neg_bin).float()
    if len(neg_histogram) < n_bins:
        neg_histogram = torch.cat([neg_histogram, torch.zeros(n_bins - len(neg_histogram)).to(preds.device)], dim=0)
    accumulated_neg = 0


    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / total_case.float()

def auc_calculate(labels, preds, n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    # negative_len = torch.sum(torch.where(labels==0,1,0))
    #negative_len = labels.size()[0] - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins+1)]
    neg_histogram = [0 for _ in range(n_bins+1)]
    bin_width = 1.0 / n_bins

    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1

    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)
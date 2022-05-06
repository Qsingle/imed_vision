# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:distance_loss
    author: 12718
    time: 2022/4/25 14:36
    tool: PyCharm
"""
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    GT = np.squeeze(GT)
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        # negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask
        pos_edt[posmask == 0] = 0
        # neg_edt = distance_transform_edt(negmask)
        # neg_edt = (np.max(neg_edt) - neg_edt) * negmask
        # res[i] = pos_edt / np.max(pos_edt) + neg_edt / np.max(neg_edt)
        pos_edt = pos_edt / np.max(pos_edt)
        res[i] = 1 + pos_edt

    return res


class DisPenalizedCE(torch.nn.Module):
    """
    Only for binary 3D segmentation
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(target.cpu().numpy() > 0.5)

        dist = torch.from_numpy(dist)
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1, )

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1, )
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss * dist
        # loss = F.nll_loss(inp_logs, target, size_average=False, reduction="sum") * dist

        return weighted_loss.mean()
# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:fa_loss
    author: 12718
    time: 2022/1/30 10:43
    tool: PyCharm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class FALoss(nn.Module):
    def __init__(self, sub_scale_rate=1/32):
        """
        Implementation of the feature affinity loss introduced in
        "Dual Super-Resolution Learning for Semantic Segmentation"
        <https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.pdf>
        Args:
            sub_scale_rate (float): the downsampling rate for the features
        """
        super(FALoss, self).__init__()
        self.subscale_rate = sub_scale_rate

    def forward(self, fe_sr, fe_seg):
        f1 = F.avg_pool2d(fe_sr, int(1/self.subscale_rate))
        f2 = F.avg_pool2d(fe_seg, int(1/self.subscale_rate))
        bs, ch, h, w = f1.size()
        f1 = f1.reshape(bs, -1, h*w)#[bs,ch,h*w]
        f2 = f2.reshape(bs, -1, h*w)
        mat1 = torch.bmm(f1.transpose(1, 2), f1)#[bs,h*w,h*w]
        mat2 = torch.bmm(f2.transpose(1, 2), f2)
        mat1 = mat1 / torch.norm(mat1, 2) #l2norm
        mat2 = mat2 / torch.norm(mat2, 2)
        fe = (mat2-mat1)
        fa_loss = torch.norm(fe, 1)
        return torch.mean(fa_loss)
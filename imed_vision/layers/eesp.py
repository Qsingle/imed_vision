# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:essp
    author: 12718
    time: 2022/2/14 10:56
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv2d


__all__ = ["EESP", "SESSP"]

class EESP(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, r_lim=7, K=4):
        """
        Implementation of the Extremely Efficient Spatial Pyramid module introduced in
        "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
        <https://arxiv.org/pdf/1811.11431.pdf>
        Parameters
        ----------
        in_ch (int): number of channels for input
        out_ch (int): number of channels for output
        stride (int): stride of the convs
        r_lim (int): A maximum value of receptive field allowed for EESP block
        K (int): number of parallel branches
        """
        super(EESP, self).__init__()
        hidden_ch = int(out_ch // K)
        hidden_ch1 = out_ch - hidden_ch*(K-1)
        assert hidden_ch1 == hidden_ch, \
            "hidden size of n={} must equal to hidden size of n1={}".format(hidden_ch, hidden_ch1)
        self.g_conv1 = Conv2d(in_ch, hidden_ch, 1, stride=1,
                              groups=K, activation=nn.PReLU(hidden_ch))

        self.spp_convs = nn.ModuleList()
        for i in range(K):
            ksize = int(3 + i * 2)
            dilation = int((ksize - 1) / 2) if ksize <= r_lim else 1
            self.spp_convs.append(nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_ch, bias=False))

        self.conv_concat = Conv2d(out_ch, out_ch, groups=K, activation=None)
        self.bn_pr = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        self.module_act = nn.PReLU(out_ch)
        self.K = K
        self.stride = stride

    def forward(self, x):
        net = self.g_conv1(x)
        outputs = [self.spp_convs[0](net)]
        for i in range(1, self.K):
            output_k = self.spp_convs[i](net)
            output_k = output_k + outputs[i-1]
            outputs.append(
                output_k
            )
        concat = torch.cat(outputs, dim=1)
        concat = self.bn_pr(concat)
        net = self.conv_concat(concat)
        if self.stride == 2:
            return net
        if net.size() == x.size():
            net = net + x
        net = self.module_act(net)
        return net

class SESSP(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, r_lim=7, K=4, refin=True, refin_ch=3):
        """
            Implementation of the Extremely Efficient Spatial Pyramid module introduced in
            "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
            <https://arxiv.org/pdf/1811.11431.pdf>
            Parameters
            ----------
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            stride (int): stride of the convs
            r_lim (int): A maximum value of receptive field allowed for EESP block
            K (int): number of parallel branches
            refin (bool): whether use the inference from input image
        """
        super(SESSP, self).__init__()
        eesp_out = out_ch - in_ch
        self.eesp = EESP(in_ch, eesp_out, stride=stride, r_lim=r_lim, K=K)
        self.avg_pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.refin = refin
        self.stride = stride
        self.activation = nn.PReLU(out_ch)
        if refin:
            self.refin_conv = nn.Sequential(
                Conv2d(refin_ch, refin_ch, ksize=3, stride=1, padding=1, activation=nn.PReLU(refin_ch)),
                Conv2d(refin_ch, out_ch, activation=None)
            )


    def forward(self, inputs, _ref=None):
        avgout = self.avg_pool(inputs)
        eesp_out = self.eesp(inputs)
        net = torch.cat([eesp_out, avgout], dim=1)
        if self.refin:
            w1 = avgout.shape[2]
            w2 = _ref.shape[2]
            while w2 != w1:
                _ref = F.avg_pool2d(_ref, kernel_size=3, stride=self.stride, padding=1)
                w2 = _ref.shape[2]
            _ref = self.refin_conv(_ref)
            net = net + _ref
        net = self.activation(net)
        return net
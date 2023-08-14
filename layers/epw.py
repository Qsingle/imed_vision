# -*- coding:utf-8 -*-
"""
    FileName: epw
    Author: 12718
    Create Time: 2023-01-08 15:20
"""
import math
import torch.nn as nn

from layers.utils import Conv2d

class EfficientPWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Efficient point wise conv
        https://github.com/sacmehta/EdgeNets/blob/2b232d3f7fb60658755dad1ebca0ffc895cc795e/nn_layers/efficient_pt.py#L10
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output

        Returns:
            None
        """
        super(EfficientPWConv, self).__init__()
        self.wt_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.groups = math.gcd(in_ch, out_ch)
        self.exp = Conv2d(in_ch, out_ch, 3, 1, 1, groups=self.groups, activation=nn.PReLU(out_ch))
        self.in_size = in_ch
        self.out_size = out_ch

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.exp(x)
        x = x*wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size}, groups={groups})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

if __name__ == "__main__":
    m = EfficientPWConv(32, 32)
    print(m)
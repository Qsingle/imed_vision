#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   convnext.py
    @Time    :   2023/09/07 14:44:45
    @Author  :   12718 
    @Version :   1.0
'''

import torch
import torch.nn as nn

from .layernorm import LayerNorm
from .dropout import DropPath

class Block(nn.Module):
    def __init__(self, dim:int, drop_path_rate:float=0., layer_scale_init_value:float=1e-6) -> None:
        """Block of the ConvNeXt
           "A ConvNet for the 2020s"<https://arxiv.org/abs/2201.03545>

        Args:
            dim (int): the number of dimension for the input tensor
            drop_path_rate (float, optional): The propabe of the drop path. Defaults to 0..
            layer_scale_init_value (float, optional): Initialization value of the layer scale. Defaults to 1e-6.
        """
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channel_last")
        #pointwise/1x1 Conv1, the official implementation use the Linear layer to do this
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value* torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0. else None
        self.drop_path = DropPath(dropout_rate=drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    def forward(self, x):
        residual = x
        net = self.dwconv(x).permute(0, 2, 3, 1) #conv->transpose
        net = self.norm(net)
        net = self.pwconv1(net)
        net = self.act(net)
        net = self.pwconv2(net)
        if self.gamma is not None:
            net = self.gamma*net
        net = net.permute(0, 3, 1, 2) #NHWC->NCHW
        net = self.drop_path(net) + residual
        return net

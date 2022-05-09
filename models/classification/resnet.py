# -*- coding:utf-8 -*-
# !/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: resnet.py
Time: 2020/10/09 09:35:55
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from collections import OrderedDict
import re

from layers.utils import *
from layers.splat import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'seresnet18', 'seresnet34',
           'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
           'seresnext101_32x8d', 'resnest50', 'resnest101', 'resnest200', 'resnest269',
           'resnest14', 'resnest26'
           ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'seresnet18': '',
    'seresnet34': '',
    'seresnet50': '',
    'seresnet101': '',
    'seresnet152': '',
    'seresnext50_32x4d': '',
    'seresnext101_32x8d': '',
    'resnest50': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth',
    'resnest101': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth',
    'resnest200': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth',
    'resnest269': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth',
    'resnest14': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth',
    'resnest26': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth'
}


def get_layers(num_layers):
    '''
        Get the number of blocks for each stage in resnet
        Args:
            num_layers (int): the number of layers for resnet
        Reference:
            "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    blocks = []
    if num_layers == 14:
        blocks = [1, 1, 1, 1]
    elif num_layers == 18 or num_layers == 26:
        blocks = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
        blocks = [3, 4, 6, 3]
    elif num_layers == 101:
        blocks = [3, 4, 23, 3]
    elif num_layers == 152:
        blocks = [3, 8, 36, 3]
    elif num_layers == 200:
        blocks = [3, 24, 36, 3]
    elif num_layers == 269:
        blocks = [3, 30, 48, 8]
    else:
        raise ValueError("Unknown number of layers {}".format(num_layers))
    return blocks


def drop_path(x: torch.Tensor, drop_rate=0.0, training=False):
    """

    Args:
        x:
        drop_rate:
        training:

    Returns:

    """
    if drop_rate > 0.0 and training:
        shape = (x.size()[0],) + (1,) * (x.ndim - 1)
        keep_rate = 1 - drop_rate
        rand = keep_rate + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        output = x.div(keep_rate) * rand
        return output
    else:
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, base_width=64,
                 groups=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 zero_bn_init=False, downsample=None, radix=0, avd=False, avd_first=False,
                 is_first=False, dropout_rate=0.0, reduction=4, semodule=None,
                 drop_path_rate=0.0, gn_groups=32):
        """
        Resnet's BasicBlock
        References:
            "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
        Args:
            inplanes (int): the number of channels of inputs
            planes (int): the base number of channels of input
            stride (Union[int, tuple]): the stride of the block. (The slide step size of the second conv)
            dilation (int): the dilation rate
            base_width (int): the base width of the block
            groups (int): the number of groups for the conv2
            norm_layer (nn.Module): the normalization layer
            activation (nn.Module): the activation
            zero_bn_init (bool): whether use zero to initialize the bn3
            downsample (nn.Module): the module to do the downample for the residual
            radix (int): the number of split in each cadinality
            avd (bool): whether use avd layer
            avd_first (bool): use the avd layer before or after conv2
            is_first (bool): whether the block is the first block for the stage
            dropout_rate (float): the dropout rate
            reduction (int): the reduction for the Split Attention Moduler
        """
        super(BasicBlock, self).__init__()
        assert base_width == 64, "BasicBlock noly support base_width = 64, but got {}".format(base_width)
        assert groups == 1, "BasicBlock only support groups = 1, but got {}".format(groups)
        if (dilation > 1):
            raise NotImplementedError("BasicBlock not support dilation > 1")
        self.conv1 = Conv2d(inplanes, planes, ksize=3, stride=stride, padding=dilation, dilation=dilation,
                            norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate,
                            gn_groups=gn_groups)
        self.conv2 = Conv2d(planes, planes, ksize=3, stride=1, padding=1, dilation=1,
                            norm_layer=norm_layer, activation=None, dropout_rate=dropout_rate,
                            gn_groups=gn_groups)
        self.activation = activation
        self.downsample = downsample
        self.se = semodule
        self.drop_path_rate = drop_path_rate
        if zero_bn_init:
            nn.init.zeros_(self.conv2.norm_layer.weight)


    def forward(self, x):
        identify = x
        net = self.conv1(x)
        net = self.conv2(net)
        if self.se is not None:
            net = self.se(net)
        if self.downsample is not None:
            identify = self.downsample(x)
        if self.drop_path_rate > 0.0:
            net = drop_path(net, self.drop_path_rate, self.training)
        net = net + identify
        net = self.activation(net)

        return net


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, base_width=64,
                 groups=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                 zero_bn_init=False, downsample=None, radix=0, avd=False, avd_first=False,
                 is_first=False, dropout_rate=0.0, reduction=4, semodule=None, drop_path_rate=0,
                 gn_groups=32):
        """
        References:
            "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
            "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
            https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
            "Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
            "ResNeSt: Split-Attention Networks"<https://arxiv.org/pdf/2004.08955.pdf>
        Args:
            inplanes (int): the number of channels of inputs
            planes (int): the base number of channels of input
            stride (Union[int, tuple]): the stride of the block. (The slide step size of the second conv)
            dilation (int): the dilation rate
            base_width (int): the base width of the block
            groups (int): the number of groups for the conv2
            norm_layer (nn.Module): the normalization layer
            activation (nn.Module): the activation
            zero_bn_init (bool): whether use zero to initialize the bn3
            downsample (nn.Module): the module to do the downample for the residual
            radix (int): the number of split in each cadinality
            avd (bool): whether use avd layer
            avd_first (bool): use the avd layer before or after conv2
            is_first (bool): whether the block is the first block for the stage
            dropout_rate (float): the dropout rate
            reduction (int): the reduction for the Split Attention Moduler
            drop_path_rate (float): the drop rate of path
        """
        super(Bottleneck, self).__init__()
        width = int((base_width / 64) * planes) * groups
        # if not provide norm_layer than use bn as the default
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv2d(inplanes, width, ksize=1, stride=1, padding=0, norm_layer=norm_layer,
                            activation=activation, dropout_rate=dropout_rate,
                            gn_groups=gn_groups)
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplAtConv2d(width, width, ksize=3, stride=stride, padding=dilation, dilation=dilation,
                                     groups=groups, radix=radix, drop_prob=dropout_rate, reduction=reduction,
                                     norm_layer=norm_layer, nolinear=activation)
        else:
            self.conv2 = Conv2d(width, width, ksize=3, stride=stride, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation=activation, dropout_rate=dropout_rate,
                                groups=groups, gn_groups=gn_groups)
        self.conv3 = Conv2d(width, planes * self.expansion, ksize=1, stride=1, padding=0,
                            norm_layer=norm_layer, activation=None, dropout_rate=dropout_rate,
                            gn_groups=gn_groups)
        self.downsample = downsample
        self.activation = activation
        self.se = semodule
        self.drop_path_rate = drop_path_rate
        if zero_bn_init:
            nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        identify = x
        net = self.conv1(x)
        # conv2
        if self.avd and self.avd_first:
            net = self.avd_layer(net)
        net = self.conv2(net)

        if self.avd and not self.avd_first:
            net = self.avd_layer(net)

        net = self.conv3(net)
        if self.se is not None:
            net = self.se(net)
        if not self.downsample is None:
            identify = self.downsample(x)
        if self.drop_path_rate > 0.0:
            net = drop_path(net, drop_rate=self.drop_path_rate, training=self.training)
        net = net + identify
        net = self.activation(net)
        return net


class ResNet(nn.Module):
    def __init__(self, block, blocks, in_ch=3, num_classes=1000, first_stride=2, light_head=False,
                 zero_init_residual=False,
                 groups=1, width_per_group=64, strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1], multi_grids=[1, 1, 1],
                 norm_layer=None,
                 se_module=None, reduction=4, radix=0, avd=False, avd_first=False, avg_layer=False, avg_down=False,
                 stem_width=64,
                 activation=nn.ReLU(inplace=True), dropout_rate=0.0, semodule_reduction=16, sigmoid=nn.Sigmoid(),
                 drop_path_rate=0.0, gn_groups=32):
        '''
            Modified resnet according to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
            Implementate  ResNet and the variation of ResNet.
            Args:
                in_ch: int, the number of channels of the input
                block: BasicBlock or Bottleneck.The block of the resnet
                num_classes: int, the number of classes to predict
                first_stride: int, the stride of the first conv layer
                light_head: boolean, whether use conv3x3 replace the conv7x7 in first conv layer
                zero_init_residual: whether initilize the residule block's batchnorm with zero
                groups: int, the number of groups for the conv in net
                width_per_group: int, the width of the conv layers
                strides: list, the list of the strides for the each stage
                dilations: list, the dilations of each block
                multi_grids: list, implementation of the multi grid layer in deeplabv3
                norm_layer: megengine.module.Module, the normalization layer, default is batch normalization
                se_module: SEModule, the Squeeze Excitation Module
                radix: int, the radix index from ResNest
                reduction: int, the reduction rate
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
                stem_width: int, the channels of the conv3x3 when use 3 conv3x3 replace conv7x7
                activation: nn.Module, the activation layer
                dropout_rate: float, the dropout rate
                semodule_reduction: int, the reduction rate of the SEModule
                sigmoid: nn.Module, the sigmoid of SEModule
                gn_groups (int): number of groups in group normalization
            References:
                "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
                "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
                https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
                deeplab v3: https://arxiv.org/pdf/1706.05587.pdf
                deeplab v3+: https://arxiv.org/pdf/1802.02611.pdf
                "Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
                "ResNeSt: Split-Attention Networks"<https://arxiv.org/pdf/2004.08955.pdf>
                "Group Normalization"<https://arxiv.org/pdf/1803.08494.pdf>
        '''
        super(ResNet, self).__init__()
        if len(dilations) != 4:
            raise ValueError("The length of dilations must be 4, but got {}".format(len(dilations)))

        if len(strides) != 4:
            raise ValueError("The length of dilations must be 4, but got {}".format(len(strides)))

        if len(multi_grids) > blocks[-1]:
            multi_grids = multi_grids[:blocks[-1]]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
        elif len(multi_grids) < blocks[-1]:
            raise ValueError(
                "The length of multi_grids must greater than or equal the number of blocks for last stage , but got {}/{}".format(
                    len(multi_grids), blocks[-1]))
        self.inplanes = stem_width * 2 if light_head else 64
        self.avg_down = avg_down
        self.avg_layer = avg_layer
        self.base_width = width_per_group
        self.multi_grids = multi_grids
        # entry part
        self.light_head = light_head
        if light_head:
            self.conv1 = nn.Sequential(
                Conv2d(in_ch, stem_width, ksize=3, stride=first_stride, padding=1, norm_layer=norm_layer,
                       activation=activation, gn_groups=gn_groups),
                Conv2d(stem_width, stem_width, ksize=3, stride=1, padding=1, norm_layer=norm_layer,
                       activation=activation, gn_groups=gn_groups),
                Conv2d(stem_width, self.inplanes, ksize=3, stride=1, padding=1, norm_layer=norm_layer,
                       activation=activation, gn_groups=gn_groups)
            )
        else:
            self.conv1 = Conv2d(in_ch, self.inplanes, ksize=7, stride=first_stride, padding=3, norm_layer=norm_layer,
                                activation=activation)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        # four stage
        self.layer1 = self._make_layer(block, 64, blocks[0], strides[0], dilation=dilations[0], semodule=se_module,
                                       avd=avd, avd_first=avd_first, radix=radix, norm_layer=norm_layer,
                                       activation=activation,
                                       zero_init_bn=zero_init_residual, groups=groups, dropout_rate=dropout_rate,
                                       reduction=reduction,
                                       semodule_reduction=semodule_reduction, sigmoid=sigmoid,
                                       drop_path_rate=drop_path_rate)
        self.layer2 = self._make_layer(block, 128, blocks[1], strides[1], dilation=dilations[1], semodule=se_module,
                                       avd=avd, avd_first=avd_first, radix=radix, norm_layer=norm_layer,
                                       activation=activation,
                                       zero_init_bn=zero_init_residual, groups=groups, dropout_rate=dropout_rate,
                                       reduction=reduction, semodule_reduction=semodule_reduction, sigmoid=sigmoid,
                                       drop_path_rate=drop_path_rate)
        self.layer3 = self._make_layer(block, 256, blocks[2], strides[2], dilation=dilations[2], semodule=se_module,
                                       avd=avd, avd_first=avd_first,
                                       radix=radix, norm_layer=norm_layer, activation=activation,
                                       zero_init_bn=zero_init_residual, groups=groups,
                                       dropout_rate=dropout_rate, reduction=reduction,
                                       semodule_reduction=semodule_reduction, sigmoid=sigmoid,
                                       drop_path_rate=drop_path_rate)
        self.layer4 = self._make_grid_layer(block, 512, blocks[3], strides[3], dilation=dilations[3],
                                            semodule=se_module,
                                            avd=avd, avd_first=avd_first,
                                            radix=radix, norm_layer=norm_layer, activation=activation,
                                            zero_init_bn=zero_init_residual, groups=groups,
                                            dropout_rate=dropout_rate, reduction=reduction,
                                            semodule_reduction=semodule_reduction, sigmoid=sigmoid,
                                            drop_path_rate=drop_path_rate)
        # exit part
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.inplanes, num_classes)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias != None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, block, planes, blocks, stride, dilation=1, semodule=None, avd=False,
                    avd_first=False, radix=0, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                    zero_init_bn=False, groups=1, dropout_rate=0.0, reduction=4, semodule_reduction=16,
                    sigmoid=nn.Sigmoid(), drop_path_rate=0.0):
        """
        Build ResNet's stage layer
        Args:
            block (nn.Module): the resnet block
            planes (int): the base channels of the block
            stride (Union[int, tuple]): the stride of the block. (The slide step size of the second conv)
            dilation (int): the dilation rate
            semodule (nn.Module): the SEModule
            avd (bool): whether use avd layer
            avd_first (bool): use the avd layer before or after conv2
            radix (int): the number of split in each cadinality
            norm_layer (nn.Module): the normalization layer
            activation (nn.Module): the activation
            zero_bn_init (bool): whether use zero to initialize the bn3
            groups (int): the number of groups for the conv2
            dropout_rate (float): the dropout rate
            reduction (int): the reduction for the Split Attention Module
            semodule_reduction (int): the reduction rate for SEModule
            sigmoid (nn.Module): the sigmoid activation function for SEModule
        Returns:
            The stage layer.
        """
        downsample = None
        if planes != self.inplanes * block.expansion or stride == 2:
            down_layers = []
            down_stride = stride
            if self.avg_layer:
                if self.avg_down:
                    avg_layer = nn.AvgPool2d(kernel_size=3, stride=down_stride, padding=1)
                    down_stride = 1
                else:
                    avg_layer = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_conv = Conv2d(self.inplanes, planes * block.expansion, ksize=1, stride=down_stride, padding=0,
                               norm_layer=norm_layer, activation=None)
            down_layers.append(down_conv)
            downsample = nn.Sequential(*down_layers)
        if semodule is not None:
            semodule = SEModule(planes * block.expansion, semodule_reduction,
                                norm_layer=norm_layer, sigmoid=sigmoid, activation=activation)
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, base_width=self.base_width, groups=groups,
                            norm_layer=norm_layer, activation=activation, zero_bn_init=zero_init_bn,
                            downsample=downsample,
                            radix=radix, avd=avd, avd_first=avd_first, is_first=True, dropout_rate=dropout_rate,
                            reduction=reduction, semodule=semodule, drop_path_rate=drop_path_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, base_width=self.base_width, groups=groups,
                                norm_layer=norm_layer, activation=activation, zero_bn_init=zero_init_bn, radix=radix,
                                avd=avd, avd_first=avd_first, dropout_rate=dropout_rate,
                                reduction=reduction, semodule=semodule, drop_path_rate=drop_path_rate))
        return nn.Sequential(*layers)

    def _make_grid_layer(self, block, planes, blocks, stride, dilation=1, semodule=None, avd=False,
                         avd_first=False, radix=0, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True),
                         zero_init_bn=False, groups=1, dropout_rate=0.0, reduction=4, semodule_reduction=16,
                         sigmoid=nn.Sigmoid(), drop_path_rate=0.0):
        """
        Build ResNet's stage layer with multi-grid from
        "Rethinking Atrous Convolution for Semantic Image Segmentation"<https://arxiv.org/abs/1706.05587>
        Args:
            block (nn.Module): the resnet block
            planes (int): the base channels of the block
            stride (Union[int, tuple]): the stride of the block. (The slide step size of the second conv)
            dilation (int): the dilation rate
            semodule (nn.Module): the SEModule
            avd (bool): whether use avd layer
            avd_first (bool): use the avd layer before or after conv2
            radix (int): the number of split in each cadinality
            norm_layer (nn.Module): the normalization layer
            activation (nn.Module): the activation
            zero_bn_init (bool): whether use zero to initialize the bn3
            groups (int): the number of groups for the conv2
            dropout_rate (float): the dropout rate
            reduction (int): the reduction for the Split Attention Module
            semodule_reduction (int): the reduction rate for SEModule
            sigmoid (nn.Module): the sigmoid activation function for SEModule
        Returns:
            The stage layer.
        """
        downsample = None
        if planes != self.inplanes * block.expansion or stride == 2:
            down_layers = []
            down_stride = stride
            if self.avg_layer:
                if self.avg_down:
                    avg_layer = nn.AvgPool2d(kernel_size=3, stride=down_stride, padding=1)
                    down_stride = 1
                else:
                    avg_layer = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_conv = Conv2d(self.inplanes, planes * block.expansion, ksize=1, stride=down_stride, padding=0,
                               norm_layer=norm_layer, activation=None)
            down_layers.append(down_conv)
            downsample = nn.Sequential(*down_layers)
        if semodule is not None:
            semodule = SEModule(planes * block.expansion, semodule_reduction,
                                norm_layer=norm_layer, sigmoid=sigmoid, activation=activation)
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation * self.multi_grids[0], base_width=self.base_width,
                  groups=groups,
                  norm_layer=norm_layer, activation=activation, zero_bn_init=zero_init_bn, downsample=downsample,
                  radix=radix, avd=avd, avd_first=avd_first, is_first=True, dropout_rate=dropout_rate,
                  reduction=reduction, semodule=semodule, drop_path_rate=drop_path_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation * self.multi_grids[i], base_width=self.base_width,
                      groups=groups,
                      norm_layer=norm_layer, activation=activation, zero_bn_init=zero_init_bn,
                      radix=radix, avd=avd, avd_first=avd_first, dropout_rate=dropout_rate,
                      reduction=reduction, semodule=semodule, drop_path_rate=drop_path_rate))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        features = []
        net = self.conv1(x)
        net = self.max_pool(net)
        net = self.layer1(net)
        features.append(net)
        net = self.layer2(net)
        features.append(net)
        net = self.layer3(net)
        features.append(net)
        net = self.layer4(net)
        features.append(net)
        return features


    def forward(self, x):
        # 执行计算的部分,构图,前向计算
        # net = self.conv1(x)
        # net = self.max_pool(net)
        # net = self.layer1(net)
        # net = self.layer2(net)
        # net = self.layer3(net)
        # net = self.layer4(net)
        features = self.forward_features(x)
        net = self.avg_pool(features[-1])
        net = self.fc(net)
        return net

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        new_state_dict = OrderedDict()

        try:
            super(ResNet, self).load_state_dict(state_dict, strict=strict)
        except:
            for key in state_dict.keys():
                match = re.match("bn?([1])\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    new_key = "conv" + groups[0] + "." + "norm_layer." + groups[1]
                    if self.light_head:
                        new_key = "conv" + groups[0] + ".2.norm_layer." + groups[1]
                    new_state_dict[new_key] = state_dict[key]
                match = re.match("conv?([0-9]).([0-9]{0,1})\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    new_key = "conv" + groups[0] + ".conv." + groups[1]
                    if len(groups) == 3:
                        map_d = {"0": "0", "3": "1", "6": "2"}
                        if groups[1] in ["0", "3", "6"]:
                            new_key = "conv" + groups[0] + "." + map_d[groups[1]] + ".conv." + groups[2]
                        if groups[1] in ["1", "4"]:
                            map_d = {"1": "0", "4": "1"}
                            new_key = "conv" + groups[0] + "." + map_d[groups[1]] + ".norm_layer." + groups[2]
                    new_state_dict[new_key] = state_dict[key]
                match = re.match("(layer[1-4].[0-9]{1,3}).bn([0-9])\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    # print(groups)
                    new_key = groups[0] + ".conv" + groups[1] + ".norm_layer." + groups[2]
                    new_state_dict[new_key] = state_dict[key]
                match = re.match("(layer[1-4].[0-9]{1,3}).conv([0-9])\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    new_key = groups[0] + ".conv" + groups[1] + ".conv." + groups[2]
                    if "conv" in groups[2] or "bn" in groups[2] or "fc" in groups[2]:
                        new_key = groups[0] + ".conv" + groups[1] + "." + groups[2]
                    # print(key)
                    new_state_dict[new_key] = state_dict[key]
                match = re.match("(layer[1-4].[0-9].downsample)\.([0-9])\\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    if groups[1] == "0":
                        new_key = groups[0] + ".0.conv." + groups[2]
                    elif groups[1] == "1" or groups[1] == "2":
                        new_key = groups[0] + ".0.norm_layer." + groups[2]
                        if self.avg_down:
                            if groups[1] != "1":
                                new_key = groups[0] + ".1.norm_layer." + groups[2]
                            else:
                                new_key = groups[0] + ".1.conv." + groups[2]
                    else:
                        raise ValueError("Unknown key:{}".format(key))
                    new_state_dict[new_key] = state_dict[key]

                match = re.match("fc\.?(.*)", key)
                if match is not None:
                    groups = match.groups()
                    new_key = "fc.1." + groups[0]
                    new_state_dict[new_key] = state_dict[key]
            try:
                super(ResNet, self).load_state_dict(new_state_dict, strict=strict)
            except Exception as e:
                self.state_dict().update(new_state_dict)


def _resnet(arch, block, blocks, pretrained=False, progress=True, **kwargs):
    '''
        Wraper to create the resnet.
        Args:
            arch (str):the arch to download the pretrained weights
            pretrained (bool): if True, download and load the pretrained weights
            progress (bool): if True, display the download progress
    '''
    model = ResNet(block, blocks, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress, model_dir="./weights")
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet("resnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_50_2", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_101_2", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def seresnet18(pretrained=False, progress=True, **kwargs):
    """SEResNet-18 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)


def seresnet34(pretrained=False, progress=True, **kwargs):
    """SEResNet-34 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)


def seresnet50(pretrained=False, progress=True, **kwargs):
    """SEResNet-50 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def seresnet101(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def seresnet152(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)


def seresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-50 32x4d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    kwargs["se_module"] = SEModule
    return _resnet("seresnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def seresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-101_32x8d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    kwargs["se_module"] = SEModule
    return _resnet("seresnext101_32x8d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def resnest14(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-14 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["stem_width"] = 32
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1]
    kwargs["light_head"] = True
    kwargs["reduction"] = 4
    return _resnet("resnest14", Bottleneck, get_layers(14), pretrained, progress, **kwargs)


def resnest26(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-26 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["stem_width"] = 32
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 2
    kwargs["light_head"] = True
    kwargs["reduction"] = 4
    return _resnet("resnest26", Bottleneck, get_layers(26), pretrained, progress, **kwargs)


def resnest50(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-50 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    kwargs["stem_width"] = 32
    kwargs["reduction"] = 4
    return _resnet("resnest50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)


def resnest101(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-101 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    kwargs["reduction"] = 4
    return _resnet("resnest101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def resnest200(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-200 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    kwargs["reduction"] = 4
    return _resnet("resnest200", Bottleneck, get_layers(200), pretrained, progress, **kwargs)


def resnest269(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-269 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 8
    kwargs["light_head"] = True
    kwargs["reduction"] = 4
    return _resnet("resnest269", Bottleneck, get_layers(269), pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = resnet18(pretrained=False)
    x = torch.randn((1, 3, 224, 224))
    device = torch.device("cpu")
    model.eval()
    x = x.to(device)
    model.to(device)
    with torch.no_grad():
        out = model(x)
        print(model)
        print(out.shape)

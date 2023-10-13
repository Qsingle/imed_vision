# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:espnets
    author: 12718
    time: 2022/2/14 11:13
    tool: PyCharm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from imed_vision.layers import Conv2d, EESP, SESSP

from .create_model import BACKBONE_REGISTER

__all__ = ["EspNetV2", "espnetv2_s_0_5", "espnetv2_s_1_0",
           "espnetv2_s_1_25", "espnetv2_s_1_5", "espnetv2_s_2_0"]

model_urls = {
    "espnetv2_s_0_5": "https://github.com/Qsingle/imed_vision/releases/download/V0.1/espnetv2_s_0_5.pth",
    "espnetv2_s_1_0": "https://github.com/Qsingle/imed_vision/releases/download/V0.1/espnetv2_s_1_0.pth",
    "espnetv2_s_1_25": "https://github.com/Qsingle/imed_vision/releases/download/V0.1/espnetv2_s_1_25.pth",
    "espnetv2_s_1_5": "https://github.com/Qsingle/imed_vision/releases/download/V0.1/espnetv2_s_1_5.pth",
    "espnetv2_s_2_0": "https://github.com/Qsingle/imed_vision/releases/download/V0.1/espnetv2_s_2_0.pth"
}


class EspNetV2(nn.Module):
    def __init__(self, in_ch=3, num_classes=1000, scale=1.0):
        """
            Implementation of the ESPNetV2 introduced in
            "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
            <https://arxiv.org/pdf/1811.11431.pdf>
            Parameters
            ----------
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            scale (float): the scale rate for the net
        """
        super(EspNetV2, self).__init__()
        reps = [0, 3, 7, 3] #how many times the essp block repeat
        r_lims = [13, 11, 9, 7, 5]
        K = [4] * len(r_lims)

        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * scale)
                base_s = math.ceil(base_s/ K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if scale <= 1.5:
            config.append(1024)
        elif scale <= 2.0:
            config.append(1280)
        else:
            ValueError("Configuration for scale={} not supported".format(scale))

        ref_input = in_ch
        self.reinf = True

        self.level1 = Conv2d(in_ch, config[0], 3, stride=2, padding=1, activation=nn.PReLU(config[0]))
        self.level2_0 = SESSP(config[0], config[1], stride=2, r_lim=r_lims[0], K=K[0],
                            refin=self.reinf, refin_ch=ref_input)

        self.level3_0 = SESSP(config[1], config[2], stride=2, r_lim=r_lims[1], K=K[1],
                              refin=self.reinf, refin_ch=ref_input)

        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, r_lim=r_lims[2], K=K[2]))

        self.level4_0 = SESSP(config[2], config[3], stride=2, r_lim=r_lims[2], K=K[2],
                              refin=self.reinf, refin_ch=ref_input)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, r_lim=r_lims[3], K=K[3]))

        self.level5_0 = SESSP(config[3], config[4], stride=2, r_lim=r_lims[3], K=K[3],
                              refin=self.reinf, refin_ch=ref_input)
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, r_lim=r_lims[4], K=K[4]))

        self.level5.append(Conv2d(config[4], config[4], ksize=3, stride=1, padding=1,
                                  groups=config[4], activation=nn.PReLU(config[4])))
        self.level5.append(Conv2d(config[4], config[5], ksize=1, stride=1, padding=0,
                                  groups=K[3], activation=nn.PReLU(config[5])))
        self.classifier = nn.Linear(config[5], num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward_features(self, inputs):
        features = []
        out_l1 = self.level1(inputs)
        features.append(out_l1)
        if not self.reinf:
            del inputs
            inputs = None
        out_l2 = self.level2_0(out_l1, inputs)
        features.append(out_l2)
        out_l3_0 = self.level3_0(out_l2, inputs)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        features.append(out_l3)
        outl4_0 = self.level4_0(out_l3, inputs)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(outl4_0)
            else:
                out_l4 = layer(out_l4)
        features.append(out_l4)
        outl5_0 = self.level5_0(out_l4, inputs)
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(outl5_0)
            else:
                out_l5 = layer(out_l5)
        features.append(out_l5)
        return features

    def forward(self, inputs):
        features = self.forward_features(inputs)
        net = F.adaptive_avg_pool2d(features[-1], 1)
        net = torch.flatten(net, 1)
        net = self.classifier(net)
        return net

def _espnerv2(arch, pretrained=False, progress=True, **kwargs):
    model = EspNetV2(**kwargs)
    if pretrained:
        if model_urls[arch] == '':
            print("The weights file of {} is not provided now, pass to load pretrained weights".format(arch))
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], model_dir="./weights", progress=progress)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_0_5(pretrained=False, progross=True, **kwargs):
    kwargs["scale"] = 0.5
    model = _espnerv2("espnetv2_s_0_5", pretrained=pretrained, progress=progross, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_0(pretrained=False, progross=True, **kwargs):
    kwargs["scale"] = 1.0
    model = _espnerv2("espnetv2_s_1_0", pretrained=pretrained, progress=progross, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_25(pretrained=False, progross=True, **kwargs):
    kwargs["scale"] = 1.25
    model = _espnerv2("espnetv2_s_1_25", pretrained=pretrained, progress=progross, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_5(pretrained=False, progross=True, **kwargs):
    kwargs["scale"] = 1.5
    model = _espnerv2("espnetv2_s_1_5", pretrained=pretrained, progress=progross, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_2_0(pretrained=False, progross=True, **kwargs):
    kwargs["scale"] = 2.0
    model = _espnerv2("espnetv2_s_2_0", pretrained=pretrained, progress=progross, **kwargs)
    return model


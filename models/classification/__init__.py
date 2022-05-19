# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:__init__
    author: 12718
    time: 2022/5/14 14:46
    tool: PyCharm
"""
from .efficientnetv2 import *
from .espnets import *
from .genet import *
from .mixformer import *
from .resnet import *
from .stdcnet import stdcnet_1, stdcnet_2

from .create_model import create_backbone, BACKBONE_REGISTER

def get_backbone_list():
    return BACKBONE_REGISTER.obj_dict.keys()
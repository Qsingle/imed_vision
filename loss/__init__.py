# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:__init__.py
    author: 12718
    time: 2022/1/30 10:42
    tool: PyCharm
"""
from .ssim import SSIMLoss
from .dice import DiceLoss
from .fa_loss import FALoss
from .tfm import TFM
from .ibloss import IBLoss
from .rmi import RMILoss
from .detail_loss import DetailLoss
from .ce import CBCE, FocalLoss
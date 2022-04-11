# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:decoder
    author: 12718
    time: 2022/4/7 15:19
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class MultiDecoder(nn.Module):
    def __init__(self, in_chs):
        super(MultiDecoder, self).__init__()
        self.layer_list = nn.ModuleList()

# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:create_model
    author: 12718
    time: 2022/5/14 14:50
    tool: PyCharm
"""
from comm.register import Register

BACKBONE_REGISTER = Register("backbone_register")

def create_backbone(name, **kwargs):
    return BACKBONE_REGISTER.get(name)(**kwargs)
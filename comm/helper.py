# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/3/3 11:11
    @filename: helper.py
    @software: PyCharm
"""
from itertools import repeat
import collections.abc

def to_tuple(input, n):
    """
    Get a tuple with n data
    Args:
        input (Union[int, tuple]): the input
        n (int): the number of data

    Returns:
        A tuple with n data
    """
    if isinstance(input, collections.abc.Iterable):
        assert len(input) == n, "tuple len is not equal to n: {}".format(input)
        spatial_axis = map(int, input)
        value = tuple(spatial_axis)
        return value
    return tuple(repeat(input, n))
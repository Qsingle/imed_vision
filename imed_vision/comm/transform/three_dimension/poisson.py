#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   poisson.py
    @Time    :   2023/09/06 13:50:50
    @Author  :   12718 
    @Version :   1.0
'''

import numpy as np
from numpy import ndarray

__all__ = ["poisson_noise_3d"]

def poisson(img, lambd=1):
    return np.random.poisson(lambd, size=img.shape)

def poisson_noise_3d(vox:ndarray, lambd_limit=(0.1, 0.3), p=0.5):
    if isinstance(lambd_limit, (tuple, list)):
        if lambd_limit[0] < 0:
            raise ValueError("Lower lambd_limit should be non negative.")
        if lambd_limit[1] < 0:
            raise ValueError("Upper lambd_limit should be non negative.")
        lambd_limit = lambd_limit
    elif isinstance(lambd_limit, (int, float)):
        if lambd_limit < 0:
            raise ValueError("lambd_limit should be non negative.")

        lambd_limit = (0, lambd_limit)
    else:
        raise TypeError(
            "Expected lambd_limit type to be one of (int, float, tuple, list), got {}".format(type(lambd_limit))
        )
    if np.random.random() < p:
        lamdb = np.random.uniform(lambd_limit[0], lambd_limit[1])
        noise = poisson(vox, lamdb)
        noised = vox + noise
        return noised
    return vox
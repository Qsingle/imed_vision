#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   gaussian.py
    @Time    :   2023/09/05 10:08:17
    @Author  :   12718 
    @Version :   1.0
'''

import numpy as np


def gaussian_3d(img, mean=0, var=1):
    sigma = var ** 0.5
    random_state = np.random.RandomState(np.random.randint(0, 2 ** 32 - 1, dtype=np.uint32))
    gaussian = random_state.normal(mean, sigma, img.shape)
    return gaussian


def gaussian_noise_3d(img, var_limit=(10.0, 50.0), mean=0, p=0.5):
    if isinstance(var_limit, (tuple, list)):
        if var_limit[0] < 0:
            raise ValueError("Lower var_limit should be non negative.")
        if var_limit[1] < 0:
            raise ValueError("Upper var_limit should be non negative.")
        var_limit = var_limit
    elif isinstance(var_limit, (int, float)):
        if var_limit < 0:
            raise ValueError("var_limit should be non negative.")

        var_limit = (0, var_limit)
    else:
        raise TypeError(
            "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
        )
    if np.random.random() < p:
        var = np.random.uniform(var_limit[0], var_limit[1])
        noised = img + gaussian_3d(img, mean, var)
        return noised
    return img
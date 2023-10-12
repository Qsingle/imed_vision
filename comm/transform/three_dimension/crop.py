#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   crop.py
    @Time    :   2023/09/04 15:24:41
    @Author  :   12718 
    @Version :   1.0
'''
import random
import numpy as np
from numpy import ndarray
from typing import Union, List, Tuple

def random_crop3d(img:ndarray, label:ndarray, img_size:Union[List, Tuple], crop_size:Union[List, Tuple]) -> Tuple[ndarray, ndarray]:
    """AI is creating summary for random_crop

    Args:
        img (ndarray): the input vox array
        label (ndarray): the input label file
        img_size (Union[List, Tuple]): the size of the image
        crop_size (Union[List, Tuple]): the crop size

    Returns:
        Tuple[ndarray, ndarray]: croped image and croped label
    """
    start_z = random.randint(0, img_size[0] - crop_size[0]) if crop_size[0] < img_size[0] else None
    start_x = random.randint(0, img_size[1] - crop_size[1]) if crop_size[1] < img_size[1] else None
    start_y = random.randint(0, img_size[2] - crop_size[2]) if crop_size[2] < img_size[2] else None
    if start_z is None or start_x is None or start_y is None:
        if (crop_size[0] - img_size[0]) % 2 == 0:
            p_z1 = p_z2 = (crop_size[0] - img_size[0]) // 2 if start_z is None else 0
        else:
            if start_z is None:
                if random.random() < 0.5:
                    p_z1 = (crop_size[0] - img_size[0]) // 2 + 1
                    p_z2 = (crop_size[0] - img_size[0]) // 2
                else:
                    p_z1 = (crop_size[0] - img_size[0]) // 2
                    p_z2 = (crop_size[0] - img_size[0]) // 2 + 1
            else:
                p_z1 = p_z2 = 0
        if (crop_size[1] - img_size[1]) % 2 == 0:
            p_x1 = p_x2 = (crop_size[1] - img_size[1]) // 2 if start_x is None else 0
        else:
            if start_x is None:
                if random.random() < 0.5:
                    p_x1 = (crop_size[1] - img_size[1]) // 2 + 1
                    p_x2 = (crop_size[1] - img_size[1]) // 2
                else:
                    p_x1 = (crop_size[1] - img_size[1]) // 2
                    p_x2 = (crop_size[1] - img_size[1]) // 2 + 1
            else:
                p_x1 = p_x2 = 0
        if (crop_size[2] - img_size[2]) % 2 == 0:
            p_y1 = p_y2 = (crop_size[2] - img_size[2]) // 2 if start_y is None else 0
        else:
            if start_y is None:
                if random.random() < 0.5:
                    p_y1 = (crop_size[2] - img_size[2]) // 2 + 1
                    p_y2 = (crop_size[2] - img_size[2]) // 2
                else:
                    p_y1 = (crop_size[2] - img_size[2]) // 2
                    p_y2 = (crop_size[2] - img_size[2]) // 2 + 1
            else:
                p_y1 = p_y2 = 0
        img = np.pad(img, [(p_z1, p_z2), (p_x1, p_x2), (p_y1, p_y2)], mode="constant")
        label = np.pad(label, [(p_z1, p_z2), (p_x1, p_x2), (p_y1, p_y2)], mode="constant")
        start_z = 0 if start_z is None else start_z
        start_x = 0 if start_x is None else start_x
        start_y = 0 if start_y is None else start_y
    crop_img = img[start_z:(start_z + crop_size[0]), start_x:(start_x + crop_size[1]), start_y:(start_y + crop_size[2])]
    crop_label = label[start_z:start_z+crop_size[0], start_x:(start_x+crop_size[1]),start_y:(start_y+crop_size[2])]
    return crop_img, crop_label
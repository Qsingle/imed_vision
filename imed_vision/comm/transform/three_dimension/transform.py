#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   transform.py
    @Time    :   2023/09/05 10:20:34
    @Author  :   12718 
    @Version :   1.0
'''
import random
import cv2
import numpy as np

__all__ = ["random_flip_3d", "random_shift_3d", "random_rotate_3d"]

def random_flip_3d(img, label=None, p=0.5):
    if random.random() < p:
        if random.random() < p:
            for i in range(img.shape[0]):
                img[i, :, :] = cv2.flip(img[i, :, :], 0)
            if not (label is None):
                for i in range(label.shape[0]):
                    label[i, :, :] = cv2.flip(label[i, :, :], 0)
        else:
            for i in range(img.shape[0]):
                img[i, :, :] = cv2.flip(img[i, :, :], 1)
            if not (label is None):
                for i in range(label.shape[0]):
                    label[i, :, :] = cv2.flip(label[i, :, :], 1)
    return img, label

def random_shift_3d(img, label, p=0.5):
    if random.random() < p:  #Shift
        vertical = np.random.randint(-img.shape[1] // 8, img.shape[1] // 8)
        horizon = np.random.randint(-img.shape[1] // 8, img.shape[1] // 8)
        M_img = np.float32([[0, 1, horizon], [1, 0, vertical]])
        for i in range(img.shape[0]):
            img[i, :, :] = cv2.warpAffine(img[i, :, :], M_img, (img.shape[1], img.shape[2]))
        for i in range(label.shape[0]):
            label[i, :, :] = cv2.warpAffine(label[i, :, :], M_img, (label.shape[1], label.shape[2]))
    return img, label

def random_rotate_3d(img, label, p=0.5, max_degree=45):
    if random.random()<p: 
        degree=np.random.randint(0, max_degree)
        M_img = cv2.getRotationMatrix2D(((img.shape[1]-1)/2.0,(img.shape[2]-1)/2.0),degree,1)
        M_label=cv2.getRotationMatrix2D(((label.shape[1]-1)/2.0,(label.shape[2]-1)/2.0),degree,1)
        for i in range(img.shape[0]):
            img[i,:,:]=cv2.warpAffine(img[i,:,:], M_img, (img.shape[1],img.shape[2]))
        for i in range(label.shape[0]):
            label[i, :, :] = cv2.warpAffine(label[i, :, :], M_label, (label.shape[1], label.shape[2]))
    return img, label
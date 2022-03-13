# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:feature_visualization
    author: 12718
    time: 2022/2/22 15:05
    tool: PyCharm
"""
import glob
import os
import torch
import cv2
from models.segmentation.unet import Unet
from albumentations import Compose, Normalize, Resize
import numpy as np

seg_cnt = 0
sr_cnt = 0
cnt = 0
dataset_name = "skin"
output_dir = "../{}".format(dataset_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_file_list = glob.glob(os.path.join("D:/workspace/datasets/segmentation/ISIC2016/ISBI2016_ISIC_Part3B_Training_Data/*.jpg"))
def seg_feature_plot(module:torch.nn.Module, input, output):
    global seg_cnt
    global dataset_name
    feature = torch.mean(output, dim=1)
    # feature = torch.tanh(feature)
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    feature = feature * 255
    feature = feature.squeeze().detach().cpu().numpy()
    feature = cv2.applyColorMap(np.uint8(feature), cv2.COLORMAP_JET)
    filename = os.path.basename(image_file_list[seg_cnt])[:-4] + "_seg_fe.png"
    cv2.imwrite(os.path.join(output_dir, filename), feature)
    seg_cnt += 1

def sr_feature_plot(module:torch.nn.Module, input, output):
    global sr_cnt
    global dataset_name
    feature = torch.mean(output, dim=1)
    # feature = torch.tanh(feature)
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    feature = feature * 255
    feature = feature.squeeze().detach().cpu().numpy()
    feature = cv2.applyColorMap(np.uint8(feature), cv2.COLORMAP_JET)
    filename = os.path.basename(image_file_list[sr_cnt])[:-4] + "_sr_fe.png"
    cv2.imwrite(os.path.join(output_dir, filename), feature)
    sr_cnt += 1

def weight_plot(module:torch.nn.Module, input, output):
    global cnt
    weights = output
    weights = weights*255
    weights = weights.squeeze().detach().cpu().numpy()
    filename = os.path.basename(image_file_list[cnt])
    filename = os.path.splitext(filename)[0]
    for i in range(weights.shape[0]):
        weight = weights[i]
        weight = np.uint8(weight)
        weight = cv2.applyColorMap(weight, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, filename+"_we_c_{}".format(i)), weight)

model = Unet(3, 2, super_reso=True, sr_seg_fusion=False, sr_layer=4, expansion=1)
state = torch.load("../unet_origin_skin_best.pth", map_location="cpu")
model.load_state_dict(state["model"])
# model.eval()

re_nor = Compose(
    [
        Resize(height=384, width=512, interpolation=cv2.INTER_CUBIC),
        Normalize([0.5]*3, [0.5]*3, always_apply=True)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.up9.register_forward_hook(seg_feature_plot)
# model.sr_up8.register_forward_hook(sr_feature_plot)
model.sr_seg_fusion_module.register_forward_hook(weight_plot)
model.to(device)

with torch.no_grad():
    for image_path in image_file_list:
        image = cv2.imread(image_path)
        x = re_nor(image=image)["image"]
        x = np.transpose(x, axes=[2, 0, 1])
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).to(device)
        model(x)
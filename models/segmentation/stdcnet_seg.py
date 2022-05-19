# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:stdcnet_seg
    author: 12718
    time: 2022/5/19 12:34
    tool: PyCharm
"""
import torch.nn as nn
import torch.nn.functional as F


from models.classification import create_backbone
from layers.bisenetv1_layers import ARM, FFM
from layers.stdc import ConvBNReLU

class SegHead(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(SegHead, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, hidden_ch, 3, 1, 1),
            nn.Conv2d(hidden_ch, out_ch, 1)
        )

    def forward(self, x):
        return self.conv(x)

class STDCNetSeg(nn.Module):
    def __init__(self, in_ch, num_classes=19, backbone="stdcnet_1", pretrained=False, checkpoint=None,
                 boost=False, use_conv_last=False):
        """
        Implementation of STDCNet for segmentation.
        "Rethinking BiSeNet For Real-time Semantic Segmentation"
        <https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf>
        Args:
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            backbone (str): name of the backbone model
            pretrained (bool): whether use pretrained backbone model
            checkpoint (str): path to the pretrained model
            boost (bool): whether use boost prediction
            use_conv_last (bool):
        """
        super(STDCNetSeg, self).__init__()
        assert backbone in ["stdcnet_1", "stdcnet_2"], "Only support stdcnet_1 and stdcnet_2"
        self.backbone = create_backbone(backbone, in_ch=in_ch, pretrained=pretrained,
                                        checkpoint=checkpoint,
                                        use_conv_last=use_conv_last)
        del self.backbone.fc
        del self.backbone.bn
        del self.backbone.linear
        self.arm_s4 = ARM(512, 128)
        self.arm_s5 = ARM(1024, 128)
        self.conv_avg = ConvBNReLU(1024, 128, 1, 1, 0)
        self.ffm = FFM(128, 256, 256)
        self.detail_head = SegHead(256, 64, 1)
        self.head = SegHead(256, 64, num_classes)
        self.boost = boost
        if boost:
            self.seghead_s4 = SegHead(512, 64, num_classes)
            self.seghead_s5 = SegHead(1024, 64, num_classes)


    def forward(self, x):
        features = self.backbone.forward_features(x)
        s3 = features[0]
        s4 = features[1]
        s5 = features[-1]
        arm_s5 = self.arm_s5(s5)
        arm_s4 = self.arm_s4(s4)
        avg = self.backbone.gpool(features[-1])
        avg = self.conv_avg(avg)
        avg = F.interpolate(avg, size=arm_s5.size()[2:], mode="nearest")
        arm_s5 = arm_s5 + avg
        arm_s5 = F.interpolate(arm_s5, size=arm_s4.size()[2:], mode="nearest")
        arm_s4 = arm_s5 + arm_s4
        arm_s4 = F.interpolate(arm_s4, size=s3.size()[2:], mode="nearest")
        feature_fuse = self.ffm(arm_s4, s3)
        seg_out = self.head(feature_fuse)
        detail = self.detail_head(s3)
        seg_out = F.interpolate(seg_out, size=x.size()[2:], mode="bilinear", align_corners=True)
        detail = F.interpolate(detail, size=x.size()[2:], mode="bilinear", align_corners=True)
        if self.boost:
            out_s4 = self.seghead_s4(s4)
            out_s5 = self.seghead_s5(s5)
            out_s4 = F.interpolate(out_s4, size=x.size()[2:], mode="bilinear", align_corners=True)
            out_s5 = F.interpolate(out_s5, size=x.size()[2:], mode="bilinear", align_corners=True)
            if self.training:
                return seg_out, detail, out_s4, out_s5
            return seg_out, out_s4, out_s5
        else:
            if self.training:
                return seg_out, detail
            else:
                return seg_out




if __name__ == "__main__":
    import torch
    import sys
    import os
    import re
    from collections import OrderedDict
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 224, 224)
    model = STDCNetSeg(3)
    model.eval()
    out = model(x)
    print(out.shape)
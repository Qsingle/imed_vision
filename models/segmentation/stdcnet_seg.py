# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:stdcnet_seg
    author: 12718
    time: 2022/5/19 12:34
    tool: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.classification import create_backbone
from layers.bisenetv1_layers import ARM, FFM
from layers.stdc import ConvBNReLU, STDC
from layers.spatial_fusion import SpatialFusion

__all__ = ["STDCNetSeg", "stdcnet_1_seg", "stdcnet_2_seg"]

class SRHead(nn.Module):
    def __init__(self, in_chs, out_ch):
        super(SRHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(sum(in_chs), 64, 1, 1, 0),
            nn.Conv2d(64, 16, 1, 1, 0),
            nn.Conv2d(16, out_ch, 1, 1, 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feas):
        size = feas[0].size()[2:]
        features = [F.interpolate(fea, size=size, mode="bilinear", align_corners=True) for fea in feas[1:]]
        features.insert(0, feas[0])
        feature = torch.cat(features, dim=1)
        net = self.conv1(feature)
        return net


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
    def __init__(self, in_ch=3, num_classes=19, backbone="stdcnet_1", pretrained=False, checkpoint=None,
                 boost=False, use_conv_last=False, super_reso=False, fusion=False, upscale_rate=2):
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
        self.super_reso = super_reso
        self.sr_fusion = fusion
        self.upscale_rate = upscale_rate
        if fusion:
            self.sr_seg_fusion = SpatialFusion(in_ch, num_classes, 64)
        if super_reso:
            self.sr_head = SRHead([256, 512], 64)
            self.sr = nn.Sequential(
                nn.Conv2d(64, 32, 1, 1),
                nn.Tanh(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.Conv2d(32, in_ch*(upscale_rate**2), 1, 1),
                nn.PixelShuffle(upscale_rate)
            )
        if boost:
            self.seghead_s4 = SegHead(512, 64, num_classes)
            self.seghead_s5 = SegHead(1024, 64, num_classes)


    def forward(self, x):
        features = self.backbone.forward_features(x)
        s3 = features[1]
        s4 = features[2]
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
        h, w = x.size()[2:]
        seg_out = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=True)
        detail = F.interpolate(detail, size=(h, w), mode="bilinear", align_corners=True)
        if self.super_reso:
            h = h * self.upscale_rate
            w = w * self.upscale_rate
            detail = F.interpolate(detail, size=(h, w), mode="bilinear", align_corners=True)
            seg_out = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=True)

        if self.boost:
            out_s4 = self.seghead_s4(s4)
            out_s5 = self.seghead_s5(s5)
            out_s4 = F.interpolate(out_s4, size=(h, w), mode="bilinear", align_corners=True)
            out_s5 = F.interpolate(out_s5, size=(h, w), mode="bilinear", align_corners=True)
            if self.super_reso and self.training:
                sr_fe = self.sr_head([s3, s4])
                sr_fe = F.interpolate(sr_fe, size=x.size()[2:], mode="bilinear", align_corners=True)
                sr = self.sr(sr_fe)
                if self.sr_fusion:
                    fusion = self.sr_seg_fusion(sr, seg_out)
                    fusion_seg_out = fusion*seg_out + seg_out
                    return seg_out, sr, fusion_seg_out, detail, out_s4, out_s5
                return seg_out, sr, detail, out_s4, out_s5
            if self.training:
                return seg_out, detail, out_s4, out_s5
            return seg_out, out_s4, out_s5
        else:
            if self.training:
                return seg_out, detail
            else:
                return seg_out

def stdcnet_1_seg(pretrained=False, checkpoint=None, **kwargs):
    kwargs["backbone"] = "stdcnet_1"
    return STDCNetSeg(pretrained=pretrained, checkpoint=checkpoint, **kwargs)

def stdcnet_2_seg(pretrained=False, checkpoint=None, **kwargs):
    kwargs["backbone"] = "stdcnet_2"
    return STDCNetSeg(pretrained=pretrained, checkpoint=checkpoint, **kwargs)

if __name__ == "__main__":
    import torch
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    x = torch.randn(1, 3, 224, 224)
    model = STDCNetSeg(3)
    model.eval()
    out = model(x)
    print(out.shape)
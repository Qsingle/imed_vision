# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cross_ma_resnet
    author: 12718
    time: 2022/9/14 15:27
    tool: PyCharm
"""
import torch
import torch.nn as nn
# from torchvision.models import resnet18
from torchvision.models import efficientnet_b3
from torchvision.models import resnet50
from torchvision.models import resnet34

def drop_path(x, dropout_rate=0., training=False, scale_by_keep: bool = True):
    """
    Implementation drop path
    Args:
        x (Tensor): input tensor
        dropout_rate (float): rate of drop out
        training (bool): whether training
    References:
        <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py>
    Returns:
        Tensor
    """
    if not training or dropout_rate == 0.:
        return x
    keep_prob = 1 - dropout_rate
    random_shape = (x.shape[0], ) + (1,) * (x.ndim -1)
    if hasattr(torch, "bernoulli"):
        random_tensor = torch.empty(random_shape, device=x.device, dtype=x.dtype)
        random_tensor = torch.bernoulli(random_tensor, p=keep_prob)
    else:
        random_tensor = torch.rand(random_shape, device=x.device, dtype=x.dtype) + keep_prob
        random_tensor.floor_()
    if scale_by_keep and keep_prob > 0.:
        random_tensor = torch.div(random_tensor, keep_prob)
    x = random_tensor * x
    return x

class DropPath(nn.Module):
    def __init__(self, dropout_rate=0., scale_by_keep=True):
        """
        Class decorator of drop path. Stochastic Depth
        Args:
            dropout_rate (float): probability of drop
            scale_by_keep (bool): whether use keep probability to scale the tensor
        """
        super(DropPath, self).__init__()
        self.drop_rate = dropout_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_rate, self.training, self.scale_by_keep)

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, reduction=0.25, norm_layer=None, sigmoid=None, activation=None):
        """
        SEModule of SENet and MobileNetV3
        Args:
            in_ch (int): the number of input channels
            reduction (int): the reduction rate
            norm_layer (nn.Module): the normalization module
            sigmoid （nn.Module): the sigmoid activation function for the last of fc
            activation (nn.Module): the middle activation function
        """
        super(SqueezeExcite, self).__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if sigmoid is None:
            sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        inter_channel = round(in_ch*reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, inter_channel, 1, 1, 0),
            activation,
            nn.Conv2d(inter_channel, in_ch, 1, 1, 0),
            sigmoid
        )
    def forward(self, x):
        net = self.avg_pool(x)
        net = self.fc(net) * x
        return net

class MBConv(nn.Module):
    """
        Inverted Bottleneck introduced in MobilenetV2
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"<https://arxiv.org/abs/1801.04381v4>
        In MobilenetV3, the block introduce the se layer.
        "Searching for MobileNetV3"<https://arxiv.org/abs/1905.02244>
    """
    def __init__(self, in_ch, out_ch, exp_ratio=4., ksize=3, stride=1, act_layer=nn.ReLU6(), norm_laryer=nn.BatchNorm2d,
                 se_layer=True, gate_layer=nn.Sigmoid(), has_skip=False, drop_path_rate=0., reduction=0.25, padding=1):
        super(MBConv, self).__init__()
        exp_ch = round(in_ch*exp_ratio)
        self.pw = nn.Conv2d(in_ch, exp_ch, 1, 1, 0)
        self.act = act_layer
        self.bn1 = norm_laryer(exp_ch)
        self.dw = nn.Conv2d(exp_ch, exp_ch, ksize, stride, dilation=1, groups=exp_ch, padding=padding)
        self.bn2 = norm_laryer(exp_ch)
        self.pw1 = nn.Conv2d(exp_ch, out_ch, 1, 1, 0)
        self.bn3 = norm_laryer(out_ch)
        self.se = SqueezeExcite(exp_ch, reduction=reduction,activation=act_layer,
                                sigmoid=gate_layer) if se_layer else nn.Identity()
        self.skip = has_skip or (in_ch != out_ch)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.shortcut = nn.Identity() if in_ch == out_ch and self.skip else nn.Conv2d(in_ch, out_ch, 1, 1)
        if stride == 2 and self.skip:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(2, 2) if stride == 2 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, 1, 1)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        net = self.pw(x)
        net = self.bn1(net)
        net = self.act(net)
        net = self.dw(net)
        net = self.bn2(net)
        net = self.se(net)
        net = self.pw1(net)
        net = self.bn3(net)
        if self.skip:
            net = self.drop_path(net) + shortcut
        return net


class MLP(nn.Module):
    def __init__(self, dim, expansion_rate=4., act_layer=nn.GELU(), drop_rate=0.):
        super(MLP, self).__init__()
        expansion_dim = int(dim * expansion_rate)
        self.fc1 = nn.Linear(dim, expansion_dim)
        self.drop1 = nn.Dropout(drop_rate)
        self.act = act_layer
        self.fc2 = nn.Linear(expansion_dim, dim)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        net = self.fc1(x)
        net = self.drop1(net)
        net = self.fc2(net)
        net = self.drop2(net)
        return net


class CrossAtt(nn.Module):
    def __init__(self, x_ch, y_ch, dim=128, num_head=4):
        super(CrossAtt, self).__init__()
        self.x_qkv = nn.Conv2d(x_ch, dim*3, 1, 1)
        self.y_qkv = nn.Conv2d(y_ch, dim*3, 1, 1)
        self.num_head = num_head
        h_dim = dim // num_head
        self.gamma = h_dim ** -0.5
        self.proj_x = nn.Conv2d(dim, x_ch, 1, 1)
        self.proj_y = nn.Conv2d(dim, y_ch, 1, 1)

    def forward(self, x, y):
        bs, ch, h_x, w_x = x.size()
        qkv_x = self.x_qkv(x).reshape(bs, 3, self.num_head, -1, h_x*w_x).permute(1, 0, 2, 4, 3) #3,bs,num_head, h*w, h_dim
        q_x, k_x, v_x = qkv_x.unbind(0)
        bs, ch, h_y, w_y = y.size()
        qkv_y = self.y_qkv(y).reshape(bs, 3, self.num_head, -1, h_y * w_y).permute(1, 0, 2, 4, 3)  # 3,bs,num_head, h*w, h_dim
        q_y, k_y, v_y = qkv_y.unbind(0)
        cross_x = q_x @ k_y.transpose(-2, -1) #bs, num_head, h_x*w_x, h_y*w_y
        cross_x = cross_x*self.gamma
        atten_x =torch.softmax(cross_x, dim=-1)
        out_x = atten_x @ v_y #bs, num_head, h_x*w_x, h_dim
        out_x = out_x.permute(0, 1, 3, 2).reshape(bs, -1, h_x, w_x)
        cross_y = q_y @ k_x.transpose(-2, -1)  # bs, num_head, h_x*w_x, h_y*w_y
        cross_y = cross_y * self.gamma
        atten_y = torch.softmax(cross_y, dim=-1)
        out_y = atten_y @ v_x  # bs, num_head, h_x*w_x, h_dim
        out_y = out_y.permute(0, 1, 3, 2).reshape(bs, -1, h_y, w_y)
        out_x = self.proj_x(out_x) + x
        out_y = self.proj_y(out_y) + y
        return out_x, out_y

class CrossGate(nn.Module):
    def __init__(self, x_ch, y_ch, dim, num_head):
        super(CrossGate, self).__init__()
        self.cross_att = CrossAtt(x_ch, y_ch, dim, num_head)

    def forward(self, x, y):
        x, y = self.cross_att(x, y)
        return x, y




class CrossMaResNet(nn.Module):
    def __init__(self, in_ch_x=3, in_ch_y=3, num_classes=10, pretrained=False):
        super(CrossMaResNet, self).__init__()
        # self.backbone_x = resnet18(pretrained=pretrained)
        # self.backbone_x = resnet50(pretrained=pretrained)
        # self.backbone_x = resnet34(pretrained=pretrained)
        self.backbone_x = efficientnet_b3(pretrained=pretrained)
        # del self.backbone_x.fc
        del self.backbone_x.classifier
        del self.backbone_x.avgpool
        if in_ch_x != 3:
            self.backbone_x.conv1 = nn.Conv2d(in_ch_x, 64, 7, 2, 3)
        self.backbone_y = resnet34(pretrained=pretrained)
        del self.backbone_y.fc
        if in_ch_y != 3:
            self.backbone_y.conv1 = nn.Conv2d(in_ch_y, 64, 7, 2, 3)
        # self.cross_att_stage3 = CrossGate(256, 256, 256, num_head=4)
        self.cross_att_stage4 = CrossGate(1536, 512, 1024, num_head=4)
        self.cross_fusion_x = nn.Sequential(
            nn.Conv2d(1536, 1536, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1536, 1536, 3, 1, 1)
        )
        self.cross_fusion_y = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1)
        )
        self.fc = nn.Sequential(
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, 1024),
            # nn.ReLU(),
            nn.Linear(2048, num_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Linear(1536, num_classes)
        self.fc_y = nn.Linear(512, num_classes)

    def backbone_forward(self, model, input):
        net = model.relu(model.bn1(model.conv1(input)))
        s1 = model.layer1(net)
        s2 = model.layer2(s1)
        s3 = model.layer3(s2)
        s4 = model.layer4(s3)
        return s4

    def forward(self, x, y):
        # x_s3 = self.backbone_forward(self.backbone_x, x)
        # x_s4 = self.backbone_x.features(x)
        # x_s4 = self.backbone_forward(self.backbone_x, x)
        x_s4 = self.backbone_x.features(x)
        y_s4 = self.backbone_forward(self.backbone_y, y)
        # y_s4 = self.backbone_y.layer4(y_s3)
        x_s4, y_s4 = self.cross_att_stage4(x_s4, y_s4)
        x_s4 = self.cross_fusion_x(x_s4)
        y_s4 = self.cross_fusion_y(y_s4)
        x_s4 = self.avg_pool(x_s4).flatten(1)
        y_s4 = self.avg_pool(y_s4).flatten(1)
        out = torch.cat([x_s4, y_s4], dim=1)
        out = self.fc(out)
        out_x = self.fc_x(x_s4)
        out_y = self.fc_y(y_s4)
        return out, out_x, out_y

if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512).cuda()
    y = torch.randn(1, 3, 256, 256).cuda()
    model = CrossMaResNet().cuda()
    out, out_x, out_y = model(x, y)
    print(out.shape)
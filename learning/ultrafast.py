from os import read
from numpy.lib.arraypad import pad
import torch
from torch import nn
import torchvision
import numpy as np
from torchvision.models.resnet import resnet18

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ultrafast(nn.Module):
    def __init__(self, size=(288,800), cls_dim=(37, 10, 4), use_aux=False, pretrained=False):
        super(ultrafast, self).__init__()
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim) # w+1 h c (lanes)
        self.cls_dim = cls_dim
        self.h = size[0]
        self.w = size[1]

        resnet = torchvision.models.resnet18(pretrained=pretrained)
        # print(resnet)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

        if self.use_aux:
            self.aux_header2 = nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1)
            )
            self.aux_header3 = nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1)
            )
            self.aux_header4 = nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1)
            )
            self.aux_combine = nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1)
            )

        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim)
        )
        self.pool = nn.Conv2d(512, 8, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        fea = self.pool(x4).view([-1, 1800])
        group_cls = self.cls(fea).view([-1, *self.cls_dim])

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(x4)
            x4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls

# model = ultrafast(use_aux=True)
# # print(model)

# inputs = torch.ones([2, 3, 288, 800])
# outputs = model(inputs)
# print(outputs[0].size()) # [2, 37, 10, 4]
# print(outputs[1].size()) # [2, 5, 36, 100]

# c = []
# a = torch.ones([2, 3])
# b = torch.zeros([2, 3])
# c.append(a)
# c.append(b)
# c = torch.cat(c)
# print (c.shape)

a = np.array([[[1,2,3,2], [4,5,6,5], [4,4,6,5]],[[1,2,3,1], [4,5,6,3], [4,5,5,5]]])
# a = torch.rand([2, 3, 4])
a = torch.tensor(a)
print(a)
print(a.shape)
# a = a.flip(1)
# print(a)


# b = a.sum(1)
# print(b)
# print(b.shape)


c = np.array([[[2]],[[3]]])
c = torch.tensor(c)
print(c)
print(c.shape)
d = a * c
print(d.shape)
print(d)

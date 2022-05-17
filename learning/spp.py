from numpy import pad
import torch
import torch.nn as nn

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(5,9,13)):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels, eps=0.001, momentum=0.03)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels * (len(kernels) + 1), out_channels, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act2 = nn.ReLU()
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in kernels])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(torch.cat([x] + [pool(x) for pool in self.pools], 1))
        x = self.bn2(x)
        x = self.act2(x)
        return x

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, k, s, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        x = self.bn(x)
        x = self.act(x)
        return x



x = torch.randn([1, 64, 32, 32])
spp = SPP(64, 128)
print(spp(x).shape)
# foc = Focus(64, 128)
# print(foc(x).shape)
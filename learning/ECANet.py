import torch
from torch import nn
import math

class ECANet(nn.Module):
    def __init__(self, channel, gamma = 12, b = 1):
        super(ECANet, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv1d(1, 1, kernel_size, padding = padding, bias = False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        pool_out = self.gap(x).view([b, 1, c])
        conv_out = self.conv(pool_out)
        out = self.sigmoid(conv_out).view([b, c, 1, 1])
        # print(out)
        return x * out

model = ECANet(512)
print(model)

inputs = torch.ones([2, 512, 26, 26])
# print(inputs)
outputs = model(inputs)
# print(outputs.size())
# print(outputs)
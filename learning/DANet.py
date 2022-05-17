import torch
from torch import nn

class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        # 将一个不可训练的Tensor转换成可训练的类型parameter, 并将该parameter绑定到module里(存在于net.parameter()中)
        self.softmax = nn.Softmax(dim=-1) # 在行上进行softmax
        # dim 维度顺序  输入二位张量  dim=0,列之和为1  dim=1,行之和为1
    
    def forward(self, x):
        b, c, h, w = x.size()
        query_out = self.query_conv(x).view([b, -1, h * w]).permute(0, 2, 1) # b h*w c' 维度换位
        key_out   = self.key_conv(x).view([b, -1, h * w])
        # calculate correlation
        energy = torch.bmm(query_out, key_out) # 矩阵乘法 bmm(a,b) 3D tensor a(b,n,m) b(b,m,p) => (b,n,p)
        # spatial normalize
        attention = self.softmax(energy) # b h*w h*w

        value_out = self.value_conv(x).view([b, -1, h * w])
        out = torch.bmm(value_out, attention.permute(0, 2, 1)) # b c h*w
        out = out.view([b, c, h, w])
        out = self.gamma * out + x

        return out

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1) # 在行上进行softmax
    
    def forward(self, x):
        b, c, h, w = x.size()
        query = x.view([b, c, h * w])
        key = x.view([b, c, h * w]).permute(0, 2, 1)
        energy = torch.bmm(query, key) # b c c
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy # ???
        # input dim keepdim: output same size as input except dim of size 1
        attention = self.softmax(energy)
        value = x.view([b, c, h * w])

        out = torch.bmm(attention, value) # b c h*w
        out = out.view([b, c, h, w])
        out = self.gamma * out + x
        return out

class DANet_attention(nn.Module):
    def __init__(self, in_channels):
        super(DANet_attention, self).__init__()
        self.channel_attention  = ChannelAttention()
        self.position_attention = PositionAttention(in_channels)
    def forward(self, x):
        ca_out = self.channel_attention(x)
        pa_out = self.position_attention(x)
        fusion = pa_out + ca_out
        return fusion

model = DANet_attention(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)
print(outputs.size())

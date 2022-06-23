import torch
from torch import nn
import numpy as np
from torch import functional as F
import time

yv, xv = torch.meshgrid(torch.arange(4), torch.arange(6))
print(yv)
print(xv)
grid = torch.stack((xv, yv), 2)
print(grid)
grid = grid.view(1, -1, 2)
print(grid)

shape = grid.shape[:2]
stride = 2.0
full = torch.full((*shape, 1), stride)
print(full)

pred = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
target = torch.tensor([5, 6, 7, 8, 1, 2, 3, 4])
pred = pred.view(-1, 4)
target = target.view(-1, 4)
p = torch.prod(pred[:, 2:], 1)
print(p)
tl = torch.min((pred[:, :2] - pred[:, 2:]), (target[:, :2] - target[:, 2:]))
br = torch.max((pred[:, :2] + pred[:, 2:]), (target[:, :2] + target[:, 2:]))
print(tl)
print(br)
en = (tl < br).type(tl.type())
print(en)
en = en.prod(dim=1)
print(en)
print(br - tl)
ai = torch.prod(br - tl, 1)
print(ai)
ai = ai * en
print(ai)

x = torch.randint(0, 6, (2, 3))
x = x.float()
print(x)

g = torch.Generator()
g.manual_seed(1)
print(torch.randperm(10, generator=g))

print(torch.arange(10))

x = torch.randn(2,3)
print(x)
print(x.view(-1))

x = torch.LongTensor(2)
print(x)


'''
x = torch.random([1, 3, 64, 64])

input_shape = [3,32,32]
dummy_input = torch.zeros([2,]+input_shape)
print(dummy_input.shape) # torch.Size([2, 3, 32, 32])

x_input = np.random.rand(1, 3, 64, 64).astype(dtype=np.float32)
print(x_input.shape) # (1, 3, 64, 64)
x_input = x_input.reshape(-1)
print(x_input.shape) # (12288,)
'''

# x = torch.rand([1,3,2,4])
# x = torch.ones([1,3,2,4])
# print(x.shape)
# print(x)
# x = x.view([3,2,-1])
# x = x.permute([0,1,3,2])
# x = x.transpose(2, 3)
# x = x.squeeze(0)
# x = x.unsqueeze(0)
# x = x.expand([5,3,2,4])
# x = torch.narrow(x, 3, 1, 2)
# x = x.repeat([2,1,1,1])
# x = x.unfold(3, 1, 2)
# x = x.resize_([3,2,4])
# x = torch.cat((x, x), 0)
# x = torch.stack((x, x), 0)

# print(x.shape)
# print(x)

# x = torch.Tensor([0.1, 0.2, 0.3])
# print(x.shape)
# x = x.repeat(4)[None]
# print(x.shape)
# print(x)

# x = torch.ones([1, 3, 4])
# y = torch.ones([2, 1, 4])
# z = x + y
# print(z.shape)

# x = torch.rand([8, 2])
# print(x)
# y = x[:, 0::2]
# z = x[:, 1::2]
# print(y)
# print(z)

# x = torch.rand([20])
# x = x[:,None]
# print(x.shape)


# x = np.array([[1,2],[3,4]])
# print(x)
# x = np.tile(x, 2)
# x = np.tile(x, (1,2))
# x = np.tile(x, (2, 1))
# print(x)

# print(x[:, ::4])
# print(x[:, 1::4])
# print(x[:, 2::4])
# print(x[:, 3::4])
# print(x[..., :])
# x = np.eye(3)[np.array([1,2,0], np.int32)]
# print(x)

# x = torch.rand([2,3])
# print(x)
# print(torch.max(x, 1))
# y = torch.rand([1,3])

# print(y)
# z = torch.max(x,y)
# print(z)

# index = torch.tensor([[1, 0], [0,2]])
# print(torch.gather(x, 1, index))

# index = torch.tensor([[1, 0, 1], [0, 1, 1]])
# print(torch.gather(x, 0, index))

# print(torch.sum(x))
# print(torch.sum(x, dim=0))
# print(torch.sum(x, dim=(1,), keepdim=True))

# print(torch.topk(x, k=2, dim=1, largest=False, sorted=False))
# print(torch.clamp(x, 0.3, 0.6))
# y = torch.rand([2,3])
# print(y)
# print(torch.where(x>0.5, x, y))

# print(x.mean(dim=0, keepdim=True))

# print(torch.mean(x, dim=0, keepdim=True))

'''
nn.AdaptiveAvgPool2d((7,7))
nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False)
nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
kernel_size = 3
stride = 1
padding = kernel_size // 2
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=padding)
nn.ReLU()
nn.Linear(in_features=10, out_features=7)
torch.float32
'''

''' conv + bn acceleration
module = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
bn_module = nn.BatchNorm2d(16, affine=True)

w = module.weight.data    # shape[16, 8, 3, 3]
b = module.bias.data      # shape[16] 可用全零代替
ws = [1] * len(w.size())  # [1, 1, 1, 1]
ws[0] = w.size()[0]       # [16, 1, 1, 1]

invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5) # shape[16]
w.mul_(invstd.view(*ws).expand_as(w))
b.add_(-bn_module.running_mean).mul_(invstd)

if bn_module.affine:
    w.mul_(bn_module.weight.data.view(*ws).expand_as(w))
    b.mul_(bn_module.weight.data).add_(bn_module.bias.data)
'''



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
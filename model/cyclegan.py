import torch
from torch.nn import Module

class ConvInstNormRelu(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, filters, stride):
        """`kernel_size` x `kernel_size` Convolution-InstanceNorm-ReLU layer with `filters` filters and `stride` stride"""
        super(self, ConvInstNormRelu).__init__([torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        torch.nn.InstanceNorm2d(), torch.nn.ReLU()])


# Added relu, not sure if needed
class ConvPair(Module):
    def __init__(self, in_channels, out_channels):
        self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return torch.relu(x)

class Generator(Module):
    def __init__(self):

        super(self, Generator).__init__()
        self.c1 = ConvInstNormRelu(256, 232, 7, 64, 1) # c7s1-64
        ConvInstNormRelu(232, 228, 3, 128, 1)  #d128
        ConvInstNormRelu(228, 224, 3, 256, 1)  # d256


    def forward(self, x):
        return x


class Discriminator(Module):
    def __init__(self):
        super(self, Discriminator).__init__()

    def forward(self, x):
        return x

#generator
"""

9 residual blocks for 256 × 256 or higher-resolution training images. Below, we follow
the naming convention used in the Johnson et al.’s Github
repository.

c7s1-k: denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1. 

dk: denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and
stride 2.

Rk: denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both
layer. 

uk: denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2


The network with 9 residual blocks consists of:
c7s1-64,d128,d256, 9 x R256,  u128, u64, c7s1-3
"""

#discriminator
"""

For discriminator networks, we use 70 × 70 PatchGAN 

Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 

After the last layer, we apply a convolution to produce a 1-dimensional output.

We do not use InstanceNorm for the first C64 layer. 

We use leaky ReLUs with a slope of 0.2. 

The discriminator architecture is:
C64-C128-C256-C512
"""
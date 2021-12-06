import torch
from icecream import ic
from torch.nn import Module


class ConvInstSigm(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """`kernel_size` x `kernel_size` Convolution-InstanceNorm-ReLU layer with `filters` filters and `stride` stride"""
        super(ConvInstSigm, self).__init__(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Sigmoid())

class ConvInstNormRelu(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """`kernel_size` x `kernel_size` Convolution-InstanceNorm-ReLU layer with `filters` filters and `stride` stride"""
        super(ConvInstNormRelu, self).__init__(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.InstanceNorm2d(num_features=out_channels), torch.nn.ReLU())

class TransposeConvInstNormRelu(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, *args, **keyargs):
        """`kernel_size` x `kernel_size` Convolution-Transpose-InstanceNorm-ReLU layer with `filters` filters and `stride` stride"""
        super(TransposeConvInstNormRelu, self).__init__(torch.nn.ConvTranspose2d( in_channels, out_channels, *args, **keyargs),
        torch.nn.InstanceNorm2d(out_channels), torch.nn.ReLU())

class ConvInstNormLeakyRelu(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """`kernel_size` x `kernel_size` Convolution-InstanceNorm-ReLU layer with `filters` filters and `stride` stride"""
        super(ConvInstNormLeakyRelu, self).__init__(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.InstanceNorm2d(out_channels), torch.nn.LeakyReLU(0.2))



class Generator(Module):
    def __init__(self):

        super(Generator, self).__init__()
        self.c1 = ConvInstNormLeakyRelu(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3) # c7s1-64
        self.c2 = ConvInstNormLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  #d128
        self.c3 = ConvInstNormLeakyRelu(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # d256

        self.r1 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256
        self.r2 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256
        self.r3 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256
        self.r4 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256
        self.r5 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256
        self.r6 = ConvInstNormLeakyRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # R256


        self.u1 = TransposeConvInstNormRelu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) #u128
        self.u2 = TransposeConvInstNormRelu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)  # u64

        self.c4 = ConvInstSigm(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)  # c7s1-3


    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)

        x = x + self.r1(x)
        x = x + self.r2(x)
        x = x + self.r3(x)
        x = x + self.r4(x)
        x = x + self.r5(x)
        x = x + self.r6(x)


        x = self.u1(x)
        x = self.u2(x)
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode='replicate')
        x = self.c4(x)
        #x = torch.sigmoid(x)
        return x


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # C64 without instance norm
        self.c2 = ConvInstNormLeakyRelu(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # C128
        self.c3 = ConvInstNormLeakyRelu(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)  # C256
        self.c4 = ConvInstNormLeakyRelu(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)  # C512
        self.o = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=2, padding=1)

    def forward(self, x):
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.o(x)
        #x = torch.sigmoid(x)
        x = x.mean([2,3]) #TODO loss for each patch
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
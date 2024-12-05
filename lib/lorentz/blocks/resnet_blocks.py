import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzConv2d,
    LorentzBatchNorm2d,
    LorentzReLU,
)


def get_Conv2d(manifold, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, LFC_normalize=False):
    return LorentzConv2d(
        manifold=manifold, 
        in_channels=in_channels+1, 
        out_channels=out_channels+1, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias, 
        LFC_normalize=LFC_normalize
    )

def get_BatchNorm2d(manifold, num_channels):
    return LorentzBatchNorm2d(manifold=manifold, num_channels=num_channels+1)

def get_Activation(manifold):
    return LorentzReLU(manifold)


class LorentzInputBlock(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, manifold: CustomLorentz, img_dim, in_channels, bias=True):
        super(LorentzInputBlock, self).__init__()

        self.manifold = manifold

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                img_dim,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, in_channels),
            get_Activation(self.manifold),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        return self.conv(x)


class LorentzBasicBlock(nn.Module):
    """ Basic Block for Lorentz ResNet-10, -18 and -34 """

    expansion = 1

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=True):
        super(LorentzBasicBlock, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold)

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels * LorentzBasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels * LorentzBasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    self.manifold, out_channels * LorentzBasicBlock.expansion
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1]-1) + res.narrow(-1, 1, res.shape[-1]-1)
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False):
        super(LorentzBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold)

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels * LorentzBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels * LorentzBottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    self.manifold, out_channels * LorentzBottleneck.expansion
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1]-1) + res.narrow(-1, 1, res.shape[-1]-1)
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out

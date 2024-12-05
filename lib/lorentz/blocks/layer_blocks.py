import torch
from torch.nn import Module

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzFullyConnected, 
    LorentzConv2d, 
    LorentzConvTranspose2d,
    LorentzBatchNorm1d,
    LorentzBatchNorm2d,
)

class LFC_Block(Module):
    """ Implementation of a hyperbolic fully-connected Block.

    Contains a hyperbolic linear layer followed by bnorm and ReLU-Activation.
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_features, 
            out_features, 
            bias=True,
            activation=None,
            normalization="None",
            LFC_normalize=False
        ):
        super(LFC_Block, self).__init__()

        self.manifold = manifold
        self.activation = activation
        self.normalization = normalization

        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = LorentzBatchNorm1d(num_features=out_features, manifold=self.manifold)

        self.linear = LorentzFullyConnected(
            manifold=self.manifold, 
            in_features=in_features, 
            out_features=out_features,
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        x = self.linear(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.manifold.lorentz_activation(x, self.activation)

        return x


class LConv2d_Block(Module):
    """ Implementation of a hyperbolic 2D-convolutional Block.

    Contains a hyperbolic 2D-convolutional layer followed by bnorm and ReLU-Activation.
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            bias=True,
            activation=None, # e.g. torch.relu
            normalization="None",
            LFC_normalize=False
        ):
        super(LConv2d_Block, self).__init__()

        self.manifold = manifold
        self.activation = activation
        self.normalization = normalization

        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = LorentzBatchNorm2d(num_channels=out_channels, manifold=self.manifold)

        self.conv = LorentzConv2d(
            manifold=self.manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias,
            LFC_normalize=LFC_normalize
        )


    def forward(self, x):
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.manifold.lorentz_activation(x, self.activation)

        return x


class LTransposedConv2d_Block(Module):
    """ Implementation of a 2D Transposed convolutional Block in hyperbolic Space.

    Contains a hyperbolic 2D-transposed convolutional layer followed by bnorm and ReLU-Activation.
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            bias=True,
            activation=None,
            normalization="None",
            LFC_normalize=False
        ):
        super(LTransposedConv2d_Block, self).__init__()

        self.manifold = manifold
        self.activation = activation
        self.normalization = normalization

        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = LorentzBatchNorm2d(num_channels=out_channels, manifold=self.manifold)

        self.trConv = LorentzConvTranspose2d(
            manifold=self.manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias,
            LFC_normalize=LFC_normalize
        )


    def forward(self, x):
        x = self.trConv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.manifold.lorentz_activation(x, self.activation)

        return x
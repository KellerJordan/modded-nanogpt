import torch
from torch import nn
import torch.nn.functional as F

class FC_Block(nn.Module):
    """ Implementation of a fully-connected block.

    Contains a standard linear layer followed by bnorm and ReLU-Activation.
    """
    def __init__(
            self, 
            in_features, 
            out_features, 
            activation=True, 
            normalization="None"
        ):
        super(FC_Block, self).__init__()

        self.activation = activation
        self.normalization = normalization

        self.linear = nn.Linear(in_features, out_features)

        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = nn.BatchNorm1d(num_features=out_features)


    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation:
            x = torch.relu(x)

        return x


class Conv2d_Block(nn.Module):
    """ Implementation of a 2D-convolutional block.

    Contains a standard 2D-convolutional layer followed by bnorm and ReLU-Activation.
    """
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            activation=None, # e.g. torch.relu
            bias=True,
            normalization="None"
        ):
        super(Conv2d_Block, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class TransposedConv2d_Block(nn.Module):
    """ Implementation of a 2D transposed convolutional block in Euclidean Space.

    Contains a standard 2D-transposed convolutional layer followed by bnorm and ReLU-Activation.
    """
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            activation=True, 
            bias=True,
            normalization="None"
        ):
        super(TransposedConv2d_Block, self).__init__()

        self.activation = activation
        self.bnorm = normalization=="batch_norm"

        self.trConv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        
        self.batch_norm = None
        if normalization=="batch_norm":
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.trConv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation:
            x = torch.relu(x)

        return x
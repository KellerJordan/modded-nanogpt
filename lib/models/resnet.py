import torch.nn as nn

from lib.Euclidean.blocks.resnet_blocks import BasicBlock, Bottleneck

from lib.lorentz.blocks.resnet_blocks import (
    LorentzBasicBlock,
    LorentzBottleneck,
    LorentzInputBlock,
)

from lib.lorentz.layers import LorentzMLR, LorentzGlobalAvgPool2d
from lib.lorentz.manifold import CustomLorentz

__all__ = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class ResNet(nn.Module):
    """ Implementation of ResNet models on manifolds. """

    def __init__(
        self,
        block,
        num_blocks,
        manifold: CustomLorentz=None,
        img_dim=[3,32,32],
        embed_dim=512,
        num_classes=100,
        bias=True,
        remove_linear=False,
    ):
        super(ResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim

        self.bias = bias
        self.block = block

        self.manifold = manifold

        self.conv1 = self._get_inConv()
        self.conv2_x = self._make_layer(block, out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block, out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2)
        self.avg_pool = self._get_GlobalAveragePooling()

        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(self.embed_dim*block.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)

        out_1 = self.conv2_x(out)
        out_2 = self.conv3_x(out_1)
        out_3 = self.conv4_x(out_2)
        out_4 = self.conv5_x(out_3)
        out = self.avg_pool(out_4)
        out = out.view(out.size(0), -1)

        if self.predictor is not None:
            out = self.predictor(out)

        return out

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            if self.manifold is None:
                layers.append(block(self.in_channels, out_channels, stride, self.bias))
            elif type(self.manifold) is CustomLorentz:
                layers.append(
                    block(
                        self.manifold,
                        self.in_channels,
                        out_channels,
                        stride,
                        self.bias
                    )
                )
            else:
                raise RuntimeError(
                    f"Manifold {type(self.manifold)} not supported in ResNet."
                )

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _get_inConv(self):
        if self.manifold is None:
            return nn.Sequential(
                nn.Conv2d(
                    self.img_dim,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=self.bias
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )

        elif type(self.manifold) is CustomLorentz:
            return LorentzInputBlock(
                self.manifold, 
                self.img_dim, 
                self.in_channels, 
                self.bias
            )

        else:
            raise RuntimeError(
                f"Manifold {type(self.manifold)} not supported in ResNet."
            )

    def _get_predictor(self, in_features, num_classes):
        if self.manifold is None:
            return nn.Linear(in_features, num_classes, bias=self.bias)

        elif type(self.manifold) is CustomLorentz:
            return LorentzMLR(self.manifold, in_features+1, num_classes)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

    def _get_GlobalAveragePooling(self):
        if self.manifold is None:
            return nn.AdaptiveAvgPool2d((1, 1))

        elif type(self.manifold) is CustomLorentz:
            return LorentzGlobalAvgPool2d(self.manifold, keep_dim=True)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

#################################################
#       Lorentz
#################################################
def Lorentz_resnet10(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-10 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBasicBlock, [1, 1, 1, 1], manifold, **kwargs)
    return model


def Lorentz_resnet18(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-18 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBasicBlock, [2, 2, 2, 2], manifold, **kwargs)
    return model


def Lorentz_resnet34(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-34 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBasicBlock, [3, 4, 6, 3], manifold, **kwargs)
    return model


def Lorentz_resnet50(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-50 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBottleneck, [3, 4, 6, 3], manifold, **kwargs)
    return model


def Lorentz_resnet101(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-101 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBottleneck, [3, 4, 23, 3], manifold, **kwargs)
    return model


def Lorentz_resnet152(k=1, learn_k=False, manifold=None, **kwargs):
    """Constructs a ResNet-152 model."""
    if not manifold:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ResNet(LorentzBottleneck, [3, 8, 36, 3], manifold, **kwargs)
    return model

#################################################
#       Euclidean
#################################################
def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

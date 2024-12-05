import torch.nn as nn

class BasicBlock(nn.Module):
    """ Basic Block for ResNet-10, -18 and -34 """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        out = out + res

        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    """ Residual block for ResNet with > 50 layers """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(Bottleneck, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        out = out + res

        out = self.activation(out)

        return out

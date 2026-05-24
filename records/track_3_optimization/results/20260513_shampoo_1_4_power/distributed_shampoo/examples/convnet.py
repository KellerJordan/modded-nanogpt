"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math

import torch
from torch import nn


class ConvNet(nn.Module):
    """Simple two-layer convolutional network for image classification.
    Takes in image represented by an order-3 tensor. Used for testing optimizers.

    Args:
        height (int): Height of image.
        width (int): Width of image.
        out_channels (int, optional): Number of output channels. Defaults to 64.
        disable_bias (bool, optional): Disable bias in the linear layer. Defaults to False.

    """

    def __init__(
        self,
        height: int,
        width: int,
        out_channels: int = 64,
        disable_bias: bool = False,
    ) -> None:
        super().__init__()
        KERNEL_SIZE, STRIDE, PADDING = 3, 1, 1
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=False,
        )
        self.activation = nn.ReLU()
        self.linear = nn.Linear(
            in_features=math.prod(
                ConvNet._infer_conv_output_shape(
                    input_shape=(height, width),
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    padding=PADDING,
                )
            )
            * out_channels,
            out_features=10,
            bias=(not disable_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.flatten(self.activation(self.conv(x)), start_dim=1))

    @staticmethod
    def _infer_conv_output_shape(
        input_shape: tuple[int, ...], kernel_size: int, stride: int, padding: int
    ) -> list[int]:
        output_shape = []
        for input_length in input_shape:
            output_length = (input_length - kernel_size + 2 * padding) / stride + 1
            assert output_length.is_integer(), (
                f"Stride {stride} is not compatible with input shape {input_shape}, kernel size {kernel_size} and padding {padding}!"
            )
            output_shape.append(int(output_length))
        return output_shape

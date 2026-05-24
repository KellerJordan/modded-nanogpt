"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math

import torch
from torch.nn.parameter import Parameter


class CombinedLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Compared to torch.nn.Linear, uses a combined parameter for both the weight and bias terms.
    This is useful for reducing the number of parameters for methods that exploit tensor structure, such as Shampoo.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        combined_weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features\_with\_bias})`.
            If :attr:`bias` is ``True``, :math:`\text{in\_features\_with\_bias} = \text{in\_features} + 1`;
            otherwise, :math:`\text{in\_features\_with\_bias} = \text{in\_features}`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias: the flag for using the bias term

    Call Args:
        input (torch.Tensor): tensor of shape (B, I),
            where B is batch size and I is number of elements in each input sample (i.e. `in_features`).

    Returns:
        output (torch.Tensor): tensor of shape (B, O), where B is batch size
            and O is number of elements in each output sample (i.e. `out_features`).

    Examples::
        >>> m = CombinedLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_with_bias: int = in_features + 1 if bias else in_features
        self.bias = bias
        self.combined_weight = Parameter(
            torch.empty(
                (self.out_features, self.in_features_with_bias),
                device=device,
                dtype=dtype,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.bias:
            torch.nn.init.kaiming_uniform_(self.combined_weight[:, :-1], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.combined_weight[:, :-1]
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.combined_weight[:, -1], -bound, bound)
        else:
            torch.nn.init.kaiming_uniform_(self.combined_weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias:
            return torch.nn.functional.linear(
                input, self.combined_weight[:, :-1], self.combined_weight[:, -1]
            )
        else:
            return torch.nn.functional.linear(input, self.combined_weight, None)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features=}, {self.out_features=}, {self.in_features_with_bias=}"
        )

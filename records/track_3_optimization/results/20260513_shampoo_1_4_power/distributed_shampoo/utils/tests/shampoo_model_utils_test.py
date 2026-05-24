"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from math import sqrt
from typing import cast

import torch
from distributed_shampoo.utils.shampoo_model_utils import CombinedLinear
from torch import nn
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class CombinedLinearTest(unittest.TestCase):
    def _init_weights(self, m: nn.Linear | CombinedLinear, seed: int) -> None:
        torch.random.manual_seed(seed)
        if isinstance(m, nn.Linear):
            bound = 1 / sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, CombinedLinear):
            bound = 1 / sqrt(m.in_features)
            if m.bias:
                torch.nn.init.uniform_(m.combined_weight[:, :-1], -bound, bound)
                torch.nn.init.uniform_(m.combined_weight[:, -1], -bound, bound)
            else:
                torch.nn.init.uniform_(m.combined_weight, -bound, bound)

    def _test_linear_forward_backward(
        self,
        feature_vector: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: bool,
        seed: int,
    ) -> None:
        # generate linear layers and initialize
        original_linear = nn.Linear(in_features, out_features, bias=bias)
        combined_linear = CombinedLinear(in_features, out_features, bias=bias)

        self._init_weights(original_linear, seed)
        self._init_weights(combined_linear, seed)

        # confirm weights are initialized equally
        if bias:
            assert torch.equal(
                original_linear.weight, combined_linear.combined_weight[:, :-1]
            )
            assert torch.equal(
                original_linear.bias, combined_linear.combined_weight[:, -1]
            )
        else:
            assert torch.equal(original_linear.weight, combined_linear.combined_weight)

        # perform forward pass
        original_output = original_linear(feature_vector)
        combined_output = combined_linear(feature_vector)

        # compute backward of sum of output
        torch.sum(original_output).backward()
        torch.sum(combined_output).backward()

        # check values are equal
        with self.subTest("Test forward"):
            torch.testing.assert_close(original_output, combined_output)

        with self.subTest("Test backward"):
            if bias:
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.weight.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad)[:, :-1],
                )
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.bias.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad)[:, -1],
                )
            else:
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.weight.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad),
                )

    @parametrize("seed", [920, 2022])
    @parametrize("bias", [False, True])
    @parametrize("out_features", [2, 10])
    @parametrize("in_features", [2, 10])
    def test_linear_forward_backward(
        self, in_features: int, out_features: int, bias: bool, seed: int
    ) -> None:
        torch.random.manual_seed(seed)
        feature_vector = torch.rand(in_features)
        self._test_linear_forward_backward(
            feature_vector, in_features, out_features, bias, seed
        )

    @parametrize("seed", [920, 2022])
    @parametrize("bias", [False, True])
    def test_initialization(self, bias: bool, seed: int) -> None:
        in_features = 10
        out_features = 20

        # generate linear layers and initialize
        torch.random.manual_seed(seed)
        original_linear = nn.Linear(in_features, out_features, bias=bias)
        torch.random.manual_seed(seed)
        combined_linear = CombinedLinear(in_features, out_features, bias=bias)

        # confirm weights are initialized equally
        if bias:
            torch.testing.assert_close(
                original_linear.weight, combined_linear.combined_weight[:, :-1]
            )
            torch.testing.assert_close(
                original_linear.bias, combined_linear.combined_weight[:, -1]
            )
        else:
            torch.testing.assert_close(
                original_linear.weight, combined_linear.combined_weight
            )

    def test_extra_repr(self) -> None:
        in_features = 10
        out_features = 20
        combined_linear = CombinedLinear(in_features, out_features, bias=True)
        self.assertEqual(
            combined_linear.extra_repr(),
            "self.in_features=10, self.out_features=20, self.in_features_with_bias=11",
        )

        combined_linear = CombinedLinear(in_features, out_features, bias=False)
        self.assertEqual(
            combined_linear.extra_repr(),
            "self.in_features=10, self.out_features=20, self.in_features_with_bias=10",
        )

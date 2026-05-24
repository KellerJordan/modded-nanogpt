"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import re
import unittest

import torch
from distributed_shampoo.distributor.shampoo_hsdp_distributor import HSDPDistributor
from torch import Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

logger: logging.Logger = logging.getLogger(__name__)


@instantiate_parametrized_tests
class SplitTensorBlockRecoveryTest(unittest.TestCase):
    def _test_split_tensor_block_recovery(
        self,
        original_tensor: Tensor,
        expected_split_tensors: list[Tensor],
        start_idx: int,
        end_idx: int,
    ) -> None:
        actual_split_tensors = HSDPDistributor._split_tensor_block_recovery(
            original_tensor.flatten()[start_idx:end_idx],
            original_tensor.size(),
            start_idx,
            end_idx,
        )

        self.assertNotEqual(len(actual_split_tensors), 0)
        torch.testing.assert_close(actual_split_tensors, expected_split_tensors)

    def test_illegal_tensor_shard_size(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape("Input tensor is not flat"),
            HSDPDistributor._split_tensor_block_recovery,
            tensor_shard=torch.randn((3, 4)),
            original_shape=torch.Size((3, 4)),
            start_idx=0,
            end_idx=16,
        )

    @parametrize(
        "start_idx, end_idx, expected_split_tensors",
        ((0, 5, [torch.arange(5)]), (1, 4, [torch.arange(1, 4)])),
    )
    def test_split_tensor_block_recovery_for_one_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(5)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    @parametrize(
        "start_idx, end_idx, expected_split_tensors",
        (
            (0, 11, [torch.arange(10).reshape(2, 5), torch.tensor([10])]),
            (3, 15, [torch.arange(3, 5), torch.arange(5, 15).reshape(2, 5)]),
            (3, 4, [torch.tensor([3])]),
        ),
    )
    def test_split_tensor_block_recovery_for_two_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(15).reshape(3, 5)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    @parametrize(
        "start_idx, end_idx, expected_split_tensors",
        (
            (0, 9, [torch.arange(9).reshape(1, 3, 3)]),
            (8, 10, [torch.tensor([8]), torch.tensor([9])]),
            (
                7,
                22,
                [
                    torch.tensor([7, 8]),
                    torch.tensor([[[9, 10, 11], [12, 13, 14], [15, 16, 17]]]),
                    torch.tensor([[18, 19, 20]]),
                    torch.tensor([21]),
                ],
            ),
        ),
    )
    def test_split_tensor_block_recovery_for_three_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(27).reshape(3, 3, 3)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

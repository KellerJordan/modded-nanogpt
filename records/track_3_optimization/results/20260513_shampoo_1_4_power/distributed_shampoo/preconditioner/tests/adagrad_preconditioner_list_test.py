"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

from collections.abc import Hashable
from typing import Any

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner.adagrad_preconditioner_list import (
    AdagradPreconditionerList,
)
from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from distributed_shampoo.preconditioner.tests.preconditioner_list_test_utils import (
    AbstractPreconditionerListTest,
)
from torch import Tensor


class AdagradPreconditionerListTest(AbstractPreconditionerListTest.Interface):
    def _instantiate_block_list(self) -> tuple[Tensor, ...]:
        # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
        # and self._params[1] are split into two blocks.
        return (
            self._params[0],
            *torch.split(self._params[1], 2, dim=0),
            self._params[2],
        )

    def _instantiate_preconditioner_list(self, **kwargs: Any) -> PreconditionerList:
        return AdagradPreconditionerList(
            block_list=self._block_list,
            state=self._state,
            block_info_list=self._block_info_list,
            **kwargs,
        )

    def setUp(self) -> None:
        self._params = (
            torch.tensor([1.0, 2.0]),
            torch.arange(6, dtype=torch.float).reshape(3, 2),
            torch.tensor(1.0),  # a 0D tensor
        )
        self._state: dict[Tensor, dict[Hashable, object]] = {
            self._params[0]: {"block_0": {}},
            self._params[1]: {"block_0": {}, "block_1": {}},
            self._params[2]: {"block_0": {}},
        }
        # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
        # and self._params[1] are split into two blocks.
        self._block_info_list = (
            BlockInfo(
                param=self._params[0],
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(
                param=self._params[1],
                composable_block_ids=(1, "block_0"),
            ),
            BlockInfo(
                param=self._params[1],
                composable_block_ids=(1, "block_1"),
            ),
            BlockInfo(
                param=self._params[2],
                composable_block_ids=(2, "block_0"),
            ),
        )
        super().setUp()

    def test_update_preconditioners_and_precondition(self) -> None:
        grad_list = (
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
            torch.tensor(1.0),
        )

        # Adagrad setting
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0, weighting_factor=1.0
            ),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_sign(grad_list),
        )

        # Adam setting
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=0.9, weighting_factor=1.0 - 0.9
            ),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_sign(grad_list),
        )

        # RMSprop setting
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=0.99,
                weighting_factor=1.0 - 0.99,
                use_bias_correction=False,
            ),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_mul(
                torch._foreach_sign(grad_list), 10.0
            ),
        )

    @property
    def _expected_numel_list(self) -> tuple[int, ...]:
        return (2, 4, 2, 1)

    @property
    def _expected_dims_list(self) -> tuple[torch.Size, ...]:
        return (torch.Size([2]), torch.Size([2, 2]), torch.Size([1, 2]), torch.Size([]))

    @property
    def _expected_num_bytes_list(self) -> tuple[int, ...]:
        return (8, 16, 8, 4)

    @property
    def _expected_numel(self) -> int:
        return 9

    @property
    def _expected_num_bytes(self) -> int:
        return 36

    @property
    def _expected_compress_list_call_count(self) -> int:
        return 1

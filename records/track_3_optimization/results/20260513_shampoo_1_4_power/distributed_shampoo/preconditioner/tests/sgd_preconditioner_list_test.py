"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import torch
from distributed_shampoo.preconditioner.sgd_preconditioner_list import (
    SGDPreconditionerList,
)
from distributed_shampoo.preconditioner.tests.preconditioner_list_test_utils import (
    AbstractPreconditionerListTest,
)
from torch import Tensor


class SGDPreconditionerListTest(AbstractPreconditionerListTest.Interface):
    def _instantiate_block_list(self) -> tuple[Tensor, ...]:
        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        )

    def _instantiate_preconditioner_list(
        self, **kwargs: object
    ) -> SGDPreconditionerList:
        return SGDPreconditionerList(block_list=self._block_list, **kwargs)

    def test_update_preconditioners_and_precondition(self) -> None:
        masked_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(),
            masked_grad_lists=[masked_grad_list],
            masked_expected_preconditioned_grad_list=masked_grad_list,
        )

    @property
    def _expected_numel_list(self) -> tuple[int, ...]:
        return (0, 0)

    @property
    def _expected_dims_list(self) -> tuple[torch.Size, ...]:
        return (torch.Size([2]), torch.Size([2, 2]))

    @property
    def _expected_num_bytes_list(self) -> tuple[int, ...]:
        return (0, 0)

    @property
    def _expected_numel(self) -> int:
        return 0

    @property
    def _expected_num_bytes(self) -> int:
        return 0

    @property
    def _expected_compress_list_call_count(self) -> int:
        return 0

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from torch import Tensor


class SGDPreconditionerList(PreconditionerList):
    """SGD (identity) preconditioners for a list of parameters.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
    ) -> None:
        super().__init__(block_list)

    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        return

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return masked_grad_list

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return

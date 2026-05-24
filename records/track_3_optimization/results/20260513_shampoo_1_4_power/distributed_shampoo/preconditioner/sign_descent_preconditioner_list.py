"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import torch
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.shampoo_types import SignDescentPreconditionerConfig
from torch import Tensor


class SignDescentPreconditionerList(PreconditionerList):
    """Preconditioner list for sign descent.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        preconditioner_config (SignDescentPreconditionerConfig): Configuration for sign descent.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        preconditioner_config: SignDescentPreconditionerConfig,
    ) -> None:
        super().__init__(block_list)
        self._preconditioner_config = preconditioner_config

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        return

    @profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return tuple(
            torch.sign(grad).mul_(self._preconditioner_config.scale_fn(grad))
            for grad in masked_grad_list
        )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return

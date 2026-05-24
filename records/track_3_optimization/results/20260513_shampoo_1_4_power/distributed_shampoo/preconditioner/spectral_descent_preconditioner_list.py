"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from distributed_shampoo.preconditioner.matrix_functions import matrix_orthogonalization
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.shampoo_types import SpectralDescentPreconditionerConfig
from torch import Tensor


class SpectralDescentPreconditionerList(PreconditionerList):
    """Preconditioner list for spectral descent.

    NOTE: This algorithm can only be used for 2D parameters, or parameters that have been reshaped to 2D.
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo.
    If all >2D parameters should be guaranteed to be reshaped to 2D, then max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2 has to be used.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        preconditioner_config (SpectralDescentPreconditionerConfig): Configuration for spectral descent.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        preconditioner_config: SpectralDescentPreconditionerConfig,
    ) -> None:
        if any(block.dim() != 2 for block in block_list):
            raise ValueError(
                "Spectral descent can only be used for 2D parameters, or parameters that have been reshaped to 2D. "
                "To guarantee that all >2D parameters are reshaped to 2D, set max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2."
            )
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
            # An error will be raised when grad is not 2D.
            matrix_orthogonalization(
                grad,
                orthogonalization_config=self._preconditioner_config.orthogonalization_config,
            )
            for grad in masked_grad_list
        )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return

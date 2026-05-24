"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
from typing import Any

import torch
from distributed_shampoo.preconditioner.matrix_functions_types import (
    DefaultNewtonSchulzOrthogonalizationConfig,
    OrthogonalizationConfig,
    SVDOrthogonalizationConfig,
)
from distributed_shampoo.preconditioner.spectral_descent_preconditioner_list import (
    SpectralDescentPreconditionerList,
)
from distributed_shampoo.preconditioner.tests.preconditioner_list_test_utils import (
    AbstractPreconditionerListTest,
)
from distributed_shampoo.shampoo_types import (
    DefaultSpectralDescentPreconditionerConfig,
    SpectralDescentPreconditionerConfig,
)
from torch import Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class SpectralDescentPreconditionerListTest(AbstractPreconditionerListTest.Interface):
    def _instantiate_block_list(self) -> tuple[Tensor, ...]:
        return (
            torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            torch.tensor([[5.0, 0.0], [0.0, 8.0]]),
        )

    def _instantiate_preconditioner_list(
        self, **kwargs: Any
    ) -> SpectralDescentPreconditionerList:
        return SpectralDescentPreconditionerList(
            block_list=self._block_list,
            preconditioner_config=SpectralDescentPreconditionerConfig(
                orthogonalization_config=SVDOrthogonalizationConfig()
            ),
            **kwargs,
        )

    def test_update_preconditioners_and_precondition(self) -> None:
        masked_grad_list = (torch.eye(2), torch.eye(2))
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(),
            masked_grad_lists=[masked_grad_list],
            masked_expected_preconditioned_grad_list=masked_grad_list,
        )

    @parametrize(
        "orthogonalization_config",
        (
            SVDOrthogonalizationConfig(),
            DefaultNewtonSchulzOrthogonalizationConfig,
        ),
    )
    def test_precondition_non_square_matrix(
        self, orthogonalization_config: OrthogonalizationConfig
    ) -> None:
        block_list = (torch.randn(3, 2), torch.randn(2, 3))
        masked_grad_list = (torch.randn(3, 2), torch.randn(2, 3))
        preconditioner_list = SpectralDescentPreconditionerList(
            block_list=block_list,
            preconditioner_config=SpectralDescentPreconditionerConfig(
                orthogonalization_config=orthogonalization_config,
            ),
        )
        preconditioner_list.precondition(masked_grad_list=masked_grad_list)

    @parametrize(
        "block_list",
        (
            (torch.randn(3),),  # 1D tensor
            (torch.randn(2, 3, 4),),  # 3D tensor
        ),
    )
    def test_non_2d_parameters_error(self, block_list: tuple[Tensor, ...]) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Spectral descent can only be used for 2D parameters, or parameters that have been reshaped to 2D. "
                "To guarantee that all >2D parameters are reshaped to 2D, set max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2."
            ),
            SpectralDescentPreconditionerList,
            block_list=block_list,
            preconditioner_config=DefaultSpectralDescentPreconditionerConfig,
        )

    @property
    def _expected_numel_list(self) -> tuple[int, ...]:
        return (0, 0)

    @property
    def _expected_dims_list(self) -> tuple[torch.Size, ...]:
        return (torch.Size([2, 2]), torch.Size([2, 2]))

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

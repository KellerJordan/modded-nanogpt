"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from collections.abc import Callable
from functools import partial

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    PreconditionerConfig,
    ShampooPT2CompileConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
)
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class DistributedShampooPytorchCompileTest(unittest.TestCase):
    @staticmethod
    def _shampoo_optim_factory(
        shampoo_pt2_compile_config: ShampooPT2CompileConfig | None,
        precondition_frequency: int,
        start_preconditioning_step: int,
        weight_decay: float,
        betas: tuple[float, float],
        grafting_config: PreconditionerConfig | None,
    ) -> Callable[[ParamsT], torch.optim.Optimizer]:
        return partial(
            DistributedShampoo,
            lr=0.01,
            betas=betas,
            epsilon=1e-10,
            weight_decay=weight_decay,
            max_preconditioner_dim=10,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay_type=WeightDecayType.DECOUPLED,
            shampoo_pt2_compile_config=shampoo_pt2_compile_config,
            grafting_config=grafting_config,
        )

    @parametrize(
        "precondition_frequency, start_preconditioning_step, total_steps",
        ((1, 1000, 5), (10, 10, 5)),
    )
    @parametrize("grafting_config", (None, AdaGradPreconditionerConfig(epsilon=1e-10)))
    @parametrize("betas", ((0.0, 1.0), (0.9, 0.999)))
    @parametrize("weight_decay", (0.0, 0.1))
    def test_pt2_shampoo_before_preconditioning(
        self,
        weight_decay: float,
        betas: tuple[float, float],
        grafting_config: PreconditionerConfig | None,
        precondition_frequency: int,
        start_preconditioning_step: int,
        total_steps: int,
    ) -> None:
        shampoo_optim_factory = partial(
            DistributedShampooPytorchCompileTest._shampoo_optim_factory,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay=weight_decay,
            betas=betas,
            grafting_config=grafting_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=shampoo_optim_factory(
                shampoo_pt2_compile_config=None
            ),
            experimental_optim_factory=shampoo_optim_factory(
                shampoo_pt2_compile_config=ShampooPT2CompileConfig()
            ),
            device=torch.device("cuda"),
            total_steps=total_steps,
        )

    @parametrize(
        "precondition_frequency, start_preconditioning_step, total_steps",
        ((1, 1000, 5), (10, 10, 5)),
    )
    @parametrize("betas", ((0.0, 1.0), (0.9, 0.999)))
    def test_pt2_shampoo_after_preconditioning(
        self,
        betas: tuple[float, float],
        precondition_frequency: int,
        start_preconditioning_step: int,
        total_steps: int,
    ) -> None:
        # NOTE: Test on steps after start_preconditioning_step.
        #       PT2 compilation with Inductor + root inverse introduces larger numerical differences
        #       compared to the non-PT2 baseline after preconditioning starts. However, these differences
        #       should NOT impact model quality.
        #       So we still want to add some numerical diff guardrails to prevent PT2 degradation.
        #       - It appears if torch.float16 precision tolerance is a good threshold:
        #       rtol = 1e-3; atol = 1e-5;
        #       - Test config specific: changing other Shampoo param vals can lead to UT failure:
        #       e.g., increase total_steps to a big val (e.g., 10000)

        shampoo_optim_factory = partial(
            DistributedShampooPytorchCompileTest._shampoo_optim_factory,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay=0.1,
            betas=betas,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1e-10),
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=shampoo_optim_factory(
                shampoo_pt2_compile_config=None,
            ),
            experimental_optim_factory=shampoo_optim_factory(
                shampoo_pt2_compile_config=ShampooPT2CompileConfig()
            ),
            device=torch.device("cuda"),
            total_steps=total_steps,
            rtol=1.0e-3,
            atol=1.0e-5,
        )

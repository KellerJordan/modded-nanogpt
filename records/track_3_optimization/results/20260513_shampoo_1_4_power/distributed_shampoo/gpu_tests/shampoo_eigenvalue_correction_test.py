"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import math
import unittest
from functools import partial
from numbers import Real
from typing import Any

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultSOAPConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_optimizer_on_cpu_and_device,
    compare_two_optimizers_on_weight_and_loss,
)
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import ParamsT
from torch.optim.rmsprop import RMSprop
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


# Note: We have to set the epsilon to a very small value (i.e., 1e-15) due to the
# the place epsilon is added in the PyTorch optimizers (i.e., AdaGrad, RMSprop, Adam, AdamW)
# and Distributed Shampoo.
# The PyTorch optimizers add epsilon outside of the square root, and Distributed Shampoo
# adds epsilon inside of the square root.


@instantiate_parametrized_tests
class DistributedShampooEigenvalueCorrectionTest(unittest.TestCase):
    @staticmethod
    def _optim_factory(
        parameters: ParamsT,
        optim_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    available_devices: tuple[torch.device, ...] = (torch.device("cpu"),) + (
        torch.device("cuda"),
    ) * torch.cuda.is_available()

    @parametrize(
        "start_preconditioning_step, preconditioner_config",
        (
            (math.inf, DefaultEigenvalueCorrectedShampooConfig),
            (math.inf, DefaultSOAPConfig),
            (
                1,
                EigenvalueCorrectedShampooPreconditionerConfig(
                    ignored_basis_change_dims={0: [0], 1: [0], 2: [0, 1]}
                ),
            ),
        ),
    )
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adagrad_eigenvalue_correction(
        self,
        weight_decay: float,
        device: torch.device,
        start_preconditioning_step: Real,
        preconditioner_config: EigenvalueCorrectedShampooPreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooEigenvalueCorrectionTest._optim_factory,
            lr=0.01,
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            betas=(0.0, 1.0),
            epsilon=1e-15,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=None,
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(optim_factory, optim_cls=Adagrad, eps=1e-15),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize(
        "start_preconditioning_step, preconditioner_config",
        (
            (math.inf, DefaultEigenvalueCorrectedShampooConfig),
            (math.inf, DefaultSOAPConfig),
            (
                1,
                EigenvalueCorrectedShampooPreconditionerConfig(
                    ignored_basis_change_dims={0: [0], 1: [0], 2: [0, 1]}
                ),
            ),
        ),
    )
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adam_eigenvalue_correction(
        self,
        weight_decay: float,
        device: torch.device,
        start_preconditioning_step: Real,
        preconditioner_config: EigenvalueCorrectedShampooPreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooEigenvalueCorrectionTest._optim_factory,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-15,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=None,
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=Adam,
                eps=1e-15,
            ),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize(
        "start_preconditioning_step, preconditioner_config",
        (
            (math.inf, DefaultEigenvalueCorrectedShampooConfig),
            (math.inf, DefaultSOAPConfig),
            (
                1,
                EigenvalueCorrectedShampooPreconditionerConfig(
                    ignored_basis_change_dims={0: [0], 1: [0], 2: [0, 1]}
                ),
            ),
        ),
    )
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adamw_eigenvalue_correction(
        self,
        weight_decay: float,
        device: torch.device,
        start_preconditioning_step: Real,
        preconditioner_config: EigenvalueCorrectedShampooPreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooEigenvalueCorrectionTest._optim_factory,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-15,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay_type=WeightDecayType.DECOUPLED,
            grafting_config=None,
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=AdamW,
                eps=1e-15,
            ),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize(
        "start_preconditioning_step, preconditioner_config",
        (
            (math.inf, DefaultEigenvalueCorrectedShampooConfig),
            (math.inf, DefaultSOAPConfig),
            (
                1,
                EigenvalueCorrectedShampooPreconditionerConfig(
                    ignored_basis_change_dims={0: [0], 1: [0], 2: [0, 1]}
                ),
            ),
        ),
    )
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_rmsprop_eigenvalue_correction(
        self,
        weight_decay: float,
        device: torch.device,
        start_preconditioning_step: Real,
        preconditioner_config: EigenvalueCorrectedShampooPreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooEigenvalueCorrectionTest._optim_factory,
            lr=0.01,
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            betas=(0.0, 0.99),
            epsilon=1e-15,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=start_preconditioning_step,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=None,
            use_bias_correction=False,
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=RMSprop,
                alpha=0.99,
                eps=1e-15,
            ),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

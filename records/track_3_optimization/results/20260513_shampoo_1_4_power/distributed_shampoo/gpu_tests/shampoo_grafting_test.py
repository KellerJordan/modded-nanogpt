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
from typing import Any

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSOAPConfig,
    EigendecomposedShampooPreconditionerConfig,
    PreconditionerConfig,
    RMSpropPreconditionerConfig,
    SGDPreconditionerConfig,
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
from torch.optim.sgd import SGD
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class DistributedShampooGraftingTest(unittest.TestCase):
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

    preconditioner_configs: tuple[PreconditionerConfig, ...] = (
        DefaultShampooConfig,
        EigendecomposedShampooPreconditionerConfig(),
        DefaultEigenvalueCorrectedShampooConfig,
        DefaultSOAPConfig,
        DefaultSignDescentPreconditionerConfig,
    )

    @parametrize("preconditioner_config", preconditioner_configs)
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adagrad_grafting(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooGraftingTest._optim_factory,
            lr=0.01,
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            betas=(0.0, 1.0),
            epsilon=1e-10,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1e-10),
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(optim_factory, optim_cls=Adagrad, eps=1e-10),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize("preconditioner_config", preconditioner_configs)
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adam_grafting(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooGraftingTest._optim_factory,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-8,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(optim_factory, optim_cls=Adam, eps=1e-8),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize("preconditioner_config", preconditioner_configs)
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_adamw_grafting(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooGraftingTest._optim_factory,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-8,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            weight_decay_type=WeightDecayType.DECOUPLED,
            grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(optim_factory, optim_cls=AdamW, eps=1e-8),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize("preconditioner_config", preconditioner_configs)
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_rmsprop_grafting(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooGraftingTest._optim_factory,
            lr=0.01,
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            betas=(0.0, 0.99),
            epsilon=1e-8,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            use_bias_correction=False,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=RMSpropPreconditionerConfig(beta2=0.99, epsilon=1e-8),
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=RMSprop,
                alpha=0.99,
                eps=1e-8,
            ),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

    @parametrize("preconditioner_config", preconditioner_configs)
    @parametrize("device", available_devices)
    @parametrize("weight_decay", (0.0, 0.3))
    def test_sgd_grafting(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        optim_factory = partial(
            DistributedShampooGraftingTest._optim_factory,
            lr=0.1,
            weight_decay=weight_decay,
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            betas=(0.0, 1.0),
            epsilon=1e-10,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            weight_decay_type=WeightDecayType.L2,
            grafting_config=SGDPreconditionerConfig(),  # type: ignore[abstract]
            preconditioner_config=preconditioner_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=SGD,
            ),
            experimental_optim_factory=experimental_optim_factory,
            # Setting model_linear_layers_dims to (10, 10) to ensure a simple model structure,
            # as SGD can be sensitive to the choice of model architecture.
            model_linear_layers_dims=(10, 10),
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory,
            # Using the same model_linear_layers_dims for consistency in testing across devices.
            model_linear_layers_dims=(10, 10),
            device=device,
        )

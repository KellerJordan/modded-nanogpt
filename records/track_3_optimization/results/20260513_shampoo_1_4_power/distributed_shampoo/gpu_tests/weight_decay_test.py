#!/usr/bin/env python3

"""
Tests for weight decay types in Distributed Shampoo.

This test module focuses exclusively on testing the different weight decay strategies:
- L2: Standard L2 regularization applied to gradient before preconditioning
- DECOUPLED: Weight decay applied independent of preconditioned gradient (AdamW-style)
- CORRECTED: Weight decay scaled by lr/peak_lr (AdamC-style)
- INDEPENDENT: Weight decay scaled by 1/peak_lr

All tests use Adam grafting as the baseline to isolate weight decay behavior.
"""

import math
import unittest
from functools import partial
from typing import Any

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdamPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSOAPConfig,
    EigendecomposedShampooPreconditionerConfig,
    PreconditionerConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_optimizer_on_cpu_and_device,
    compare_two_optimizers_on_weight_and_loss,
)
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class DistributedShampooWeightDecayTest(unittest.TestCase):
    """Test suite for weight decay types in Distributed Shampoo.

    Each test verifies that a specific weight decay type produces results
    equivalent to the corresponding PyTorch optimizer baseline.

    Weight decay formulas:
    - L2: Adds weight_decay * params to gradient before preconditioning
    - DECOUPLED: params *= (1 - lr * weight_decay)
    - CORRECTED: params *= (1 - lr * (lr / peak_lr) * weight_decay)
    - INDEPENDENT: params *= (1 - (lr / peak_lr) * weight_decay)
    """

    @staticmethod
    def _optim_factory(
        parameters: ParamsT,
        optim_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    available_devices: tuple[torch.device, ...] = (torch.device("cpu"),) + (
        (torch.device("cuda"),) if torch.cuda.is_available() else ()
    )

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
    def test_l2_weight_decay(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        """Test L2 weight decay matches PyTorch Adam with L2 regularization.

        L2 weight decay adds weight_decay * params to the gradient before
        preconditioning. This is equivalent to standard Adam with weight_decay.
        """
        optim_factory = partial(
            DistributedShampooWeightDecayTest._optim_factory,
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
    def test_decoupled_weight_decay(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        """Test DECOUPLED weight decay matches PyTorch AdamW.

        Decoupled weight decay applies params *= (1 - lr * weight_decay)
        independent of the preconditioned gradient. This is the AdamW formula.
        """
        optim_factory = partial(
            DistributedShampooWeightDecayTest._optim_factory,
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
    def test_corrected_weight_decay(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        """Test CORRECTED weight decay matches AdamW when lr == peak_lr.

        Corrected weight decay applies params *= (1 - lr * (lr / peak_lr) * weight_decay).
        When lr == peak_lr (constant learning rate), this simplifies to AdamW's formula:
        params *= (1 - lr * weight_decay).
        """
        optim_factory = partial(
            DistributedShampooWeightDecayTest._optim_factory,
            lr=0.001,
            betas=(0.9, 0.999),
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-8,
            weight_decay=weight_decay,
            weight_decay_type=WeightDecayType.CORRECTED,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
            preconditioner_config=preconditioner_config,
        )

        # When lr == peak_lr, CORRECTED reduces to DECOUPLED/AdamW
        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=AdamW,
                eps=1e-8,
                weight_decay=weight_decay,
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
    def test_independent_weight_decay(
        self,
        weight_decay: float,
        device: torch.device,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        """Test INDEPENDENT weight decay matches AdamW with scaled weight_decay.

        Independent weight decay applies params *= (1 - (lr / peak_lr) * weight_decay).
        When lr == peak_lr (constant learning rate), this simplifies to:
        params *= (1 - weight_decay)

        This is equivalent to AdamW with effective_weight_decay = weight_decay / lr:
        params *= (1 - lr * (weight_decay / lr)) = (1 - weight_decay)
        """
        lr = 0.001
        optim_factory = partial(
            DistributedShampooWeightDecayTest._optim_factory,
            lr=lr,
            betas=(0.9, 0.999),
        )
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            epsilon=1e-8,
            weight_decay=weight_decay,
            weight_decay_type=WeightDecayType.INDEPENDENT,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=math.inf,
            grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
            preconditioner_config=preconditioner_config,
        )

        # INDEPENDENT with weight_decay is equivalent to AdamW with weight_decay / lr
        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=partial(
                optim_factory,
                optim_cls=AdamW,
                eps=1e-8,
                weight_decay=weight_decay / lr if lr != 0.0 else 0.0,
            ),
            experimental_optim_factory=experimental_optim_factory,
            device=device,
        )

        compare_optimizer_on_cpu_and_device(
            optim_factory=experimental_optim_factory, device=device
        )

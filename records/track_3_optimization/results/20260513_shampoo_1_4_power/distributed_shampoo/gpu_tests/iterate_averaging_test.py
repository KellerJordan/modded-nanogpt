#!/usr/bin/env python3

"""Tests for validating iterate averaging (GPA and Schedule-Free) equivalence.

This module contains tests that validate the theoretical equivalence between Shampoo's
iterate averaging implementations and:
1. PyTorch's SGD with momentum (for GPA-momentum equivalence)
2. GPA-AdamW from gpa (GPA-AdamW and Schedule-Free equivalence)

The tests use SGD/Adam grafting to isolate the iterate averaging behavior from preconditioning,
allowing direct comparison between implementations.

Note: The hyperparameters in GeneralizedPrimalAveragingConfig and ScheduleFreeConfig need to
be set appropriately to achieve equivalence.

See the following papers for the theoretical background:
- GPA: https://arxiv.org/pdf/2512.17131
- Schedule-Free: https://arxiv.org/abs/2405.15682
"""

import enum
import unittest
from functools import partial
from typing import Any

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdamPreconditionerConfig,
    GeneralizedPrimalAveragingConfig,
    IterateAveragingConfig,
    ScheduleFreeConfig,
    SGDPreconditionerConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
)
from gpa.gpa_adamw import GPAAdamW
from gpa.gpa_types import IterateAveragingType as GPAIterateAveragingType
from torch.optim.optimizer import ParamsT
from torch.optim.sgd import SGD
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@enum.unique
class IterateAveragingType(enum.Enum):
    """Enum for selecting the type of iterate averaging in tests."""

    GPA = enum.auto()  # Generalized Primal Averaging
    SCHEDULE_FREE = enum.auto()  # Schedule-Free


@instantiate_parametrized_tests
class IterateAveragingTest(unittest.TestCase):
    """Tests for iterate averaging equivalence (GPA-SGD, GPA-AdamW, and Schedule-Free)."""

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

    @parametrize("device", available_devices)
    def test_gpa_vs_sgd_momentum(
        self,
        device: torch.device,
    ) -> None:
        """Test that Shampoo with GPA produces equivalent results to SGD with momentum.

        This test validates the theoretical equivalence between Generalized Primal
        Averaging and heavy-ball momentum as described in https://arxiv.org/pdf/2512.17131.

        The test uses SGD grafting to ensure the preconditioner is a no-op, isolating
        the iterate averaging behavior.

        Args:
            device: Device to run the test on (CPU or CUDA).
        """
        lr = 0.1
        momentum = 0.9
        weight_decay = 0.0
        optim_factory = partial(
            IterateAveragingTest._optim_factory,
            weight_decay=weight_decay,
        )

        # Control: SGD with momentum
        control_optim_factory = partial(
            optim_factory,
            optim_cls=SGD,
            lr=lr,
            momentum=momentum,
        )

        # Experimental: Shampoo with GPA + SGD grafting
        # The GPA coefficients are set to achieve equivalence with SGD momentum.
        # TODO: User needs to set the correct hyperparameters for equivalence.
        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            lr=lr / (1 - momentum),  # Adjust lr to account for momentum
            betas=(0.0, 1.0),  # No momentum filtering in Shampoo's base
            epsilon=1e-10,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            weight_decay_type=WeightDecayType.L2,
            preconditioner_config=SGDPreconditionerConfig(),  # type: ignore[abstract]
            grafting_config=None,
            iterate_averaging_config=GeneralizedPrimalAveragingConfig(
                # TODO: Set these coefficients based on momentum value for equivalence.
                # See https://arxiv.org/pdf/2512.17131 for the relationship.
                eval_interp_coeff=momentum,
                train_interp_coeff=1.0,
            ),
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=control_optim_factory,
            experimental_optim_factory=experimental_optim_factory,
            # Using simple model architecture for SGD-based comparison.
            model_linear_layers_dims=(10, 10),
            device=device,
        )

    @parametrize("device", available_devices)
    @parametrize(
        "iterate_averaging_type",
        [IterateAveragingType.GPA, IterateAveragingType.SCHEDULE_FREE],
    )
    def test_iterate_averaging_vs_gpa_adamw(
        self,
        device: torch.device,
        iterate_averaging_type: IterateAveragingType,
    ) -> None:
        """Test that Shampoo with iterate averaging + Adam grafting produces equivalent results to GPA-AdamW.

        This test validates the theoretical equivalence between Shampoo's iterate
        averaging implementations (GPA or Schedule-Free) with Adam grafting and the
        standalone GPA-AdamW optimizer.

        Both optimizers should produce the same results when configured with matching
        hyperparameters since they implement the same algorithm.

        Args:
            device: Device to run the test on (CPU or CUDA).
            iterate_averaging_type: Type of iterate averaging to test (GPA or Schedule-Free).
        """
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        weight_decay = 0.01
        train_interp_coeff = 0.9

        # GPA uses a fixed eval_interp_coeff, Schedule-Free uses eval_interp_coeff=0
        # which triggers the polynomial weighting formula.
        eval_interp_coeff = (
            0.9 if iterate_averaging_type == IterateAveragingType.GPA else 0.0
        )

        optim_factory = partial(
            IterateAveragingTest._optim_factory,
            weight_decay=weight_decay,
        )

        # Control: GPA-AdamW optimizer
        gpa_iterate_averaging_type = (
            GPAIterateAveragingType.GPA
            if iterate_averaging_type == IterateAveragingType.GPA
            else GPAIterateAveragingType.SCHEDULE_FREE
        )
        control_optim_factory = partial(
            optim_factory,
            optim_cls=GPAAdamW,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=epsilon,
            eval_interp_coeff=eval_interp_coeff,
            train_interp_coeff=train_interp_coeff,
            iterate_averaging_type=gpa_iterate_averaging_type,
        )

        # Experimental: Shampoo with iterate averaging + Adam grafting
        # Using Adam grafting and disabling Shampoo preconditioning to isolate iterate averaging behavior.
        iterate_averaging_config: IterateAveragingConfig
        if iterate_averaging_type == IterateAveragingType.GPA:
            iterate_averaging_config = GeneralizedPrimalAveragingConfig(
                eval_interp_coeff=eval_interp_coeff,
                train_interp_coeff=train_interp_coeff,
            )
        else:  # SCHEDULE_FREE
            iterate_averaging_config = ScheduleFreeConfig(
                train_interp_coeff=train_interp_coeff,
            )

        experimental_optim_factory = partial(
            optim_factory,
            optim_cls=DistributedShampoo,
            lr=lr,
            betas=(beta1, beta2),
            epsilon=epsilon,
            weight_decay_type=WeightDecayType.DECOUPLED,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            use_bias_correction=True,
            preconditioner_config=AdamPreconditionerConfig(
                beta2=beta2,
                epsilon=epsilon,
            ),
            grafting_config=None,
            iterate_averaging_config=iterate_averaging_config,
        )

        compare_two_optimizers_on_weight_and_loss(
            control_optim_factory=control_optim_factory,
            experimental_optim_factory=experimental_optim_factory,
            model_linear_layers_dims=(10, 10),
            device=device,
        )

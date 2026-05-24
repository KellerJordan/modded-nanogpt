"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import gc
import logging
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.preconditioner.matrix_functions_types import (
    DefaultNewtonSchulzOrthogonalizationConfig,
    EigenConfig,
    OrthogonalizationConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSpectralDescentPreconditionerConfig,
    DistributedConfig,
    EigendecomposedKLShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    GeneralizedPrimalAveragingConfig,
    IterateAveragingConfig,
    PreconditionerConfig,
    RootInvKLShampooPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    ScheduleFreeConfig,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    WeightDecayType,
)
from torch import nn, Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class DistributedShampooInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )

    def test_invalid_preconditioner_config(self) -> None:
        @dataclass
        class NotSupportedPreconditionerConfig(PreconditionerConfig):
            """A dummy preconditioner config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            r"preconditioner_config=.*\.NotSupportedPreconditionerConfig\(.*\) not supported!",
            DistributedShampoo,
            self._model.parameters(),
            preconditioner_config=NotSupportedPreconditionerConfig(),
        )

    @parametrize(
        "incorrect_hyperparameter_setting, expected_error_msg",
        [
            (
                {"lr": -0.1},
                "Invalid param_group[LR]=-0.1. Must be >= 0.0.",
            ),
            (
                {"betas": (-0.1, 1.0)},
                "Invalid param_group[BETAS][0]=-0.1. Must be in [0.0, 1.0).",
            ),
            (
                {"betas": (0.9, -0.1)},
                "Invalid param_group[BETAS][1]=-0.1. Must be in [0.0, 1.0].",
            ),
            (
                {"beta3": -0.1},
                "Invalid param_group[BETA3]=-0.1. Must be in [0.0, 1.0).",
            ),
            (
                {
                    "epsilon": 0.1,
                    "preconditioner_config": RootInvShampooPreconditionerConfig(
                        amortized_computation_config=EigenConfig(
                            rank_deficient_stability_config=PseudoInverseConfig()
                        )
                    ),
                },
                "Invalid param_group[EPSILON]=0.1. Must be == 0.0 when PseudoInverseConfig is used.",
            ),
            (
                {"epsilon": 0.0},
                "Invalid param_group[EPSILON]=0.0. Must be > 0.0.",
            ),
            (
                {"weight_decay": -0.1},
                "Invalid param_group[WEIGHT_DECAY]=-0.1. Must be >= 0.0.",
            ),
            (
                {"max_preconditioner_dim": 3.14},
                "Invalid param_group[MAX_PRECONDITIONER_DIM]=3.14. Must be an integer or math.inf.",
            ),
            (
                {"max_preconditioner_dim": 0},
                "Invalid param_group[MAX_PRECONDITIONER_DIM]=0. Must be >= 1.",
            ),
            (
                {"precondition_frequency": 0},
                "Invalid param_group[PRECONDITION_FREQUENCY]=0. Must be >= 1.",
            ),
            (
                {"start_preconditioning_step": -2},
                "Invalid param_group[START_PRECONDITIONING_STEP]=-2. Must be >= -1.",
            ),
            (
                {"start_preconditioning_step": 10, "precondition_frequency": 100},
                "Invalid param_group[START_PRECONDITIONING_STEP]=10. Must be >= param_group[PRECONDITION_FREQUENCY]=100.",
            ),
        ],
    )
    def test_invalid_with_incorrect_hyperparameter_setting(
        self, incorrect_hyperparameter_setting: dict[str, Any], expected_error_msg: str
    ) -> None:
        # Test the incorrect hyperparameter setting in the default hyperparameter setting.
        self.assertRaisesRegex(
            ValueError,
            re.escape(expected_error_msg),
            DistributedShampoo,
            self._model.parameters(),
            **incorrect_hyperparameter_setting,
        )

        # Test the incorrect hyperparameter setting in the param_group setting.
        with self.assertLogs(level="INFO") as cm:
            self.assertRaisesRegex(
                ValueError,
                re.escape(expected_error_msg),
                DistributedShampoo,
                [
                    {"params": []},  # param_group 0 is valid
                    {
                        "params": self._model.parameters(),
                        **incorrect_hyperparameter_setting,  # We intentionally let param_group 1 fail to test error detection
                    },
                    {"params": []},  # param_group 2 is valid
                ],
            )

            msgs = [r.msg for r in cm.records if r.levelname == "INFO"]

        self.assertEqual(
            msgs,
            [
                "Checking param_group 0 hyperparameters...",
                "Checking param_group 1 hyperparameters...",
                # We don't see param_group 2 message because validation stops after finding the first invalid param_group
            ],
        )

    @parametrize(
        "noop_hyperparameter_setting, expected_warning_msgs",
        [
            (
                {
                    "betas": (0.9, 0.999),
                    "epsilon": 1e-8,
                    "precondition_frequency": 100,
                    "preconditioner_config": DefaultSpectralDescentPreconditionerConfig,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=1,
                    ),
                },
                [
                    "param_group[BETAS][1]=0.999 does not have any effect when SpectralDescentPreconditionerConfig is used.",
                    "param_group[EPSILON]=1e-08 does not have any effect when SpectralDescentPreconditionerConfig is used.",
                    "param_group[PRECONDITION_FREQUENCY]=100 does not have any effect when SpectralDescentPreconditionerConfig is used. Setting precondition_frequency to 1...",
                    "param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=1 is not equal to 2. Setting target_parameter_dimensionality to 2...",
                ],
            ),
            (
                {
                    "betas": (0.9, 0.999),
                    "epsilon": 1e-8,
                    "precondition_frequency": 100,
                    "preconditioner_config": DefaultSignDescentPreconditionerConfig,
                },
                [
                    "param_group[BETAS][1]=0.999 does not have any effect when SignDescentPreconditionerConfig is used.",
                    "param_group[EPSILON]=1e-08 does not have any effect when SignDescentPreconditionerConfig is used.",
                    "param_group[PRECONDITION_FREQUENCY]=100 does not have any effect when SignDescentPreconditionerConfig is used. Setting precondition_frequency to 1...",
                ],
            ),
        ],
    )
    def test_noop_hyperparameter_setting_warnings(
        self,
        noop_hyperparameter_setting: dict[str, Any],
        expected_warning_msgs: list[str],
    ) -> None:
        with self.assertLogs(level="WARNING") as cm:
            DistributedShampoo(
                self._model.parameters(),
                **noop_hyperparameter_setting,
            )
            recorded_warning_msgs = [r.msg for r in cm.records]
            for expected_warning_msg in expected_warning_msgs:
                with self.subTest(
                    noop_hyperparameter_setting=noop_hyperparameter_setting,
                    expected_warning_msg=expected_warning_msg,
                    recorded_warning_msgs=recorded_warning_msgs,
                ):
                    self.assertIn(
                        expected_warning_msg,
                        recorded_warning_msgs,
                    )

    def test_invalid_distributed_config(self) -> None:
        @dataclass
        class NotSupportedDistributedConfig(DistributedConfig):
            """A dummy distributed config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            r"group\[DISTRIBUTED_CONFIG\]=.*\.NotSupportedDistributedConfig\(.*\) not supported!",
            DistributedShampoo,
            params=self._model.parameters(),
            distributed_config=NotSupportedDistributedConfig(),
        )


class DistributedShampooTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            # Explicitly set grafting_config=None to test the case that no grafting config is used.
            grafting_config=None,
        )

    def test_step_with_closure(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        # Test the case without closure, the loss returned by step() is None.
        self._optimizer.zero_grad()
        layer_weight.grad = torch.rand_like(layer_weight)
        self.assertIsNone(self._optimizer.step(closure=None))

        # Test the case that the closure returns a scalar.
        def closure() -> float:
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            return 1.0

        self.assertEqual(self._optimizer.step(closure=closure), 1.0)

    def test_optimizer_zero_grad(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)

        # Store the data pointer of the current gradient to check if it gets freed later.
        grad_data_ptr = layer_weight.grad.data_ptr()

        self._optimizer.step()

        # Call zero_grad with set_to_none=True to explicitly release gradient memory rather than just zeroing it out.
        self._optimizer.zero_grad(set_to_none=True)

        # Verify that the gradient has been set to None.
        self.assertIsNone(layer_weight.grad)

        # Get all tensor objects currently tracked by the garbage collector.
        all_alive_tensors = tuple(
            obj
            for obj in gc.get_objects()
            # Using type(obj) here to prevent the garbage collector from including non-real tensors like FakeTensor.
            if type(obj) in (torch.Tensor, nn.Parameter)
        )

        # Check that the stored gradient data pointer is not in the list of alive tensors, ensuring it was freed.
        self.assertNotIn(
            grad_data_ptr,
            (t.data_ptr() for t in all_alive_tensors),
            msg="Found gradients space is still not freed, check Shampoo code for properly free gradients pointers.",
        )


class AbstractTest:
    class StateDictTestBase(abc.ABC, unittest.TestCase):
        @property
        @abc.abstractmethod
        def _preconditioner_config(self) -> PreconditionerConfig: ...

        @property
        @abc.abstractmethod
        def _ref_state_dict(self) -> dict[str, Any]: ...

        def setUp(self) -> None:
            self._model = nn.Sequential(
                nn.Linear(5, 10, bias=False),
            )
            # Initialize weights to zeros to ensure deterministic state dict values.
            with torch.no_grad():
                cast(torch.Tensor, self._model[0].weight).zero_()
            self._optimizer = DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                weight_decay=0.0,
                max_preconditioner_dim=5,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                distributed_config=replace(
                    DefaultSingleDeviceDistributedConfig,
                    # distributed_config.target_parameter_dimensionality=2 is necessary to prevent SpectralDescentPreconditionerConfig assertion error.
                    target_parameter_dimensionality=2,
                ),
                grafting_config=AdaGradPreconditionerConfig(
                    epsilon=0.001,
                ),
                preconditioner_config=self._preconditioner_config,
            )

        def test_setstate_call(self) -> None:
            """Test that __setstate__ is properly called during load_state_dict operation."""

            class MockDistributedShampoo(DistributedShampoo):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    # Flag to track if __setstate__ was called
                    self._shampoo_setstate_called = False

                def __setstate__(self, state: dict[str, Any]) -> None:
                    # Mark that __setstate__ was invoked
                    self._shampoo_setstate_called = True
                    super().__setstate__(state)

            # Create a mock optimizer instance
            mocked_shampoo_optimizer = MockDistributedShampoo(self._model.parameters())
            # Get the current state dictionary
            optim_state_dict = mocked_shampoo_optimizer.state_dict()

            # Load the state dictionary, which should trigger __setstate__
            mocked_shampoo_optimizer.load_state_dict(optim_state_dict)

            # Verify that __setstate__ was called during load_state_dict
            self.assertTrue(mocked_shampoo_optimizer._shampoo_setstate_called, True)

        def test_state_dict(self) -> None:
            """
            Test that the state dict is correct by comparing
            optimizer.state_dict() and the reference state dict.
            """
            state_dict = self._optimizer.state_dict()
            ref_state_dict = self._ref_state_dict
            self.assertEqual(state_dict.keys(), {"state", "param_groups"})

            torch.testing.assert_close(
                state_dict["state"],
                ref_state_dict["state"],
            )
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_load_state_dict(self) -> None:
            """
            Test that load_state_dict() loads the correct state dict by comparing
            optimizer.state_dict() and the reference state dict. Note that load_state_dict()
            calls __setstate__, which we override in Shampoo.
            """
            ref_state_dict = self._ref_state_dict
            self._optimizer.load_state_dict(
                state_dict=ref_state_dict,
            )

            state_dict = self._optimizer.state_dict()

            self.assertEqual(state_dict.keys(), ref_state_dict.keys())
            torch.testing.assert_close(state_dict["state"], ref_state_dict["state"])
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

    class NoPreconditionerStateDictTestBase(StateDictTestBase):
        """A base class for methods that do not have a preconditioner."""

        @property
        def _ref_state_dict(self) -> dict[str, Any]:
            return {
                "state": {
                    0: {
                        "block_0": {
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "block_1": {
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "step": torch.tensor(0),
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.01,
                        "betas": (0.9, 1.0),
                        "beta3": 0.9,
                        "epsilon": 1e-12,
                        "weight_decay": 0.0,
                        "peak_lr": 0.01,
                        "weight_decay_type": WeightDecayType.DECOUPLED,
                        "max_preconditioner_dim": 5,
                        "precondition_frequency": 1,
                        "start_preconditioning_step": 1,
                        "use_bias_correction": True,
                        "iterate_averaging_config": None,
                        "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                        "use_pin_memory": False,
                        "distributed_config": SingleDeviceDistributedConfig(
                            target_parameter_dimensionality=2
                        ),
                        "preconditioner_config": self._preconditioner_config,
                        "params": [0],
                    }
                ],
            }


class ShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return DefaultShampooConfig

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "inv_factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "inv_factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class EigendecomposedShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> EigendecomposedShampooPreconditionerConfig:
        return EigendecomposedShampooPreconditionerConfig()

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvalues": {
                                0: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                                1: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvalues": {
                                0: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                                1: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class EigenvalueCorrectedShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return DefaultEigenvalueCorrectedShampooConfig

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "corrected_eigenvalues": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "corrected_eigenvalues": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class RootInvKLShampooStateDictTest(ShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvKLShampooPreconditionerConfig:
        return RootInvKLShampooPreconditionerConfig()


class EigendecomposedKLShampooStateDictTest(EigendecomposedShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> EigendecomposedKLShampooPreconditionerConfig:
        return EigendecomposedKLShampooPreconditionerConfig()


class SignDescentStateDictTest(AbstractTest.NoPreconditionerStateDictTestBase):
    @property
    def _preconditioner_config(self) -> SignDescentPreconditionerConfig:
        return DefaultSignDescentPreconditionerConfig

    def test_state_dict_warning(self) -> None:
        """
        When Shampoo's `post_state_dict_hook` is fired during
        `state_dict()` call, it should issue a warning if a lambda function is detected,
        since it cannot pickled. This test checks that the warning is issued.
        """
        osd = self._optimizer.state_dict()
        self.assertCountEqual(osd.keys(), ["state", "param_groups"])

        @dataclass(kw_only=True)
        class SignDescentPreconditionerConfigWithLambda(
            SignDescentPreconditionerConfig
        ):
            """
            Creating a preconditioner config with a dummy lambda function to make sure the
            warning from `_post_state_dict_hook` emit.
            """

            scale_fn: Callable[[Tensor], float | Tensor] = lambda grad: 1.0

        self._optimizer.param_groups[0]["preconditioner_config"] = (
            SignDescentPreconditionerConfigWithLambda()
        )
        logger = logging.getLogger("distributed_shampoo.distributed_shampoo")
        with self.assertLogs(logger, level="WARNING") as cm:
            osd = self._optimizer.state_dict()
        self.assertIn(
            "Note that lambda function cannot be pickled. torch.save() cannot serialize lambda functions, "
            "because it relies on Python's pickle module for serialization, and pickle does not support lambda functions",
            cm.output[0],
        )


class SpectralDescentStateDictTest(AbstractTest.NoPreconditionerStateDictTestBase):
    @property
    def _preconditioner_config(self) -> SpectralDescentPreconditionerConfig:
        return DefaultSpectralDescentPreconditionerConfig

    def test_state_dict_warning(self) -> None:
        """
        When Shampoo's `post_state_dict_hook` is fired during
        `state_dict()` call, it should issue a warning if a lambda function is detected,
        since it cannot pickled. This test checks that the warning is issued.
        """
        osd = self._optimizer.state_dict()
        self.assertCountEqual(osd.keys(), ["state", "param_groups"])

        @dataclass(kw_only=True)
        class SpectralDescentPreconditionerConfigWithLambda(PreconditionerConfig):
            """
            Creating a orthogonalization config with a dummy lambda function to make sure the
            warning from `_post_state_dict_hook` emit.
            """

            orthogonalization_config: OrthogonalizationConfig = field(
                default_factory=lambda: DefaultNewtonSchulzOrthogonalizationConfig
            )

        self._optimizer.param_groups[0]["orthogonalization_config"] = (
            SpectralDescentPreconditionerConfigWithLambda()
        )
        logger = logging.getLogger("distributed_shampoo.distributed_shampoo")
        with self.assertLogs(logger, level="WARNING") as cm:
            osd = self._optimizer.state_dict()
        self.assertIn(
            "Note that lambda function cannot be pickled. torch.save() cannot serialize lambda functions, "
            "because it relies on Python's pickle module for serialization, and pickle does not support lambda functions",
            cm.output[0],
        )


class AbstractIterateAveragingTest:
    """Abstract base classes for testing iterate averaging configurations (GPA and Schedule-Free)."""

    class IterateAveragingStateDictTestBase(abc.ABC, unittest.TestCase):
        """Base class for testing state dict with iterate averaging enabled.

        When iterate averaging is enabled, the optimizer stores a weight_buffer
        for each parameter block that contains the "z" sequence.
        """

        @property
        @abc.abstractmethod
        def _iterate_averaging_config(self) -> IterateAveragingConfig: ...

        @property
        def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
            return DefaultShampooConfig

        def setUp(self) -> None:
            self._model = nn.Sequential(
                nn.Linear(5, 10, bias=False),
            )
            # Initialize weights to zeros to ensure deterministic state dict values.
            with torch.no_grad():
                cast(torch.Tensor, self._model[0].weight).zero_()
            self._optimizer = DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                weight_decay=0.0,
                max_preconditioner_dim=5,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                iterate_averaging_config=self._iterate_averaging_config,
                distributed_config=replace(
                    DefaultSingleDeviceDistributedConfig,
                    target_parameter_dimensionality=2,
                ),
                grafting_config=AdaGradPreconditionerConfig(
                    epsilon=0.001,
                ),
                preconditioner_config=self._preconditioner_config,
            )

        @property
        def _ref_state_dict(self) -> dict[str, Any]:
            return {
                "state": {
                    0: {
                        "block_0": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: torch.tensor(
                                        [
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                        ]
                                    ),
                                    1: torch.tensor(
                                        [
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                        ]
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: torch.tensor(
                                        [
                                            [1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0],
                                        ]
                                    ),
                                    1: torch.tensor(
                                        [
                                            [1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0],
                                        ]
                                    ),
                                },
                            },
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "weight_buffer": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "block_1": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: torch.tensor(
                                        [
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                        ]
                                    ),
                                    1: torch.tensor(
                                        [
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0],
                                        ]
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: torch.tensor(
                                        [
                                            [1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0],
                                        ]
                                    ),
                                    1: torch.tensor(
                                        [
                                            [1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0],
                                        ]
                                    ),
                                },
                            },
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "weight_buffer": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "step": torch.tensor(0),
                        "train_mode": torch.tensor(True),
                        "lr_sum": torch.tensor(0.0),
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.01,
                        "betas": (0.9, 1.0),
                        "beta3": 0.9,
                        "epsilon": 1e-12,
                        "weight_decay": 0.0,
                        "peak_lr": 0.01,
                        "weight_decay_type": WeightDecayType.DECOUPLED,
                        "max_preconditioner_dim": 5,
                        "precondition_frequency": 1,
                        "start_preconditioning_step": 1,
                        "use_bias_correction": True,
                        "iterate_averaging_config": self._iterate_averaging_config,
                        "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                        "use_pin_memory": False,
                        "distributed_config": SingleDeviceDistributedConfig(
                            target_parameter_dimensionality=2
                        ),
                        "preconditioner_config": self._preconditioner_config,
                        "params": [0],
                    }
                ],
            }

        def test_state_dict(self) -> None:
            """Test that the state dict contains weight_buffer when iterate averaging is enabled."""
            state_dict = self._optimizer.state_dict()
            ref_state_dict = self._ref_state_dict
            self.assertEqual(state_dict.keys(), {"state", "param_groups"})

            torch.testing.assert_close(
                state_dict["state"],
                ref_state_dict["state"],
            )
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_load_state_dict(self) -> None:
            """Test that load_state_dict() correctly restores weight_buffer."""
            ref_state_dict = self._ref_state_dict
            self._optimizer.load_state_dict(
                state_dict=ref_state_dict,
            )

            state_dict = self._optimizer.state_dict()

            self.assertEqual(state_dict.keys(), ref_state_dict.keys())
            torch.testing.assert_close(state_dict["state"], ref_state_dict["state"])
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_weight_buffer_in_state(self) -> None:
            """Test that weight_buffer is present in each block's state."""
            state_dict = self._optimizer.state_dict()
            for block_key in ["block_0", "block_1"]:
                self.assertIn(
                    "weight_buffer",
                    state_dict["state"][0][block_key],
                    f"weight_buffer should be present in {block_key} when iterate averaging is enabled",
                )


class GPAShampooStateDictTest(
    AbstractIterateAveragingTest.IterateAveragingStateDictTestBase
):
    """Test state dict with Generalized Primal Averaging (GPA) enabled.

    See https://arxiv.org/pdf/2512.17131 for details on GPA.
    """

    @property
    def _iterate_averaging_config(self) -> GeneralizedPrimalAveragingConfig:
        return GeneralizedPrimalAveragingConfig(
            eval_interp_coeff=0.5,
            train_interp_coeff=0.9,
        )


class ScheduleFreeShampooStateDictTest(
    AbstractIterateAveragingTest.IterateAveragingStateDictTestBase
):
    """Test state dict with Schedule-Free enabled.

    See https://arxiv.org/abs/2405.15682 for details on Schedule-Free.
    """

    @property
    def _iterate_averaging_config(self) -> ScheduleFreeConfig:
        return ScheduleFreeConfig(
            train_interp_coeff=0.9,
        )


@instantiate_parametrized_tests
class DistributedShampooTrainEvalModeTest(unittest.TestCase):
    """Test train/eval mode switching with iterate averaging configurations."""

    @parametrize(
        "iterate_averaging_config",
        (
            GeneralizedPrimalAveragingConfig(),
            ScheduleFreeConfig(),
        ),
    )
    def test_train_eval_mode_switching(
        self,
        iterate_averaging_config: IterateAveragingConfig,
    ) -> None:
        """
        Test that train() and eval() mode switching works correctly with iterate averaging.
        This verifies that the mode switching updates the train_mode flag appropriately.
        """
        model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=iterate_averaging_config,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Get the full state dictionary.
        initial_state = optimizer.state_dict()["state"]

        # Find all parameters that contain train_mode keys.
        # We expect exactly 1 parameter (the first in each group) to have train_mode stored.
        train_mode_param_keys = []
        all_train_mode_keys: dict[str, list[str]] = {}
        for param_key, param_state in initial_state.items():
            keys_with_train_mode = [
                key for key in param_state.keys() if "train_mode" in key
            ]
            if keys_with_train_mode:
                train_mode_param_keys.append(param_key)
                all_train_mode_keys[param_key] = keys_with_train_mode

        # There should be exactly 1 parameter with train_mode keys (one per param group).
        self.assertEqual(
            len(train_mode_param_keys),
            1,
            msg=f"Expected exactly 1 parameter with train_mode keys, got {len(train_mode_param_keys)}",
        )

        train_mode_param_key = train_mode_param_keys[0]
        train_mode_keys = all_train_mode_keys[train_mode_param_key]

        # Verify initial training mode (should be True after training)
        for key in train_mode_keys:
            self.assertTrue(
                initial_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be True after training",
            )

        # Switch to eval mode
        optimizer.eval()

        # Verify eval mode
        eval_state = optimizer.state_dict()["state"]
        for key in train_mode_keys:
            self.assertFalse(
                eval_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be False after eval()",
            )

        # Switch back to train mode
        optimizer.train()

        # Verify train mode again
        train_state = optimizer.state_dict()["state"]
        for key in train_mode_keys:
            self.assertTrue(
                train_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be True after train()",
            )

    @parametrize(
        "iterate_averaging_config",
        (
            GeneralizedPrimalAveragingConfig(),
            ScheduleFreeConfig(),
        ),
    )
    @parametrize(
        "save_in_eval,load_in_eval",
        (
            (False, False),  # Save in train, load in train → stay in train.
            (True, False),  # Save in eval, load in train → stay in train.
            (False, True),  # Save in train, load in eval → stay in eval.
            (True, True),  # Save in eval, load in eval → stay in eval.
        ),
    )
    def test_load_state_dict_preserves_mode(
        self,
        iterate_averaging_config: IterateAveragingConfig,
        save_in_eval: bool,
        load_in_eval: bool,
    ) -> None:
        """Test that load_state_dict() preserves the caller's train/eval mode."""
        model = nn.Sequential(nn.Linear(5, 10, bias=False))
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=iterate_averaging_config,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps to populate state.
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Save checkpoint in the specified mode.
        if save_in_eval:
            optimizer.eval()
        state_dict = optimizer.state_dict()

        # Switch to the load mode.
        if load_in_eval:
            optimizer.eval()
        else:
            optimizer.train()

        # Load the checkpoint.
        optimizer.load_state_dict(state_dict)

        # Verify the optimizer preserved the load mode (not the save mode).
        expected_train_mode = not load_in_eval
        state = optimizer.state_dict()["state"]
        for param_state in state.values():
            for key in param_state:
                if "train_mode" in key:
                    self.assertEqual(
                        param_state[key].item(),
                        expected_train_mode,
                        msg=f"Expected train_mode to be {expected_train_mode} "
                        f"(save_in_eval={save_in_eval}, load_in_eval={load_in_eval})",
                    )

    def test_train_eval_mode_without_iterate_averaging(self) -> None:
        """
        Test that train() and eval() are no-ops when iterate_averaging_config is None.
        This verifies that calling these methods doesn't raise a KeyError.
        """
        model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=None,  # No iterate averaging
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Calling train() and eval() should not raise any errors
        optimizer.train()
        optimizer.eval()
        optimizer.train()

        # Verify state_dict works without iterate averaging
        state_dict = optimizer.state_dict()
        self.assertIn("state", state_dict)
        self.assertIn("param_groups", state_dict)


class DistributedShampooNoneGradTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(backend="eager"),
            distributed_config=DefaultSingleDeviceDistributedConfig,
            # Explicitly set grafting_config=None to test the case that no grafting config is used.
            grafting_config=None,
        )

    def test_step_with_consistent_grads(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        with self.assertNoLogs(level="WARNING"):
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

    def test_step_with_none_grads(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        expected_msg = "PT2 will recompile because the gradient selection of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
        ending_msg = "Changed gradient selector indices: [0, 1]"
        with self.assertLogs(level="WARNING") as cm:
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

            self._optimizer.zero_grad()  # Implicitly set grad=None in second step
            self._optimizer.step()
            msgs = [r.msg for r in cm.records]

        self.assertEqual(len(msgs), 1)
        self.assertIn(expected_msg, msgs[0])
        self.assertIn(ending_msg, msgs[0])

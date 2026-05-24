"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import math
import re
import unittest
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field, replace
from functools import partial
from unittest import mock

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner import shampoo_preconditioner_list
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EighEigendecompositionConfig,
    PerturbationConfig,
    QREigendecompositionConfig,
)
from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import (
    BaseShampooPreconditionerList,
    EigendecomposedKLShampooPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    RootInvKLShampooPreconditionerList,
    RootInvShampooPreconditionerList,
)
from distributed_shampoo.preconditioner.tests.preconditioner_list_test_utils import (
    AbstractPreconditionerListTest,
)
from distributed_shampoo.shampoo_types import (
    BaseShampooPreconditionerConfig,
    ClassicShampooPreconditionerConfig,
    DefaultShampooConfig,
    DefaultSOAPConfig,
    EigendecomposedKLShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    PreconditionerValueError,
    RootInvKLShampooPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
)
from distributed_shampoo.utils.abstract_dataclass import AbstractDataclass
from distributed_shampoo.utils.shampoo_utils import compress_list
from torch import Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@dataclass(init=False)
class AmortizedComputationProperties(AbstractDataclass):
    """Dataclass for properties of amortized computation functions."""

    amortized_computation_function_name: str = field(init=False)
    invalid_amortized_computation_return_values: (
        tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    ) = field(init=False)
    valid_amortized_computation_return_value: Tensor | tuple[Tensor, Tensor] = field(
        init=False
    )


@dataclass
class InverseRootProperties(AmortizedComputationProperties):
    """Dataclass for properties of matrix_inverse_root function."""

    amortized_computation_function_name: str = "matrix_inverse_root"
    invalid_amortized_computation_return_values: tuple[Tensor, Tensor] = (
        torch.tensor([torch.nan]),
        torch.tensor([torch.inf]),
    )
    valid_amortized_computation_return_value: Tensor = torch.tensor([1.0])


@dataclass
class EigendecompositionProperties(AmortizedComputationProperties):
    """Dataclass for properties of matrix_eigendecomposition function."""

    amortized_computation_function_name: str = "matrix_eigendecomposition"
    invalid_amortized_computation_return_values: tuple[
        tuple[Tensor, Tensor], tuple[Tensor, Tensor]
    ] = (
        (torch.tensor([torch.nan]), torch.tensor([torch.nan])),
        (torch.tensor([torch.inf]), torch.tensor([torch.inf])),
    )
    valid_amortized_computation_return_value: tuple[Tensor, Tensor] = (
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )


# Use outer class as wrapper to avoid running the abstract test.
class AbstractTest:
    @instantiate_parametrized_tests
    class BaseShampooPreconditionerListTest(
        AbstractPreconditionerListTest.Interface, abc.ABC
    ):
        """
        BaseShampooPreconditionerListTest is an abstract class for testing Shampoo preconditioner lists.

        Users should override the following abstract methods:
            1. _amortized_computation_properties - Provides properties for amortized computation functions.
            2. _default_preconditioner_config - Returns the default configuration for the preconditioner.
            3. _preconditioner_list_factory - Factory method to create instances of PreconditionerList.
        """

        @property
        @abc.abstractmethod
        def _amortized_computation_properties(
            self,
        ) -> AmortizedComputationProperties: ...

        @property
        @abc.abstractmethod
        def _default_preconditioner_config(self) -> BaseShampooPreconditionerConfig: ...

        @property
        @abc.abstractmethod
        def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]: ...

        def test_adaptive_amortized_computation_frequency(self) -> None:
            # Setup the preconditioner list with the adaptive amortized computation frequency.
            self._preconditioner_list = self._instantiate_preconditioner_list(
                preconditioner_config=replace(
                    self._default_preconditioner_config,
                    amortized_computation_config=EighEigendecompositionConfig(
                        tolerance=0.01,  # Any tolerance in (0.0, 1.0] works here.
                    ),
                )
            )

            # Create the masked gradient list for the test.
            masked_grad_list = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / torch.tensor(2.0).sqrt(),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor(1.0),
            )

            # Setup the constants for the mock functions.

            # Count total number of factor matrices across all parameters
            NUM_FACTOR_MATRICES = sum(grad.dim() for grad in masked_grad_list)
            # If criterion is False, amortized computation is performed.
            # If criterion is True, amortized computation is not performed.
            CRITERION_RESULTS = [False, True, False, False, True]
            assert len(CRITERION_RESULTS) == NUM_FACTOR_MATRICES
            # We will perform one update step, and we will skip the update when the criterion result is mocked to be True, i.e. sum(CRITERION_RESULTS_STEP_ONE) times.
            # Hence, the number of calls of the amortized computation function should be:
            NUM_AMORTIZED_COMPUTATION_CALLS = NUM_FACTOR_MATRICES - sum(
                CRITERION_RESULTS
            )

            with (
                # This mock replaces torch.sqrt with a function that returns:
                # -math.inf when CRITERION_RESULTS is True (meaning we want to skip computation)
                # math.inf when CRITERION_RESULTS is False (meaning we want to perform computation)
                # This simulates the adaptive computation criterion where negative values
                # indicate we should skip the computation, and positive values indicate we should compute
                mock.patch.object(
                    torch,
                    "sqrt",
                    side_effect=(
                        -math.inf
                        if v
                        else math.inf  # Return -inf for True (skip) and inf for False (compute)
                        for v in CRITERION_RESULTS
                    ),
                ) as mock_criterion,
                # Mock torch.linalg.eigh because it's the main function used in the amortized computation
                # after passing the adaptive criterion. By mocking it, we can assert that the number of times
                # the computation is performed matches our expectations based on the CRITERION_RESULTS.
                mock.patch.object(
                    torch.linalg,
                    "eigh",
                ) as mock_amortized_computation,
            ):
                self._preconditioner_list.update_preconditioners(
                    masked_grad_list=masked_grad_list,
                    step=torch.tensor(1),
                    perform_amortized_computation=True,
                )

                # Verify criterion called for each factor matrix
                self.assertEqual(mock_criterion.call_count, NUM_FACTOR_MATRICES)

                # Verify only non-skipped matrices were computed
                self.assertEqual(
                    mock_amortized_computation.call_count,
                    NUM_AMORTIZED_COMPUTATION_CALLS,
                )

        def _instantiate_block_list(self) -> tuple[Tensor, ...]:
            # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
            # and self._params[1] are split into two blocks.
            return (
                self._params[0],
                *torch.split(self._params[1], 2, dim=0),
                self._params[2],
            )

        def setUp(self) -> None:
            self._params = (
                torch.tensor([1.0, 2.0]),
                torch.arange(6, dtype=torch.float).reshape(3, 2),
                torch.tensor(1.0),  # a 0D tensor
            )
            self._state: dict[Tensor, dict[Hashable, object]] = {
                self._params[0]: {"block_0": {}},
                self._params[1]: {"block_0": {}, "block_1": {}},
                self._params[2]: {"block_0": {}},
            }
            # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
            # and self._params[1] are split into two blocks.
            self._block_info_list = (
                BlockInfo(
                    param=self._params[0],
                    composable_block_ids=(0, "block_0"),
                ),
                BlockInfo(
                    param=self._params[1],
                    composable_block_ids=(1, "block_0"),
                ),
                BlockInfo(
                    param=self._params[1],
                    composable_block_ids=(1, "block_1"),
                ),
                BlockInfo(
                    param=self._params[2],
                    composable_block_ids=(2, "block_0"),
                ),
            )

            super().setUp()

        def _instantiate_preconditioner_list(
            self, **kwargs: object
        ) -> PreconditionerList:
            return self._preconditioner_list_factory(
                block_list=self._block_list,
                state=self._state,
                block_info_list=self._block_info_list,
                **{"preconditioner_config": self._default_preconditioner_config}
                | kwargs,
            )

        @parametrize("invalid_value", (torch.nan, torch.inf))
        def test_raise_invalid_value_in_factor_matrix(
            self, invalid_value: float
        ) -> None:
            self.assertRaisesRegex(
                PreconditionerValueError,
                re.escape("Encountered nan/inf values in factor matrix"),
                self._preconditioner_list.update_preconditioners,
                masked_grad_list=(
                    torch.tensor([invalid_value, invalid_value]),
                    torch.eye(2) / math.sqrt(2.0),
                    torch.tensor([[invalid_value, invalid_value]]),
                    torch.tensor(invalid_value),
                ),
                step=torch.tensor(1),
                perform_amortized_computation=True,
            )

        def test_raise_nan_and_inf_in_inv_factor_matrix_amortized_computation(
            self,
        ) -> None:
            for invalid_value in self._amortized_computation_properties.invalid_amortized_computation_return_values:
                with (
                    self.subTest(invalid_value=invalid_value),
                    # Mock the amortized computation function to simulate inv factor matrix with nan/inf values.
                    mock.patch.object(
                        shampoo_preconditioner_list,
                        self._amortized_computation_properties.amortized_computation_function_name,
                        side_effect=(invalid_value,),
                    ) as mock_amortized_computation,
                ):
                    self.assertRaisesRegex(
                        PreconditionerValueError,
                        re.escape("Encountered nan or inf values in"),
                        self._preconditioner_list.update_preconditioners,
                        masked_grad_list=(
                            torch.tensor([1.0, 0.0]),
                            torch.eye(2) / math.sqrt(2.0),
                            torch.tensor([[1.0, 0.0]]),
                            torch.tensor(1.0),
                        ),
                        step=torch.tensor(1),
                        perform_amortized_computation=True,
                    )
                mock_amortized_computation.assert_called_once()

        def test_raise_nan_and_inf_in_inv_factor_matrix_amortized_computation_but_fail_saving(
            self,
        ) -> None:
            expected_torch_save_failures = RuntimeError("Failed to save")

            for invalid_value in self._amortized_computation_properties.invalid_amortized_computation_return_values:
                with (
                    self.subTest(invalid_value=invalid_value),
                    # Mock the amortized computation function to simulate inv factor matrix with nan/inf values.
                    mock.patch.object(
                        shampoo_preconditioner_list,
                        self._amortized_computation_properties.amortized_computation_function_name,
                        side_effect=(invalid_value,),
                    ) as mock_amortized_computation,
                    # Mock the torch.save function to simulate a failure when saving.
                    mock.patch.object(
                        torch,
                        "save",
                        side_effect=expected_torch_save_failures,
                    ) as mock_save,
                    self.assertLogs(level="WARNING") as cm,
                ):
                    self.assertRaisesRegex(
                        PreconditionerValueError,
                        re.escape("Encountered nan or inf values in"),
                        self._preconditioner_list.update_preconditioners,
                        masked_grad_list=(
                            torch.tensor([1.0, 0.0]),
                            torch.eye(2) / math.sqrt(2.0),
                            torch.tensor([[1.0, 0.0]]),
                            torch.tensor(1.0),
                        ),
                        step=torch.tensor(1),
                        perform_amortized_computation=True,
                    )

                mock_save.assert_called_once()
                mock_amortized_computation.assert_called_once()

                # Check that the warning message contains the expected failure reason.
                self.assertCountEqual(
                    # Only extracts the exception reason in the warning message for simple comparison.
                    [r.msg.split(": ", maxsplit=1)[-1] for r in cm.records],
                    [str(expected_torch_save_failures)],
                )

        def test_amortized_computation_internal_failure(self) -> None:
            with (
                mock.patch.object(
                    shampoo_preconditioner_list,
                    self._amortized_computation_properties.amortized_computation_function_name,
                    # Simulate the situation throws an exception (not nan and inf) to test the warning
                    side_effect=ZeroDivisionError,
                ) as mock_amortized_computation,
                self.assertLogs(level="WARNING") as cm,
            ):
                self._preconditioner_list.update_preconditioners(
                    masked_grad_list=(
                        torch.tensor([1.0, 0.0]),
                        torch.eye(2) / math.sqrt(2.0),
                        torch.tensor([[1.0, 0.0]]),
                        torch.tensor(1.0),
                    ),
                    step=torch.tensor(1),
                    perform_amortized_computation=True,
                )

            self.assertCountEqual(
                # Only extracts the first sentence in the warning message for simple comparison.
                [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                [
                    "Matrix computation failed for factor matrix 0.block_0.0 with exception=ZeroDivisionError()",
                    "Matrix computation failed for factor matrix 1.block_0.0 with exception=ZeroDivisionError()",
                    "Matrix computation failed for factor matrix 1.block_0.1 with exception=ZeroDivisionError()",
                    "Matrix computation failed for factor matrix 1.block_1.0 with exception=ZeroDivisionError()",
                    "Matrix computation failed for factor matrix 1.block_1.1 with exception=ZeroDivisionError()",
                ],
            )
            mock_amortized_computation.assert_called()
            mock_amortized_computation.reset_mock()

        def test_amortized_computation_failure_tolerance(self) -> None:
            self._preconditioner_list = self._instantiate_preconditioner_list()
            masked_grad_list0 = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor(1.0),
            )
            masked_grad_list = (
                torch.tensor([0.0, 1.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor(1.0),
            )

            # Number of calls to the amortized computation function per update.
            NUM_AMORTIZED_COMPUTATION_CALLS = 5

            # Initialize step counter.
            step = 1
            # Define the side effect for each call of the amortized computation function.
            fail = ValueError
            success = self._amortized_computation_properties.valid_amortized_computation_return_value
            all_but_one_fail = (fail,) * (NUM_AMORTIZED_COMPUTATION_CALLS - 1) + (
                success,
            )
            all_fail = (fail,) * NUM_AMORTIZED_COMPUTATION_CALLS
            all_success = (success,) * NUM_AMORTIZED_COMPUTATION_CALLS
            with (
                mock.patch.object(
                    shampoo_preconditioner_list,
                    self._amortized_computation_properties.amortized_computation_function_name,
                    # Note that the cases causally depend on each other.
                    side_effect=[
                        # Case 1: amortized computation fails less often than tolerance.
                        *all_but_one_fail,  # Success for a single Kronecker factor is not enough to reset counter.
                        # Case 2: amortized computation fails exactly as often as tolerance (3).
                        *all_fail,
                        *all_fail,
                        # Case 3: amortized computation succeeds after tolerance hit (counter is reset).
                        *all_success,
                        # Case 4: amortized computation fails more often than tolerance.
                        *all_fail,
                        *all_fail,
                        *all_fail,
                        fail,  # One failure is enough to raise an exception in this case.
                    ],
                ) as mock_amortized_computation
            ):
                # Accumulate factor matrices for valid amortized computation.
                self._preconditioner_list.update_preconditioners(
                    masked_grad_list=masked_grad_list0,
                    step=torch.tensor(step),
                    perform_amortized_computation=False,
                )
                self.assertEqual(mock_amortized_computation.call_count, 0)
                step += 1

                # Case 1: amortized computation fails less often than tolerance -> no error.
                with self.assertLogs(level="WARNING") as cm:
                    self._preconditioner_list.update_preconditioners(
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                # Check that warnings are logged for four failed amortized computations.
                # The fifth one doesn't raise an exception (see the definition of the side effect), so no warning is logged.
                self.assertCountEqual(
                    # Only extracts the first sentence in the warning message for simple comparison.
                    [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                    [
                        "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                    ],
                )
                step += 1

                # Case 2: amortized computation fails exactly as often as tolerance (3) -> no error.
                for _ in range(2):
                    with self.assertLogs(level="WARNING") as cm:
                        self._preconditioner_list.update_preconditioners(
                            masked_grad_list=masked_grad_list,
                            step=torch.tensor(step),
                            perform_amortized_computation=True,
                        )
                    # Check that warnings are logged for all failed amortized computations.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.1 with exception=ValueError()",
                        ],
                    )
                    step += 1

                # Case 3: amortized computation succeeds after tolerance hit (test reset) -> no error.
                with self.assertNoLogs(level="WARNING"):
                    self._preconditioner_list.update_preconditioners(
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                step += 1

                # Case 4: amortized computation fails more often than tolerance -> error.
                for _ in range(3):
                    with self.assertLogs(level="WARNING") as cm:
                        self._preconditioner_list.update_preconditioners(
                            masked_grad_list=masked_grad_list,
                            step=torch.tensor(step),
                            perform_amortized_computation=True,
                        )
                    # Check that warnings are logged for four failed amortized computations.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.1 with exception=ValueError()",
                        ],
                    )
                    step += 1
                # Exactly at failure tolerance now.
                with self.assertLogs(level="WARNING") as cm:
                    self.assertRaisesRegex(
                        ValueError,
                        r"The number of failed amortized computations for factors \('0\.block_0\.0',\) exceeded the allowed tolerance\. The last seen exception was .*",
                        self._preconditioner_list.update_preconditioners,
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                    # Check that the warning is logged for the failed amortized computation of the first matrix.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                        ],
                    )

        # Compare the results of preconditioning the gradient with both setups for different contract dimensions.
        @parametrize("dims", (([0], [0]), ([0], [1])))
        def test_precondition_grad(self, dims: tuple[list[int], list[int]]) -> None:
            # Generate a random gradient tensor with shape (2, 3, 4, 5, 6, 7).
            grad = torch.randn((2, 3, 4, 5, 6, 7))

            # Define selectors for which dimensions to precondition in the experimental setup.
            # Note that in the control setup, we will precondtion all dimensions normally except for the `False` ones with identity matrices.
            experimental_preconditioned_dims_selector = (
                True,
                False,
                False,
                True,
                True,
                False,
            )
            # Define selectors for which dimensions to precondition in the control setup.
            control_preconditioned_dims_selector = (True,) * grad.ndim

            # Create a list of random preconditioner matrices for each dimension of the gradient.
            preconditioner_list = [torch.randn((d, d)) for d in grad.shape]

            # Compress the preconditioner list based on experimental_preconditioned_dims_selector.
            experimental_preconditioner_list = compress_list(
                preconditioner_list,
                experimental_preconditioned_dims_selector,
            )

            # Create a control preconditioner list, using identity matrices where not preconditioning.
            control_preconditioner_list = tuple(
                preconditioner
                if should_precondition
                else torch.eye(preconditioner.shape[0])
                for preconditioner, should_precondition in zip(
                    preconditioner_list,
                    experimental_preconditioned_dims_selector,
                    strict=True,
                )
            )

            assert isinstance(self._preconditioner_list, BaseShampooPreconditionerList)
            precondition_grad = partial(
                self._preconditioner_list._precondition_grad, grad=grad, dims=dims
            )
            torch.testing.assert_close(
                precondition_grad(
                    preconditioned_dims_selector=experimental_preconditioned_dims_selector,
                    preconditioner_list=experimental_preconditioner_list,
                ),
                precondition_grad(
                    preconditioned_dims_selector=control_preconditioned_dims_selector,
                    preconditioner_list=control_preconditioner_list,
                ),
            )

        @property
        def _expected_numel_list(self) -> tuple[int, ...]:
            return (8, 16, 10, 0)

        @property
        def _expected_dims_list(self) -> tuple[torch.Size, ...]:
            return (
                torch.Size([2]),
                torch.Size([2, 2]),
                torch.Size([1, 2]),
                torch.Size([]),
            )

        @property
        def _expected_num_bytes_list(self) -> tuple[int, ...]:
            return (48, 96, 60, 0)

        @property
        def _expected_numel(self) -> int:
            return 34

        @property
        def _expected_num_bytes(self) -> int:
            return 204

        @property
        def _expected_compress_list_call_count(self) -> int:
            return 3

    class ClassicShampooPreconditionerListTest(BaseShampooPreconditionerListTest):
        @property
        @abc.abstractmethod
        def _default_preconditioner_config(
            self,
        ) -> ClassicShampooPreconditionerConfig: ...

        def test_update_preconditioners_and_precondition(self) -> None:
            """
            We provide examples where we update the preconditioners twice using specially
            chosen gradients such that we get a scalar * identity matrix for both Kronecker
            factor matrices for all parameters of interest.

            Specifically, for the beta2 = 1 and weighting_factor = 1 case, we have 3 parameters and define their gradients
            as the following in order to get the expected preconditioned gradient list:

            (1) Tensor of Size 2
                G1 = [1, 0]^T
                G2 = [0, 1]^T

                L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
                P = L^{-1/2} G2 = [0, 1]^T = G2

            (2) Tensor of Size 2 x 2
                G1 = [[1, 0], [0, 1]] / sqrt(2)
                G2 = [[1, 0], [0, 1]] / sqrt(2)

                L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
                R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
                P = L^{-1/4} G2 R^{-1/4} = [[1, 0], [0, 1]] / sqrt(2) = G2

            (3) Tensor of Size 1 x 2
                G1 = [[1, 0]]
                G2 = [[0, 1]]

                L = G1 * G1^T + G2 * G2^T = 2
                R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
                P = L^{-1/4} G2 R^{-1/4} = 2^{-1/4} * [[0, 1]] = 2^{-1/4} G2

            (4) Tensor of Size 0
                G1 = 3
                G2 = 2

                P = G2 = 2 because there is no preconditioner due to 0D tensor.

            """
            masked_grad_list1 = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor(3.0),
            )
            masked_grad_list2 = (
                torch.tensor([0.0, 1.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor(2.0),
            )

            masked_expected_preconditioned_grad_list = [
                preconditioned_grad.clone() for preconditioned_grad in masked_grad_list2
            ]
            masked_expected_preconditioned_grad_list[2] /= 2.0 ** (1 / 4)

            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                    use_bias_correction=True,
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list
                ),
            )

            """
            For the other two cases (beta2 < 1 and weighting_factor = 1 - beta2), note:

                L = beta2 * weighting_factor * G1 * G1^T + weighting_factor * G2 * G2^T
                R = beta2 * weighting_factor * G1^T * G1 + weighting_factor * G2^T * G2

            Therefore, in order to retain the identity matrix, we simply need to scale each gradient by:

                G1 -> G1 / sqrt(beta2 * weighting_factor)
                G2 -> G2 / sqrt(weighting_factor).

            """
            beta2 = 0.9
            weighting_factor = 1 - beta2

            beta2_compensated_grad_list1 = torch._foreach_div(
                masked_grad_list1, math.sqrt(beta2 * weighting_factor)
            )
            beta2_compensated_grad_list2 = torch._foreach_div(
                masked_grad_list2, math.sqrt(weighting_factor)
            )

            masked_expected_preconditioned_grad_list = [
                preconditioned_grad.clone()
                for preconditioned_grad in beta2_compensated_grad_list2
            ]
            masked_expected_preconditioned_grad_list[2] /= 2.0 ** (1 / 4)

            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=beta2,
                    weighting_factor=weighting_factor,
                    use_bias_correction=False,
                ),
                masked_grad_lists=[
                    beta2_compensated_grad_list1,
                    beta2_compensated_grad_list2,
                ],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list
                ),
            )

            """
            For the last case of including bias correction, we re-scale the entire matrix by the
            bias correction at iteration 2.

                L -> L / (1 - beta2^2)
                R -> R / (1 - beta2^2).

            Therefore, it is sufficient to additionally scale by this value:

                G1 -> sqrt(1 - beta2^2) * G1
                G2 -> sqrt(1 - beta2^2) * G2.

            """
            bias_compensated_grad_list1 = torch._foreach_mul(
                beta2_compensated_grad_list1, math.sqrt(1 - beta2**2)
            )
            bias_compensated_grad_list2 = torch._foreach_mul(
                beta2_compensated_grad_list2, math.sqrt(1 - beta2**2)
            )

            masked_expected_preconditioned_grad_list = [
                preconditioned_grad.clone()
                for preconditioned_grad in bias_compensated_grad_list2
            ]
            masked_expected_preconditioned_grad_list[2] /= 2.0 ** (1 / 4)

            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=beta2,
                    weighting_factor=weighting_factor,
                    use_bias_correction=True,
                ),
                masked_grad_lists=[
                    bias_compensated_grad_list1,
                    bias_compensated_grad_list2,
                ],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list
                ),
            )

            """
            For the case of beta2 = 1 and use_trace_scaling = True, trace scaling normalizes
            the factor matrix by sqrt(trace) before computing the inverse root.

            With trace scaling enabled:
            - The factor matrix L is normalized to L' = L / sqrt(trace(L))
            - The inverse root is computed on L' instead of L

            For this test, we reuse the same gradients as the baseline case (beta2 = 1).
            The factor matrices are L = R = I (identity), so trace(L) = trace(R) = 2.
            With trace scaling: L' = R' = I / sqrt(2)
                L'^{-1/2} = (I / sqrt(2))^{-1/2} = I * 2^{1/4}
                L'^{-1/4} = R'^{-1/4} = I * 2^{1/8}

            Therefore:
            (1) Tensor of Size 2:
                P = L'^{-1/2} G2 = 2^{1/4} * G2

            (2) Tensor of Size 2 x 2:
                P = L'^{-1/4} G2 R'^{-1/4} = 2^{1/8} * G2 * 2^{1/8} = 2^{1/4} * G2

            (3) Tensor of Size 1 x 2:
                L = 2 (scalar), R = I, trace(L) = 2, trace(R) = 2
                L' = 2 / sqrt(2) = sqrt(2), R' = I / sqrt(2)
                L'^{-1/4} = 2^{-1/8}, R'^{-1/4} = 2^{1/8}
                P = L'^{-1/4} G2 R'^{-1/4} = G2 (no change due to cancellation)

            (4) Tensor of Size 0:
                No preconditioner is applied. P = G2 = 2.

            """
            masked_expected_preconditioned_grad_list_trace_scaling = [
                masked_grad_list2[0] * 2.0 ** (1 / 4),
                masked_grad_list2[1] * 2.0 ** (1 / 4),
                masked_grad_list2[2],  # No change due to cancellation
                masked_grad_list2[3],  # 0D tensor, no preconditioner
            ]

            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                    use_bias_correction=True,
                    preconditioner_config=replace(
                        self._default_preconditioner_config,
                        use_trace_scaling=True,
                    ),
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list_trace_scaling
                ),
            )

        def test_update_preconditioners_and_precondition_with_epsilon(self) -> None:
            """
            We provide examples where we deliberately choose a large epsilon. This is to ensure that
            the matrix inverse computation behaves as expected. Below we update the preconditioners twice
            for 4 different blocks and check if the preconditioned gradients are as expected. When
            performing the inverse computation, epsilon is chosen to be 80 in (L + epsilon * I) and
            (R + epsilon * I).

            G_{ij}: at the i-th step, the gradient for the j-th block. For example,
            G12 is the gradient for the second block, at the first step.

            epsilon = 80.0

            Gradients for block 1: (no right preconditioner)
            (1) 1D Tensor of Size 2
                G11 = [1, 0]^T
                G21 = [0, 1]^T

                L = G11 * G11^T + G21 * G21^T = [[1, 0], [0, 1]]
                P = (L + epsilon * I)^{-1/2} G21 = [[81, 0], [0, 81]]^{-1/2} [0, 1]^T
                = [0, 1/9]^T.

            Gradients for block 2: (both left and right preconditioner)
            (2) Tensor of Size 2 x 2
                G12 = [[1, 0], [0, 1]] / sqrt(2)
                G22 = [[1, 0], [0, 1]] / sqrt(2)

                L = G12 * G12^T + G22 * G22^T = [[1, 0], [0, 1]]
                R = G12^T * G12 + G22^T * G22 = [[1, 0], [0, 1]]
                P = (L + epsilon * I)^{-1/4} G22 (R + epsilon * I)^{-1/4}
                = [[1/3, 0], [0, 1/3]] * G22 * [[1/3, 0], [0, 1/3]] = I / (9 * sqrt(2))

            Gradients for block 3: (both left and right preconditioner)
            (3) Tensor of Size 1 x 2
                G13 = [[1, 0]]
                G23 = [[0, 1]]

                L = G13 * G13^T + G23 * G23^T = I
                R = G13^T * G13 + G23^T * G23 = 2
                P = (L + epsilon * I)^{-1/4} G22 (R + epsilon * I)^{-1/4}
                = [[1/3, 0], [0, 1/3]] * G22 * (80 + 2)^{-1/4} =
                = [[0.0, 1.0/3.0 * 82.0 ** (-1/4)]]

            Gradients for block 4: (no preconditioner)
            (4) Tensor of Size 0
                G14 = 1
                G24 = 1

                No preconditioner is applied. Expected gradient is 1.

            """

            epsilon = 80.0

            # Blocked gradients at the first step: masked_grad_list1 = (G11, G12, G13, G14)
            masked_grad_list1 = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / math.sqrt(2),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor(1.0),
            )

            # Blocked gradients at the second step: masked_grad_list2 = (G21, G22, G23, G24)
            masked_grad_list2 = (
                torch.tensor([0.0, 1.0]),
                torch.eye(2) / math.sqrt(2),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor(1.0),
            )

            # Manually apply the preconditioners to the gradients at the second step (masked_grad_list2) with epsilon.
            # The result is stored in masked_expected_preconditioned_grad_list.

            masked_expected_preconditioned_grad_list = (
                torch.tensor([0.0, 1.0 / 9.0]),
                torch.eye(2) / (9 * math.sqrt(2)),
                torch.tensor([[0.0, 1.0 / 3.0 * 82.0 ** (-1 / 4)]]),
                torch.tensor(1.0),
            )

            # Apply preconditioner to the last step (masked_grad_list2) with epsilon. The result should be the same as the expected preconditioned grad list.
            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                    use_bias_correction=True,
                    epsilon=epsilon,
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list
                ),
            )

        def test_update_preconditioners_and_precondition_with_dims_ignored(
            self,
        ) -> None:
            """

            (1) Tensor of Size 2
                G1 = [4, 0]^T
                G2 = [0, 4]^T

                L = G1 * G1^T + G2 * G2^T = [[4*4, 0], [0, 4*4]]
                P = L^{-1/2} G2 = [[1/4, 0], [0, 1/4]] G2 = [0, 1]^T

            (2) Tensor of Size 2 x 2
                G1 = [[3, 0], [0, 3]]
                G2 = [[4, 0], [0, 4]]

                L = G1 * G1^T + G2 * G2^T = [[3*3+4*4, 0], [0, 3*3+4*4]]
                R = G1^T * G1 + G2^T * G2 = [[3*3+4*4, 0], [0, 3*3+4*4]]
                P = L^{-1/4} G2 R^{-1/4} = [[1/sqrt(5), 0], [0, 1/sqrt(5)]] G2 [[1/sqrt(5), 0], [0, 1/sqrt(5)]] = G2 / 5

            (3) Tensor of Size 1 x 2
                G1 = [[2, 0]]
                G2 = [[0, 2]]

                L = G1 * G1^T + G2 * G2^T = 2*2+2*2 = 8
                R = G1^T * G1 + G2^T * G2 = [[4, 0], [0, 4]]
                P = L^{-1/4} G2 R^{-1/4} = 8^{-1/4} G2 [[1/sqrt(2), 0], [0, 1/sqrt(2)]] = G2 / (sqrt(2 * sqrt(8)))

            (4) Tensor of Size 0
                G1 = 3
                G2 = 2

                P = G2 = 2 because there is no preconditioner due to 0D tensor.

            """
            masked_grad_list1 = (
                torch.tensor([4.0, 0.0]),
                torch.eye(2) * 3,
                torch.tensor([[2.0, 0.0]]),
                torch.tensor(3.0),
            )
            masked_grad_list2 = (
                torch.tensor([0.0, 4.0]),
                torch.eye(2) * 4,
                torch.tensor([[0.0, 2.0]]),
                torch.tensor(2.0),
            )

            masked_expected_preconditioned_grad_list = [
                torch.tensor([0.0, 1.0]),
                masked_grad_list2[1] / 5,
                masked_grad_list2[2] / math.sqrt(2 * math.sqrt(8)),
                torch.tensor(2.0),
            ]

            # The default case where we do not ignore any dimensions.
            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=tuple(
                    masked_expected_preconditioned_grad_list
                ),
            )

            # When ignoring all the dimensions by setting all inverse exponent override values to 0.0, the preconditioner should be the identity matrix, and the expected preconditioned gradient should be the same as the input gradient.
            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                    preconditioner_config=replace(
                        self._default_preconditioner_config,
                        inverse_exponent_override={
                            0: {0: 0.0},
                            1: {0: 0.0},
                            2: 0.0,
                        },
                    ),
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=masked_grad_list2,
            )

        def test_inverse_exponent_override(self) -> None:
            """
            For this example, we modify the one given above such that the inverse_exponent_override = {0: 1.0, 1: 1.0, 2: 1.0}.
            This effectively means all tensors in this test setting will use the inverse root of 1 rather than the default.
            This should result in the following behavior:

            (1) Tensor of Size 2
                G1 = [1, 0]^T
                G2 = [0, 2]^T

                L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
                P = L^{-1} G2 = [0, 0.5]^T

            (2) Tensor of Size 2 x 2
                G1 = [[1, 0], [0, 1]] / sqrt(2)
                G2 = [[1, 0], [0, 1]] / sqrt(2)

                L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
                R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
                P = L^{-1} G2 R^{-1} = [[1, 0], [0, 1]] / sqrt(2) = G2

            (3) Tensor of Size 1 x 2
                G1 = [[1, 0]]
                G2 = [[0, 2]]

                L = G1 * G1^T + G2 * G2^T = 5
                R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
                P = L^{-1} G2 R^{-1} =  [[0, 0.1]]

            (4) Tensor of Size 0
                G1 = 3
                G2 = 2

                P = G2 = 2 because there is no preconditioner due to 0D tensor.

            """

            preconditioner_config = replace(
                self._default_preconditioner_config,
                inverse_exponent_override={
                    0: {0: 1.0},
                    1: {0: 1.0},
                    2: 1.0,
                },
            )

            masked_grad_list1 = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[1.0, 0.0]]),
                torch.tensor(3.0),
            )
            masked_grad_list2 = (
                torch.tensor([0.0, 2.0]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[0.0, 2.0]]),
                torch.tensor(2.0),
            )

            masked_expected_preconditioned_grad_list = (
                torch.tensor([0, 0.5]),
                torch.eye(2) / math.sqrt(2.0),
                torch.tensor([[0, 0.1]]),
                torch.tensor(2.0),
            )

            self._verify_preconditioner_updates(
                preconditioner_list=self._instantiate_preconditioner_list(
                    beta2=1.0,
                    weighting_factor=1.0,
                    use_bias_correction=True,
                    preconditioner_config=preconditioner_config,
                ),
                masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
            )


class RootInvShampooPreconditionerListTest(
    AbstractTest.ClassicShampooPreconditionerListTest
):
    @property
    def _amortized_computation_properties(self) -> AmortizedComputationProperties:
        return InverseRootProperties()

    @property
    def _default_preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return replace(
            DefaultShampooConfig,
            factor_matrix_dtype=torch.float64,
            inv_factor_matrix_dtype=torch.float64,
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return RootInvShampooPreconditionerList

    @unittest.skip(
        "RootInvShampooPreconditionerList does not support adaptive computation frequency."
    )
    def test_adaptive_amortized_computation_frequency(self) -> None: ...


class EigendecomposedShampooPreconditionerListTest(
    AbstractTest.ClassicShampooPreconditionerListTest
):
    @property
    def _amortized_computation_properties(self) -> AmortizedComputationProperties:
        return EigendecompositionProperties()

    @property
    def _default_preconditioner_config(  # type: ignore[override]
        self,
    ) -> EigendecomposedShampooPreconditionerConfig:
        return EigendecomposedShampooPreconditionerConfig(
            amortized_computation_config=QREigendecompositionConfig(),
            factor_matrix_dtype=torch.float64,
            factor_matrix_eigenvectors_dtype=torch.float64,
            factor_matrix_eigenvalues_dtype=torch.float64,
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return EigendecomposedShampooPreconditionerList


class EigenvalueCorrectedShampooPreconditionerListTest(
    AbstractTest.BaseShampooPreconditionerListTest
):
    @property
    def _amortized_computation_properties(self) -> AmortizedComputationProperties:
        return EigendecompositionProperties()

    @property
    def _default_preconditioner_config(
        self,
    ) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return replace(
            DefaultSOAPConfig,
            factor_matrix_dtype=torch.float64,
            factor_matrix_eigenvectors_dtype=torch.float64,
            corrected_eigenvalues_dtype=torch.float64,
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return EigenvalueCorrectedShampooPreconditionerList

    def test_update_preconditioners_and_precondition(self) -> None:
        """
        We provide examples where we update the preconditioners twice using specially
        chosen gradients such that we get a scalar * identity matrix for both Kronecker
        factor matrices for all parameters of interest.

        Specifically, for the beta2 = 1 and weighting_factor = 1 case, we have 3 parameters and define their gradients
        as the following in order to get the expected preconditioned gradient list:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 2]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
            B = [[1, 0], [0, 1]]  # eigenvectors of L
            E = G1^2 + (B G2)^2   # corrected eigenvalues
            P = B ((B G2) / sqrt(E + eps)) = G2 / sqrt(E + eps)

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = [[1, 0], [0, 1]]     # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps)

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 5
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps)

        (4) Tensor of Size 0
            G1 = 1
            G2 = 2

            E = G1^2 + G2^2            # identical to Adam on 0D tensors
            P = G2 / (E + eps) = 2 / (5 + eps) ≈ 0.4

        """
        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor(1.0),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 2.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[0.0, 2.0]]),
            torch.tensor(2.0),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor(2 / math.sqrt(5)),
        )
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                weighting_factor=1.0,
                use_bias_correction=True,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
        )

        """
        For the other two cases (beta2 < 1 and weighting_factor = 1 - beta2), note:

            E = beta2 * weighting_factor G1^2 + weighting_factor G2^2

        Therefore, in order to retain the identity matrix, we simply need to scale each gradient by:

            G1 -> G1 / sqrt(beta2 * weighting_factor)
            G2 -> G2 / sqrt(weighting_factor).

        """
        beta2 = 0.9
        weighting_factor = 1 - beta2

        beta2_compensated_grad_list1 = torch._foreach_div(
            masked_grad_list1, math.sqrt(beta2 * weighting_factor)
        )
        beta2_compensated_grad_list2 = torch._foreach_div(
            masked_grad_list2, math.sqrt(weighting_factor)
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor(2 / math.sqrt(5)),
        )
        # Fix scaling due to EMA.
        torch._foreach_div_(
            masked_expected_preconditioned_grad_list, math.sqrt(weighting_factor)
        )

        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                weighting_factor=weighting_factor,
                use_bias_correction=False,
            ),
            masked_grad_lists=[
                beta2_compensated_grad_list1,
                beta2_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

        """
        For the last case of including bias correction, we re-scale the entire matrix by the
        bias correction at iteration 2.

            E -> E / (1 - beta2^2).

        Therefore, it is sufficient to additionally scale by this value:

            G1 -> sqrt(1 - beta2^2) * G1
            G2 -> sqrt(1 - beta2^2) * G2.

        """
        bias_compensated_grad_list1 = torch._foreach_mul(
            beta2_compensated_grad_list1, math.sqrt(1 - beta2**2)
        )
        bias_compensated_grad_list2 = torch._foreach_mul(
            beta2_compensated_grad_list2, math.sqrt(1 - beta2**2)
        )

        # Fix scaling due to bias correction.
        torch._foreach_mul_(
            masked_expected_preconditioned_grad_list, math.sqrt(1 - beta2**2)
        )

        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                weighting_factor=weighting_factor,
                use_bias_correction=True,
            ),
            masked_grad_lists=[
                bias_compensated_grad_list1,
                bias_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

    def test_inverse_exponent_override(self) -> None:
        """
        For this example, we modify the one given above such that the inverse_exponent_override = {0: 1.0, 1: 1.0, 2: 1.0}.
        This effectively means all tensors in this test setting will use the inverse root of 1 rather than the default, 1/2.
        This should result in the following behavior:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 2]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
            B = [[1, 0], [0, 1]]  # eigenvectors of L
            E = G1^2 + (B G2)^2   # corrected eigenvalues
            P = B ((B G2) / (E + eps) = G2 / (E + eps) ≈ [0, 0.5]^T

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = [[1, 0], [0, 1]]     # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / (E + eps) B_R = G2 / (E + eps) ≈ G2

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 5
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / (E + eps)) B_R = G2 / (E + eps) ≈ [[0, 0.5]]

        (4) Tensor of Size 0
            G1 = 1
            G2 = 2

            E = G1^2 + G2^2            # identical to Adam on 0D tensors
            P = G2 / (E + eps) = 2 / (5 + eps) ≈ 0.4

        """

        preconditioner_config = EigenvalueCorrectedShampooPreconditionerConfig(
            inverse_exponent_override={0: 1.0, 1: 1.0, 2: 1.0},
            amortized_computation_config=EighEigendecompositionConfig(
                rank_deficient_stability_config=PerturbationConfig(
                    perturb_before_computation=False
                )
            ),
        )

        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor(1.0),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 2.0]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[0.0, 2.0]]),
            torch.tensor(2.0),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0, 0.5]),
            torch.eye(2) / math.sqrt(2.0),
            torch.tensor([[0, 0.5]]),
            torch.tensor(0.4),
        )

        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                weighting_factor=1.0,
                use_bias_correction=True,
                preconditioner_config=preconditioner_config,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
        )


class RootInvKLShampooPreconditionerListTest(RootInvShampooPreconditionerListTest):
    @property
    def _default_preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return replace(
            RootInvKLShampooPreconditionerConfig(),
            factor_matrix_dtype=torch.float64,
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return RootInvKLShampooPreconditionerList

    @unittest.skip(
        "RootInvKLShampooPreconditionerList does not support adaptive computation frequency."
    )
    def test_adaptive_amortized_computation_frequency(self) -> None: ...


class EigendecomposedKLShampooPreconditionerListTest(
    EigendecomposedShampooPreconditionerListTest
):
    @property
    def _default_preconditioner_config(  # type: ignore[override]
        self,
    ) -> EigendecomposedKLShampooPreconditionerConfig:
        return EigendecomposedKLShampooPreconditionerConfig(
            amortized_computation_config=QREigendecompositionConfig(),
            factor_matrix_dtype=torch.float64,
            factor_matrix_eigenvectors_dtype=torch.float64,
            factor_matrix_eigenvalues_dtype=torch.float64,
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return EigendecomposedKLShampooPreconditionerList

    def test_update_preconditioners_and_precondition_with_epsilon(self) -> None:
        """
        We provide examples where we deliberately choose a large epsilon. This is to ensure that
        the matrix inverse computation behaves as expected. Below we update the preconditioners twice
        for 4 different blocks and check if the preconditioned gradients are as expected. When
        performing the inverse computation, epsilon is chosen to be 80 in (L + epsilon * I) and
        (R + epsilon * I).

        G_{ij}: at the i-th step, the gradient for the j-th block. For example,
        G12 is the gradient for the second block, at the first step.

        For KL-Shampoo, we will correct the scale of the 2D gradients such that the Kronecker factors
        match the ones of regular Shampoo with a scale correction s.

        epsilon = 80.0
        s = epsilon^{1/4}

        Gradients for block 1: (no right preconditioner)
        (1) 1D Tensor of Size 2
            G11 = [1, 0]^T
            G21 = [0, 1]^T

            L = G11 * G11^T + G21 * G21^T = [[1, 0], [0, 1]]
            P = (L + epsilon * I)^{-1/2} G21 = [[81, 0], [0, 81]]^{-1/2} [0, 1]^T
            = [0, 1/9]^T.

        Gradients for block 2: (both left and right preconditioner)
        (2) Tensor of Size 2 x 2
            G12 = s * [[1, 0], [0, 1]] / sqrt(2)
            G22 = s * [[1, 0], [0, 1]] / sqrt(2)

            L = G12 * G12^T + G22 * G22^T = [[1, 0], [0, 1]]
            R = G12^T * G12 + G22^T * G22 = [[1, 0], [0, 1]]
            P = (L + epsilon * I)^{-1/4} G22 (R + epsilon * I)^{-1/4}
            = [[1/3, 0], [0, 1/3]] * G22 * [[1/3, 0], [0, 1/3]] = s * I / (9 * sqrt(2))

        Gradients for block 3: (both left and right preconditioner)
        (3) Tensor of Size 1 x 2
            G13 = s * [[1, 0]]
            G23 = s * [[0, 1]]

            L = G13 * G13^T + G23 * G23^T = I
            R = G13^T * G13 + G23^T * G23 = 2
            P = (L + epsilon * I)^{-1/4} G22 (R + epsilon * I)^{-1/4}
            = [[1/3, 0], [0, 1/3]] * G22 * (80 + 2)^{-1/4} =
            = s * [[0.0, 1.0/3.0 * 82.0 ** (-1/4)]]

        Gradients for block 4: (no preconditioner)
        (4) Tensor of Size 0
            G14 = 1
            G24 = 1

            No preconditioner is applied. Expected gradient is 1.

        """

        epsilon = 80.0
        scale_correction = epsilon**0.25

        # Blocked gradients at the first step: masked_grad_list1 = (G11, G12, G13, G14)
        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            scale_correction * torch.eye(2) / math.sqrt(2),
            scale_correction * torch.tensor([[1.0, 0.0]]),
            torch.tensor(1.0),
        )

        # Blocked gradients at the second step: masked_grad_list2 = (G21, G22, G23, G24)
        masked_grad_list2 = (
            torch.tensor([0.0, 1.0]),
            scale_correction * torch.eye(2) / math.sqrt(2),
            scale_correction * torch.tensor([[0.0, 1.0]]),
            torch.tensor(1.0),
        )

        # Manually apply the preconditioners to the gradients at the second step (masked_grad_list2) with epsilon.
        # The result is stored in masked_expected_preconditioned_grad_list.
        masked_expected_preconditioned_grad_list = (
            torch.tensor([0.0, 1.0 / 9.0]),
            scale_correction * torch.eye(2) / (9 * math.sqrt(2)),
            scale_correction * torch.tensor([[0.0, 1.0 / 3.0 * 82.0 ** (-1 / 4)]]),
            torch.tensor(1.0),
        )

        # Apply preconditioner to the last step (masked_grad_list2) with epsilon. The result should be the same as the expected preconditioned grad list.
        self._verify_preconditioner_updates(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                weighting_factor=1.0,
                use_bias_correction=True,
                epsilon=epsilon,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

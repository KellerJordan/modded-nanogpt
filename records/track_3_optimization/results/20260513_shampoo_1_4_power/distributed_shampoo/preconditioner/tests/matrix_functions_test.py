"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import itertools
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from types import ModuleType
from unittest import mock

import torch
from distributed_shampoo.preconditioner import matrix_functions
from distributed_shampoo.preconditioner.matrix_functions import (
    _check_2d_tensor,
    _check_square_matrix,
    _matrix_perturbation,
    matrix_eigendecomposition,
    matrix_inverse_root,
    matrix_inverse_root_from_eigendecomposition,
    matrix_orthogonalization,
    NewtonConvergenceFlag,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    EigenConfig,
    EigendecompositionConfig,
    EighEigendecompositionConfig,
    NewtonSchulzOrthogonalizationConfig,
    OrthogonalizationConfig,
    PerturbationConfig,
    PseudoInverseConfig,
    QREigendecompositionConfig,
    RankDeficientStabilityConfig,
    RootInvConfig,
    SVDOrthogonalizationConfig,
)
from torch import Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class Check2DTensorTest(unittest.TestCase):
    @staticmethod
    @_check_2d_tensor
    def check_tensor_func(A: Tensor) -> Tensor:
        """Helper function decorated with _check_2d_tensor for testing."""
        return A

    @parametrize("A", (torch.tensor(5), torch.zeros((2, 2, 2))))
    def test_check_tensor_for_non_two_dim_matrix(self, A: Tensor) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            self.check_tensor_func,
            A=A,
        )

    def test_check_tensor_for_square_matrix(self) -> None:
        A = torch.eye(2)
        # Verify the function was called and returned the input
        torch.testing.assert_close(A, self.check_tensor_func(A=A))

    def test_check_tensor_for_rectangle_matrix(self) -> None:
        A = torch.zeros((2, 3))
        # Verify the function was called and returned the input
        torch.testing.assert_close(A, self.check_tensor_func(A=A))


class CheckSquareMatrixTest(Check2DTensorTest):
    @staticmethod
    @_check_square_matrix
    def check_tensor_func(A: Tensor) -> Tensor:
        """Helper function decorated with _check_square_matrix for testing."""
        return A

    def test_check_tensor_for_rectangle_matrix(self) -> None:
        """Override the test from Check2DTensorTest to test non-square matrix."""
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            super().test_check_tensor_for_rectangle_matrix,
        )


@instantiate_parametrized_tests
class MatrixPerturbationTest(unittest.TestCase):
    def test_matrix_perturbation_not_is_eigenvalues(self) -> None:
        A = torch.eye(2)
        torch.testing.assert_close(
            A * 1.1, _matrix_perturbation(A=A, epsilon=0.1, is_eigenvalues=False)
        )

    @parametrize("A", (torch.ones(5), torch.eye(2)))
    def test_matrix_perturbation_is_eigenvalues(self, A: Tensor) -> None:
        torch.testing.assert_close(
            A + 0.1, _matrix_perturbation(A=A, epsilon=0.1, is_eigenvalues=True)
        )


@instantiate_parametrized_tests
class MatrixInverseRootFromEigendecomposition(unittest.TestCase):
    @parametrize("perturb_before_computation", (True, False))
    def test_perturbation_before_computation(
        self, perturb_before_computation: bool
    ) -> None:
        L = torch.tensor([0.1, 3.1])
        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # The stabilized eigenvalues is [1.0, 4.0] and root = 2, inv_power_L is [1.0, 0.5].
        torch.testing.assert_close(
            torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
            matrix_inverse_root_from_eigendecomposition(
                L=L,
                Q=Q,
                root=Fraction(2),
                epsilon=1.0 - torch.min(L).item() * (not perturb_before_computation),
                rank_deficient_stability_config=PerturbationConfig(
                    perturb_before_computation=perturb_before_computation
                ),
            ),
        )

    @parametrize("rank_rtol", (None, 1e-6))
    def test_pseudoinverse(self, rank_rtol: float | None) -> None:
        L = torch.tensor([1.0, 4.0, 0.0])
        Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]),
            matrix_inverse_root_from_eigendecomposition(
                L=L,
                Q=Q,
                root=Fraction(2),
                epsilon=0.0,
                rank_deficient_stability_config=PseudoInverseConfig(
                    rank_rtol=rank_rtol
                ),
            ),
        )

    @parametrize("perturb_before_computation", (True, False))
    def test_test_perturbation_before_computation_with_disportionate_epsilon(
        self, perturb_before_computation: bool
    ) -> None:
        # This test verifies that matrix_inverse_root_from_eigenvalues_and_eigenvectors handles matrices with large values correctly.
        # Note that the smallest entries in L have absolute magnitude of 100000.0, which is much larger than epsilon (1e-5).
        # In a numerically unstable implementation, such a small epsilon would get "absorbed" or lost when added to these large values, causing potential numerical instability.
        L = torch.tensor(
            [
                [30000.0, 30000.0, -100000.0],
                [40000.0, 40000.0, -100000.0],
                [-100000.0, -100000.0, 400000.0],
            ]
        )
        Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        inverse_root = matrix_inverse_root_from_eigendecomposition(
            L=L,
            Q=Q,
            root=Fraction(2),
            epsilon=1e-5,
            rank_deficient_stability_config=PerturbationConfig(
                perturb_before_computation=perturb_before_computation,
            ),
        )
        self.assertTrue(torch.isfinite(inverse_root).all())

    def test_pseudoinverse_with_invalid_epsilon(self) -> None:
        L = torch.tensor([1.0, 4.0, 0.0])
        Q = torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
        epsilon = 1e-8
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"{epsilon=} should be 0.0 when using pseudo-inverse!"),
            matrix_inverse_root_from_eigendecomposition,
            L=L,
            Q=Q,
            root=Fraction(2),
            epsilon=epsilon,
            rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None),
        )

    def test_invalid_rank_deficient_stability_config(self) -> None:
        @dataclass
        class NotSupportedRankDeficientStabilityConfig(RankDeficientStabilityConfig):
            """A dummy rank_deficient_stability_config that is not supported."""

            unsupported_mode: str = ""

        L = torch.tensor([1.0, 4.0, 0.0])
        Q = torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
        self.assertRaisesRegex(
            NotImplementedError,
            r"rank_deficient_stability_config=.*\.NotSupportedRankDeficientStabilityConfig\(.*\) is not supported\.",
            matrix_inverse_root_from_eigendecomposition,
            L=L,
            Q=Q,
            root=Fraction(2),
            rank_deficient_stability_config=NotSupportedRankDeficientStabilityConfig(),
        )


@instantiate_parametrized_tests
class MatrixInverseRootTest(unittest.TestCase):
    @parametrize(
        "root_inv_config",
        (
            EigenConfig(),  # perturb_before_computation=True by default
            CoupledNewtonConfig(),
            EigenConfig(
                rank_deficient_stability_config=PerturbationConfig(
                    perturb_before_computation=False
                )
            ),
            EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),  # equivalent behavior when test matrices are full rank
            EigenConfig(eigendecomposition_offload_device="cpu"),
            *(
                CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0, order=order)
                for order in range(2, 7)
            ),
        ),
    )
    @parametrize("exp", (1, 2))
    @parametrize(
        "A, expected_root",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, -3.0], [1.0, -3.0, 18.0]]),
            ),
        ),
    )
    def test_matrix_inverse_root(
        self, A: Tensor, expected_root: Tensor, exp: int, root_inv_config: RootInvConfig
    ) -> None:
        atol = 0.05
        rtol = 1e-2

        torch.testing.assert_close(
            torch.linalg.matrix_power(expected_root, exp),
            matrix_inverse_root(
                A=A,
                root=Fraction(2, exp),
                root_inv_config=root_inv_config,
            ),
            atol=atol,
            rtol=rtol,
        )

    def test_matrix_inverse_root_higher_order_blowup(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape(
                "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
            ),
            matrix_inverse_root,
            A=A,
            root=Fraction(1, 20),
            root_inv_config=CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
        )

    def test_matrix_inverse_root_with_no_effect_exponent_multiplier(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        exp = 3
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"root.denominator={exp} must be equal to 1 to use coupled inverse Newton iteration!"
            ),
            matrix_inverse_root,
            A=A,
            root=Fraction(2, exp),
            root_inv_config=CoupledNewtonConfig(),
        )

    @parametrize(
        "root_inv_config, msg",
        [
            (CoupledNewtonConfig(), "Newton"),
            (
                CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
                "Higher order method",
            ),
        ],
    )
    def test_matrix_inverse_root_reach_max_iterations(
        self, root_inv_config: RootInvConfig, msg: str
    ) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
        with (
            mock.patch.object(
                matrix_functions,
                "_assign_function_args_from_config",
                return_value=lambda *args, **kwargs: (
                    None,
                    None,
                    NewtonConvergenceFlag.REACHED_MAX_ITERS,
                    None,
                    None,
                ),
            ),
            self.assertLogs(level="WARNING") as cm,
        ):
            matrix_inverse_root(A=A, root=root, root_inv_config=root_inv_config)
            self.assertIn(
                f"{msg} did not converge and reached maximum number of iterations!",
                [r.msg for r in cm.records],
            )

    def test_matrix_inverse_root_higher_order_tf32_preservation(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, float("inf")]])
        root = Fraction(2)
        tf32_flag_before = torch.backends.cuda.matmul.allow_tf32
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape("Input matrix has entries close to inf"),
            matrix_inverse_root,
            A=A,
            root=root,
            root_inv_config=CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
        )
        tf32_flag_after = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(tf32_flag_before, tf32_flag_after)

    def test_matrix_inverse_root_higher_order_error_blowup_before_powering(
        self,
    ) -> None:
        # Trigger this error by using an ill-conditioned matrix.
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        root = Fraction(2)
        self.assertRaisesRegex(
            ArithmeticError,
            r"Error in matrix inverse root \(before powering for fractions\) [+-]?([0-9]*[.])?[0-9]+ exceeds threshold 1e-1, raising an exception!",
            matrix_inverse_root,
            A=A,
            root=root,
            # Set max_iterations to 0 to fast forward to the error check before powering.
            root_inv_config=CoupledHigherOrderConfig(
                rel_epsilon=0.0, abs_epsilon=0.0, max_iterations=0
            ),
        )

    def test_matrix_inverse_root_with_invalid_root_inv_config(self) -> None:
        @dataclass
        class NotSupportedRootInvConfig(RootInvConfig):
            """A dummy root inv config that is not supported."""

            unsupported_root: int = -1

        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
        self.assertRaisesRegex(
            NotImplementedError,
            r"Root inverse config is not implemented! Specified root inverse config is root_inv_config=.*\.NotSupportedRootInvConfig\(.*\)\.",
            matrix_inverse_root,
            A=A,
            root=root,
            root_inv_config=NotSupportedRootInvConfig(),
        )


feasible_alpha_beta_pairs: tuple[tuple[float, float], ...] = tuple(
    (alpha, beta)
    for alpha, beta in itertools.product((0.001, 0.01, 0.1, 1.0, 10.0, 100.0), repeat=2)
    if 2 * beta <= alpha
)


@instantiate_parametrized_tests
class EigenRootTest(unittest.TestCase):
    def _test_eigen_root_multi_dim(
        self,
        A: Callable[[int], Tensor],
        n: int,
        root: int,
        epsilon: float,
        tolerance: float,
    ) -> None:
        X = matrix_inverse_root(
            A=A(n),
            root=Fraction(root),
            root_inv_config=DefaultEigenConfig,
            epsilon=epsilon,
        )
        abs_error = torch.dist(torch.linalg.matrix_power(X, -root), A(n), p=torch.inf)
        A_norm = torch.linalg.norm(A(n), ord=torch.inf)
        rel_error = abs_error / torch.maximum(torch.tensor(1.0), A_norm)
        self.assertLessEqual(rel_error.item(), tolerance)

    @parametrize("root", [1, 2, 4, 8])
    @parametrize("n", [10, 100])
    def test_eigen_root_identity(self, n: int, root: int) -> None:
        self._test_eigen_root_multi_dim(
            A=torch.eye,
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-6,
        )

    @parametrize("alpha, beta", feasible_alpha_beta_pairs)
    @parametrize("root", [1, 2, 4, 8])
    @parametrize("n", [10, 100])
    def test_eigen_root_tridiagonal(
        self, n: int, root: int, alpha: float, beta: float
    ) -> None:
        def A_tridiagonal_1(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] += beta
            diag[n - 1] += beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_eigen_root_multi_dim(
            A=partial(A_tridiagonal_1, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-4,
        )

        def A_tridiagonal_2(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] -= beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_eigen_root_multi_dim(
            A=partial(A_tridiagonal_2, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-4,
        )

    def test_eigen_root_nonfull_rank(self) -> None:
        A = torch.tensor([[2.0, 1.0], [2.0, 1.0]])
        root = Fraction(2)
        epsilon = 0.0

        M_default = matrix_inverse_root(
            A=A, root=root, root_inv_config=EigenConfig(), epsilon=epsilon
        )
        self.assertTrue(torch.all(torch.isinf(M_default)))

        M_pseudoinverse = matrix_inverse_root(
            A=A,
            root=root,
            root_inv_config=EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),
            epsilon=epsilon,
        )
        self.assertTrue(torch.all(torch.isreal(M_pseudoinverse)))

    def test_matrix_root_eigen_nonpositive_root(self) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = Fraction(-1)
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Root {root} should be positive!"),
            matrix_inverse_root,
            A=A,
            root=root,
        )

    torch_linalg_module: ModuleType = torch.linalg

    @mock.patch.object(
        torch_linalg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_no_retry_double_precision_raise_exception(
        self, mock_eigh: mock.Mock
    ) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        self.assertRaisesRegex(
            RuntimeError,
            re.escape("Mock Eigen Error"),
            matrix_inverse_root,
            A=A,
            root=Fraction(2),
            root_inv_config=EigenConfig(retry_double_precision=False),
            epsilon=0.0,
        )
        mock_eigh.assert_called_once()

    @mock.patch.object(
        torch_linalg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        self.assertRaisesRegex(
            RuntimeError,
            re.escape("Mock Eigen Error"),
            matrix_inverse_root,
            A=A,
            root=Fraction(2),
            epsilon=0.0,
        )
        mock_eigh.assert_called()
        self.assertEqual(mock_eigh.call_count, 2)

    def test_retry_double_precision_double_precision(self) -> None:
        """Test that matrix_inverse_root retries with double precision when eigh fails."""
        # Store the original eigh function to use in our mock
        original_eigh: Callable[..., tuple[Tensor, Tensor]] = torch.linalg.eigh

        # Mock the eigh function
        mock_eigh: mock.Mock
        with mock.patch.object(torch.linalg, "eigh") as mock_eigh:
            # Define a side effect function that fails on first call but succeeds on subsequent calls
            def only_first_call_runtime_error(
                *args: object, **kwargs: object
            ) -> tuple[Tensor, Tensor]:
                if mock_eigh.call_count == 1:
                    raise RuntimeError("Mock Eigen Error")
                return original_eigh(*args, **kwargs)

            mock_eigh.side_effect = only_first_call_runtime_error

            # Create identity matrix for testing
            A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

            # Call the function under test - should retry after first failure
            X = matrix_inverse_root(
                A=A,
                root=Fraction(2),
                epsilon=0.0,
            )

            # Verify the result is correct (identity matrix)
            torch.testing.assert_close(X, torch.eye(2))

            # Verify eigh was called and retried
            mock_eigh.assert_called()
            self.assertEqual(mock_eigh.call_count, 2)


@instantiate_parametrized_tests
class NewtonRootInverseTest(unittest.TestCase):
    def _test_newton_root_inverse_multi_dim(
        self,
        A: Callable[[int], Tensor],
        n: int,
        root: int,
        epsilon: float,
        max_iterations: int,
        A_tol: float,
        M_tol: float,
    ) -> None:
        X = matrix_inverse_root(
            A=A(n),
            root=root,
            root_inv_config=CoupledNewtonConfig(
                max_iterations=max_iterations, tolerance=M_tol
            ),
            epsilon=epsilon,
        )
        abs_A_error = torch.dist(torch.linalg.matrix_power(X, -root), A(n), p=torch.inf)
        A_norm = torch.linalg.norm(A(n), ord=torch.inf)
        rel_A_error = abs_A_error / torch.maximum(torch.tensor(1.0), A_norm)
        self.assertLessEqual(rel_A_error.item(), A_tol)

    @parametrize("root", [2, 4, 8])
    @parametrize("n", [10, 100])
    def test_newton_root_inverse_identity(self, n: int, root: int) -> None:
        max_iterations = 1000

        self._test_newton_root_inverse_multi_dim(
            A=torch.eye,
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-6,
            M_tol=1e-6,
        )

    @parametrize("alpha, beta", feasible_alpha_beta_pairs)
    @parametrize("root", [2, 4, 8])
    @parametrize("n", [10, 100])
    def test_newton_root_inverse_tridiagonal(
        self, n: int, root: int, alpha: float, beta: float
    ) -> None:
        max_iterations = 1000

        def A_tridiagonal_1(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] += beta
            diag[n - 1] += beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_newton_root_inverse_multi_dim(
            A=partial(A_tridiagonal_1, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-4,
            M_tol=1e-6,
        )

        def A_tridiagonal_2(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] -= beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_newton_root_inverse_multi_dim(
            A=partial(A_tridiagonal_2, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-4,
            M_tol=1e-6,
        )


class CoupledHigherOrderRootInverseTest(unittest.TestCase):
    def test_root_with_big_numerator_denominator(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(13, 15)
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            matrix_inverse_root(
                A=A,
                root=root,
                root_inv_config=CoupledHigherOrderConfig(
                    rel_epsilon=0.0, abs_epsilon=0.0
                ),
            )
        self.assertIn(
            "abs(root.numerator)=13 and abs(root.denominator)=15 are probably too big for best performance.",
            [r.msg for r in cm.records],
        )


@instantiate_parametrized_tests
class MatrixEigendecompositionTest(unittest.TestCase):
    def test_pseudoinverse_with_invalid_epsilon(self) -> None:
        A = torch.ones((2, 2))
        epsilon = 1e-8
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"{epsilon=} should be 0.0 when using pseudo-inverse!"),
            matrix_eigendecomposition,
            A=A,
            epsilon=epsilon,
            eigendecomposition_config=EighEigendecompositionConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),
        )

    @parametrize("perturb_before_computation", (False, True))
    @parametrize("eigendecomposition_offload_device", ("cpu", ""))
    @parametrize(
        "A, expected_eigenvalues, expected_eigenvectors",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([1.0, 4.0]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([2.9008677229e-03, 1.7424316704e-01, 1.9828229980e03]),
                torch.tensor(
                    [
                        [0.0460073575, -0.6286827326, 0.7762997746],
                        [-0.1751257628, -0.7701635957, -0.6133345366],
                        [0.9834705591, -0.1077321917, -0.1455317289],
                    ]
                ),
            ),
        ),
    )
    def test_matrix_eigendecomposition_with_eigh(
        self,
        A: Tensor,
        expected_eigenvalues: Tensor,
        expected_eigenvectors: Tensor,
        perturb_before_computation: bool,
        eigendecomposition_offload_device: str,
    ) -> None:
        atol = 1e-4
        rtol = 1e-5

        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            matrix_eigendecomposition(
                A=A,
                eigendecomposition_config=EighEigendecompositionConfig(
                    rank_deficient_stability_config=PerturbationConfig(
                        perturb_before_computation=perturb_before_computation
                    ),
                    eigendecomposition_offload_device=eigendecomposition_offload_device,
                ),
            ),
            atol=atol,
            rtol=rtol,
        )

    def test_matrix_eigendecomposition_with_eigh_and_eigenvectors_estimate(
        self,
    ) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        eigendecomposition_config = EighEigendecompositionConfig(tolerance=0.01)
        expected_eigenvalues, expected_eigenvectors = (
            torch.tensor([1.0, 4.0]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            matrix_eigendecomposition(
                A=A,
                eigendecomposition_config=eigendecomposition_config,
                eigenvectors_estimate=torch.eye(2),
            ),
        )

    @parametrize(
        "initialization_fn",
        (
            lambda A: torch.eye(A.shape[0], dtype=A.dtype, device=A.device),
            lambda A: matrix_eigendecomposition(A)[1],
        ),
    )
    @parametrize(
        "A, expected_eigenvalues, expected_eigenvectors",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([1.0, 4.0]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([2.9008677229e-03, 1.7424316704e-01, 1.9828229980e03]),
                torch.tensor(
                    [
                        [0.0460073575, -0.6286827326, 0.7762997746],
                        [-0.1751257628, -0.7701635957, -0.6133345366],
                        [0.9834705591, -0.1077321917, -0.1455317289],
                    ]
                ),
            ),
        ),
    )
    def test_matrix_eigendecomposition_with_qr(
        self,
        A: Tensor,
        expected_eigenvalues: Tensor,
        expected_eigenvectors: Tensor,
        initialization_fn: Callable[[Tensor], Tensor],
    ) -> None:
        atol = 2e-3
        rtol = 1e-5

        qr_config = QREigendecompositionConfig(max_iterations=10_000)
        eigenvalues_estimate, eigenvectors_estimate = matrix_eigendecomposition(
            A=A,
            eigenvectors_estimate=initialization_fn(A),
            eigendecomposition_config=qr_config,
        )

        # Ensure that the signs of the eigenvectors are consistent.
        eigenvectors_estimate[
            :,
            expected_eigenvectors[0, :] / eigenvectors_estimate[0, :] < 0,
        ] *= -1
        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            (eigenvalues_estimate, eigenvectors_estimate),
            atol=atol,
            rtol=rtol,
        )

    def test_invalid_eigendecomposition_config(self) -> None:
        @dataclass
        class NotSupportedEigendecompositionConfig(EigendecompositionConfig):
            """A dummy class eigendecomposition config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                f"Eigendecomposition config is not implemented! Specified eigendecomposition config is {NotSupportedEigendecompositionConfig.__name__}."
            ),
            matrix_eigendecomposition,
            A=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            eigendecomposition_config=NotSupportedEigendecompositionConfig(),
        )

    def test_non_zero_tolerance_eigh_without_eigenvectors_estimate(self) -> None:
        eigendecomposition_config = EighEigendecompositionConfig(tolerance=0.01)
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "eigenvectors_estimate should be passed to matrix_eigendecomposition when using tolerance != 0.0."
            ),
            matrix_eigendecomposition,
            A=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            eigendecomposition_config=eigendecomposition_config,
        )


@instantiate_parametrized_tests
class MatrixOrthogonalizationTest(unittest.TestCase):
    @parametrize("scale_by_nuclear_norm", (True, False))
    @parametrize(
        "scale_by_dims_fn",
        (lambda d_in, d_out: 1.0, lambda d_in, d_out: d_out * d_in),
    )
    def test_orthogonalization_with_svd(
        self, scale_by_nuclear_norm: bool, scale_by_dims_fn: Callable[[int, int], float]
    ) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        expected_singular_values = torch.tensor([4.0, 1.0])
        expected_orthogonalized_matrix = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            .mul_(expected_singular_values.sum() if scale_by_nuclear_norm else 1.0)
            .mul_(scale_by_dims_fn(A.shape[1], A.shape[0]))
        )

        torch.testing.assert_close(
            expected_orthogonalized_matrix,
            matrix_orthogonalization(
                A=A,
                orthogonalization_config=SVDOrthogonalizationConfig(
                    scale_by_nuclear_norm=scale_by_nuclear_norm,
                    scale_by_dims_fn=scale_by_dims_fn,
                ),
            ),
        )

    @parametrize(
        "scale_by_dims_fn",
        # Choose different function compared to test_orthogonalization_with_svd here to allow for the same atol.
        (lambda d_in, d_out: 1.0, lambda d_in, d_out: 1 / (d_out * d_in)),
    )
    def test_orthogonalization_with_newton_schulz(
        self, scale_by_dims_fn: Callable[[int, int], float]
    ) -> None:
        atol = 0.3
        rtol = 1e-6

        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        expected_orthogonalized_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).mul_(
            scale_by_dims_fn(A.shape[1], A.shape[0])
        )

        torch.testing.assert_close(
            expected_orthogonalized_matrix,
            matrix_orthogonalization(
                A=A,
                orthogonalization_config=NewtonSchulzOrthogonalizationConfig(
                    scale_by_dims_fn=scale_by_dims_fn,
                ),
            ),
            atol=atol,
            rtol=rtol,
        )

    @parametrize(
        "matrix",
        (torch.randn(3, 2), torch.randn(2, 3)),
    )
    @parametrize(
        "orthogonalization_config",
        (
            SVDOrthogonalizationConfig(),
            NewtonSchulzOrthogonalizationConfig(),
        ),
    )
    def test_orthogonalization_non_square_matrix(
        self, matrix: Tensor, orthogonalization_config: OrthogonalizationConfig
    ) -> None:
        orthogonalized_matrix = matrix_orthogonalization(
            matrix, orthogonalization_config=orthogonalization_config
        )
        self.assertEqual(orthogonalized_matrix.shape, matrix.shape)

    def test_invalid_orthogonalization_config(self) -> None:
        @dataclass
        class NotSupportedOrthogonalizationConfig(OrthogonalizationConfig):
            """A dummy class orthogonalization config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                f"Orthogonalization config is not implemented! Specified orthogonalization config is {NotSupportedOrthogonalizationConfig.__name__}."
            ),
            matrix_orthogonalization,
            A=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            orthogonalization_config=NotSupportedOrthogonalizationConfig(),
        )

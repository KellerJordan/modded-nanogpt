"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import inspect
import logging
import math
import time
from collections.abc import Callable
from dataclasses import fields
from fractions import Fraction
from functools import partial, wraps
from math import isfinite
from typing import TypeVar

import torch
from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    DefaultNewtonSchulzOrthogonalizationConfig,
    DefaultPerturbationConfig,
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

logger: logging.Logger = logging.getLogger(__name__)


@enum.unique
class NewtonConvergenceFlag(enum.Enum):
    """
    Enum class for the state of the Newton / higher-order iteration method.

    REACHED_MAX_ITERS: Reached maximum iteration count without meeting other exit criteria (rare, unexpected).
    CONVERGED: Met the tolerance criterion (expected).
    EARLY_STOP: Error in residual stopped improving (unexpected).
    """

    REACHED_MAX_ITERS = enum.auto()
    CONVERGED = enum.auto()
    EARLY_STOP = enum.auto()


_FuncReturnType = TypeVar("_FuncReturnType")


def _assign_function_args_from_config(
    func: Callable[..., _FuncReturnType], config: object
) -> Callable[..., _FuncReturnType]:
    """
    Creates a partial function with arguments from config that match func's parameter names.

    This function examines the fields in the config object and creates a partial function
    that pre-fills parameters of func with matching values from config. Only fields that
    are present in both the config object and the function's parameter list are used.

    Args:
        func (Callable[..., _FuncReturnType]): The function to partially apply arguments to.
        config (object): A dataclass object containing configuration values.

    Returns:
        decorated_func (Callable[..., _FuncReturnType]): A partial function with pre-filled arguments from config.
    """
    return partial(
        func,
        **{
            field.name: getattr(config, field.name)
            for field in fields(config)  # type: ignore[arg-type]
            if field.name in inspect.getfullargspec(func).args
        },
    )


def _check_2d_tensor(
    func: Callable[..., _FuncReturnType],
) -> Callable[..., _FuncReturnType]:
    """
    Decorator to check if the input tensor is 2-dimensional.

    This decorator checks if the input tensor `A` is 2-dimensional.
    If not, it raises a ValueError. If the tensor is valid, it calls the decorated function.

    Args:
        func (Callable[..., FuncReturnType]): The function to be decorated.

    Returns:
        wrapped_func (Callable[..., FuncReturnType]): The wrapped function that includes the 2D tensor check.

    """

    @wraps(func)
    def wrapper(A: Tensor, *args: object, **kwargs: object) -> _FuncReturnType:
        if len(A.shape) != 2:
            raise ValueError(f"Matrix is not 2-dimensional! {A.shape=}")
        return func(A, *args, **kwargs)

    return wrapper


def _check_square_matrix(
    func: Callable[..., _FuncReturnType],
) -> Callable[..., _FuncReturnType]:
    """
    Decorator to check if the input matrix is square.

    This decorator first checks if input A is 2-dimensional, then checks if A is square.
    If not, it raises a ValueError. If the matrix is valid, it calls the decorated function.

    Args:
        func (Callable[..., FuncReturnType]): The function to be decorated.

    Returns:
        wrapped_func (Callable[..., FuncReturnType]): The wrapped function that includes the square matrix check.

    """

    @_check_2d_tensor
    @wraps(func)
    def wrapper(A: Tensor, *args: object, **kwargs: object) -> _FuncReturnType:
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix is not square! {A.shape=}")
        return func(A, *args, **kwargs)

    return wrapper


def _matrix_perturbation(
    A: Tensor,
    epsilon: float = 0.0,
    is_eigenvalues: bool = True,
) -> Tensor:
    """Add epsilon * I to matrix (if square) or epsilon (if vector).

    Args:
        A (Tensor): Matrix of interest.
        epsilon (float): Value to add to matrix for perturbation/regularization. (Default: 0.0)
        is_eigenvalues (bool): Whether A is a matrix of eigenvalues (true) or a full matrix (false). In the former case (true), add epsilon to all values; in the latter (false), add epsilon along the diagonal. (Default: True)

    Returns:
        A_ridge (Tensor): Matrix with perturbation/regularization.

    """
    return (
        (
            A.add(torch.eye(A.shape[0], dtype=A.dtype, device=A.device), alpha=epsilon)
            if not is_eigenvalues
            else A + epsilon
        )
        if epsilon != 0
        else A  # Fast path when epsilon is 0.0, return A without modification
    )


def matrix_inverse_root_from_eigendecomposition(
    L: Tensor,
    Q: Tensor,
    root: Fraction,
    epsilon: float = 0.0,
    rank_deficient_stability_config: RankDeficientStabilityConfig = DefaultPerturbationConfig,
) -> Tensor:
    """Compute A^(-1/root) from eigendecomposition A = Q diag(L) Q^T.

    This function computes A^(-1/root) given the eigendecomposition of A = Q diag(L) Q^T.
    It handles rank-deficient matrices through either pseudo-inverse or perturbation approaches.

    Args:
        L (Tensor): Eigenvalues of the matrix.
        Q (Tensor): Eigenvectors of the matrix (orthogonal matrix).
        root (Fraction): Root of interest (e.g., 2 for inverse square root).
        epsilon (float): Regularization parameter for numerical stability. (Default: 0.0)
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for
            handling rank-deficient matrices. (Default: DefaultPerturbationConfig)

    Returns:
        X (Tensor): Inverse root of matrix, computed as Q * diag(L^(-1/root)) * Q^T.

    Raises:
        ValueError: If epsilon is not 0.0 when using pseudo-inverse.
        NotImplementedError: If rank_deficient_stability_config is not a supported config type.
    """

    def compute_eigenvalue_threshold(
        L: Tensor,
        rank_rtol: float | None = None,
        rank_atol: float = 0.0,
    ) -> float:
        """Compute threshold for filtering eigenvalues based on numerical rank.

        Determines which eigenvalues should be considered numerically zero based on
        relative and absolute tolerances. Follows the approach used in torch.linalg.matrix_rank.

        Args:
            L (Tensor): Eigenvalues of matrix.
            rank_rtol (float | None): Relative tolerance for determining numerical rank.
                If None, uses machine epsilon scaled by tensor size. (Default: None)
            rank_atol (float): Absolute tolerance for determining numerical rank. (Default: 0.0)

        Returns:
            threshold (float): Threshold value below which eigenvalues are treated as zero.
        """
        if rank_rtol is None:
            rtol = L.numel() * torch.finfo(L.dtype).eps
        else:
            rtol = rank_rtol
        return max(rank_atol, rtol * L.max().relu().item())

    match rank_deficient_stability_config:
        case PseudoInverseConfig():
            if epsilon != 0.0:
                raise ValueError(f"{epsilon=} should be 0.0 when using pseudo-inverse!")

            spectrum_cutoff = compute_eigenvalue_threshold(
                L=L,
                rank_rtol=rank_deficient_stability_config.rank_rtol,
                rank_atol=rank_deficient_stability_config.rank_atol,
            )
            inv_power_L = torch.where(
                L <= spectrum_cutoff,
                torch.zeros_like(L),
                L.pow(-1.0 / root),
            )
        case PerturbationConfig():
            lambda_min = torch.min(L).item()

            # make eigenvalues > 0 (if necessary)
            # Note that our input matrix may not be PSD, even though it mathematically should be in Shampoo.
            # So, exercise great care in the below logic!
            if rank_deficient_stability_config.perturb_before_computation:
                # The happy path/mathematically ideal case: lambda_min >= epsilon, do nothing!
                # The unhappy path: lambda_min could be anything (even large negative value)
                # In that case, do a 2 step perturbation: first by -lambda_min, and then by epsilon
                if lambda_min < epsilon:
                    L = _matrix_perturbation(
                        L, epsilon=-lambda_min, is_eigenvalues=True
                    )
                    L = _matrix_perturbation(L, epsilon=epsilon, is_eigenvalues=True)
            else:
                # In that case, do a 2 step perturbation: first by -min(lambda_min, 0.0), and then by epsilon;
                # this approach is more stable when dealing with matrices that have large magnitude eigenvalues because it ensures epsilon doesn't get "absorbed" by large values.
                L = _matrix_perturbation(
                    L, epsilon=-min(lambda_min, 0.0), is_eigenvalues=True
                )
                L = _matrix_perturbation(L, epsilon=epsilon, is_eigenvalues=True)

            inv_power_L = L.pow(-1.0 / root)
        case _:
            raise NotImplementedError(
                f"{rank_deficient_stability_config=} is not supported."
            )

    return Q * inv_power_L.unsqueeze(0) @ Q.T


@_check_square_matrix
def matrix_inverse_root(
    A: Tensor,
    root: Fraction,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
    epsilon: float = 0.0,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (Fraction): Root of interest. Any rational number.
        root_inv_config (RootInvConfig): Configuration for root inverse computation. (Default: DefaultEigenConfig)
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)

    Returns:
        X (Tensor): Inverse root of matrix A.

    Raises:
        ValueError: If the matrix is not 2-dimensional or not square, or if the root denominator is not 1 for CoupledNewtonConfig.
        NotImplementedError: If the root inverse config is not implemented.

    """

    def matrix_inverse_root_eigen(
        A: Tensor,
        root: Fraction,
        epsilon: float = 0.0,
        rank_deficient_stability_config: RankDeficientStabilityConfig = DefaultPerturbationConfig,
        retry_double_precision: bool = True,
        eigendecomposition_offload_device: str = "",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute matrix inverse root using eigendecomposition of symmetric positive (semi-)definite matrix.

                A^{-1/r} = Q L^{-1/r} Q^T

        Assumes matrix A is symmetric.

        Args:
            A (Tensor): Square matrix of interest.
            root (Fraction): Root of interest. Any rational number.
            epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
            rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
            retry_double_precision (bool): Flag for retrying eigendecomposition with higher precision if lower precision fails due
                to CuSOLVER failure. (Default: True)
            eigendecomposition_offload_device (str): Device to offload eigendecomposition computation. If value is empty string, do not perform offloading. (Default: "")

        Returns:
            X (Tensor): (Inverse) root of matrix. Same dimensions as A.
            L (Tensor): Eigenvalues of A.
            Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

        Raises:
            ValueError: If the root is not a positive integer.
            ValueError: If epsilon is 0.0 when using pseudo-inverse.

        """

        # check if root is positive integer
        if root <= 0:
            raise ValueError(f"Root {root} should be positive!")

        # compute eigendecomposition and compute minimum eigenvalue
        L, Q = matrix_eigendecomposition(
            A=A,
            epsilon=epsilon,
            eigendecomposition_config=EighEigendecompositionConfig(
                rank_deficient_stability_config=rank_deficient_stability_config,
                retry_double_precision=retry_double_precision,
                eigendecomposition_offload_device=eigendecomposition_offload_device,
                tolerance=0.0,
            ),
        )

        return (
            matrix_inverse_root_from_eigendecomposition(
                L=L,
                Q=Q,
                root=root,
                epsilon=epsilon,
                rank_deficient_stability_config=rank_deficient_stability_config,
            ),
            L,
            Q,
        )

    def matrix_inverse_root_newton(
        A: Tensor,
        root: int,
        epsilon: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
        """Compute matrix inverse root using coupled inverse Newton iteration.

            alpha <- -1 / p
            X <- 1/c * I
            M <- 1/c^p * A
            repeat until convergence
                M' <- (1 - alpha) * I + alpha * M
                X <- X * M'
                M <- M'^p * M

        where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
        We will instead use z = (p + 1) / (2 * |A|_F).

        NOTE: Exponent multiplier not compatible with coupled inverse Newton iteration!

        Args:
            A (Tensor): Matrix of interest.
            root (int): Root of interest. Any natural number.
            epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
            max_iterations (int): Maximum number of iterations. (Default: 100)
            tolerance (float): Tolerance. (Default: 1e-6)

        Returns:
            A_root (Tensor): Inverse square root of matrix.
            M (Tensor): Coupled matrix.
            termination_flag (NewtonConvergenceFlag): Specifies convergence.
            iteration (int): Number of iterations.
            error (Tensor): Final error between M and I.

        """

        # initialize iteration, dimension, and alpha
        iteration = 0
        dim = A.shape[0]
        alpha = -1 / root
        identity = torch.eye(dim, dtype=A.dtype, device=A.device)

        # add regularization
        A_ridge = _matrix_perturbation(A, epsilon=epsilon, is_eigenvalues=False)

        # initialize matrices
        A_nrm = torch.linalg.norm(A_ridge)
        z = (root + 1) / (2 * A_nrm)
        X = z ** (-alpha) * identity
        M = z * A_ridge
        error = torch.dist(M, identity, p=torch.inf)

        # main for loop
        while error > tolerance and iteration < max_iterations:
            iteration += 1
            M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
            X = X @ M_p
            M = torch.linalg.matrix_power(M_p, root) @ M
            error = torch.dist(M, identity, p=torch.inf)

        # determine convergence flag
        termination_flag = (
            NewtonConvergenceFlag.CONVERGED
            if error <= tolerance
            else NewtonConvergenceFlag.REACHED_MAX_ITERS
        )

        return X, M, termination_flag, iteration, error

    def matrix_inverse_root_higher_order(
        A: Tensor,
        root: Fraction,
        rel_epsilon: float = 0.0,
        abs_epsilon: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-20,
        order: int = 3,  # 2 represents Newton's method
        disable_tf32: bool = True,
    ) -> tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
        """Compute matrix inverse root using coupled iterations, similar to above but generalized to support higher order.

            Rough sketch (at order = 2, i.e., Newton)

            alpha <- -1 / p
            X <- 1/c * I
            M <- 1/c^p * A
            repeat until convergence
                M' <- (1 - alpha) * I + alpha * M
                X <- X * M'
                M <- M'^p * M

        where c = (k |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
        We will instead use z = (p + 1) / (k * |A|_F).
        Here, k > 1, and typically lies in [1, 2]. It is picked internally in this method.

        NOTE: Exponent multiplier not compatible with coupled iterations!

        Args:
            A (Tensor): Matrix of interest.
            root (Fraction): Root of interest. Any rational number. Use small numerator, denominator for best numerics as well as performance.
            rel_epsilon (float): Adds epsilon * lambda_max * I to matrix before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue. (Default: 0.0)
            abs_epsilon (float): Adds epsilon * I to matrix before taking matrix root. When both "abs_epsilon" and "rel_epsilon" are specified, max(rel_epsilon * lambda_max, abs_epsilon) * I is added to the matrix.
                Generally recommend setting according to A.dtype (1e-3 for tf32, 1e-5 for fp32, 1e-9 for fp64) (Default: 0.0)
            max_iterations (int): Maximum number of iterations. Typically we need < 20 iterations. (Default: 100)
            tolerance (float): Tolerance for determining exit criterion from iterations. (Default: 1e-20, which in practice guarantees they run to convergence)
            order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations), but can take more matmuls per iteration. (Default: 3)
            disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True, since tf32 is challenging numerically here. (Default: True)

        Returns:
            A_root (Tensor): Inverse root of matrix (A^{-1/root}).
            M (Tensor): Coupled matrix.
            termination_flag (NewtonConvergenceFlag): Specifies convergence.
            iteration (int): Number of iterations.
            error (Tensor): Final error, measured as |A * A_root^(p/q) - I|_Inf, where root = -q/p.

        Raises:
            ArithmeticError: If the computed result is inaccurate, i.e., error > 1e-1 or if there is an internal error.
            ArithmeticError: If the input matrix has entries close to infinity.
            ArithmeticError: If NaN/Inf is found in the matrix inverse root after powering for fractions.

        """

        # TODO(irisz): This save/modify/restore pattern is not thread-safe.
        # Concurrent calls (e.g., from multi-threaded optimizer step) can race
        # on this global flag. Revisit this for D97459682.
        tf32_flag = torch.backends.cuda.matmul.allow_tf32
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
        logger.debug(
            f"Using tf32 precision for fp32 matmul: {torch.backends.cuda.matmul.allow_tf32}"
        )

        try:
            t_iter_begin = time.perf_counter()
            p = root.numerator
            q = root.denominator
            dtype = A.dtype

            if min(abs(p), abs(q)) >= 10:
                logger.warning(
                    f"{abs(root.numerator)=} and {abs(root.denominator)=} are probably too big for best performance."
                )

            # develop the b coefficients array first (ref: Lakic's paper)
            b = torch.zeros(order, dtype=A.dtype, device=A.device)
            b[0] = 1
            num = 1
            denom = 1
            for i in range(1, order):
                num *= 1 + (i - 1) * p
                denom *= i * p
                b[i] = num / denom

            # initialize iteration, dimension, and s
            iteration = 0
            n = A.shape[0]
            s = -1 / p

            # We add a diagonal term to condition the matrix better
            # We follow the Google style conditioning (in spirit) and scale by an upper bound on the max eigenvalue
            # NOTE: this is different from other parts of Shampoo for now
            # Simply use the basic upper bound on the spectral radius of A via infinity norm (should not underflow)
            # NOTE: One may wish to use a cheap (|A^4|_inf)**0.25 to get a tighter upper bound, but beware of fp32 underflow!
            lambda_max_approx = torch.linalg.matrix_norm(A, torch.inf)

            # We have not seen lambda_max being Inf in practice, however there is not a whole lot we can do in this pathological case and its good to bail early
            if not isfinite(lambda_max_approx):
                raise ArithmeticError(
                    "Input matrix has entries close to inf, exiting root inverse"
                )

            # Now scale and setup our variables
            epsilon = max(rel_epsilon * lambda_max_approx, abs_epsilon)
            identity = torch.eye(n, dtype=dtype, device=A.device)
            A_ridge = _matrix_perturbation(A, epsilon=epsilon, is_eigenvalues=False)
            lambda_max_approx += epsilon

            # Figure out a constant that gives good starting location
            # We stick to a conservative setting that gives very good accuracy
            # For a ref, see https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/matrix_functions.py#L114
            z = 1.0 / torch.trace(A_ridge).item()
            X = (z ** (-s)) * identity
            M = z * A_ridge
            error = torch.linalg.vector_norm(M - identity, torch.inf)
            t_iter_end = time.perf_counter()
            logger.debug(
                f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
            )

            # Do one iteration of basic Newton first. This is used to mathematically guarantee convergence of higher order method.
            # TODO: we may be able to get rid of this with a more careful analysis of the convergence region
            t_iter_begin = time.perf_counter()
            M_p = M.mul(s).add_(identity, alpha=(1 - s))
            X = X @ M_p
            M = torch.linalg.matrix_power(M_p, p) @ M
            error = torch.linalg.vector_norm(M - identity, torch.inf)
            n_matmul = math.ceil(math.log2(p)) + 2
            iteration += 1
            t_iter_end = time.perf_counter()
            logger.debug(
                f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
            )

            # main while loop
            while error > tolerance and iteration < max_iterations:
                t_iter_begin = time.perf_counter()
                iteration += 1

                # create M_p via Horner's rule
                base_matrix = identity - M
                M_p = base_matrix.mul(b[order - 1]).add_(
                    identity, alpha=float(b[order - 2])
                )
                for i in reversed(range(order - 2)):
                    M_p = torch.addmm(identity, M_p, base_matrix, beta=float(b[i]))

                # rest is same as Newton
                X = X @ M_p
                M = torch.linalg.matrix_power(M_p, p) @ M
                new_error = torch.linalg.vector_norm(M - identity, torch.inf)
                n_matmul += math.ceil(math.log2(p)) + order

                # TODO: 1.2 is the value from the Google code, can be tuned
                if new_error > error * 1.2 or (new_error == error and error < 1e-3):
                    logger.debug(
                        f"Coupled inverse Newton is stagnating or diverging based on comparing current error {new_error.item()} against last iteration's error {error.item()}."
                        f"(We assume divergence if the new error > 1.2 * previous error, and assume stagnation if they are equal.)"
                    )
                    termination_flag = NewtonConvergenceFlag.EARLY_STOP
                    break
                error = new_error

                t_iter_end = time.perf_counter()
                logger.debug(
                    f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
                )
            else:
                # determine convergence flag based on error and tolerance because the main while loop exited with False condition.
                termination_flag = (
                    NewtonConvergenceFlag.REACHED_MAX_ITERS
                    if error > tolerance
                    else NewtonConvergenceFlag.CONVERGED
                )

            # compute a cheap error proxy
            true_error = torch.linalg.vector_norm(
                A_ridge @ torch.linalg.matrix_power(X, p) - identity, torch.inf
            )
            n_matmul += math.ceil(math.log2(p)) + 1

            # If the error is too high, let us log and raise an exception for investigation. This should be relatively infrequent (if epsilon isn't too small)
            if true_error > 1e-1:
                raise ArithmeticError(
                    f"Error in matrix inverse root (before powering for fractions) {true_error} exceeds threshold 1e-1, raising an exception!"
                )

            # Now power the root to q
            if q > 1:
                X = torch.linalg.matrix_power(X, q)
                n_matmul += math.ceil(math.log2(q))

            logger.debug(f"Upper bound on maximum eigenvalue: {lambda_max_approx}")
            logger.debug(f"Number of matmuls: {n_matmul}")
            logger.debug(f"Number of iterations: {iteration}")
            logger.debug(f"Error before powering: {true_error}")
            logger.debug(f"Termination Flag: {termination_flag}")

            # If we have inf/nan in our answer also raise an arithmetic exception.
            # Usually, this is due to the powering to q > 1 which can blow up entries.
            # We have not seen this yet for q = 1 in Shampoo.
            if not torch.isfinite(X).all():
                raise ArithmeticError(
                    "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
                )

        finally:
            # Always restore tf32 mode unconditionally, so we skip the
            # disable_tf32 check. When disable_tf32=False, this is a no-op
            # since tf32_flag already equals the current value. When
            # disable_tf32=True, this restores the original value.
            torch.backends.cuda.matmul.allow_tf32 = tf32_flag

        return X, M, termination_flag, iteration, true_error

    match root_inv_config:
        case EigenConfig():
            X, _, _ = _assign_function_args_from_config(
                func=matrix_inverse_root_eigen, config=root_inv_config
            )(A=A, root=root, epsilon=epsilon)
        case CoupledNewtonConfig():
            # NOTE: Use Fraction.is_integer() instead when downstream applications are Python 3.12+ available
            if root.denominator != 1:
                raise ValueError(
                    f"{root.denominator=} must be equal to 1 to use coupled inverse Newton iteration!"
                )

            X, _, termination_flag, _, _ = _assign_function_args_from_config(
                func=matrix_inverse_root_newton, config=root_inv_config
            )(A=A, root=root.numerator, epsilon=epsilon)
            if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
                logger.warning(
                    "Newton did not converge and reached maximum number of iterations!"
                )
        case CoupledHigherOrderConfig():
            X, _, termination_flag, _, _ = _assign_function_args_from_config(
                func=matrix_inverse_root_higher_order, config=root_inv_config
            )(A=A, root=root)
            if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
                logger.warning(
                    "Higher order method did not converge and reached maximum number of iterations!"
                )
        case _:
            raise NotImplementedError(
                f"Root inverse config is not implemented! Specified root inverse config is {root_inv_config=}."
            )

    return X


@_check_square_matrix
def matrix_eigendecomposition(
    A: Tensor,
    epsilon: float = 0.0,
    eigendecomposition_config: EigendecompositionConfig = DefaultEigendecompositionConfig,
    eigenvectors_estimate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the eigendecomposition of a symmetric matrix.

    Args:
        A (Tensor): The input symmetric matrix.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root for numerical stability. (Default: 0.0)
        eigendecomposition_config (EigendecompositionConfig): Determines how eigendecomposition is computed. (Default: DefaultEigendecompositionConfig)
        eigenvectors_estimate (Tensor | None): Current estimate of eigenvectors. (Default: None)

    Returns:
        eigenvalues (Tensor): The eigenvalues of the input matrix.
        eigenvectors (Tensor): The eigenvectors of the input matrix.

    Raises:
        ValueError: If the matrix is not 2-dimensional or not square.
        ValueError: If epsilon is 0.0 when using pseudo-inverse.
        ValueError: If eigenvectors_estimate is None and eigendecomposition_config.tolerance != 0.0.
        NotImplementedError: If the eigendecomposition config is not implemented.

    """

    def eigh_eigenvalue_decomposition(
        A: Tensor,
        retry_double_precision: bool = True,
        eigendecomposition_offload_device: str = "",
    ) -> tuple[Tensor, Tensor]:
        """Compute the eigendecomposition of a symmetric matrix using torch.linalg.eigh.

        Args:
            A (Tensor): The input symmetric matrix.
            retry_double_precision (bool): Whether to retry the computation in double precision if it fails in the current precision. (Default: True)
            eigendecomposition_offload_device (str): Device to offload eigendecomposition computation. If value is empty string, do not perform offloading. (Default: "")

        Returns:
            eigenvalues (Tensor): The eigenvalues of the input matrix A.
            eigenvectors (Tensor): The eigenvectors of the input matrix A.

        Raises:
            Exception: If the eigendecomposition fails and retry_double_precision is False or fails in double precision.

        """
        # Create a function that will convert tensors back to the original device and dtype of A
        # This is used at the end to ensure the returned tensors match the input specifications
        restore_original_format = partial(Tensor.to, device=A.device, dtype=A.dtype)

        if eigendecomposition_offload_device != "":
            A = A.to(device=eigendecomposition_offload_device)

        try:
            # Attempt to compute the eigendecomposition in the current precision
            L, Q = torch.linalg.eigh(A)

        except Exception as exception:
            # If the computation fails and retry_double_precision is True, retry in double precision
            # Higher precision can help with numerical stability issues
            if retry_double_precision and A.dtype != torch.float64:
                logger.warning(
                    f"Failed to compute eigendecomposition in {A.dtype} precision with exception {exception}! Retrying in double precision..."
                )
                L, Q = torch.linalg.eigh(A.double())
            else:
                # If retry_double_precision is False or the computation fails in double precision, raise the exception
                raise exception

        # Convert the results back to the original device and dtype before returning
        # This ensures consistency with the input tensor's specifications
        return restore_original_format(L), restore_original_format(Q)

    def eigenvalues_estimate_criterion_below_or_equal_tolerance(
        eigenvalues_estimate: Tensor, tolerance: float
    ) -> bool:
        """Evaluates if a criterion using estimated eigenvalues is below or equal to the tolerance.

        Let Q^T A Q =: B be the estimate of the eigenvalues of the matrix A, where Q is the matrix containing the last computed eigenvectors.
        The criterion based on the estimated eigenvalues is defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
        The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

        This convergence criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
        We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
        Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
        Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

        Args:
            eigenvalues_estimate (Tensor): The estimated eigenvalues.
            tolerance (float): The tolerance for the criterion.

        Returns:
            is_below_tolerance (bool): True if the criterion is below or equal to the tolerance, False otherwise.

        """
        norm = torch.linalg.norm(eigenvalues_estimate)
        diagonal_norm = torch.linalg.norm(eigenvalues_estimate.diag())
        off_diagonal_norm = torch.sqrt(norm**2 - diagonal_norm**2)
        return bool(off_diagonal_norm <= tolerance * norm)

    def qr_algorithm(
        A: Tensor,
        eigenvectors_estimate: Tensor,
        max_iterations: int = 1,
        tolerance: float = 0.01,
    ) -> tuple[Tensor, Tensor]:
        """Approximately compute the eigendecomposition of a symmetric matrix by performing the QR algorithm.

        Given an initial estimate of the eigenvectors Q of matrix A, QR iterations are performed until the criterion based on the estimated eigenvalues is below or equal to the specified tolerance or until the maximum number of iterations is reached.

        Note that if the criterion based on the estimated eigenvalues is already below or equal to the tolerance given the initial eigenvectors_estimate, the QR iterations will be skipped.

        Args:
            A (Tensor): The symmetric input matrix.
            eigenvectors_estimate (Tensor): The current estimate of the eigenvectors of A.
            max_iterations (int): The maximum number of iterations to perform. (Default: 1)
            tolerance (float): The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the eigenvalue estimate.
                (Default: 0.01)

        Returns:
            eigenvalues_estimate (Tensor): The estimated eigenvalues of the input matrix A.
            eigenvectors_estimate (Tensor): The estimated eigenvectors of the input matrix A.

        Raises:
            AssertionError: If the data types of Q and A do not match.

        """
        # Perform orthogonal/simultaneous iterations (QR algorithm).
        Q = eigenvectors_estimate

        # This assertion provides a more clear error message than the internal error message in `torch.mm`, and assertion makes sure that user-side is unable to catch the error.
        assert Q.dtype == A.dtype, (
            f"Q and A must have the same dtype! {Q.dtype=} {A.dtype=}"
        )

        eigenvalues_estimate = Q.T @ A @ Q
        iteration = 0
        # NOTE: This will skip the QR iterations if the criterion is already below or equal to the tolerance given the initial eigenvectors_estimate.
        while (
            iteration < max_iterations
            and not eigenvalues_estimate_criterion_below_or_equal_tolerance(
                eigenvalues_estimate, tolerance
            )
        ):
            Q, R = torch.linalg.qr(eigenvalues_estimate)
            eigenvalues_estimate = R @ Q
            eigenvectors_estimate = eigenvectors_estimate @ Q
            iteration += 1

        # Ensure consistent ordering of estimated eigenvalues and eigenvectors.
        eigenvalues_estimate, indices = eigenvalues_estimate.diag().sort(stable=True)
        eigenvectors_estimate = eigenvectors_estimate[:, indices]

        return eigenvalues_estimate, eigenvectors_estimate

    # TODO: reduce redundant code when rank_deficient_stability_config is generalized to all methods
    # check epsilon is 0 when using pseudo-inverse
    if (
        isinstance(
            eigendecomposition_config.rank_deficient_stability_config,
            PseudoInverseConfig,
        )
        and epsilon != 0.0
    ):
        raise ValueError(f"{epsilon=} should be 0.0 when using pseudo-inverse!")

    # Add epsilon to the diagonal to help with numerical stability of the eigenvalue decomposition
    # Only do it when perturb_before_computation is True.
    A_ridge = _matrix_perturbation(
        A=A,
        # If perturb_before_computation is False, we take the fast path in _matrix_perturbation() by effectively setting epsilon to 0, avoiding the perturbation step.
        # If the perturb_before_computation field doesn't exist in the config, default to 0 (equivalent to False).
        epsilon=epsilon
        * getattr(
            eigendecomposition_config.rank_deficient_stability_config,
            "perturb_before_computation",
            0,
        ),
        is_eigenvalues=False,
    )

    match eigendecomposition_config:
        case EighEigendecompositionConfig():
            if eigendecomposition_config.tolerance != 0.0:
                if eigenvectors_estimate is None:
                    raise ValueError(
                        "eigenvectors_estimate should be passed to matrix_eigendecomposition when using tolerance != 0.0."
                    )
                eigenvalues_estimate = (
                    eigenvectors_estimate.T @ A_ridge @ eigenvectors_estimate
                )
                if eigenvalues_estimate_criterion_below_or_equal_tolerance(
                    eigenvalues_estimate=eigenvalues_estimate,
                    tolerance=eigendecomposition_config.tolerance,
                ):
                    return eigenvalues_estimate.diag(), eigenvectors_estimate
            return _assign_function_args_from_config(
                func=eigh_eigenvalue_decomposition, config=eigendecomposition_config
            )(A=A_ridge)
        case QREigendecompositionConfig():
            assert eigenvectors_estimate is not None, (
                "eigenvectors_estimate should not be None when QR algorithm is used."
            )
            return _assign_function_args_from_config(
                func=qr_algorithm, config=eigendecomposition_config
            )(A=A_ridge, eigenvectors_estimate=eigenvectors_estimate)
        case _:
            raise NotImplementedError(
                f"Eigendecomposition config is not implemented! Specified eigendecomposition config is {type(eigendecomposition_config).__name__}."
            )


@_check_2d_tensor
def matrix_orthogonalization(
    A: Tensor,
    orthogonalization_config: OrthogonalizationConfig = DefaultNewtonSchulzOrthogonalizationConfig,
) -> Tensor:
    """Compute the orthogonalization of a matrix.

    Args:
        A (Tensor): The input matrix.
        orthogonalization_config (OrthogonalizationConfig): Determines how orthogonalization is computed.
            (Default: DefaultNewtonSchulzOrthogonalizationConfig)

    Returns:
        orthogonalized_matrix (Tensor): The orthogonalized matrix.

    Raises:
        NotImplementedError: If the orthogonalization config is not implemented.

    """

    def svd_orthogonalization(
        A: Tensor,
        scale_by_nuclear_norm: bool = False,
    ) -> Tensor:
        """Compute the orthogonalization of a matrix using SVD.

        Args:
            A (Tensor): The input matrix.
            scale_by_nuclear_norm (bool): Whether to scale the orthogonalized matrix by the nuclear norm of A.
                (Default: False)

        Returns:
            A_orthogonal (Tensor): The approximated orthogonalized matrix.

        """
        # Orthogonalize A via reduced SVD.
        U, s, V_T = torch.linalg.svd(A, full_matrices=False)
        A_orthogonal = U @ V_T

        # Scale by nuclear norm if specified.
        if scale_by_nuclear_norm:
            A_orthogonal.mul_(s.sum())

        return A_orthogonal

    def newton_schulz(
        A: Tensor,
        num_iterations: int = 5,
        coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ) -> Tensor:
        """
        Perform quintic Newton-Schulz iteration to compute the semi-orthogonalization of a matrix A.

        This iteratively performs the iteration:
            X <- p(X) = a * X + b * X * X^T * X + c * (X * X^T)^2 * X.

        NOTE: In order to guarantee convergence, the coefficients must satisfy p(1) = 1.
            This is not true for the Muon coefficients, which only guarantee convergence to [0.7, 1.3].
            Another alternative is to use the coefficients (3., -16./5., 6./5.) as used in Modula.

        References:
            - https://arxiv.org/abs/2409.20325
            - https://kellerjordan.github.io/posts/muon/
            - https://docs.modula.systems/algorithms/newton-schulz/

        Args:
            A (Tensor): The input matrix to be semi-orthogonalized.
            num_iterations (int): Number of iterations for the Newton-Schulz iteration.
            coefficients (tuple[float, float, float]): Coefficients for the quintic Newton-Schulz iteration.
                (Default: (3.4445, -4.7750, 2.0315) based on suggestion in Muon.)

        Returns:
            X (Tensor): The semi-orthogonalized matrix.

        """
        # Normalize the matrix A in order to ensure spectral norm <= 1.
        X = A / max(torch.linalg.matrix_norm(A), 1e-8)

        a, b, c = coefficients

        # Use transpose optimization for wide matrices.
        # When A is wider than tall (more columns than rows), it's more efficient to transpose the matrix before performing the Newton-Schulz iterations.
        if transpose := A.shape[0] < A.shape[1]:
            X = X.T

        # Compute X <- p(X) = a * X + b * X * X^T * X + c * (X * X^T)^2 * X.
        for _ in range(num_iterations):
            # B = X^T * X (intermediate matrix for computing orthogonalization)
            B = X.T @ X
            # B = b*B + c*B^2 = b*(X^T * X) + c*(X^T * X)^2 (coefficient matrix)
            B = torch.addmm(B, B, B, beta=b, alpha=c)
            # X = a*X + X*B = a*X + X*(b*(X^T * X) + c*(X^T * X)^2) (Newton-Schulz iteration)
            X = torch.addmm(X, X, B, beta=a, alpha=1.0)

        if transpose:
            X = X.T

        return X

    # Compute scaling based on dimensions of A.
    d_in, d_out = A.shape[1], A.shape[0]
    scaling = orthogonalization_config.scale_by_dims_fn(d_in, d_out)

    match orthogonalization_config:
        case SVDOrthogonalizationConfig():
            return _assign_function_args_from_config(
                func=svd_orthogonalization, config=orthogonalization_config
            )(A=A).mul_(scaling)
        case NewtonSchulzOrthogonalizationConfig():
            return _assign_function_args_from_config(
                func=newton_schulz, config=orthogonalization_config
            )(A=A).mul_(scaling)
        case _:
            raise NotImplementedError(
                f"Orthogonalization config is not implemented! Specified orthogonalization config is {type(orthogonalization_config).__name__}."
            )

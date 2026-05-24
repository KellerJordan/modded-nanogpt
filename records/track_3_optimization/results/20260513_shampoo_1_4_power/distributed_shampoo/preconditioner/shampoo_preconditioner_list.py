"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from functools import partial, reduce
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Any, Generic, get_args, NoReturn, overload, TypeAlias, TypeVar

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner.matrix_functions import (
    matrix_eigendecomposition,
    matrix_inverse_root,
    matrix_inverse_root_from_eigendecomposition,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigendecompositionConfig,
    MatrixFunctionConfig,
    RootInvConfig,
)
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.shampoo_types import (
    BaseShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    PreconditionerValueError,
    RootInvShampooPreconditionerConfig,
)
from distributed_shampoo.utils.dict_zip_iterator import DictZipIterator
from distributed_shampoo.utils.optimizer_modules import OptimizerModule
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
from torch import Tensor


logger: logging.Logger = logging.getLogger(__name__)

SHAMPOO = "shampoo"
INVERSE_EXPONENT_OVERRIDE = "inverse_exponent_override"


_SubStateValueType = TypeVar("_SubStateValueType")
# NOTE: Use type _StateValueType instead when downstream applications are Python 3.12+ available
_StateValueType: TypeAlias = dict[Hashable, _SubStateValueType]


@dataclass
class BaseShampooKroneckerFactorsState(OptimizerModule):
    """Base class for Shampoo Kronecker factors (wrapped).

    Attributes:
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "BaseShampooKroneckerFactorsState":
        """
        Creates a BaseShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (BaseShampooKroneckerFactorsState): An instance of BaseShampooKroneckerFactorsState with initialized factor matrices and indices.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]

        return cls(
            factor_matrices=tuple(
                block_info.allocate_zeros_tensor(
                    size=(dim, dim),
                    dtype=factor_matrix_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            factor_matrix_indices=tuple(
                ".".join((*map(str, block_info.composable_block_ids), str(k)))
                for k in range(len(preconditioned_dims))
            ),
        )

    def __post_init__(self) -> None:
        super().__init__()  # Add this because the synthesized __init__() does not call super().__init__().
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)


@dataclass
class BaseShampooKroneckerFactorsUnwrapped:
    """Base class for Shampoo Kronecker factors (unwrapped).

    This class represents the unwrapped version of Kronecker factors used in Shampoo optimization.
    Unwrapped tensors are used during the actual computation phase of the optimizer, as opposed
    to the wrapped versions which are stored in the optimizer state.

    Attributes:
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
            computation of matrix operations, specifying algorithms and parameters for
            eigendecomposition or matrix inverse computation.
        epsilon (float): Small constant added to matrices to ensure numerical stability
            during matrix operations like inversion or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]
    roots: tuple[float, ...]
    amortized_computation_config: MatrixFunctionConfig
    epsilon: float
    num_tolerated_failed_amortized_computations: int
    use_trace_scaling: bool = False
    _failed_amortized_computation_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # NOTE: Due to EigenvalueCorrectedShampooKroneckerFactorsState's roots usage, which is one root only applied on corrected eigenvalues,
        # there is no check of roots with other fields.
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)

    def _get_field_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing shallow copies of this dataclass's fields.

        This method creates a dictionary where keys are field names and values are
        the corresponding field values from the dataclass. Since this is a shallow copy,
        any modifications to the returned dictionary's values will affect the original
        dataclass fields.

        Returns:
            dict[str, Any]: A dictionary mapping field names to their values
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in (
                "amortized_computation_config",
                "epsilon",
                "num_tolerated_failed_amortized_computations",
                "use_trace_scaling",
                "_failed_amortized_computation_counter",
            )
        }

    @abstractmethod
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Performs computationally expensive matrix operations for Shampoo preconditioners.

        This method handles the heavy computational work that is too expensive to perform
        at every optimization step. Instead, these operations are "amortized" - performed
        periodically (e.g., every N steps) to update the preconditioner matrices.

        Different Shampoo variants implement this method differently:
        - RootInvShampooPreconditionerList: Computes matrix inverse roots
        - EigendecomposedShampooPreconditionerList: Performs eigendecomposition
        - EigenvalueCorrectedShampooPreconditionerList: Computes eigenvectors

        The method includes error handling to gracefully recover from numerical issues
        that may occur during matrix operations.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied. This is the matrix on which the computationally expensive
                operations (like eigendecomposition or matrix inverse) will be performed.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing factor matrices
                and related data needed for the computation. The exact contents depend on the
                specific Shampoo implementation, but typically include factor matrices,
                their indices, and other relevant tensors.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary mapping from tensor names to computed tensors. The keys and values depend on the specific implementation.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.
        """

    @profile_decorator
    def amortized_computation(self, bias_correction2: float) -> None:
        """Performs amortized computation for Shampoo preconditioners.

        This method orchestrates the execution of the computationally expensive matrix operations
        that are amortized over multiple optimization steps. It applies bias correction to the
        factor matrices and calls the specialized _amortized_computation method for each factor.

        The method handles exceptions that may occur during computation, keeping track of failures
        and raising an exception if the number of failures exceeds the configured tolerance.

        Args:
            bias_correction2 (float): The bias correction factor to apply to the factor matrices
                before performing the amortized computation.

        Raises:
            PreconditionerValueError: If NaN or infinity values are encountered in the factor matrices.
            ValueError: If the number of failed amortized computations exceeds the configured tolerance.
        """
        last_seen_exception: Exception | None = None
        for kronecker_factors_iter_dict in DictZipIterator(data=self._get_field_dict()):
            bias_corrected_factor_matrix, factor_matrix_index = (
                # Incorporate bias correction.
                kronecker_factors_iter_dict["factor_matrices"] / bias_correction2,
                kronecker_factors_iter_dict["factor_matrix_indices"],
            )

            # Apply trace scaling if enabled.
            # This normalizes the factor matrix by 1/sqrt(trace) before computing
            # the inverse root, which can improve numerical stability by bringing
            # different factor matrices to a similar scale.
            # Credit to https://arxiv.org/pdf/2506.03595.
            #
            # Numerics can cause the trace to be negative. This is unlikely
            # for normal Shampoo but could become an issue for KL-Shampoo.
            # We clamp the trace to epsilon rather than silently replacing
            # non-positive values with 1.0 (which skipped scaling entirely).
            #
            # TODO(irisz): We do not error out on negative trace for now since it should
            # not be mathematically possible given that the factor matrices are
            # symmetric PSD. If we want to add conditioning / positive definiteness
            # checks in the future, they should be incorporated in a different
            # part of the code (not here in the trace scaling logic).
            #
            # TODO(irisz):
            # approach in prod_beta:
            #    non-positive trace → replaced with 1.0 → rsqrt(1.0) = 1.0 → scaling is effectively a no-op
            # approach in dev:
            #    non-positive trace → clamped to epsilon → rsqrt(epsilon) produces a very large
            #    scaling factor → could amplify the factor matrix significantly
            # From a theoretical standpoint, this shouldn't matter much since negative traces should never
            # occur for symmetric PSD matrices. However, we need to compare these two behaviors empircally.
            if self.use_trace_scaling:
                trace = torch.trace(bias_corrected_factor_matrix)
                safe_trace = torch.clamp(trace, min=self.epsilon)
                bias_corrected_factor_matrix = (
                    bias_corrected_factor_matrix * torch.rsqrt(safe_trace)
                )

            # Check for nan or inf values.
            if not torch.isfinite(bias_corrected_factor_matrix).all():
                raise PreconditionerValueError(
                    f"Encountered nan/inf values in factor matrix {factor_matrix_index}! "
                    "To mitigate, check if nan inputs are being passed into the network or nan gradients are being passed to the optimizer. "
                    "Otherwise, in some cases, this may be due to divergence of the algorithm. To mitigate, try decreasing the learning rate or increasing grafting epsilon. "
                    f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                    f"{torch.min(bias_corrected_factor_matrix)=}, {torch.max(bias_corrected_factor_matrix)=}, "
                    f"{bias_corrected_factor_matrix.isinf().any()=}, {bias_corrected_factor_matrix.isnan().any()=}."
                )

            computed_quantity_to_result, exception = self._amortized_computation(
                bias_corrected_factor_matrix=bias_corrected_factor_matrix,
                kronecker_factors_iter_dict=kronecker_factors_iter_dict,
            )
            if exception:
                last_seen_exception = exception

                BaseShampooPreconditionerList._save_and_handle_matrix_error(
                    factor_matrix_index=factor_matrix_index,
                    source_matrix=bias_corrected_factor_matrix,
                    error_handler=partial(
                        logger.warning,
                        f"Matrix computation failed for factor matrix {factor_matrix_index} with {exception=}. To investigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=} Using previous preconditioner and continuing...",
                    ),
                )

            # Check if we encounter NaN or inf values in computed quantities.
            for (
                computed_quantity_name,
                computed_result,
            ) in computed_quantity_to_result.items():
                if not torch.isfinite(computed_result).all():
                    # Define a closure to handle the error with proper variable capture
                    def raise_preconditioner_value_error(
                        factor_matrix_index: str = factor_matrix_index,
                        bias_corrected_factor_matrix: Tensor = bias_corrected_factor_matrix,
                        computed_quantity_name: str = computed_quantity_name,
                    ) -> NoReturn:
                        quantity_name = f"{computed_quantity_name=}".split("=")[
                            0
                        ].split("_")[-1]
                        raise PreconditionerValueError(
                            f"Encountered nan or inf values in {quantity_name} of factor matrix {factor_matrix_index}! "
                            f"To mitigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=}"
                        )

                    BaseShampooPreconditionerList._save_and_handle_matrix_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=bias_corrected_factor_matrix,
                        error_handler=raise_preconditioner_value_error,
                    )

                kronecker_factors_iter_dict[computed_quantity_name].copy_(
                    computed_result
                )

        if last_seen_exception is None:
            # Reset counter for failed amortized computations.
            self._failed_amortized_computation_counter = 0
        else:
            # Increment counter for failed amortized computations.
            self._failed_amortized_computation_counter += 1
            # Raise the exception if the tolerance is exceeded.
            if (
                self._failed_amortized_computation_counter
                > self.num_tolerated_failed_amortized_computations
            ):
                raise ValueError(
                    f"The number of failed amortized computations for factors {self.factor_matrix_indices} exceeded the allowed tolerance. The last seen exception was {last_seen_exception}."
                ) from last_seen_exception


@dataclass(kw_only=True)
class RootInvShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "RootInvShampooKroneckerFactorsState":
        """
        Creates a RootInvShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            preconditioner_config (RootInvShampooPreconditionerConfig): Configuration for the preconditioner.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (RootInvShampooKroneckerFactorsState): An instance of RootInvShampooKroneckerFactorsState with initialized inverse factor matrices.
        """
        block_info: BlockInfo = kwargs["block_info"]
        preconditioner_config: RootInvShampooPreconditionerConfig = kwargs[
            "preconditioner_config"
        ]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=preconditioner_config.factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize inv_factor_matrices as identity matrices.
            inv_factor_matrices=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=preconditioner_config.inv_factor_matrix_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass(kw_only=True)
class RootInvShampooKroneckerFactorsUnwrapped(BaseShampooKroneckerFactorsUnwrapped):
    """Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    This class implements the Root Inverse variant of Shampoo, which directly computes
    the inverse root of factor matrices for preconditioning.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse
            of the factor matrices. These are the preconditioners that are applied to gradients.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
            computation of matrix operations, specifying algorithms and parameters for
            matrix inverse computation.
        epsilon (float): Small constant added to matrices to ensure numerical stability
            during matrix operations like inversion.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: RootInvConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
        use_trace_scaling: bool = False,
    ) -> "RootInvShampooKroneckerFactorsUnwrapped":
        """
        Constructs a RootInvShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for matrix operations.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of RootInvShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to each factor matrix during preconditioner computation.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of matrix operations, specifying algorithms and parameters for
                matrix inverse computation.
            epsilon (float): Small constant added to matrices to ensure numerical stability
                during matrix operations like inversion.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.
            use_trace_scaling (bool): Flag for whether to normalize the factor matrix by its trace's
                sqrt before computing the inverse root. (Default: False)

        Returns:
            kronecker_factors_unwrapped (RootInvShampooKroneckerFactorsUnwrapped): An instance of
                RootInvShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
        """
        assert isinstance(kronecker_factors_state, RootInvShampooKroneckerFactorsState)
        return cls(
            inv_factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.inv_factor_matrices,
                )
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
            use_trace_scaling=use_trace_scaling,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Computes matrix inverse roots for Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the computation of matrix inverse roots for the RootInvShampoo variant. It applies
        the matrix_inverse_root function to each factor matrix with the appropriate root value.

        The computation is performed on the bias-corrected factor matrices and uses the
        configuration specified in amortized_computation_config. Error handling is included
        to gracefully recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                inv_factor_matrices and roots values for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed inverse factor matrices.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        inv_factor_matrix, root = (
            kronecker_factors_iter_dict["inv_factor_matrices"],
            kronecker_factors_iter_dict["roots"],
        )

        try:
            # Compute inverse preconditioners
            return {
                "inv_factor_matrices": matrix_inverse_root(
                    A=bias_corrected_factor_matrix,
                    root=Fraction(root),
                    root_inv_config=self.amortized_computation_config,
                    epsilon=self.epsilon,
                ).to(dtype=inv_factor_matrix.dtype)
            }, None
        except Exception as exception:
            return {"inv_factor_matrices": inv_factor_matrix}, exception

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.roots)
            == len(self.factor_matrices)
            == len(self.inv_factor_matrices)
        )


@dataclass(kw_only=True)
class EigendecomposedShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Eigendecomposed Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the eigenvalues of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "EigendecomposedShampooKroneckerFactorsState":
        """
        Creates an EigendecomposedShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            preconditioner_config (EigendecomposedShampooPreconditionerConfig): Configuration for the preconditioner.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (EigendecomposedShampooKroneckerFactorsState): An instance of EigendecomposedShampooKroneckerFactorsState.
        """
        block_info: BlockInfo = kwargs["block_info"]
        preconditioner_config: EigendecomposedShampooPreconditionerConfig = kwargs[
            "preconditioner_config"
        ]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=preconditioner_config.factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize factor_matrices_eigenvectors as identity matrices.
            factor_matrices_eigenvectors=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=preconditioner_config.factor_matrix_eigenvectors_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            # Initialize factor_matrices_eigenvalues all ones.
            factor_matrices_eigenvalues=tuple(
                block_info.allocate_ones_tensor(
                    size=(dim,),
                    dtype=preconditioner_config.factor_matrix_eigenvalues_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.factor_matrices)
            == len(self.factor_matrices_eigenvectors)
            == len(self.factor_matrices_eigenvalues)
        )


@dataclass(kw_only=True)
class EigendecomposedShampooKroneckerFactorsUnwrapped(
    BaseShampooKroneckerFactorsUnwrapped
):
    """Eigendecomposed Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    This class implements the Eigendecomposed variant of Shampoo, which computes and stores
    the eigendecomposition of factor matrices for more efficient and numerically stable
    preconditioning operations.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvectors of the factor matrices. These are used to transform gradients into
            the eigenspace and back.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvalues of the factor matrices. These are used to scale gradients in the eigenspace.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
        amortized_computation_config (EigendecompositionConfig): Configuration for the amortized
            computation of eigendecomposition, specifying algorithms and parameters.
        epsilon (float): Small constant added to eigenvalues to ensure numerical stability
            during matrix operations.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: EigendecompositionConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
        use_trace_scaling: bool = False,
    ) -> "EigendecomposedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigendecomposedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for eigendecomposition.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of EigendecomposedShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to each factor matrix during preconditioner computation.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of eigendecomposition, specifying algorithms and parameters.
            epsilon (float): Small constant added to eigenvalues to ensure numerical stability
                during matrix operations.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.
            use_trace_scaling (bool): Flag for whether to normalize the factor matrix by its trace's
                sqrt before computing the eigendecomposition. (Default: False)

        Returns:
            kronecker_factors_unwrapped (EigendecomposedShampooKroneckerFactorsUnwrapped): An instance of
                EigendecomposedShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
        """
        assert isinstance(
            kronecker_factors_state, EigendecomposedShampooKroneckerFactorsState
        )
        return cls(
            factor_matrices_eigenvectors=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvectors,
                )
            ),
            factor_matrices_eigenvalues=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvalues,
                )
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
            use_trace_scaling=use_trace_scaling,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Performs eigendecomposition for Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the eigendecomposition for the EigendecomposedShampoo variant. It computes both
        eigenvalues and eigenvectors for each factor matrix.

        The computation uses the configuration specified in amortized_computation_config,
        with special handling for QR-based eigendecomposition which requires the previous
        eigenvectors as an initial estimate. Error handling is included to gracefully
        recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                factor_matrices_eigenvalues and factor_matrices_eigenvectors for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed eigenvalues and eigenvectors.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        (
            factor_matrix_eigenvectors,
            factor_matrix_eigenvalues,
        ) = (
            kronecker_factors_iter_dict["factor_matrices_eigenvectors"],
            kronecker_factors_iter_dict["factor_matrices_eigenvalues"],
        )

        try:
            # Compute inverse preconditioner.
            computed_eigenvalues, computed_eigenvectors = matrix_eigendecomposition(
                A=bias_corrected_factor_matrix,
                eigendecomposition_config=self.amortized_computation_config,
                # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                eigenvectors_estimate=factor_matrix_eigenvectors.to(
                    dtype=bias_corrected_factor_matrix.dtype
                ),
                epsilon=self.epsilon,
            )

            return {
                "factor_matrices_eigenvalues": computed_eigenvalues.to(
                    dtype=factor_matrix_eigenvalues.dtype
                ),
                "factor_matrices_eigenvectors": computed_eigenvectors.to(
                    dtype=factor_matrix_eigenvectors.dtype
                ),
            }, None
        except Exception as exception:
            return {
                "factor_matrices_eigenvalues": factor_matrix_eigenvalues,
                "factor_matrices_eigenvectors": factor_matrix_eigenvectors,
            }, exception

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.roots)
            == len(self.factor_matrices)
            == len(self.factor_matrices_eigenvectors)
            == len(self.factor_matrices_eigenvalues)
        )


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Eigenvalue-corrected Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_block(
        cls, **kwargs: Any
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsState":
        """
        Creates an EigenvalueCorrectedShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            preconditioner_config (EigenvalueCorrectedShampooPreconditionerConfig): Configuration for the preconditioner.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.
            dims (tuple[int, ...]): Dimensions of the block.

        Returns:
            kronecker_factors_state (EigenvalueCorrectedShampooKroneckerFactorsState): An instance of EigenvalueCorrectedShampooKroneckerFactorsState.
        """
        block_info: BlockInfo = kwargs["block_info"]
        preconditioner_config: EigenvalueCorrectedShampooPreconditionerConfig = kwargs[
            "preconditioner_config"
        ]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        dims: tuple[int, ...] = kwargs["dims"]

        return EigenvalueCorrectedShampooKroneckerFactorsState(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=preconditioner_config.factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize factor_matrices_eigenvectors as identity matrices.
            factor_matrices_eigenvectors=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=preconditioner_config.factor_matrix_eigenvectors_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            corrected_eigenvalues=block_info.allocate_zeros_tensor(
                # Note that the corrected eigenvalues are not affected by the preconditioned_dims.
                size=tuple(dims),
                dtype=preconditioner_config.corrected_eigenvalues_dtype,
                device=block_info.param.device,
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooKroneckerFactorsUnwrapped(
    BaseShampooKroneckerFactorsUnwrapped
):
    """Eigenvalue-corrected Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    This class implements the Eigenvalue-Corrected variant of Shampoo, which computes eigenvectors
    of factor matrices but maintains a separate tensor of corrected eigenvalues that are updated
    directly from gradients. This approach can provide better conditioning and convergence properties
    in certain optimization scenarios.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvectors of the factor matrices. These are used to transform gradients into
            the eigenspace and back.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues that are
            updated directly from squared gradients in the eigenspace. This is a single tensor
            rather than a tuple of tensors per factor matrix.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to the corrected eigenvalues during preconditioner computation.
            Note that for eigenvalue-corrected Shampoo, this always contains only a single value
            since all eigenvalues are corrected using the same exponent.
        amortized_computation_config (EigendecompositionConfig): Configuration for the amortized
            computation of eigendecomposition, specifying algorithms and parameters.
        epsilon (float): Small constant added to corrected eigenvalues to ensure numerical stability
            during preconditioning operations.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: EigendecompositionConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
        use_trace_scaling: bool = False,
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigenvalueCorrectedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for eigendecomposition
        and eigenvalue correction.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of EigenvalueCorrectedShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to the corrected eigenvalues during preconditioner computation.
                For eigenvalue-corrected Shampoo, this always contains only a single value
                since all eigenvalues are corrected using the same exponent.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of eigendecomposition, specifying algorithms and parameters.
            epsilon (float): Small constant added to corrected eigenvalues to ensure numerical stability
                during preconditioning operations.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.
            use_trace_scaling (bool): Flag for whether to normalize the factor matrix by its trace's
                sqrt before computing the eigendecomposition. (Default: False)

        Returns:
            kronecker_factors_unwrapped (EigenvalueCorrectedShampooKroneckerFactorsUnwrapped): An instance of
                EigenvalueCorrectedShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
        """
        assert isinstance(
            kronecker_factors_state, EigenvalueCorrectedShampooKroneckerFactorsState
        )
        return cls(
            factor_matrices_eigenvectors=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvectors,
                )
            ),
            corrected_eigenvalues=unwrapped_tensor_getter(
                kronecker_factors_state.corrected_eigenvalues
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
            use_trace_scaling=use_trace_scaling,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Computes eigenvectors for eigenvalue-corrected Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the computation of eigenvectors for the EigenvalueCorrectedShampoo variant. Unlike
        the EigendecomposedShampoo variant, this only computes eigenvectors and not eigenvalues,
        as the eigenvalues are corrected separately during the optimization process.

        The computation uses the configuration specified in amortized_computation_config,
        with special handling for QR-based eigendecomposition which requires the previous
        eigenvectors as an initial estimate. Error handling is included to gracefully
        recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                factor_matrices_eigenvectors for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed eigenvectors.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        factor_matrix_eigenvectors = kronecker_factors_iter_dict[
            "factor_matrices_eigenvectors"
        ]

        try:
            # Compute eigenvectors of factor matrix.
            return {
                "factor_matrices_eigenvectors": matrix_eigendecomposition(
                    A=bias_corrected_factor_matrix,
                    eigendecomposition_config=self.amortized_computation_config,
                    # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                    eigenvectors_estimate=factor_matrix_eigenvectors.to(
                        dtype=bias_corrected_factor_matrix.dtype
                    ),
                    epsilon=self.epsilon,
                )[1].to(dtype=factor_matrix_eigenvectors.dtype)
            }, None
        except Exception as exception:
            return {
                "factor_matrices_eigenvectors": factor_matrix_eigenvectors
            }, exception

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)
        assert len(self.roots) == 1

    def _get_field_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing shallow copies of this dataclass's fields, excluding specific fields.

        This method overrides the parent class's _get_field_dict method to exclude fields that don't
        align with the per-factor iteration pattern used in amortized computation:

        1. 'corrected_eigenvalues' is a single tensor that doesn't align with the per-factor iteration pattern
           since it represents eigenvalues across all dimensions rather than per-factor.
        2. 'roots' contains a single value for eigenvalue correction, unlike other fields which have
           one entry per factor matrix.

        Returns:
            dict[str, Any]: A dictionary mapping field names to their values, excluding
                'corrected_eigenvalues' and 'roots'.
        """
        return {
            key: value
            for key, value in super()._get_field_dict().items()
            if key not in ("corrected_eigenvalues", "roots")
        }


_ShampooKroneckerFactorsStateType = TypeVar(
    "_ShampooKroneckerFactorsStateType",
    RootInvShampooKroneckerFactorsState,
    EigendecomposedShampooKroneckerFactorsState,
    EigenvalueCorrectedShampooKroneckerFactorsState,
)
_ShampooKroneckerFactorsUnwrappedType = TypeVar(
    "_ShampooKroneckerFactorsUnwrappedType",
    RootInvShampooKroneckerFactorsUnwrapped,
    EigendecomposedShampooKroneckerFactorsUnwrapped,
    EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
)


class BaseShampooPreconditionerList(
    PreconditionerList,
    Generic[_ShampooKroneckerFactorsStateType, _ShampooKroneckerFactorsUnwrappedType],
):
    """Base class for Shampoo preconditioners.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad-like Shampoo gradient outer products accumulations, set beta2 = 1.0 and weighting_factor = 1.0.
    To enable RMSprop-like Shampoo gradient outer products accumulations, set beta2 = 0.999 and weighting_factor = 1 - beta2.
    To enable Adam-like Shampoo gradient outer products accumulations, set beta2 = 0.999, weighting_factor = 1 - beta2, and use_bias_correction = True.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, _StateValueType]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        preconditioner_config (BaseShampooPreconditionerConfig): Configuration for preconditioner computation.
        beta2 (float): The decay rate of exponential moving average factor for Shampoo factor matrices. (Default: 1.0)
        weighting_factor (float): The weighting factor for the current Shampoo gradient outer products. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioner_config: BaseShampooPreconditionerConfig,
        beta2: float = 1.0,
        weighting_factor: float = 1.0,
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._preconditioner_config = preconditioner_config
        self._beta2 = beta2
        self._weighting_factor = weighting_factor
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = tuple(
            self._create_preconditioned_dims_selector(dims)
            # Traverse through each block's dims.
            for dims in self._dims_list
        )
        preconditioned_dims_list: tuple[tuple[int, ...], ...] = tuple(
            compress_list(dims, preconditioned_dims_selector)
            for dims, preconditioned_dims_selector in zip(
                self._dims_list, preconditioned_dims_selector_list, strict=True
            )
        )

        # Create the Kronecker factors.
        kronecker_factors_unwrapped: list[_ShampooKroneckerFactorsUnwrappedType] = (
            self._create_kronecker_factors_state(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioned_dims_list=preconditioned_dims_list,
                preconditioned_dims_selector_list=preconditioned_dims_selector_list,
            )
        )

        # Initialize state lists.
        self._initialize_state_lists(
            block_list=block_list,
            kronecker_factors_unwrapped=kronecker_factors_unwrapped,
            preconditioned_dims_list=preconditioned_dims_list,
            preconditioned_dims_selector_list=preconditioned_dims_selector_list,
        )

    @abstractmethod
    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        """
        Creates a preconditioned dimensions selectors for a block.

        Args:
            dims (torch.Size): The dimensions of the block.

        Returns:
            preconditioned_dims_selector (tuple[bool, ...]): A preconditioned dimensions selectors for a block.
        """

    def _create_kronecker_factors_state(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...],
    ) -> list[_ShampooKroneckerFactorsUnwrappedType]:
        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        # NOTE: We need to instantiate the Kronecker factor states within the optimizer's state dictionary,
        # and do not explicitly store them as RootInvShampooPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but RootInvShampooPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        kronecker_factors_unwrapped = []
        for (
            block,
            block_info,
            dims,
            preconditioned_dims,
            preconditioned_dims_selector,
        ) in zip(
            block_list,
            block_info_list,
            self._dims_list,
            preconditioned_dims_list,
            preconditioned_dims_selector_list,
            strict=True,
        ):
            param_index, block_index = block_info.composable_block_ids
            assert block_index in state[block_info.param], (
                f"{block_index=} not found in {state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within DistributedShampoo._initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                "within Distributor for the correctness of block_index."
            )
            block_state = state[block_info.param][block_index]
            # NOTE: Use types.get_original_bases() instead of self.__orig_bases__ when downstream applications are Python 3.12+ available
            kronecker_factors_state_type, kronecker_factors_state_unwrapped_type = (
                get_args(attrgetter("__orig_bases__")(self)[0])
            )
            block_state[SHAMPOO] = kronecker_factors_state_type.from_block(
                block_info=block_info,
                preconditioner_config=self._preconditioner_config,
                preconditioned_dims=preconditioned_dims,
                dims=dims,
            )
            kronecker_factors_unwrapped.append(
                kronecker_factors_state_unwrapped_type.from_kronecker_factors_state(
                    kronecker_factors_state=block_state[SHAMPOO],
                    unwrapped_tensor_getter=block_info.get_tensor,
                    roots=self._get_inverse_roots_from_override(
                        preconditioned_dims_selector
                    ),
                    amortized_computation_config=self._preconditioner_config.amortized_computation_config,
                    epsilon=self._epsilon,
                    num_tolerated_failed_amortized_computations=self._preconditioner_config.num_tolerated_failed_amortized_computations,
                    use_trace_scaling=self._preconditioner_config.use_trace_scaling,
                )
            )

            # Note: the block_info.param.shape is the shape of the local parameter if the original parameter is a DTensor.
            logger.info(
                f"Instantiated Shampoo Preconditioner {str(param_index) + '.' + str(block_index)} for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_unwrapped

    @abstractmethod
    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        """
        Retrieves the inverse roots from the override parameter for a block.

        For a block, we compute the inverse root from the inverse exponent override parameter according to its order.
        If the order is not present in the inverse exponent override parameter, the default value is used for the inverse exponent override.
        The inverse root is then computed as 1 / inverse exponent override.

        Args:
            preconditioned_dims_selector (tuple[bool, ...]): A selector indicating which dimensions are preconditioned for a block.

        Returns:
            inverse_roots (tuple[float, ...]): Inverse roots for each preconditioner of a block.
        """

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """
        Updates the preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.
            step (Tensor): The current step.
            perform_amortized_computation (bool): Whether to perform an amortized computation.

        Returns:
            None
        """
        # Update the Kronecker factor matrices.
        self._update_factor_matrices(masked_grad_list=masked_grad_list)

        # Update bias correction term based on step.
        if self._use_bias_correction and self._beta2 < 1.0:
            self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

        # In Shampoo, this is equivalent to computing the inverse factor matrix.
        # In eigenvalue-corrected Shampoo, this is equivalent to computing the eigenvectors of the factor matrix.
        if perform_amortized_computation:
            for kronecker_factors_unwrapped in self._masked_kronecker_factors_unwrapped:
                kronecker_factors_unwrapped.amortized_computation(
                    bias_correction2=self._bias_correction2
                )

    def _initialize_state_lists(
        self,
        block_list: tuple[Tensor, ...],
        kronecker_factors_unwrapped: list[_ShampooKroneckerFactorsUnwrappedType],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...],
    ) -> None:
        # Initialize local lists.
        self._local_kronecker_factors_unwrapped: tuple[
            _ShampooKroneckerFactorsUnwrappedType,
            ...,
        ] = tuple(kronecker_factors_unwrapped)
        self._local_order_list: tuple[int, ...] = tuple(
            block.dim() for block in block_list
        )
        self._local_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            preconditioned_dims_selector_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: tuple[int, ...] = self._local_order_list
        self._masked_kronecker_factors_unwrapped: tuple[
            _ShampooKroneckerFactorsUnwrappedType,
            ...,
        ] = self._local_kronecker_factors_unwrapped
        self._masked_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            self._local_preconditioned_dims_selector_list
        )

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._numel_list: tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in preconditioned_dims)
            for preconditioned_dims in preconditioned_dims_list
        )
        self._num_bytes_list: tuple[int, ...] = tuple(
            numel
            * (
                get_dtype_size(self._preconditioner_config.factor_matrix_dtype)
                + get_dtype_size(block.dtype)
            )
            // 2
            for numel, block in zip(self._numel_list, block_list, strict=True)
        )

    @profile_decorator
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        self._masked_order_list = compress_list(
            self._local_order_list, local_grad_selector
        )
        self._masked_kronecker_factors_unwrapped = compress_list(
            self._local_kronecker_factors_unwrapped, local_grad_selector
        )
        self._masked_preconditioned_dims_selector_list = compress_list(
            self._local_preconditioned_dims_selector_list, local_grad_selector
        )

    @profile_decorator
    def _compute_outer_product_list(
        self,
        grad: Tensor,
        order: int,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: _ShampooKroneckerFactorsUnwrappedType,
    ) -> tuple[Tensor, ...]:
        # Construct outer product list for updating Kronecker factors.
        return tuple(
            torch.tensordot(
                grad,
                grad,
                # Contracts across all dimensions except for k.
                dims=[[*chain(range(k), range(k + 1, order))]] * 2,  # type: ignore[has-type]
            )
            for k in compress_list(range(order), preconditioned_dims_selector)
        )

    @profile_decorator
    def _update_factor_matrices(self, masked_grad_list: tuple[Tensor, ...]) -> None:
        # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
        # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
        # full list of gradients/optimizer states).
        for grad, order, preconditioned_dims_selector, kronecker_factors in zip(
            masked_grad_list,
            self._masked_order_list,
            self._masked_preconditioned_dims_selector_list,
            self._masked_kronecker_factors_unwrapped,
            strict=True,
        ):
            # Because of preconditioned_dims_selector, we may have no factor matrices to update.
            if not kronecker_factors.factor_matrices:
                continue

            outer_product_list = self._compute_outer_product_list(
                grad, order, preconditioned_dims_selector, kronecker_factors
            )

            if self._beta2 != 1.0:
                torch._foreach_mul_(kronecker_factors.factor_matrices, self._beta2)

            torch._foreach_add_(
                kronecker_factors.factor_matrices,
                outer_product_list,
                alpha=self._weighting_factor,
            )

    @staticmethod
    def _precondition_grad(
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        preconditioner_list: tuple[Tensor, ...],
        dims: tuple[list[int], list[int]] = ([0], [0]),
    ) -> Tensor:
        # TODO: Need to refactor this function to be more efficient. Ideally eliminate those branches.
        # Might consider einsum?
        assert sum(preconditioned_dims_selector) == len(preconditioner_list), (
            f"The number of dimensions to precondition ({sum(preconditioned_dims_selector)}) must match the number of preconditioners ({len(preconditioner_list)})."
        )

        # Extract all dtypes and assert they are unique
        assert len(unique_dtypes := {p.dtype for p in preconditioner_list}) <= 1, (
            f"All preconditioners must have the same dtype, but found: {unique_dtypes}"
        )

        # Use the single dtype if preconditioners exist, otherwise use grad dtype
        target_dtype = next(iter(unique_dtypes), grad.dtype)
        preconditioner_list_iter = iter(preconditioner_list)

        return reduce(
            lambda grad, should_precondition: torch.tensordot(
                # Use the single target dtype for all operations
                grad.to(dtype=target_dtype),
                # Use the actual iterator for the operation
                next(preconditioner_list_iter),
                dims=dims,
            )
            if should_precondition
            # Perform a left rotation on grad if not preconditioned.
            else grad.permute(*range(1, grad.ndim), 0),
            preconditioned_dims_selector,
            grad,
        ).to(dtype=grad.dtype)

    @overload
    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], NoReturn],
    ) -> NoReturn: ...

    @overload
    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], None],
    ) -> None: ...

    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], NoReturn | None],
    ) -> NoReturn | None:
        """
        Saves a problematic matrix for debugging and configures detailed tensor printing.

        When numerical issues occur in matrix operations, this method:
        1. Creates a temporary directory and saves the matrix for later analysis
        2. Configures PyTorch's print options to show full tensor details
        3. Executes the provided error handler function

        This approach facilitates debugging of numerical instabilities in preconditioner computations.

        Args:
            factor_matrix_index: Identifier for the factor matrix used in the filename
            source_matrix: The problematic matrix to be saved
            error_handler: Function to execute after saving the matrix. This function may:
                - Raise an exception (NoReturn), which will propagate to the caller
                - Return None to continue execution
                - Implement other error handling logic

        Returns:
            NoReturn: If the error_handler raises an exception
            None: If the error_handler returns normally

        Examples of error_handler:
            - Raise a custom exception with a detailed error message.
            - Log a warning message and continue execution.
            - Trigger a fallback mechanism to use default values.
        """
        # Save the problematic matrix to a file for debugging.
        tmp_dir = Path("/tmp").resolve()
        tmp_dir.mkdir(exist_ok=True)
        file_path = tmp_dir / f"{factor_matrix_index.replace('.', '_')}.pt"
        try:
            torch.save(source_matrix, file_path)
            logger.info(f"Matrix has been saved to {file_path} for debugging.")
        except Exception as e:
            logger.warning(f"Failed to save matrix to {file_path}: {str(e)}")

        torch.set_printoptions(
            precision=16,  # Set the precision for floating point numbers to 16 decimal places.
            linewidth=10000,  # Set the line width to 10000, allowing for long lines without wrapping.
            profile="full",  # Use the 'full' profile to display all elements of tensors.
        )
        error_handler()

    @abstractmethod
    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: _ShampooKroneckerFactorsUnwrappedType,
    ) -> Tensor:
        """
        Applies the Shampoo preconditioner to a gradient tensor.

        This method is implemented by subclasses to perform the actual preconditioning
        operation using the specific preconditioner implementation (root inverse,
        eigendecomposed, or eigenvalue-corrected).

        Args:
            grad (Tensor): The gradient tensor to be preconditioned.
            preconditioned_dims_selector (tuple[bool, ...]): A boolean tuple indicating which
                dimensions of the gradient should be preconditioned. Dimensions with True
                values will be preconditioned, while dimensions with False values will not.
            kronecker_factors (_ShampooKroneckerFactorsUnwrappedType): The unwrapped Kronecker
                factors containing the necessary matrices for preconditioning.

        Returns:
            preconditioned_grad (Tensor): The preconditioned gradient tensor.
        """

    @profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the Shampoo preconditioner that rely on ClassicShampooPreconditionerConfig.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        return tuple(
            self._compute_preconditioned_gradient(
                grad=masked_grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                kronecker_factors=kronecker_factors,
            )
            for masked_grad, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_unwrapped,
                strict=True,
            )
        )


_ClassicShampooKroneckerFactorsStateType = TypeVar(
    "_ClassicShampooKroneckerFactorsStateType",
    RootInvShampooKroneckerFactorsState,
    EigendecomposedShampooKroneckerFactorsState,
)

_ClassicShampooKroneckerFactorsUnwrappedType = TypeVar(
    "_ClassicShampooKroneckerFactorsUnwrappedType",
    RootInvShampooKroneckerFactorsUnwrapped,
    EigendecomposedShampooKroneckerFactorsUnwrapped,
)


class ClassicShampooPreconditionerList(
    BaseShampooPreconditionerList[
        _ClassicShampooKroneckerFactorsStateType,
        _ClassicShampooKroneckerFactorsUnwrappedType,
    ]
):
    """Base class for Shampoo preconditioners that rely on ClassicShampooPreconditionerConfig.

    This class factors out common implementations for Shampoo preconditioners that use
    ClassicShampooPreconditionerConfig to determine inverse exponent overrides and preconditioned dimensions.
    It provides methods to retrieve inverse exponent overrides based on dimension and order,
    and to create preconditioned dimension selectors.

    """

    def _get_inverse_exponent(self, dimension: int, order: int) -> float:
        """
        Retrieves the inverse exponent override based on the dimension and order.

        Args:
            dimension (int): The dimension for which the inverse exponent override is needed.
            order (int): The order of the preconditioner.

        Returns:
            float: The inverse exponent override value for the given dimension and order.
        """
        inverse_exponent_override_on_order: dict[int, float] | float = attrgetter(
            INVERSE_EXPONENT_OVERRIDE
        )(self._preconditioner_config).get(order, {})
        if isinstance(inverse_exponent_override_on_order, dict):
            return inverse_exponent_override_on_order.get(
                dimension, 1 / (2 * max(order, 1))
            )
        assert isinstance(inverse_exponent_override_on_order, float), (
            f"Expected inverse_exponent_override_on_order to be a float or a dict, but got {type(inverse_exponent_override_on_order)} instead."
        )
        return inverse_exponent_override_on_order

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            self._get_inverse_exponent(dimension=d, order=len(dims)) != 0.0
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        return tuple(
            # Compute the inverse root, 1 / inverse_exponent{_override}, accordingly for each required dim.
            1
            / self._get_inverse_exponent(
                dimension=k, order=len(preconditioned_dims_selector)
            )
            # Traverse through each dim of a block that requires precondition.
            for k, should_precondition in enumerate(preconditioned_dims_selector)
            if should_precondition
        )


class RootInvShampooPreconditionerList(
    ClassicShampooPreconditionerList[
        RootInvShampooKroneckerFactorsState, RootInvShampooKroneckerFactorsUnwrapped
    ]
):
    """Root inverse Shampoo preconditioners for list of parameters."""

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: RootInvShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        return self._precondition_grad(
            grad=grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.inv_factor_matrices,
        )


class EigendecomposedShampooPreconditionerList(
    ClassicShampooPreconditionerList[
        EigendecomposedShampooKroneckerFactorsState,
        EigendecomposedShampooKroneckerFactorsUnwrapped,
    ]
):
    """Eigendecomposed Shampoo preconditioners for list of parameters."""

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: EigendecomposedShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        # TODO: remove assertion when rank_deficient_stability_config is generalized to MatrixFunctionConfig
        assert isinstance(
            self._preconditioner_config.amortized_computation_config,
            EigendecompositionConfig,
        )
        rank_deficient_stability_config = self._preconditioner_config.amortized_computation_config.rank_deficient_stability_config

        return self._precondition_grad(
            grad=grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=tuple(
                matrix_inverse_root_from_eigendecomposition(
                    L=eigenvalues,
                    Q=eigenvectors,
                    root=Fraction(root),
                    epsilon=self._epsilon,
                    rank_deficient_stability_config=rank_deficient_stability_config,
                )
                for eigenvectors, eigenvalues, root in zip(
                    kronecker_factors.factor_matrices_eigenvectors,
                    kronecker_factors.factor_matrices_eigenvalues,
                    kronecker_factors.roots,
                    strict=True,
                )
            ),
        )


class EigenvalueCorrectedShampooPreconditionerList(
    BaseShampooPreconditionerList[
        EigenvalueCorrectedShampooKroneckerFactorsState,
        EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
    ]
):
    """Eigenvalue-corrected Shampoo preconditioners for list of parameters."""

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            d
            not in attrgetter("ignored_basis_change_dims")(
                self._preconditioner_config
            ).get(len(dims), [])
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        # NOTE: In eigenvalue-corrected Shampoo, there is only a single inverse root that is applied to the corrected eigenvalues.
        return (
            # Compute the inverse root, 1 / eigenvalue_inverse_exponent{_override}.
            1
            / attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config).get(
                len(preconditioned_dims_selector), 1 / 2
            ),
        )

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """
        Updates the preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.
            step (Tensor): The current step.
            perform_amortized_computation (bool): Whether to perform an amortized computation.

        Returns:
            None
        """
        super().update_preconditioners(
            masked_grad_list=masked_grad_list,
            step=step,
            perform_amortized_computation=perform_amortized_computation,
        )

        # Update the eigenvalue corrections of Shampoo's preconditioner.
        for grad, preconditioned_dims_selector, kronecker_factors in zip(
            masked_grad_list,
            self._masked_preconditioned_dims_selector_list,
            self._masked_kronecker_factors_unwrapped,
            strict=True,
        ):
            # Transform the gradient to eigenbasis of Shampoo's factor matrices.
            # Because of preconditioned_dims_selector, this might be a no-op.
            grad = self._precondition_grad(
                grad=grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
            )
            # Update corrected eigenvalues (squared gradient in eigenbasis of Shampoo preconditioner).
            if self._beta2 != 1.0:
                kronecker_factors.corrected_eigenvalues.mul_(self._beta2)

            # NOTE: The case when self._weighting_factor == 1.0 is not well tested and might not be stable.
            kronecker_factors.corrected_eigenvalues.addcmul_(
                grad, grad, value=self._weighting_factor
            )

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        # Clone the masked gradient to avoid modifying the original tensor.
        # This is only relevant when _precondition_grad is a no-op.
        preconditioned_grad = grad.clone()
        # Transform the gradient to eigenbasis of Shampoo's factor matrices.
        preconditioned_grad = self._precondition_grad(
            grad=preconditioned_grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
        )

        # Precondition with inverse root of corrected eigenvalues.
        preconditioned_grad.div_(
            kronecker_factors.corrected_eigenvalues.div(self._bias_correction2)
            .pow_(1 / kronecker_factors.roots[0])
            .add_(self._epsilon)
        )
        # Convert back to basis of the parameters.
        return self._precondition_grad(
            grad=preconditioned_grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
            dims=([0], [1]),
        )


class RootInvKLShampooPreconditionerList(RootInvShampooPreconditionerList):
    """Root inverse KL-Shampoo preconditioners for list of parameters."""

    @profile_decorator
    def _compute_outer_product_list(
        self,
        grad: Tensor,
        order: int,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: RootInvShampooKroneckerFactorsUnwrapped,
    ) -> tuple[Tensor, ...]:
        # Construct outer product list for updating Kronecker factors.
        outer_product_list = []
        for idx_of_k, k in enumerate(
            compress_list(range(order), preconditioned_dims_selector)
        ):
            # KL-Shampoo uses the gradient preconditioned (along all dimensions that are contracted) with the inverse root of the factor matrices to compute the outer products.
            local_preconditioned_dims_selector = list(preconditioned_dims_selector)
            local_preconditioned_dims_selector[k] = False
            preconditioned_grad = self._precondition_grad(
                grad=grad,
                preconditioned_dims_selector=tuple(local_preconditioned_dims_selector),
                preconditioner_list=tuple(
                    inv_factor_matrix
                    for idx, inv_factor_matrix in enumerate(
                        kronecker_factors.inv_factor_matrices
                    )
                    if idx != idx_of_k
                ),
            )
            outer_product_list.append(
                torch.tensordot(
                    preconditioned_grad,
                    preconditioned_grad,
                    # Contracts across all dimensions except for k.
                    dims=[[*chain(range(k), range(k + 1, order))]] * 2,  # type: ignore[has-type]
                )
            )
        return tuple(outer_product_list)


class EigendecomposedKLShampooPreconditionerList(
    EigendecomposedShampooPreconditionerList
):
    """Eigendecomposed KL-Shampoo preconditioners for list of parameters."""

    @profile_decorator
    def _compute_outer_product_list(
        self,
        grad: Tensor,
        order: int,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: EigendecomposedShampooKroneckerFactorsUnwrapped,
    ) -> tuple[Tensor, ...]:
        # TODO: remove assertion when rank_deficient_stability_config is generalized to MatrixFunctionConfig
        assert isinstance(
            self._preconditioner_config.amortized_computation_config,
            EigendecompositionConfig,
        )
        rank_deficient_stability_config = self._preconditioner_config.amortized_computation_config.rank_deficient_stability_config

        # Construct outer product list for updating Kronecker factors.
        outer_product_list = []
        for idx_of_k, k in enumerate(
            compress_list(range(order), preconditioned_dims_selector)
        ):
            # KL-Shampoo uses the gradient preconditioned (along all dimensions that are contracted) with the inverse root of the factor matrices to compute the outer products.
            local_preconditioned_dims_selector = list(preconditioned_dims_selector)
            local_preconditioned_dims_selector[k] = False
            preconditioned_grad = self._precondition_grad(
                grad=grad,
                preconditioned_dims_selector=tuple(local_preconditioned_dims_selector),
                preconditioner_list=tuple(
                    matrix_inverse_root_from_eigendecomposition(
                        L=eigenvalues,
                        Q=eigenvectors,
                        root=Fraction(root),
                        epsilon=self._epsilon,
                        rank_deficient_stability_config=rank_deficient_stability_config,
                    )
                    for idx, (eigenvalues, eigenvectors, root) in enumerate(
                        zip(
                            kronecker_factors.factor_matrices_eigenvalues,
                            kronecker_factors.factor_matrices_eigenvectors,
                            kronecker_factors.roots,
                            strict=True,
                        )
                    )
                    if idx != idx_of_k
                ),
            )
            outer_product = torch.tensordot(
                preconditioned_grad,
                preconditioned_grad,
                # Contracts across all dimensions except for k.
                dims=[[*chain(range(k), range(k + 1, order))]] * 2,  # type: ignore[has-type]
            )
            outer_product_list.append(outer_product)
        return tuple(outer_product_list)

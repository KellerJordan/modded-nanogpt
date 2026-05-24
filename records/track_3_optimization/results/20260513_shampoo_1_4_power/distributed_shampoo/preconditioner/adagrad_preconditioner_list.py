"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Hashable, Mapping
from typing import TypeVar

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.utils.shampoo_utils import compress_list
from torch import Tensor


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"


_SubStateValueType = TypeVar("_SubStateValueType")
_StateValueType = dict[Hashable, _SubStateValueType]


class AdagradPreconditionerList(PreconditionerList):
    """Adagrad / RMSprop / Adam preconditioners for a list of parameters.

    Operations are performed in-place with foreach operators.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0 and weighting_factor = 1.0.
    To enable RMSprop, set beta2 = 0.999 and weighting_factor = 1 - beta2.
    To enable Adam, set beta2 = 0.999, weighting_factor = 1 - beta2, and use_bias_correction = True.

    Other variants can also be specified.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, _StateValueType]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        beta2 (float): The decay rate of exponential moving average factor for Adam/RMSprop second moment state. (Default: 1.0)
        weighting_factor (float): The weighting factor for the current squared gradients. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        beta2: float = 1.0,
        weighting_factor: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Instantiate scalar hyperparameters.
        self._beta2 = beta2
        self._weighting_factor: float = weighting_factor
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Instantiate (blocked) AdaGrad preconditioners and construct preconditioner list.
        # NOTE: We need to instantiate the AdaGrad preconditioner states within the optimizer's state dictionary,
        # and do not explicitly store them as AdagradPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but AdagradPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        preconditioner_list: list[Tensor] = []
        for block, block_info in zip(block_list, block_info_list, strict=True):
            param_index, block_index = block_info.composable_block_ids
            assert block_index in state[block_info.param], (
                f"{block_index=} not found in {state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within DistributedShampoo._initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                "within Distributor for the correctness of block_index."
            )
            block_state = state[block_info.param][block_index]

            # Instantiate AdaGrad optimizer state for this block.
            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[ADAGRAD] = block_info.allocate_zeros_tensor(
                size=block.size(),
                dtype=block.dtype,
                device=block.device,
            )
            preconditioner_list.append(block_info.get_tensor(block_state[ADAGRAD]))

            # Note: the block_info.param.shape is the shape of the local parameter if the original parameter is a DTensor.
            logger.info(
                f"Instantiated Adagrad Preconditioner {preconditioner_index} ({block_state[ADAGRAD].shape} with dtype {block_state[ADAGRAD].dtype}) for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._local_preconditioner_list: tuple[Tensor, ...] = tuple(preconditioner_list)
        self._masked_preconditioner_list: tuple[Tensor, ...] = (
            self._local_preconditioner_list
        )

        # Construct lists of numels and bytes for logging purposes.
        self._numel_list: tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in self._local_preconditioner_list
        )
        self._num_bytes_list: tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in self._local_preconditioner_list
        )

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        if self._beta2 != 1.0:
            torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)

        torch._foreach_addcmul_(
            self._masked_preconditioner_list,
            masked_grad_list,
            masked_grad_list,
            value=self._weighting_factor,
        )

        # Update bias correction term based on step list.
        if self._use_bias_correction and self._beta2 < 1.0:
            self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    @profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions the gradient list using the AdaGrad preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A tuple of gradients with None values removed.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        masked_bias_corrected_preconditioner_list = torch._foreach_div(
            self._masked_preconditioner_list,
            self._bias_correction2,
        )
        torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
        torch._foreach_add_(masked_bias_corrected_preconditioner_list, self._epsilon)
        return torch._foreach_div(
            masked_grad_list, masked_bias_corrected_preconditioner_list
        )

    @profile_decorator
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        self._masked_preconditioner_list = compress_list(
            self._local_preconditioner_list, local_grad_selector
        )

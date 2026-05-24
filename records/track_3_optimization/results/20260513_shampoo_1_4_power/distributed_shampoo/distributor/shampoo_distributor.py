"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from operator import attrgetter
from typing import Any, Literal, overload

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    ShampooRuntimeConfig,
)
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    merge_small_dims,
    multi_dim_split,
)
from torch import Tensor


###### DISTRIBUTOR CLASSES ######
class DistributorInterface(ABC):
    """Distributor interface.

    Functionally specifies the API for Distributor classes.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        runtime_config (ShampooRuntimeConfig): Runtime configurations for the distributor, e.g., debugging, pt2 compile options.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        self._param_group = param_group
        self._runtime_config: ShampooRuntimeConfig = (
            runtime_config if runtime_config is not None else ShampooRuntimeConfig()
        )
        # Merge and block parameters creates self._global_blocked_params and self._global_num_blocks_per_param
        # Global blocked params are all the blocked parameters after merging and blocking.
        # Global num blocks per param stores the number of blocks for each global parameter.
        # Global merged dims list stores the merged dimensions for each global parameter.
        self._merge_and_block_parameters()
        # Global grad selector masks all global gradients that are None.
        self._global_grad_selector: tuple[bool, ...] = (True,) * len(
            self._global_blocked_params
        )
        # In order to avoid redundant computation, we store the previous global grad selector.
        self._previous_global_grad_selector: tuple[bool, ...] | None = None

        # Declare properties that will be populated by subclasses.
        # Distributor selector masks all global parameter blocks that are NOT assigned to the local device.
        self._distributor_selector: tuple[bool, ...]
        # Local grad selector masks all local gradients (i.e., already masked by distributor selector) that are None.
        self._local_grad_selector: tuple[bool, ...]
        # Local blocked params are the parameters masked by the distributor selector.
        self._local_blocked_params: tuple[Tensor, ...]
        # Local masked blocked params are the parameters masked by the distributor selector AND the local grad selector.
        self._local_masked_blocked_params: tuple[Tensor, ...]
        # Local block info list contains information about each block masked by the distributor selector.
        self._local_block_info_list: tuple[BlockInfo, ...]

    @abstractmethod
    @torch.no_grad()
    def update_params(
        self,
        blocked_search_directions: tuple[Tensor, ...],
        use_masked_tensors: bool = True,
    ) -> None: ...

    @property
    def local_grad_selector(self) -> tuple[bool, ...]:
        return self._local_grad_selector

    @property
    def local_blocked_params(self) -> tuple[Tensor, ...]:
        return self._local_blocked_params

    @property
    def local_masked_blocked_params(self) -> tuple[Tensor, ...]:
        return self._local_masked_blocked_params

    @property
    def local_block_info_list(self) -> tuple[BlockInfo, ...]:
        return self._local_block_info_list

    def _construct_composable_block_ids(
        self,
        param_index: int,
        block_index: int,
        rank: int | None = None,
    ) -> tuple[int, str]:
        """Construct composable block ids.

        Args:
            param_index (int): Index of the parameter in self._param_group[PARAMS].
            block_index (int): Index of the tensor block within a given parameter.
            rank (int | None): Rank of this process group; used in FSDP/HSDP. (Default: None)

        Returns:
            composable_block_ids (tuple[int, str]): Composable block id tuple containing global block index and local block name.
                The latter will be used to identify blocks in the masked tensor.

        Examples:
            (0, "block_2") - For parameter index 0, block index 2, no rank
            (1, "rank_3-block_0") - For parameter index 1, block index 0, rank 3

        """
        return (
            param_index,
            "-".join(
                filter(
                    None,
                    (
                        f"rank_{rank}" if rank is not None else None,
                        f"block_{block_index}",
                    ),
                )
            ),
        )

    @overload
    @torch.no_grad()
    def _get_params_or_grads(
        self, get_grad: Literal[True]
    ) -> Iterable[Tensor | None]: ...

    @overload
    @torch.no_grad()
    def _get_params_or_grads(
        self, get_grad: Literal[False] = False
    ) -> Iterable[Tensor]: ...

    @torch.no_grad()
    def _get_params_or_grads(self, get_grad: bool = False) -> Iterable[Tensor | None]:
        """Helper function that gets params or grads from the parameter group.

        NOTE: The purpose of this function is for FullyShardDistributor (supporting Shampoo on
        per-parameter FSDP, a.k.a. FSDP2 or FullyShard) to override, in order to get the local
        params/grads from DTensors.

        By default, we just return the original params/grads.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param. (Default: False)

        Returns:
            local (Iterable[Tensor | None]): Local params (or gradients) from the param_group. Note
              that gradients can be None.
        """

        return (
            map(attrgetter("grad"), self._param_group[PARAMS])
            if get_grad
            else self._param_group[PARAMS]
        )

    @torch.no_grad()
    def _merge_and_block_with_params(
        self, params: Iterable[Tensor]
    ) -> tuple[list[Tensor], list[int]]:
        """Merges small dimensions and blocks parameters into manageable chunks.

        This function processes each parameter tensor by:
        1. Merging small dimensions based on a threshold
        2. Splitting large tensors into blocks that don't exceed the maximum preconditioner dimension
        3. Detaching the resulting blocks to prevent gradient tracking

        Args:
            params (Iterable[Tensor]): Parameter tensors to be processed

        Returns:
            blocked_params (list[Tensor]) : List of blocked parameter tensors.
            num_blocks_per_param (list[int]) : List of block counts for each original parameter.
        """
        # Generate blocked parameters list and number of blocks per parameter.
        blocked_params: list[Tensor] = []
        num_blocks_per_param: list[int] = []
        merge_dims = partial(
            merge_small_dims,
            threshold=self._param_group[MAX_PRECONDITIONER_DIM],
            target_tensor_dimensionality=self._param_group[
                DISTRIBUTED_CONFIG
            ].target_parameter_dimensionality,
        )

        for param in params:
            # Obtain blocks for each parameter after merging.
            blocks_within_param = multi_dim_split(
                param.view(merge_dims(tensor_shape=param.size())),
                self._param_group[MAX_PRECONDITIONER_DIM],
            )

            # Generate and extend blocked parameters list.
            blocked_params.extend(
                # Note: We are using tensor.detach() here to explicitly set block_param (a view of the original
                # parameter) to requires_grad = False in order to prevent errors with print and PT2 compile.
                # Remove this tensor.detach() once https://github.com/pytorch/pytorch/issues/113793 is fixed.
                block_param.detach()
                for block_param in blocks_within_param
            )
            num_blocks_per_param.append(len(blocks_within_param))

        return blocked_params, num_blocks_per_param

    def _merge_and_block_parameters(self) -> None:
        """Merges small dimensions and blocks parameters, storing results as instance attributes.

        NOTE: FSDP may modify this function.
        """
        self._global_blocked_params: tuple[Tensor, ...]
        self._global_num_blocks_per_param: tuple[int, ...]
        self._global_blocked_params, self._global_num_blocks_per_param = map(  # type: ignore[assignment]
            partial(tuple),
            self._merge_and_block_with_params(params=self._get_params_or_grads()),
        )

    @abstractmethod
    def merge_and_block_gradients(
        self,
    ) -> tuple[Tensor, ...]: ...

    def _merge_and_block_gradients(
        self,
    ) -> tuple[Tensor, ...]:
        """Merges small dims and blocks gradients.

        NOTE: FSDP Distributor may modify this function.

        Returns:
            local_masked_blocked_grads (tuple[Tensor, ...]): Local gradients with grad not None.

        """

        local_masked_blocked_grads: list[Tensor] = []
        global_grad_selector = []
        merge_dims = partial(
            merge_small_dims,
            threshold=self._param_group[MAX_PRECONDITIONER_DIM],
            target_tensor_dimensionality=self._param_group[
                DISTRIBUTED_CONFIG
            ].target_parameter_dimensionality,
        )

        for grad, num_blocks, (block_index, next_block_index) in zip(
            self._get_params_or_grads(get_grad=True),
            self._global_num_blocks_per_param,
            generate_pairwise_indices(self._global_num_blocks_per_param),
            strict=True,
        ):
            param_distributor_selector = self._distributor_selector[
                block_index:next_block_index
            ]

            # Note: Gradients that are None or empty (grad.numel() == 0) still belong to a block
            # with corresponding block_info, but are filtered out for all updates.
            is_invalid_grad = grad is None or grad.numel() == 0
            # Update the selector
            global_grad_selector.extend([not is_invalid_grad] * num_blocks)

            if is_invalid_grad or not any(param_distributor_selector):
                # Skip multi_dim_split if this blocked grad will not be used locally.
                continue

            assert grad is not None

            if self._runtime_config.eager_nan_check:
                assert torch.isfinite(grad).all(), (
                    f"Encountered gradient containing NaN/Inf in parameter with shape {attrgetter('shape')(grad)}. Check your model for numerical instability or consider gradient clipping."
                )

            # Obtain blocks for each gradient after merging.
            blocks_within_grad = multi_dim_split(
                grad.view(merge_dims(tensor_shape=grad.size())),
                self._param_group[MAX_PRECONDITIONER_DIM],
            )
            # Generate block-to-parameter metadata and extend blocked parameters list.
            local_masked_blocked_grads.extend(
                compress_list(blocks_within_grad, param_distributor_selector)
            )

        # Set global grad selector as tuple.
        self._global_grad_selector = tuple(global_grad_selector)

        return tuple(local_masked_blocked_grads)


class Distributor(DistributorInterface):
    """Default Distributor class.

    Handles merging and blocking of the parameters at instantiation, and the gradients
    at each iteration. Note that no communication is performed since it assumes only
    single-GPU training.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        runtime_config (ShampooRuntimeConfig): Runtime configurations for the distributor, e.g., debugging, pt2 compile options.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        super().__init__(param_group, runtime_config)

        # Initialize selectors and local blocked (masked) parameters.
        self._local_grad_selector: tuple[bool, ...] = (True,) * len(
            self._global_blocked_params
        )
        self._distributor_selector: tuple[bool, ...] = self._local_grad_selector
        self._local_blocked_params: tuple[Tensor, ...] = self._global_blocked_params
        self._local_masked_blocked_params: tuple[Tensor, ...] = (
            self._local_blocked_params
        )
        self._local_block_info_list: tuple[BlockInfo, ...] = (
            self._construct_local_block_info_list()
        )

    @torch.no_grad()
    def update_params(
        self,
        blocked_search_directions: tuple[Tensor, ...],
        use_masked_tensors: bool = True,
    ) -> None:
        """Update params stored inside this distributor according to the input search directions argument.

        Args:
            blocked_search_directions (tuple[Tensor, ...]): Search directions for each local blocked parameter.
                This tuple might be empty if the parameters are not receiving gradients.
            use_masked_tensors (bool): If True (default), operates on masked blocked params.
                If False, operates on all local blocked params regardless of gradient masking.

        """
        target_params = (
            self._local_masked_blocked_params
            if use_masked_tensors
            else self._local_blocked_params
        )

        assert len(blocked_search_directions) == len(target_params), (
            f"Expected {len(blocked_search_directions)=} to be equal to {len(target_params)=}."
        )

        # torch._foreach only accepts non-empty list
        if blocked_search_directions:
            torch._foreach_add_(
                target_params,
                blocked_search_directions,
            )

    @torch.no_grad()
    def _construct_local_block_info_list_with_params(
        self, params: Iterable[Tensor], rank: int | None = None
    ) -> tuple[BlockInfo, ...]:
        return tuple(
            BlockInfo(
                param=param,
                composable_block_ids=self._construct_composable_block_ids(
                    param_index=param_index,
                    block_index=block_index,
                    rank=rank,
                ),
            )
            # Block index that is accumulated across all parameters within a parameter group.
            for ((param_index, param), num_blocks_within_param) in zip(
                enumerate(params),
                self._global_num_blocks_per_param,
                strict=True,
            )
            for block_index in range(num_blocks_within_param)
        )

    @torch.no_grad()
    def _construct_local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Construct the local block info list.

        This method creates a list of BlockInfo objects, which contain information about each parameter block,
        including its composable block IDs and the original parameter it belongs to. The BlockInfo objects
        are used throughout the optimizer to track and manage parameter blocks.

        Returns:
            block_info_list (tuple[BlockInfo, ...]): A tuple of BlockInfo objects for each parameter block.
        """
        return self._construct_local_block_info_list_with_params(
            params=self._get_params_or_grads()
        )

    def merge_and_block_gradients(
        self,
    ) -> tuple[Tensor, ...]:
        """Merge and block gradients.

        NOTE: This function MUST be called in the step function of the optimizer after the
        gradient has been updated.

        Returns:
            local_masked_blocked_grads (tuple[Tensor, ...]): Local blocked gradients masked with grad existence.

        """
        local_masked_blocked_grads = self._merge_and_block_gradients()

        if self._previous_global_grad_selector != self._global_grad_selector:
            self._previous_global_grad_selector = self._global_grad_selector

            # Update _local_grad_selector and _local_masked_blocked_params only when global_grad_selector is changed.
            self._local_grad_selector = compress_list(
                self._global_grad_selector,
                self._distributor_selector,
            )
            self._local_masked_blocked_params = compress_list(
                self._local_blocked_params, self._local_grad_selector
            )

        return local_masked_blocked_grads

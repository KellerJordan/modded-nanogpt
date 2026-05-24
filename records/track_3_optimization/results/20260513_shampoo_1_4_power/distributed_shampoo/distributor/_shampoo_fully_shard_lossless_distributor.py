"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Iterable
from typing import Any, Literal, overload

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_dist_utils import shampoo_comm_profiler
from distributed_shampoo.distributor.shampoo_distributor import Distributor
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    PARAMS,
    ShampooRuntimeConfig,
)
from distributed_shampoo.utils.shampoo_utils import (
    prepare_update_param_buffers,
    redistribute_and_update_params,
)
from torch import distributed as dist, Tensor

logger: logging.Logger = logging.getLogger(__name__)


class FullyShardLosslessDistributor(Distributor):
    """FullyShard Lossless Distributor class.

    On top of FullyShardDistributor, this distributor handles the parameter assignment to exchange the gradients
    and parameter updates across the shards to achieve lossless numerical results compared to default Shampoo.

    .. note::
        FullyShardLosslessDistributor is experimental and subject to change.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        distributed_config: FullyShardDistributedConfig = param_group[
            DISTRIBUTED_CONFIG
        ]
        self._param_assignment_strategy: FSDPParamAssignmentStrategy = (
            distributed_config.param_assignment_strategy
        )
        logger.info(
            f"Shampoo FullyShardLosslessDistributor {self._param_assignment_strategy=}",
        )

        self._group_size: int = dist.get_world_size()
        self._dist_group: dist.ProcessGroup = dist.new_subgroups(
            group_size=self._group_size
        )[0]
        self._group_rank: int = dist.get_rank(group=self._dist_group)

        def should_assign_param_idx(i: int) -> bool:
            if (
                self._param_assignment_strategy
                == FSDPParamAssignmentStrategy.ROUND_ROBIN
            ):
                return i % self._group_size == self._group_rank
            return True

        self._assigned_params_mask: tuple[bool, ...] = tuple(
            should_assign_param_idx(idx) for idx in range(len(param_group[PARAMS]))
        )

        # Collects and stores the model parameters assigned to this rank.
        # Note that we explicitly disable the unnecessary gradient tracking for the all-gather collectives
        # used to initialize the full parameters.
        with (
            torch.no_grad(),
            shampoo_comm_profiler(f"{self.__class__.__name__}::full_tensor_calls"),
        ):
            full_params: list[Tensor] = [p.full_tensor() for p in param_group[PARAMS]]

        # TODO (irisz): eagerly initialize the _assigned_full_params cannot handle dead layers parameters correctly,
        # as we do not have gradient information during initialization. We need a way to handle dead layers parameters
        # before doing performance optimization on the full_tensor call above (e.g. change full_tensor call to all_to_all).
        self._assigned_full_params: list[Tensor] = [
            p
            for p, assigned in zip(full_params, self._assigned_params_mask)
            if assigned
        ]

        # For ROUND_ROBIN strategy, creates a buffer for receiving the updated param shards.
        self._update_param_buffers: list[Tensor] | None = (
            prepare_update_param_buffers(param_group[PARAMS], self._group_size)
            if self._param_assignment_strategy
            == FSDPParamAssignmentStrategy.ROUND_ROBIN
            else None
        )

        super().__init__(param_group, runtime_config)

        if logger.isEnabledFor(logging.DEBUG):
            # logging local blocked info list for easier debugging
            local_block_id_list = tuple(
                block_info.composable_block_ids
                for block_info in self._local_block_info_list
            )
            logger.debug(
                f"Local blocked params[size={len(local_block_id_list)}]: {local_block_id_list}"
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
        """Helper function to get the assigned full params (or gradients) from the param_group.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param.
        Returns:
            local (Iterable[Tensor | None]): assigned full params (or grad) from the param_group.
        """

        if get_grad:
            # Getting grads at every optimizer step triggers implicit all-gather. Note that p.numel()
            # returns total number of elements in the tensor (as opposed to local shard of DTensor).
            with shampoo_comm_profiler(f"{self.__class__.__name__}::grad_full_tensors"):
                full_grads = [
                    None if p.grad is None else p.grad.full_tensor()
                    for p in self._param_group[PARAMS]
                ]
            return (
                full_grad
                for full_grad, assigned in zip(
                    full_grads, self._assigned_params_mask, strict=True
                )
                if assigned and (full_grad is None or full_grad.numel() > 0)
            )

        else:
            return filter(
                lambda p: isinstance(p, Tensor) and p.numel() > 0,
                self._assigned_full_params,
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
        super().update_params(blocked_search_directions, use_masked_tensors)

        # Copy the updated full parameters to the original parameters in the param group.
        # For example, when the strategy is REPLICATE, we need to take each updated full parameter `full_param`,
        # redistribute it according to the device mesh to get the locally assigned slice, and copy the slice to the
        # corresponding local parameter `local_param` in the param group.
        if self._param_assignment_strategy == FSDPParamAssignmentStrategy.ROUND_ROBIN:
            with shampoo_comm_profiler(
                f"{self.__class__.__name__}::redistribute_and_update_params"
            ):
                redistribute_and_update_params(
                    self._param_group[PARAMS],
                    self._assigned_full_params,
                    self._update_param_buffers,  # type: ignore
                    self._dist_group,
                )

        elif self._param_assignment_strategy == FSDPParamAssignmentStrategy.REPLICATE:
            local_params = list(
                filter(lambda p: p.numel() > 0, self._param_group[PARAMS])
            )
            full_param_slices = [
                # When param assignment strategy is REPLICATE, explicitly set `src_data_rank` to None to avoid
                # triggering communication and simply use the local copy of replicated parameters.
                dist.tensor.distribute_tensor(
                    full_param,
                    local_param.device_mesh,
                    local_param.placements,
                    src_data_rank=None,
                ).to_local()
                for local_param, full_param in zip(
                    local_params,
                    self._get_params_or_grads(),
                    strict=True,
                )
            ]
            # torch._foreach_copy_ requires both lists of tensors to be local tensors.
            torch._foreach_copy_(
                [p.to_local() for p in local_params], full_param_slices
            )

    @torch.no_grad()
    def _construct_local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Construct local block info list from param_group."""
        return self._construct_local_block_info_list_with_params(
            params=(
                p
                for assigned, p in zip(
                    self._assigned_params_mask, self._param_group[PARAMS], strict=True
                )
                if assigned and p.numel() > 0
            ),
            rank=None,
        )

    @torch.no_grad()
    def refresh_assigned_full_params(self) -> None:
        """Refresh the cached full parameter tensors from current model parameters.

        This method should be called after loading a checkpoint to ensure the
        distributor's cached tensors reflect the updated model parameter values.
        Without calling this, the optimizer may use stale parameter values.

        This method refreshes:
        1. _assigned_full_params - the cached full tensor copies
        2. _global_blocked_params and _local_blocked_params - views of the full tensors
        """
        full_params: list[Tensor] = [p.full_tensor() for p in self._param_group[PARAMS]]
        self._assigned_full_params = [
            p
            for p, assigned in zip(full_params, self._assigned_params_mask)
            if assigned
        ]
        # Also re-block the parameters since _global_blocked_params are views of the old
        # _assigned_full_params tensors
        self._merge_and_block_parameters()
        # Update _local_blocked_params and _local_masked_blocked_params which are set
        # in parent's __init__ but don't get updated by _merge_and_block_parameters()
        self._local_blocked_params = self._global_blocked_params
        self._local_masked_blocked_params = self._local_blocked_params

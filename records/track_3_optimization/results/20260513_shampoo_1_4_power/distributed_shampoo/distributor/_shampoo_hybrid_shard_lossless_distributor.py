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
from distributed_shampoo.distributor.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.distributor.shampoo_hybrid_shard_distributor import (
    HybridShardDistributor,
)
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParamAssignmentStrategy,
    HybridShardDistributedConfig,
    PARAMS,
    ShampooRuntimeConfig,
)
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    prepare_update_param_buffers,
    redistribute_and_update_params,
)
from torch import distributed as dist, Tensor
from torch.distributed import tensor as dtensor

logger: logging.Logger = logging.getLogger(__name__)


class HybridShardLosslessDistributor(HybridShardDistributor):
    """HybridShard Lossless Distributor class.

    On top of the HybridShardDistributor, this distributor handles the parameter assignment to exchange the gradients
    and parameter updates across the shards to achieve lossless numerical results compared to default Shampoo.

    .. note::
        HybridShardLosslessDistributor is experimental and subject to change.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        distributed_config: HybridShardDistributedConfig = param_group[
            DISTRIBUTED_CONFIG
        ]
        self._param_assignment_strategy: FSDPParamAssignmentStrategy = (
            distributed_config.param_assignment_strategy
        )
        logger.info(
            f"Shampoo HybridShardLosslessDistributor {self._param_assignment_strategy=}",
        )

        self._shard_group_size: int = distributed_config.device_mesh.size(1)
        # Initialize distributed group for communicating across the shard dimension.
        self._shard_dist_group: dist.ProcessGroup = (
            distributed_config.device_mesh.get_group(mesh_dim=1)
        )
        self._shard_group_rank: int = dist.get_rank(self._shard_dist_group)

        # Stores full parameters (as opposed to DTensors) for the model parameters assigned to this rank.
        # For example, when the strategy is REPLICATE, it stores the full parameters on all ranks.
        def should_assign_param_idx(i: int) -> bool:
            if (
                self._param_assignment_strategy
                == FSDPParamAssignmentStrategy.ROUND_ROBIN
            ):
                return i % self._shard_group_size == self._shard_group_rank
            return True

        with torch.no_grad():
            self._assigned_params_mask: tuple[bool, ...] = tuple(
                should_assign_param_idx(idx) for idx in range(len(param_group[PARAMS]))
            )

        # Collects and stores the model parameters assigned to this rank.
        with torch.no_grad():
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
            prepare_update_param_buffers(param_group[PARAMS], self._shard_group_size)
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

    def _get_composable_block_id_rank(self) -> int | None:
        """For lossless shampoo distributor, it's unnecessary to include rank in block id."""
        return None

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
        """Helper function to get the assigned params (or gradients) from the param_group.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param.
        Returns:
            local (Iterable[Tensor | None]): assigned params (or grad) from the param_group.
        """
        if get_grad:
            # NOTE: getting grads at every optimizer step triggers implicit all-gather.
            full_grads = (
                None if p.grad is None else p.grad.full_tensor()
                for p in self._param_group[PARAMS]
            )
            return (
                full_grad
                for full_grad, assigned in zip(
                    full_grads, self._assigned_params_mask, strict=True
                )
                if assigned and (full_grad is None or full_grad.numel() > 0)
            )
        else:
            return (p for p in self._assigned_full_params if p.numel() > 0)

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
            redistribute_and_update_params(
                self._param_group[PARAMS],
                self._assigned_full_params,
                self._update_param_buffers,  # type: ignore
                self._shard_dist_group,
            )

        # Copy the updated full parameters to the original parameters.
        elif self._param_assignment_strategy == FSDPParamAssignmentStrategy.REPLICATE:
            local_params = list(
                filter(lambda p: p.numel() > 0, self._param_group[PARAMS])
            )
            full_param_slices = [
                # When param assignment strategy is REPLICATE, explicitly set `src_data_rank` to None to avoid
                # triggering communication and simply use the local copy of replicated parameters.
                dtensor.distribute_tensor(
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
    def _construct_local_block_info_list(
        self, group_source_ranks: tuple[int, ...], group_rank: int
    ) -> tuple[DTensorBlockInfo, ...]:
        """Construct the local block info list."""

        return self._construct_local_block_info_list_with_params(
            (
                param
                for assigned, param in zip(
                    self._assigned_params_mask, self._param_group[PARAMS], strict=True
                )
                if assigned and param.numel() > 0
            ),
            group_source_ranks,
            group_rank,
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
        3. _global_masked_blocked_params - HybridShardDistributor's reference to global blocked params
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
        # Update _local_blocked_params which is set via compress_list in HybridShardDistributor.__init__
        self._local_blocked_params = compress_list(
            self._global_blocked_params, self._distributor_selector
        )
        self._local_masked_blocked_params = self._local_blocked_params
        # Update _global_masked_blocked_params which is set in HybridShardDistributor.__init__
        # This is used by HybridShardDistributor.update_params() for all-gather communication
        self._global_masked_blocked_params = self._global_blocked_params

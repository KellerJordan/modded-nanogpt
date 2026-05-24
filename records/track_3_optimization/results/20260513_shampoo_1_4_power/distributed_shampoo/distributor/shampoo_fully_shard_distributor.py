"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Iterable
from typing import Literal, overload

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_distributor import Distributor
from distributed_shampoo.shampoo_types import PARAMS
from torch import distributed as dist, Tensor
from torch.distributed.tensor import DTensor


class FullyShardDistributor(Distributor):
    """FullyShard Distributor class.

    Handles merging and blocking of the tensor blocks at instantiation, and the gradients at each iteration.
    Note that parameters for module wrapped by `fully_shard` are represented as DTensors, sharded at dim-0:
    https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor.
    No communication is performed in FullyShard Distributor.

    """

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
        """Helper function to get the local params (or grad) from the param_group, where params are represented as DTensors.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param.

        Returns:
            local (Iterable[Tensor | None]): Local params (or grad) from the param_group.
        """
        # If a parameter is in a "dead layer", it won't have any gradient. In this case, we
        # should return `None` for the gradient.
        return (
            (None if p.grad is None else p.grad.to_local()) if get_grad else local_p
            for p in self._param_group[PARAMS]
            if (local_p := p.to_local()).numel() > 0
        )

    @torch.no_grad()
    def _construct_local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Construct the local block info list.

        This method creates a list of BlockInfo objects for DTensor parameters, which contain information
        about each parameter block, including its composable block IDs and the original DTensor parameter
        it belongs to. The BlockInfo objects are used throughout the optimizer to track and manage
        parameter blocks in the fully sharded distributed training setup.

        Returns:
            block_info_list (tuple[BlockInfo, ...]): A tuple of BlockInfo objects for each parameter block.
        """
        return self._construct_local_block_info_list_with_params(
            # Call `super()._get_param_or_grads()` instead of `self._get_param_or_grads()` because the need to to record the original DTensor parameter in the BlockInfo.
            params=filter(
                lambda p: isinstance(p, DTensor) and p.to_local().numel() > 0,
                super()._get_params_or_grads(),
            ),
            rank=dist.get_rank(),
        )

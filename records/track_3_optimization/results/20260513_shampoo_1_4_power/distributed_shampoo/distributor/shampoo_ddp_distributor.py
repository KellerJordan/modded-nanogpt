"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Iterable
from functools import partial
from itertools import islice
from typing import Any, Literal, overload

import torch
import torch.distributed as dist
from distributed_shampoo.distributor.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.distributor.shampoo_dist_utils import (
    get_device_mesh,
    shampoo_comm_profiler,
)
from distributed_shampoo.distributor.shampoo_distributor import DistributorInterface
from distributed_shampoo.shampoo_types import (
    DDPDistributedConfig,
    DISTRIBUTED_CONFIG,
    PARAMS,
    ShampooRuntimeConfig,
)
from distributed_shampoo.utils.commons import batched
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    distribute_buffer_sizes,
    generate_pairwise_indices,
    get_dtype_size,
)
from torch import Tensor
from torch.distributed import tensor as dtensor
from torch.distributed.tensor import DTensor, zeros as dtensor_zeros

logger: logging.Logger = logging.getLogger(__name__)


"""
The following is a visualization of how DDP Distributor computing parallelism functions.
Users specify num_trainers_per_group, which is the number of workers within a model replication group to manage computing parallelism.
Note that num_trainers_per_group must be a divisor of the total number of workers.

Assuming we have 4 GPUs within a model replication group, each GPU holds identical model parameters,
and there are 8 computationally expensive tasks for computing search directions or weight updates associated with the model.
In practice, tasks may not be evenly distributed among workers.

One approach is to let each worker perform these tasks independently without any subsequent communication.
This occurs when num_trainers_per_group = 1.

## num_trainers_per_group = 1

      GPU 0           GPU 1           GPU 2           GPU 3
     -------         -------         -------         -------
    | W0 W1 |       | W0 W1 |       | W0 W1 |       | W0 W1 |
    | W2 W3 |       | W2 W3 |       | W2 W3 |       | W2 W3 |
    | W4 W5 |       | W4 W5 |       | W4 W5 |       | W4 W5 |
    | W6 W7 |       | W6 W7 |       | W6 W7 |       | W6 W7 |
     -------         -------         -------         -------



Alternatively, we could partition these 8 tasks evenly among the 4 GPUs,
allowing each GPU to compute its own partition.

## num_trainers_per_group = -1 or 4

  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  +   GPU 0           GPU 1           GPU 2           GPU 3   +
  +  -------         -------         -------         -------  +
  + | W0 W1 |       |       |       |       |       |       | +
  + |       |       | W2 W3 |       |       |       |       | +
  + |       |       |       |       | W4 W5 |       |       | +
  + |       |       |       |       |       |       | W6 W7 | +
  +  -------         -------         -------         -------  +
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  After all-gather, each GPU will have all 8 computed search directions or weight updates:
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  +   GPU 0           GPU 1           GPU 2           GPU 3   +
  +  -------         -------         -------         -------  +
  + | W0 W1 |       | W0 W1 |       | W0 W1 |       | W0 W1 | +
  + | W2 W3 |       | W2 W3 |       | W2 W3 |       | W2 W3 | +
  + | W4 W5 |       | W4 W5 |       | W4 W5 |       | W4 W5 | +
  + | W6 W7 |       | W6 W7 |       | W6 W7 |       | W6 W7 | +
  +  -------         -------         -------         -------  +
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



We could also find a middle ground by splitting the 4 GPUs into two independent groups,
each completing their own 8 search directions or weight updates independently.
Here, GPU 0 and 1 form one group, and GPU 2 and 3 form another group.
For GPU 0 and 1, each needs to compute 4 search directions or weight updates, and similarly for GPU 2 and 3.

## num_trainers_per_group = 2

  +++++++++++++++++++++++++++++   +++++++++++++++++++++++++++++
  +   GPU 0           GPU 1   +   +   GPU 2           GPU 3   +
  +  -------         -------  +   +  -------         -------  +
  + | W0 W1 |       |       | +   + | W0 W1 |       |       | +
  + | W2 W3 |       |       | +   + | W2 W3 |       |       | +
  + |       |       | W4 W5 | +   + |       |       | W4 W5 | +
  + |       |       | W6 W7 | +   + |       |       | W6 W7 | +
  +  -------         -------  +   +  -------         -------  +
  +++++++++++++++++++++++++++++   +++++++++++++++++++++++++++++


  Within each group, once the computations are complete, each group performs its own synchronizations independently.
  This approach balances computational and communication costs.

  After all-gather, the result is:
  +++++++++++++++++++++++++++++   +++++++++++++++++++++++++++++
  +   GPU 0           GPU 1   +   +   GPU 2           GPU 3   +
  +  -------         -------  +   +  -------         -------  +
  + | W0 W1 |       | W0 W1 | +   + | W0 W1 |       | W0 W1 | +
  + | W2 W3 |       | W2 W3 | +   + | W2 W3 |       | W2 W3 | +
  + | W4 W5 |       | W4 W5 | +   + | W4 W5 |       | W4 W5 | +
  + | W6 W7 |       | W6 W7 | +   + | W6 W7 |       | W6 W7 | +
  +  -------         -------  +   +  -------         -------  +
  +++++++++++++++++++++++++++++   +++++++++++++++++++++++++++++

"""


class DDPDistributor(DistributorInterface):
    """DDP Distributor class.

    Handles merging, blocking, and distributing of the parameters at instantiation.
    The constructor internally sets up process groups, so torch.distributed must be initialized in advance.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        runtime_config (ShampooRuntimeConfig): Runtime configurations for the distributor, e.g., debugging, pt2 compile options.
        dist_group (dist.ProcessGroup | None): Optional process group for distributed operations. (Default: dist.distributed_c10d.GroupMember.WORLD)

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
        dist_group: dist.ProcessGroup | None = dist.distributed_c10d.GroupMember.WORLD,
    ) -> None:
        super().__init__(param_group, runtime_config)
        distributed_config: DDPDistributedConfig = param_group[DISTRIBUTED_CONFIG]

        # Construct global masked blocked parameters (which is DDP-specific).
        self._global_masked_blocked_params: tuple[Tensor, ...] = (
            self._global_blocked_params
        )

        # Check num_trainers_per_group and get global and group sizes.
        # NOTE: If num_trainers_per_group = -1, then we use the global world size.
        self._global_size: int = dist.get_world_size(group=dist_group)

        if distributed_config.num_trainers_per_group == -1:
            logger.info(
                f"Note that {distributed_config.num_trainers_per_group=}! Defaulting to world size {self._global_size}."
            )
        self._group_size: int = (
            distributed_config.num_trainers_per_group
            if distributed_config.num_trainers_per_group != -1
            else self._global_size
        )

        # Create flag for distributing parameters instead of search directions.
        self._communicate_params: bool = distributed_config.communicate_params

        # Initialize _dist_group and _group_rank.
        self._dist_group: dist.ProcessGroup = dist.new_subgroups(
            group_size=self._group_size, group=dist_group
        )[0]
        group_rank: int = dist.get_rank(group=self._dist_group)

        # blocked_params created on meta device with communication dtype (no actual data).
        blocked_params = tuple(
            block.to(device="meta", dtype=distributed_config.communication_dtype)
            for block in self._global_blocked_params
        )

        buffer_size_ranks = distribute_buffer_sizes(
            blocked_params=blocked_params,
            group_size=self._group_size,
            load_balancing_config=distributed_config.load_balancing_config,
        )

        self._local_block_info_list: tuple[DTensorBlockInfo, ...] = (
            self._construct_local_block_info_list(
                group_source_ranks=tuple(
                    group_source_rank for _, group_source_rank in buffer_size_ranks
                ),
                group_rank=group_rank,
            )
        )
        # Initialize selectors and local blocked (masked) parameters.
        self._distributor_selector: tuple[bool, ...] = tuple(
            group_source_rank == group_rank
            for _, group_source_rank in buffer_size_ranks
        )
        self._local_blocked_params: tuple[Tensor, ...] = compress_list(
            self._global_blocked_params, self._distributor_selector
        )
        self._local_masked_blocked_params: tuple[Tensor, ...] = (
            self._local_blocked_params
        )
        self._local_grad_selector: tuple[bool, ...] = (True,) * len(
            self._local_blocked_params
        )

        self._construct_distributed_buffers(
            buffer_size_ranks=buffer_size_ranks,
            communication_dtype=distributed_config.communication_dtype,
            group_rank=group_rank,
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

        Overrides the base DistributorInterface implementation to handle DTensor
        params/grads produced by the composable replicate() API. When using the
        old DistributedDataParallel, params/grads are plain Tensors and are
        returned as-is. When using the new replicate() (built on FSDP
        infrastructure), params/grads are DTensors and need .to_local().

        NOTE: Unlike old DDP which guarantees gradients on all params, replicate()
        only registers backward hooks on params that participate in the forward
        output. Unused params will have param.grad = None, hence the None check.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param. (Default: False)

        Returns:
            local (Iterable[Tensor | None]): Local params (or gradients) from the param_group.
              Gradients can be None for unused parameters under replicate().
        """
        if get_grad:
            return (
                param.grad.to_local()
                if isinstance(param.grad, DTensor)
                # Can be None: replicate() only registers backward hooks on
                # params used in forward, unlike old DDP.
                else param.grad
                for param in self._param_group[PARAMS]
            )
        else:
            return (
                param.to_local() if isinstance(param, DTensor) else param
                for param in self._param_group[PARAMS]
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

        # NOTE: Remove @torch.compiler.disable once PT2 supports all_gather with functional collective
        @torch.compiler.disable
        def all_gather_into_tensor() -> None:
            with shampoo_comm_profiler(
                f"{self.__class__.__name__}::all_gather_into_tensor"
            ):
                dist.all_gather_into_tensor(
                    output_tensor=self._global_dist_buffer,
                    input_tensor=self._local_dist_buffer,
                    group=self._dist_group,
                )

        # Select target params and buffers based on flag
        if use_masked_tensors:
            local_params = self._local_masked_blocked_params
            local_buffers = self._local_masked_dist_blocked_buffers
            global_params = self._global_masked_blocked_params
            global_buffers = self._global_masked_dist_blocked_buffers
        else:
            local_params = self._local_blocked_params
            local_buffers = self._local_dist_blocked_buffers
            global_params = self._global_blocked_params
            global_buffers = self._global_dist_blocked_buffers

        if self._communicate_params:
            assert len(local_params) == len(blocked_search_directions), (
                f"Expected {len(local_params)=} to be equal to {len(blocked_search_directions)=}."
            )

            # torch._foreach only accepts non-empty list
            if blocked_search_directions:
                # Perform your update to your local masked parameters and copy into buffers.
                torch._foreach_add_(
                    local_params,
                    blocked_search_directions,
                )
                torch._foreach_copy_(
                    local_buffers,
                    local_params,
                )

            all_gather_into_tensor()

            # torch._foreach only accepts non-empty list
            if global_params:
                # Copy updated blocked params in global_masked_dist_blocked_buffers into global_masked_blocked_params.
                torch._foreach_copy_(
                    global_params,
                    global_buffers,
                )

        else:
            assert len(local_buffers) == len(blocked_search_directions), (
                f"Expected {len(local_buffers)=} to be equal to {len(blocked_search_directions)=}."
            )

            # torch._foreach only accepts non-empty list
            if blocked_search_directions:
                # Search directions multiplied by alpha are distributed.
                # Copy the local search directions to the communication buffer.
                torch._foreach_copy_(
                    local_buffers,
                    blocked_search_directions,
                )

            all_gather_into_tensor()

            # torch._foreach only accepts non-empty list
            if global_params:
                # Add search directions in global_masked_dist_blocked_buffers to global_masked_blocked_params.
                torch._foreach_add_(
                    global_params,
                    global_buffers,
                )

    @torch.no_grad()
    def _construct_local_block_info_list(
        self, group_source_ranks: tuple[int, ...], group_rank: int
    ) -> tuple[DTensorBlockInfo, ...]:
        """Construct the local block info list.

        This method creates a list of DTensorBlockInfo objects, which contain information about each parameter block,
        including its composable block IDs, functions to allocate tensors, and a method to retrieve tensors.

        Args:
            group_source_ranks (tuple[int, ...]): A list of assigned ranks for each block.
            group_rank (int): Rank of the current process group.

        Returns:
            block_info_list (tuple[DTensorBlockInfo, ...]): A tuple of DTensorBlockInfo objects for each parameter block.
        """
        return tuple(
            DTensorBlockInfo(
                param=param,
                composable_block_ids=self._construct_composable_block_ids(
                    param_index=param_index, block_index=block_index
                ),
                # Curry a function to capture a local variable "group_source_rank".
                allocate_zeros_tensor=partial(
                    self._allocate_zeros_distributed_tensor,
                    group_source_rank=group_source_rank,
                ),
            )
            for (
                (param_index, param),
                (buffer_size_ranks_start, buffer_size_ranks_end),
            ) in zip(
                enumerate(self._param_group[PARAMS]),
                generate_pairwise_indices(self._global_num_blocks_per_param),
                strict=True,
            )
            for block_index, group_source_rank in enumerate(
                islice(
                    group_source_ranks, buffer_size_ranks_start, buffer_size_ranks_end
                )
            )
            if group_source_rank == group_rank
        )

    @staticmethod
    def _split_local_dist_buffers(
        buffer_size_ranks: tuple[tuple[int, int], ...],
        local_dist_buffers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Split distributed buffers for each local rank into views for each assigned block.

        Args:
            buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            local_dist_buffers (tuple[torch.Tensor, ...]): A list of local distributed buffers that
                correspond to each rank. Each distributed buffer will be split according to the
                assigned tensor blocks.

        Returns:
            splitted_local_dist_buffers (tuple[torch.Tensor, ...]): A list of tuples containing a view of the
                local distributed buffer for each tensor block.

        Example:
            tensor0 = tensor(512)
            tensor1 = tensor(512)
            buffer_size_ranks = [(128, 0), (64, 0), (512, 1), (256, 0)]
            local_dist_buffers = [tensor0, tensor1]
            -> splitted_local_dist_buffers = [
                tensor0's view(  0-128 bytes),
                tensor0's view(128-192 bytes),
                tensor1's view(  0-512 bytes),
                tensor0's view(192-448 bytes),
            ]

        """

        # Create list of lists containing local views of each split tensor for each rank.
        split_tensors_list = []
        for rank, local_dist_buffer in enumerate(local_dist_buffers):
            required_buffer_sizes = [s for s, r in buffer_size_ranks if r == rank]
            remainder_size = local_dist_buffer.size(0) - sum(required_buffer_sizes)
            assert remainder_size >= 0, (
                f"Local distributed buffer size {local_dist_buffer.size(0)} is "
                f"not larger than or equal to the sum of buffer sizes {sum(required_buffer_sizes)}!"
            )
            split_tensors = torch.split(
                local_dist_buffer, required_buffer_sizes + [remainder_size]
            )
            split_tensors_list.append(split_tensors)

        split_tensors_iterators = list(map(iter, split_tensors_list))
        return tuple(
            next(split_tensors_iterators[rank]) for _, rank in buffer_size_ranks
        )

    def _construct_distributed_buffers(
        self,
        buffer_size_ranks: tuple[tuple[int, int], ...],
        communication_dtype: torch.dtype,
        group_rank: int,
    ) -> None:
        """Construct the distributed buffers for AllGather communications.

        Note that this function will construct the distributed buffer for the AllGather
        communication. In addition, it massages the distributed buffer to obtain views
        of the buffer corresponding to each block assigned to the current rank.

        Args:
            buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            communication_dtype (torch.dtype): Data type used for communication.
            group_rank (int): Rank of the current process group.

        """

        # Calculate buffer size each rank needs.
        local_buffer_sizes = tuple(
            sum(buffer_size for buffer_size, rank in buffer_size_ranks if rank == i)
            for i in range(self._group_size)
        )

        # Calculate the whole buffer size and obtain buffers for every rank.
        max_buffer_size_sum = max(local_buffer_sizes)
        total_buffer_size = max_buffer_size_sum * self._group_size
        self._global_dist_buffer = torch.zeros(
            total_buffer_size,
            dtype=torch.int8,
            device=self._global_blocked_params[0].device,
        )
        local_dist_buffers = torch.split(self._global_dist_buffer, max_buffer_size_sum)
        splitted_local_dist_buffers = DDPDistributor._split_local_dist_buffers(
            buffer_size_ranks, local_dist_buffers
        )

        # Get local buffer for specific group rank.
        self._local_dist_buffer = local_dist_buffers[group_rank]

        # Obtain the list of buffers corresponding to each block (ignoring padding).
        # Note that each buffer is reshaped into the block's shape and viewed in terms
        # of the communication data type.
        self._global_dist_blocked_buffers = tuple(
            buffer.split(blocked_param.numel() * get_dtype_size(communication_dtype))[0]
            .view(communication_dtype)
            .view(blocked_param.shape)
            for buffer, blocked_param in zip(
                splitted_local_dist_buffers, self._global_blocked_params, strict=True
            )
        )
        self._local_dist_blocked_buffers = compress_list(
            self._global_dist_blocked_buffers, self._distributor_selector
        )
        self._global_masked_dist_blocked_buffers = self._global_dist_blocked_buffers
        self._local_masked_dist_blocked_buffers = self._local_dist_blocked_buffers

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

            # Re-compress DDP-specific tensor lists using the updated selector.
            self._global_masked_blocked_params = compress_list(
                self._global_blocked_params, self._global_grad_selector
            )
            self._global_masked_dist_blocked_buffers = compress_list(
                self._global_dist_blocked_buffers, self._global_grad_selector
            )
            self._local_masked_dist_blocked_buffers = compress_list(
                self._local_dist_blocked_buffers, self._local_grad_selector
            )

        return local_masked_blocked_grads

    def _allocate_zeros_distributed_tensor(
        self,
        size: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group_source_rank: int,
    ) -> torch.Tensor:
        """Instantiates distributed tensor using DTensor.

        Args:
            size (tuple[int, ...]): Shape of desired tensor.
            dtype (torch.dtype): DType of desired tensor.
            device (torch.device): Device of desired tensor.
            group_source_rank (int): Desired source rank of allocated zeros tensor within the process group.

        Returns:
            out (Tensor): Desired DTensor.

        """
        ranks_in_group = dist.get_process_group_ranks(group=self._dist_group)
        device_mesh_2d = get_device_mesh(
            device_type=device.type,
            mesh=tuple(batched(iterable=ranks_in_group, n=self._group_size)),
            mesh_dim_names=("replicate", "shard"),
        )
        replicate_submesh = device_mesh_2d._get_all_submeshes(  # type: ignore[attr-defined]
            mesh_dim_name="replicate"
        )[group_source_rank]

        return dtensor_zeros(
            size,
            dtype=dtype,
            device_mesh=replicate_submesh,
            placements=[dtensor.Replicate()],
        )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from functools import partial
from itertools import islice
from math import prod
from typing import Any

import torch
from distributed_shampoo.distributor.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.distributor.shampoo_dist_utils import get_device_mesh
from distributed_shampoo.distributor.shampoo_distributor import DistributorInterface
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParameterMetadata,
    HSDPDistributedConfig,
    LoadBalancingConfig,
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    ShampooRuntimeConfig,
)
from distributed_shampoo.utils.commons import batched
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    distribute_buffer_sizes,
    generate_pairwise_indices,
    get_dtype_size,
    merge_small_dims,
    multi_dim_split,
)
from torch import distributed as dist, Tensor
from torch.distributed import tensor as dtensor
from torch.distributed.tensor import zeros as dtensor_zeros
from torch.nn import Parameter

logger: logging.Logger = logging.getLogger(__name__)


class HSDPDistributor(DistributorInterface):
    """HSDP Distributor class.

    Handles split tensor block recovery of different parameters, then merging and blocking of
    the tensor blocks, as well as distributing of the parameters at instantiation.

    The constructor internally sets up `DeviceMesh` objects as necessary for distributing memory
    and computation, so torch.distributed must be initialized in advance.

    Unlike FSDPDistributor, HSDPDistributor requires the user to pass in a device mesh specifying the model level parallelism used for HSDP.
    For example, suppose we have 48 GPUs and the HSDP group size is 8. Then:

    HSDP Device Mesh with (Replicate, Shard) = (6, 8); note that the Replicate and Shard here is referring to the model level parallelism:

        device_mesh = [[ 0,  1,  2,  3,  4,  5,  6,  7]
                       [ 8,  9, 10, 11, 12, 13, 14, 15]
                       [16, 17, 18, 19, 20, 21, 22, 23]
                       [24, 25, 26, 27, 28, 29, 30, 31]
                       [32, 33, 34, 35, 36, 37, 38, 39]
                       [40, 41, 42, 43, 44, 45, 46, 47]]

    For example, if my device is rank 11, then:
        device_mesh["replicate"] = [3, 11, 19, 27, 35, 43]
        device_mesh["shard"] = [8, 9, 10, 11, 12, 13, 14, 15]

    Since the parameters are sharded along the "shard" dimension, we would normally replicate the
    computation along the "replicate" dimension. With HSDP Shampoo, we instead want to distribute
    the computation and memory requirements across the "replicate" dimension of the original HSDP
    device mesh.

    For example, suppose that the num_trainers_per_group = 3. We want to form a (2, 3)-submesh on
    the ranks [3, 11, 19, 27, 35, 43] (and similar).

    HSDPDistributor 2D Sub-Mesh Example with (Replicate, Shard) = (2, 3); note that the Replicate and Shard here is referring to the optimizer level computing parallelism:

        submesh = [[ 3, 11, 19]
                   [27, 35, 43]]

    In this case, optimizer states will live on different "replicate" meshes: {[3, 27], [11, 35],
    [19, 43]}. In order to synchronize the optimizer step, we will communicate along the "shard"
    mesh {[3, 11, 19], [27, 35, 43]}.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        runtime_config (ShampooRuntimeConfig): Runtime configurations for the distributor, e.g., debugging, pt2 compile options.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        distributed_config: HSDPDistributedConfig = param_group[DISTRIBUTED_CONFIG]
        self._param_to_metadata: dict[Parameter, FSDPParameterMetadata] = (
            distributed_config.param_to_metadata
        )
        self._hsdp_device_mesh: torch.distributed.device_mesh.DeviceMesh = (
            distributed_config.device_mesh
        )
        self._global_num_splits_per_param: tuple[int, ...] = ()
        self._global_num_blocks_per_split_param: tuple[int, ...] = ()

        super().__init__(param_group, runtime_config)
        if not dist.is_initialized():
            raise RuntimeError(
                "HSDPDistributor needs torch.distributed to be initialized!"
            )

        # Construct global masked blocked parameters.
        self._global_masked_blocked_params: tuple[Tensor, ...] = (
            self._global_blocked_params
        )

        # Check num_trainers_per_group and replicated group size.
        # NOTE: If num_trainers_per_group = -1, then we use the replicated group size.
        self._replicated_group_size: int = self._hsdp_device_mesh.size(0)

        if not (
            1
            <= distributed_config.num_trainers_per_group
            <= self._replicated_group_size
            or distributed_config.num_trainers_per_group == -1
        ):
            raise ValueError(
                f"Invalid number of trainers per group: {distributed_config.num_trainers_per_group}. "
                f"Must be between [1, {self._replicated_group_size}] or set to -1."
            )
        if distributed_config.num_trainers_per_group == -1:
            logger.info(
                f"Note that {distributed_config.num_trainers_per_group=}! Defaulting to replicated group size {self._replicated_group_size}."
            )
        elif (
            not self._replicated_group_size % distributed_config.num_trainers_per_group
            == 0
        ):
            raise ValueError(
                f"{distributed_config.num_trainers_per_group=} must divide {self._replicated_group_size=}!"
            )

        # Group size for distributing computation / memory requirements.
        self._dist_group_size: int = (
            distributed_config.num_trainers_per_group
            if distributed_config.num_trainers_per_group != -1
            else self._replicated_group_size
        )

        # Create flag for distributing parameters instead of search directions.
        self._communicate_params: bool = distributed_config.communicate_params

        # Initialize _dist_group and _group_rank.
        # Note that this requires initializing all process groups.
        # Splits replicated ranks group into smaller groups of size self._dist_group_size.
        # Instantiates this by using DeviceMesh.
        ranks_in_all_replicated_groups = self._hsdp_device_mesh.mesh.T
        for ranks_in_replicated_group in ranks_in_all_replicated_groups:
            device_mesh = get_device_mesh(
                device_type=self._hsdp_device_mesh.device_type,
                mesh=tuple(
                    map(
                        partial(tuple),
                        ranks_in_replicated_group.view(
                            -1, self._dist_group_size
                        ).tolist(),
                    )
                ),
                mesh_dim_names=("replicate", "shard"),
            )
            if dist.get_rank() in ranks_in_replicated_group:
                # NOTE: We want the process group in the device mesh that the current rank
                # belongs to but solely along the "shard" dimension for communications.
                #
                # For example, if the current rank is 11, then I want the process group
                # that contains the ranks [3, 11, 19].
                self._comms_dist_group: dist.ProcessGroup = device_mesh.get_group(
                    "shard"
                )

        comms_group_rank: int = dist.get_rank(self._comms_dist_group)

        # blocked_params created on meta device with communication dtype (no actual data).
        blocked_params = tuple(
            block.to(device="meta", dtype=distributed_config.communication_dtype)
            for block in self._global_blocked_params
        )

        buffer_size_ranks = distribute_buffer_sizes(
            blocked_params=blocked_params,
            group_size=self._dist_group_size,
            load_balancing_config=LoadBalancingConfig(),
        )

        self._local_block_info_list: tuple[DTensorBlockInfo, ...] = (
            self._construct_local_block_info_list(
                group_source_ranks=tuple(
                    group_source_rank for _, group_source_rank in buffer_size_ranks
                ),
                group_rank=comms_group_rank,
            )
        )
        # Initialize selectors and local blocked (masked) parameters.
        self._distributor_selector: tuple[bool, ...] = tuple(
            group_source_rank == comms_group_rank
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
            comms_group_rank=comms_group_rank,
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

        # NOTE: Remove this function once PT2 supports all_gather with functional collective
        @torch.compiler.disable
        def all_gather_into_tensor() -> None:
            dist.all_gather_into_tensor(
                output_tensor=self._global_dist_buffer,
                input_tensor=self._local_dist_buffer,
                group=self._comms_dist_group,
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
        # Note that for HSDP, we want to get the rank within each sharded group for the block id.
        # When using a device mesh, 0 corresponds to the replicated group and 1 corresponds to the sharded group.
        sharded_group_rank = self._hsdp_device_mesh.get_local_rank(1)
        return tuple(
            DTensorBlockInfo(
                param=param,
                composable_block_ids=self._construct_composable_block_ids(
                    param_index=param_index,
                    block_index=block_index,
                    rank=sharded_group_rank,
                ),
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

    def _merge_and_block_parameters(self) -> None:
        """Split, merge, and block parameters."""
        global_blocked_params: list[Tensor] = []
        # self._global_num_splits_per_param refers to the total number of splits within each
        # flattened parameter (obtained by split tensor block recovery).
        # This has the same length as the number of flattened parameters contained in
        # self._param_group[PARAMS].
        global_num_splits_per_param = []
        # self._global_num_blocks_per_split refers to the total number of blocks within each
        # split parameter.
        # This has the same length as the number of split parameters.
        global_num_blocks_per_split_param = []

        for flattened_param in self._param_group[PARAMS]:
            # Split flattened parameters into valid tensor blocks of the parameter.
            split_params = HSDPDistributor._split_tensor_block_recovery(
                flattened_param,
                self._param_to_metadata[flattened_param].shape,
                self._param_to_metadata[flattened_param].start_idx,
                self._param_to_metadata[flattened_param].end_idx,
            )
            global_num_splits_per_param.append(len(split_params))

            (blocked_params, num_blocks_per_split_param) = (
                self._merge_and_block_with_params(params=split_params)
            )
            global_blocked_params.extend(blocked_params)
            global_num_blocks_per_split_param.extend(num_blocks_per_split_param)

        # Check that the number of blocks for each parameter equals to the summation of the number of blocks
        # from each split parameter.
        self._global_num_blocks_per_param = tuple(
            sum(global_num_blocks_per_split_param[block_index:next_block_index])
            for (block_index, next_block_index) in generate_pairwise_indices(
                global_num_splits_per_param
            )
        )

        # Set lists as tuples.
        self._global_blocked_params = tuple(global_blocked_params)
        self._global_num_splits_per_param = tuple(global_num_splits_per_param)
        self._global_num_blocks_per_split_param = tuple(
            global_num_blocks_per_split_param
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
        comms_group_rank: int,
    ) -> None:
        """Construct the distributed buffers for AllGather communications.

        Note that this function will construct the distributed buffer for the AllGather
        communication. In addition, it massages the distributed buffer to obtain views
        of the buffer corresponding to each block assigned to the current rank.

        Args:
            buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            communication_dtype (torch.dtype): The data type used for communication.
            comms_group_rank (int): The rank of the current group within the comms group.

        """

        # Calculate buffer size each rank needs.
        local_buffer_sizes = tuple(
            sum(buffer_size for buffer_size, rank in buffer_size_ranks if rank == i)
            for i in range(self._dist_group_size)
        )

        # Calculate the whole buffer size and obtain buffers for every rank.
        max_buffer_size_sum = max(local_buffer_sizes)
        total_buffer_size = max_buffer_size_sum * self._dist_group_size
        self._global_dist_buffer = torch.zeros(
            total_buffer_size,
            dtype=torch.int8,
            device=self._global_blocked_params[0].device,
        )
        local_dist_buffers = torch.split(self._global_dist_buffer, max_buffer_size_sum)
        splitted_local_dist_buffers = HSDPDistributor._split_local_dist_buffers(
            buffer_size_ranks, local_dist_buffers
        )

        # Get local buffer for specific group rank.
        self._local_dist_buffer = local_dist_buffers[comms_group_rank]

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

    def _merge_and_block_gradients(
        self,
    ) -> tuple[Tensor, ...]:
        """Split, merge, and block gradients.

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

        for (
            flattened_param,
            num_blocks,
            (block_index, next_block_index),
            (split_index, next_split_index),
        ) in zip(
            self._param_group[PARAMS],
            self._global_num_blocks_per_param,
            generate_pairwise_indices(self._global_num_blocks_per_param),
            generate_pairwise_indices(self._global_num_splits_per_param),
            strict=True,
        ):
            flattened_grad = flattened_param.grad
            param_distributor_selector = self._distributor_selector[
                block_index:next_block_index
            ]

            # Note: Gradients that are None or empty (grad.numel() == 0) still belong to a block
            # with corresponding block_info, but are filtered out for all updates.
            is_invalid_grad = flattened_grad is None or flattened_grad.numel() == 0
            # Update the selector.
            global_grad_selector.extend([not is_invalid_grad] * num_blocks)

            if is_invalid_grad or not any(param_distributor_selector):
                # Skip split_tensor_block_recovery and multi_dim_split if this blocked grad will not be used locally.
                continue

            assert flattened_grad is not None

            if self._runtime_config.eager_nan_check:
                assert torch.isfinite(flattened_grad).all(), (
                    f"Encountered gradient containing NaN/Inf in parameter with shape {flattened_grad.shape}. Check your model for numerical instability or consider gradient clipping."
                )

            # Split flattened gradients into valid tensor blocks of the gradient.
            split_grads = HSDPDistributor._split_tensor_block_recovery(
                flattened_grad,
                self._param_to_metadata[flattened_param].shape,
                self._param_to_metadata[flattened_param].start_idx,
                self._param_to_metadata[flattened_param].end_idx,
            )

            # Get the number of blocks for each split gradient.
            num_blocks_within_split_grads = self._global_num_blocks_per_split_param[
                split_index:next_split_index
            ]

            for grad, (
                blocks_within_split_index,
                next_blocks_within_split_index,
            ) in zip(
                split_grads,
                generate_pairwise_indices(num_blocks_within_split_grads),
                strict=True,
            ):
                # Obtain blocks for each split gradient after merging.
                blocks_within_grad = multi_dim_split(
                    grad.view(merge_dims(tensor_shape=grad.size())),
                    self._param_group[MAX_PRECONDITIONER_DIM],
                )
                # Generate block-to-parameter metadata and extend blocked parameters list.
                local_masked_blocked_grads.extend(
                    compress_list(
                        blocks_within_grad,
                        param_distributor_selector[
                            blocks_within_split_index:next_blocks_within_split_index
                        ],
                    )
                )

        # Set global grad selector as tuple.
        self._global_grad_selector = tuple(global_grad_selector)

        return tuple(local_masked_blocked_grads)

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

            # Re-compress tensor lists using the updated selector.
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

    @staticmethod
    def _split_tensor_block_recovery(
        tensor_shard: Tensor,
        original_shape: torch.Size,
        start_idx: int,
        end_idx: int,
    ) -> list[Tensor]:
        """Chunks flattened tensor in order to re-construct valid blocks with respect to the original
        multi-dimensional tensor shape and parameter boundaries.

        Starting from the first dimension, the largest possible slices in each dimension
        (with the remaining dimensions on the right retaining the original shape) are split off.

        The following is an example of how the function works for a 2-D tensor shard:

        Given an original tensor with shape (7, 14) in Fig. 1, we receive a flattened tensor shard from HSDP
        corresponding to Fig. 4. Note that this flattened tensor shard corresponds to the shard of the tensor
        in Fig. 2. In order to respect the tensor shape, we need to split the tensor into up to three blocks
        (as in Fig. 5). This requires splitting the tensor in Fig. 2 (see flattened tensor shard in Fig. 4)
        then reshaping each flattened split tensor into its original shape (see reshaped split tensors in Fig.
        3 and 6).

          ______________
         |       _______|                        _______                         _______
         |______|       |                 ______|       |                 ______|_______|
         |              |       ->       |              |       ->       |              |
         |           ___|                |           ___|                |______________|
         |__________|   |                |__________|                    |__________|
         |______________|

          original tensor                  tensor_shard                    split tensors

              Fig. 1                           Fig. 2                          Fig. 3

        Flattened original tensor in Fig. 1:
         ________________________________________________________________
        |____________________|_________________________|_________________|
                             ^       tensor_shard      ^
                          start_idx                 end_idx

                                    Fig. 4

         ________________________________________________________________
        |____________________|______|_______________|__|_________________|
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    ^^^ denoted the flattened split tensors in Fig. 3.

                                    Fig. 5

        Reshaped split tensors (i.e., the tensors in Fig. 3):
                                     _______
                                    |_______|  <- left split
                             ______________
                            |              |  <- center split
                            |______________|
                             __________
                            |__________|      <- right split

                                    Fig. 6

        Args:
            tensor_shard (Tensor): A shard of the flattened version of original tensor to split.
            original_shape (torch.Size): Shape of original tensor that tensor_shard is a slice of.
            start_idx (int): Flattened index in the original tensor where tensor starts (inclusive).
            end_idx (int): Flattened index in the original tensor where tensor ends (exclusive).

        Returns:
            split_tensors (list[Tensor]): List of tensors.

        """
        if len(tensor_shard.size()) != 1:
            raise ValueError(
                f"Input tensor is not flat, has shape {tensor_shard.size()=}."
            )

        def block_within_tensor_shard_recovery(
            block_within_tensor_shard: Tensor,
            dimension: int,
            block_start_idx: int,
            block_end_idx: int,
        ) -> list[Tensor]:
            assert (
                block_end_idx - block_start_idx == block_within_tensor_shard.numel()
            ), (
                f"Start/end indices do not match tensor size: {block_start_idx=}, "
                f"{block_end_idx=}, {block_within_tensor_shard.numel()=}!"
            )

            if block_end_idx == block_start_idx:
                return []

            # Handle case where shape is one-dimensional.
            # Because it reached the last dimension, we can simply return the flattened tensor.
            if dimension == len(original_shape) - 1:
                return [block_within_tensor_shard]

            # Instantiate list of tensor blocks.
            center_split_tensor_blocks = []

            # Instantiates flattened indices for recursion.
            remaining_size = prod(original_shape[dimension + 1 :])

            """
             ________________________________________________________________
            |____________________|______|_______________|__|_________________|
                                 ^      ^               ^  ^
                       block_start_idx  |               | block_end_idx
                                        |               |
                            center_split_start_idx      |
                                                center_split_end_idx

            This came from Fig. 4 above.

            """
            # Get starting index of the center split of the tensor shard. (See figure above.)
            # This is equal to ceil(block_start_idx / remaining_size) * remaining_size.
            center_split_start_idx = (
                (block_start_idx + remaining_size - 1) // remaining_size
            ) * remaining_size
            # Similarly, get end index of the center split of the tensor shard.
            # This is equal to floor(block_end_idx / remaining_size) * remaining_size.
            center_split_end_idx = block_end_idx // remaining_size * remaining_size

            # Handles largest convex partition in the center.
            if center_split_start_idx < center_split_end_idx:
                center_split_start_idx_in_block = (
                    center_split_start_idx - block_start_idx
                )
                length_of_center_split = center_split_end_idx - center_split_start_idx
                new_shape = [-1] + list(original_shape[dimension + 1 :])
                # NOTE: We use Tensor.narrow() instead of slicing in order to guarantee
                # there is no copy of the tensor.
                center_split_tensor_blocks.append(
                    block_within_tensor_shard.narrow(
                        0,
                        center_split_start_idx_in_block,
                        length_of_center_split,
                    ).view(new_shape)
                )
            elif center_split_start_idx > center_split_end_idx:
                # Recursively call split tensor block recovery on the full
                # flattened tensor ignoring the first dimension of the original
                # tensor shape.
                return block_within_tensor_shard_recovery(
                    block_within_tensor_shard=block_within_tensor_shard,
                    dimension=dimension + 1,
                    block_start_idx=block_start_idx,
                    block_end_idx=block_end_idx,
                )

            # Recursively call split tensor block recovery on the left and right
            # splits of the flattened tensor.
            left_split_start_idx_in_block = 0
            left_split_tensor_size = center_split_start_idx - block_start_idx
            left_split_tensor_blocks = block_within_tensor_shard_recovery(
                block_within_tensor_shard=block_within_tensor_shard.narrow(
                    0,
                    start=left_split_start_idx_in_block,
                    length=left_split_tensor_size,
                ),
                dimension=dimension + 1,
                block_start_idx=block_start_idx,
                block_end_idx=center_split_start_idx,
            )

            center_split_end_idx_in_block = center_split_end_idx - block_start_idx
            right_split_tensor_size = block_end_idx - center_split_end_idx
            right_split_tensor_blocks = block_within_tensor_shard_recovery(
                block_within_tensor_shard=block_within_tensor_shard.narrow(
                    0,
                    start=center_split_end_idx_in_block,
                    length=right_split_tensor_size,
                ),
                dimension=dimension + 1,
                block_start_idx=center_split_end_idx,
                block_end_idx=block_end_idx,
            )

            return (
                left_split_tensor_blocks
                + center_split_tensor_blocks
                + right_split_tensor_blocks
            )

        return block_within_tensor_shard_recovery(
            block_within_tensor_shard=tensor_shard,
            dimension=0,
            block_start_idx=start_idx,
            block_end_idx=end_idx,
        )

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
            group_source_rank (int): Group rank (with respect to the sharded group of
                the 2D submesh) that determines which ranks the DTensor is allocated on.

        Returns:
            out (Tensor): Desired Tensor.

        """
        ranks_in_replicated_group = dist.get_process_group_ranks(
            self._hsdp_device_mesh.get_group(0)
        )
        device_mesh_2d = get_device_mesh(
            device_type=device.type,
            # NOTE: Use itertools.batched(ranks_in_replicated_group, self._dist_group_size) when downstream applications are Python 3.12+ available
            mesh=tuple(batched(ranks_in_replicated_group, self._dist_group_size)),
            mesh_dim_names=("replicate", "shard"),
        )
        # NOTE: We get all submeshes along the "replicate" dimension, then pick out
        # the sub-mesh that the optimizer state is assigned to.
        #
        # For the example above, this would give me submeshes [[3, 27], [11, 35], [19, 43]].
        # Note that the group source rank must belong to {0, 1, 2} in this case.
        # Suppose the group_source_rank = 1, then this would get the submesh [11, 35].
        replicate_submesh = device_mesh_2d._get_all_submeshes(  # type: ignore[attr-defined]
            mesh_dim_name="replicate"
        )[group_source_rank]

        return dtensor_zeros(
            size,
            dtype=dtype,
            device_mesh=replicate_submesh,
            placements=[dtensor.Replicate()],
        )

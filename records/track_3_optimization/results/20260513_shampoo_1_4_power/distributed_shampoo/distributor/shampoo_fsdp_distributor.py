"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from functools import partial
from math import prod
from typing import Any

import torch
import torch.distributed as dist
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_distributor import Distributor
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPDistributedConfig,
    FSDPParameterMetadata,
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
from torch.nn import Parameter


class FSDPDistributor(Distributor):
    """FSDP Distributor class.

    Handles split tensor block recovery of different parameters, then merging and blocking of
    the tensor blocks at instantiation, and the gradients at each iteration.
    Note that no communication is performed in FSDP Distributor.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        runtime_config (ShampooRuntimeConfig): Runtime configurations for the distributor, e.g., debugging, pt2 compile options.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        runtime_config: ShampooRuntimeConfig | None = None,
    ) -> None:
        distributed_config: FSDPDistributedConfig = param_group[DISTRIBUTED_CONFIG]
        self._param_to_metadata: dict[Parameter, FSDPParameterMetadata] = (
            distributed_config.param_to_metadata
        )
        self._global_num_splits_per_param: tuple[int, ...] = ()
        self._global_num_blocks_per_split_param: tuple[int, ...] = ()

        super().__init__(param_group, runtime_config)

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
            split_params = FSDPDistributor._split_tensor_block_recovery(
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
            split_grads = FSDPDistributor._split_tensor_block_recovery(
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

        Given an original tensor with shape (7, 14) in Fig. 1, we receive a flattened tensor shard from FSDP
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

    @torch.no_grad()
    def _construct_local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Construct the local block info list with rank information.

        This method overrides the parent class to include rank information in the block IDs,
        which is necessary for FSDP to properly identify blocks across different ranks.

        Returns:
            block_info_list (tuple[BlockInfo, ...]): A tuple of BlockInfo objects for each parameter block.
        """
        return self._construct_local_block_info_list_with_params(
            params=self._get_params_or_grads(), rank=dist.get_rank()
        )

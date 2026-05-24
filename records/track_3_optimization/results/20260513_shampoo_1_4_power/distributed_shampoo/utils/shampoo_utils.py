"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import heapq
import logging
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from functools import cache, partial, reduce
from itertools import accumulate, chain, compress, islice, pairwise
from types import TracebackType
from typing import Any, TypeVar

import torch
from distributed_shampoo.shampoo_types import LoadBalancingConfig
from distributed_shampoo.utils.load_balancing_utils import DefaultCostModel
from torch import distributed as dist, Tensor
from torch.distributed.tensor import DTensor

logger: logging.Logger = logging.getLogger(__name__)


@cache
def merge_small_dims(
    tensor_shape: tuple[int, ...],
    threshold: int,
    target_tensor_dimensionality: int | float,
) -> tuple[int, ...]:
    """Reshapes tensor by merging small dimensions.

    This function merges adjacent dimensions of a tensor when their product is below
    the specified threshold, which helps optimize operations on tensors with many
    small dimensions.

    Note:
    - Shampoo will promote 0D tensor (torch.Size([]) into an 1D tensor (torch.Size([1])).
    - Empty tensors (with a dimension of size 0) will return a shape of (0,).
    - Dimensions of size 1 are removed (squeezed) before merging.
    - If all dimensions are 1, it returns (1,).
    - Dimensions are merged in reverse order to accommodate PyTorch's tensor layout.

    Args:
        tensor_shape (tuple[int, ...]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.
        target_tensor_dimensionality (int | float): Target dimensionality of the tensor. Only merge until the target dimensionality is reached.
            If target_tensor_dimensionality > len(tensor_shape), then no merging occurs. The only float that can be used is math.inf.
            Note that the output tensor will NOT necessarily have dimension equal to target_tensor_dimensionality.
            The merging will stop before reaching target_tensor_dimensionality if the threshold is small.

    Returns:
        new_tensor_shape (tuple[int, ...]): New tensor shape after merging dimensions.

    Raises:
        AssertionError: If target_tensor_dimensionality is a float but not math.inf.

    Example:
        - merge_small_dims((1, 2, 5, 1), threshold=10, target_tensor_dimensionality=1) -> (10,)
          All dimensions are merged as their product (10) is equal to the threshold.

        - merge_small_dims((1, 2, 5, 1), threshold=1, target_tensor_dimensionality=1) -> (2, 5)
          Dimensions of size 1 are removed, and no merging occurs as 2*5 > threshold.

        - merge_small_dims((32, 3, 64, 64), threshold=8192, target_tensor_dimensionality=1) -> (96, 4096)
          For convolution-like dimensions, merges into (32*3, 64*64) as 96 < threshold
          but 96*4096 > threshold.

        - merge_small_dims((32, 3, 64, 64), threshold=1_000_000, target_tensor_dimensionality=2) -> (32, 12_288)
          For convolution-like dimensions, merges into (32, 3*64*64) despite 32*3*64*64 < threshold because
          target_tensor_dimensionality is 2. This is useful for spectral descent methods like Muon.

    """
    if 0 in tensor_shape:
        return (0,)

    if isinstance(target_tensor_dimensionality, float):
        assert target_tensor_dimensionality == math.inf, (
            f"{target_tensor_dimensionality=} has to be an integer or math.inf."
        )
        return tensor_shape

    # Squeeze tensor shape to remove dimension with 1; if all dimensions are 1,
    # then add a 1 to the tensor shape.
    # We merge dimensions in reverse order to accommodate PyTorch's general tensor layout.
    # This is particularly useful for convolution operations where kernel sizes are typically
    # placed at the end of the tensor shape.
    squeezed_tensor_shape = list(filter(lambda t: t != 1, reversed(tensor_shape))) or [
        1
    ]
    squeezed_dimensionality = len(squeezed_tensor_shape)
    new_tensor_shape = [squeezed_tensor_shape[0]]
    for num_processed_dimensions, next_tensor_shape in enumerate(
        islice(squeezed_tensor_shape, 1, None), start=1
    ):
        current_dimensionality = len(new_tensor_shape)
        remaining_dimensions = squeezed_dimensionality - num_processed_dimensions
        potential_dimensionality_before_merge = (
            current_dimensionality + remaining_dimensions
        )
        if (
            potential_dimensionality_before_merge > target_tensor_dimensionality
            and (new_dimension := new_tensor_shape[-1] * next_tensor_shape) <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)
    return tuple(reversed(new_tensor_shape))


def multi_dim_split(tensor: Tensor, split_size: int | float) -> tuple[Tensor, ...]:
    """Chunks tensor across multiple dimensions based on splits.

    This function recursively splits a tensor along all of its dimensions using the
    specified split size. It applies torch.split() to each dimension sequentially,
    resulting in a tuple of smaller tensors.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        split_size (int | float): Size of a single chunk along each dimension.
            If math.inf is provided, no splitting occurs.

    Returns:
        split_tensors (tuple[Tensor, ...]): Tuple of tensors after splitting.
            If split_size is greater than or equal to any dimension size,
            no splitting occurs along that dimension.

    Example:
        - multi_dim_split(tensor of shape (5, 2), split_size=3):
          Returns (tensor([0, 1, 2], [0, 1]), tensor([3, 4], [0, 1]))
          Splits only along dimension 0 since split_size > dimension 1 size.

        - multi_dim_split(tensor of shape (5, 3), split_size=2):
          First splits along dimension 0:
          [(0-1, 0-2), (2-3, 0-2), (4, 0-2)]

          Then splits each chunk along dimension 1:
          [(0-1, 0-1), (0-1, 2), (2-3, 0-1), (2-3, 2), (4, 0-1), (4, 2)]

          Returns 6 smaller tensors.

        - multi_dim_split(tensor of shape (5, 3), split_size=5):
          Returns (original tensor,) since split_size ≥ all dimensions.

        - multi_dim_split(tensor of shape (5, 3), split_size=math.inf):
          Returns (original tensor,) since math.inf means no splitting.

    """
    if isinstance(split_size, float):
        assert split_size == math.inf, (
            f"{split_size=} has to be an integer or math.inf."
        )
        return (tensor,)

    return reduce(
        lambda split_tensors, dim: tuple(
            s for t in split_tensors for s in torch.split(t, split_size, dim=dim)
        ),
        range(tensor.dim()),
        (tensor,),
    )


_CompressListType = TypeVar("_CompressListType")


def compress_list(
    complete_list: Sequence[_CompressListType], selector: Sequence[bool]
) -> tuple[_CompressListType, ...]:
    """Compresses sequence based on selector.

    NOTE: Despite the name, this function can compress both lists and tuples, but will always return
    a tuple in order to ensure downstream compatibility.

    Args:
        complete_list (Sequence[CompressListType]): Complete tuple of candidates.
        selector (Sequence[bool]): Mask that is True if state is active, False otherwise.

    Returns:
        compressed_tuple (tuple[CompressListType, ...]): Compressed list of candidates based on selector.

    Example:
        complete_list = ['a', 'b', 'c', 'd'] and selector = [True, False, True, False]:
        Result: ('a', 'c')

        Only elements from complete_list where the corresponding selector is True are included.

    """
    assert len(complete_list) == len(selector), (
        f"Inconsistent lengths between complete_list {len(complete_list)} and selector {len(selector)}!"
    )
    return tuple(compress(complete_list, selector))


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size (bytes) of a given data type."""
    if dtype is torch.bool:
        return 1
    # Fast ceiling of bits/8 using (bits + 7) // 8
    return (
        (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits + 7
    ) // 8


def generate_pairwise_indices(input_list: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Generates accumulated pairwise indices for a given input list.

    This is useful for generating interval indices for iterating through a list given the
    number of blocks within each parameter.

    Args:
        input_list (Sequence[int]): A list of integers specifying the number of elements within each partition.

    Returns:
        partition_indices (Iterator[tuple[int, int]]): An iterator containing pairs of indices which specify
            the start and the ending indices of each partition specified in the input_list.

    Example:
        If input_list = (1, 3, 2),
            - First element (1) generates index range [0, 1)
            - Second element (3) generates index range [1, 4)
            - Third element (2) generates index range [4, 6)

        then this will output [(0, 1), (1, 4), (4, 6)].

    """
    return pairwise(accumulate(chain([0], input_list)))


_ParameterizeEnterExitContextType = TypeVar("_ParameterizeEnterExitContextType")


class ParameterizeEnterExitContext:
    """ParameterizeEnterExitContext is used for automatically invoking the enter and exit methods on the input within this context.

    Args:
        input_with_enter_exit_context (ParameterizeEnterExitContextType): Input whose state will be changed while entering and exiting the context by enter_method_caller and exit_method_caller respectively.
        enter_method_caller (Callable[[ParameterizeEnterExitContextType], Any]): Method caller for entering the context.
        exit_method_caller (Callable[[ParameterizeEnterExitContextType], Any]): Method caller for exiting the context.

    """

    def __init__(
        self,
        input_with_enter_exit_context: _ParameterizeEnterExitContextType,
        enter_method_caller: Callable[[_ParameterizeEnterExitContextType], Any],
        exit_method_caller: Callable[[_ParameterizeEnterExitContextType], Any],
    ) -> None:
        self._enter_method: Callable[[], Any] = partial(
            enter_method_caller, input_with_enter_exit_context
        )
        self._exit_method: Callable[[], Any] = partial(
            exit_method_caller, input_with_enter_exit_context
        )

    def __enter__(self) -> "ParameterizeEnterExitContext":
        self._enter_method()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._exit_method()


def distribute_buffer_sizes(
    blocked_params: tuple[Tensor, ...],
    group_size: int,
    load_balancing_config: LoadBalancingConfig,
) -> tuple[tuple[int, int], ...]:
    """Distribute given param blocks across ranks in a group.

    Param blocks are distributed such that the total assigned load of each rank is as even as possible.
    The load of a param block is determined by ``load_balance_config``.
    By default, the load is measured purely by buffer size. If ``load_balance_config`` specifies
    a compute-based strategy, the distribution will instead weigh each buffer by its estimated
    computational cost (e.g., cost of processing or kernel execution time) rather than size alone.
    This is currently performed using a greedy algorithm.

    Note: A better distribution strategy should try to minimize the delta of buffer sizes
    between the most and the least allocated groups.

    Args:
        blocked_params (tuple[Tensor, ...]): A list of blocked parameters.
        group_size (int): Number of groups to distribute across.
        load_balancing_config (LoadBalancingConfig): Memory or compute load balancing config.

    Returns:
        buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
            buffer size for each block and its assigned rank.
    """
    buffer_sizes_aligned = tuple(
        int(DefaultCostModel.cost(blocked_param)) for blocked_param in blocked_params
    )

    param_block_costs = tuple(
        load_balancing_config.cost_model.cost(block) for block in blocked_params
    )
    param_block_ranks = [-1] * len(blocked_params)
    allocated_load_sizes = [(0.0, group_index) for group_index in range(group_size)]
    heapq.heapify(allocated_load_sizes)

    for index, block_cost in sorted(
        enumerate(param_block_costs),
        key=operator.itemgetter(1),
        reverse=True,
    ):
        # Greedily find the group with the least allocated load and its group index
        # in order to allocate buffers on that group.
        (
            min_allocated_load,
            min_allocated_load_group_index,
        ) = heapq.heappop(allocated_load_sizes)

        new_load_size = min_allocated_load + block_cost

        heapq.heappush(
            allocated_load_sizes,
            (
                new_load_size,
                min_allocated_load_group_index,
            ),
        )
        param_block_ranks[index] = min_allocated_load_group_index

    buffer_size_ranks = tuple(zip(buffer_sizes_aligned, param_block_ranks, strict=True))

    return buffer_size_ranks


def prepare_update_param_buffers(
    params: tuple[DTensor, ...], group_size: int
) -> list[Tensor]:
    """Allocates a persistent shadow copy of updated parameters."""
    if any(p.dtype != params[0].dtype for p in params):
        raise NotImplementedError(
            "When using round-robin assignment in FSDP Shampoo, parameters of "
            "different dtypes are not currently supported."
        )

    param_sizes = [p.to_local().numel() for p in params]
    buffer_size = sum(param_sizes)
    buffer = params[0].to_local().new_zeros(buffer_size)
    buffer_offsets = list(accumulate(param_sizes))

    def round_up_to_multiple_of(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    pad_len = round_up_to_multiple_of(len(buffer_offsets), group_size) - len(
        buffer_offsets
    )

    # The padding logic below handles when a rank has some parameters but fewer than group size.
    # For example, if group size is 4 and there are 3 parameters, it will pad a 0-sized tensor at the end.
    # Example:
    #   Assume we have 3 parameters and group size is 4. param0: 100, param1: 200, param2: 300.
    #   buffer_offsets = [100, 300, 600, 600] (note that the last element is 600)
    #   This buffer for communication have 4 chunks.
    #   - Rank 0: [0, 100)
    #   - Rank 1: [100, 300)
    #   - Rank 2: [300, 600)
    #   - Rank 3: [600, 600) (empty tensor)
    # Pad the list with empty tensors to ensure each rank participates in all-to-all.
    buffer_offsets.extend([buffer_size] * pad_len)
    # Drop the last element as torch.tensor_split takes indices as split points.
    buffer_offsets = buffer_offsets[:-1]

    return list(torch.tensor_split(buffer, buffer_offsets))


def redistribute_and_update_params(
    params: tuple[DTensor, ...],
    local_full_params: list[Tensor],
    update_param_buffers: list[Tensor],
    dist_group: torch.distributed.ProcessGroup,
) -> None:
    """Redistributes updated parameters to each parameter's rank."""
    group_size = dist_group.size()

    # Run all-to-all collectives to exchange the updated parameters across
    # ranks in group. This implementation runs multiple rounds of a2a ops
    # if the number of parameters is larger than the world size.
    a2a_rounds = range(len(update_param_buffers) // group_size)
    logger.info(f"Running {len(a2a_rounds)} rounds of a2a ops.")
    for a2a_round in a2a_rounds:
        # Send either a valid full parameter, or a padding zero tensor.
        send_param = (
            local_full_params[a2a_round]
            if a2a_round < len(local_full_params)
            else params[0].to_local().new_zeros(0)
        )
        # Chunk the send_param to exactly group_size slices to distribute to
        # all ranks. We need to manually pad the result of torch.chunk since
        # it does not guarantee that the result has the desired chunks.
        send_list = [t.flatten() for t in torch.chunk(send_param, group_size, dim=0)]
        if len(send_list) < group_size:
            # NOTE: Intentionally use `torch.tensor_split` here to do a trivial
            # split to ensure that the padding is in contiguous memory space as
            # is required for all-to-all collectives.
            append_len = group_size - len(send_list)
            last_t = send_list[-1]
            split_indices = [send_list[-1].shape[0]] * append_len
            send_list.extend(torch.tensor_split(last_t, split_indices, dim=0)[1:])
        assert len(send_list) == group_size

        # Specify receive list as a range of update_param_buffers.
        recv_list = update_param_buffers[
            a2a_round * group_size : (a2a_round + 1) * group_size
        ]

        dist.all_to_all(recv_list, send_list, dist_group)

    torch._foreach_copy_(
        [p.to_local().flatten() for p in params], update_param_buffers[: len(params)]
    )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor


@dataclass
class BlockInfo:
    """Utilities and metadata for each parameter block.

    Attributes:
        param (Tensor): The original parameter that contains the block.
        composable_block_ids (tuple[int, str]): Tuple containing the per-parameter, per-block index tuple.
            In the DDP case, this will contain (param_index, block_index), where the param_index corresponds to
            the index of the parameter in the parameter group, and the block_index is the index of the block within
            the parameter.

            Example: If we have a model with two parameters, p1 and p2, with 2 and 3 blocks respectively, then the
                possible values of composable_block_ids are (0, "block_0"), (0, "block_1"), (1, "block_0"), (1, "block_1"),
                (1, "block_2").

                For FSDP, the block index is constructed as a string containing rank information. For example, block 0 of
                parameter p1 on rank 0 will have the composable_block_ids being (0, "rank_0-block_0"), while block 0 of parameter p1
                on rank 1 will have composable_block_ids being (0, "rank_1-block_0").
        allocate_zeros_tensor (Callable[..., Tensor]): A function that returns a zero-initialized tensor.
            This tensor must be saved in the state dictionary for checkpointing.
            This tensor might be DTensor. get_tensor() must be used to access the value.
            Its function signature is (size, dtype, device) -> Tensor.
            (Default: lambda size, dtype, device: torch.zeros(size, dtype=dtype, device=device))
        allocate_ones_tensor (Callable[..., Tensor]): A function that returns a tensor filled with ones.
            This tensor is used for operations requiring a tensor of ones and might be a DTensor.
            Its function signature is (size, dtype, device) -> Tensor.
            (Default: lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device))
        allocate_eye_tensor (Callable[..., Tensor]): A function that returns an identity matrix tensor.
            This tensor is used for operations requiring an identity matrix and might be a DTensor.
            Its function signature is (n, dtype, device) -> Tensor.
            (Default: torch.eye)
        get_tensor (Callable[..., Tensor]): A function that takes a tensor allocated by allocator and returns its local tensor.
            Its function signature is (tensor: Tensor) -> Tensor.
            (Default: lambda input_tensor: input_tensor)
    """

    param: Tensor
    composable_block_ids: tuple[int, str]
    allocate_zeros_tensor: Callable[..., Tensor] = partial(torch.zeros)
    allocate_ones_tensor: Callable[..., Tensor] = field(
        init=False, default_factory=lambda: torch.ones
    )
    allocate_eye_tensor: Callable[..., Tensor] = field(
        init=False, default_factory=lambda: torch.eye
    )
    get_tensor: Callable[..., Tensor] = field(
        init=False, default_factory=lambda: lambda input_tensor: input_tensor
    )


@dataclass
class DTensorBlockInfo(BlockInfo):
    """Utilities and metadata for each parameter block specific using DTensor.

    Attributes:
        param (Tensor): The original parameter that contains the block.
        composable_block_ids (tuple[int, str]): Tuple containing the per-parameter, per-block index tuple.
            In the DDP case, this will contain (param_index, block_index), where the param_index corresponds to
            the index of the parameter in the parameter group, and the block_index is the index of the block within
            the parameter.

            Example: If we have a model with two parameters, p1 and p2, with 2 and 3 blocks respectively, then the
                possible values of composable_block_ids are (0, "block_0"), (0, "block_1"), (1, "block_0"), (1, "block_1"),
                (1, "block_2").

                For FSDP, the block index is constructed as a string containing rank information. For example, block 0 of
                parameter p1 on rank 0 will have the composable_block_ids being (0, "rank_0-block_0"), while block 0 of parameter p1
                on rank 1 will have composable_block_ids being (0, "rank_1-block_0").
        allocate_zeros_tensor (Callable[..., Tensor]): A function that returns a zero-initialized tensor.
            This tensor must be saved in the state dictionary for checkpointing.
            This tensor might be DTensor. get_tensor() must be used to access the value.
            Its function signature is (size, dtype, device) -> Tensor.
            (Default: lambda size, dtype, device: torch.zeros(size, dtype=dtype, device=device))
        allocate_ones_tensor (Callable[..., Tensor]): A function that returns a tensor filled with ones.
            This tensor is used for operations requiring a tensor of ones and might be a DTensor.
            Its function signature is (size, dtype, device) -> Tensor.
            (Default: lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device))
        allocate_eye_tensor (Callable[..., Tensor]): A function that returns an identity matrix tensor.
            This tensor is used for operations requiring an identity matrix and might be a DTensor.
            Its function signature is (n, dtype, device) -> Tensor.
            (Default: torch.eye)
        get_tensor (Callable[..., Tensor]): A function that takes a tensor allocated by allocator and returns its local tensor.
            Its function signature is (tensor: Tensor) -> Tensor.
            (Default: lambda input_tensor: input_tensor.to_local())
    """

    get_tensor: Callable[..., Tensor] = field(
        init=False, default_factory=lambda: lambda input_tensor: input_tensor.to_local()
    )

    def __post_init__(self) -> None:
        # Due to the lack of `torch.ones`-like support for DTensor, we need to manually construct a matrix with all ones.
        def allocate_ones_tensor(
            size: tuple[int, ...],
            dtype: torch.dtype | None = None,
            device: torch.device | None = None,
        ) -> Tensor:
            self.get_tensor(
                ones := self.allocate_zeros_tensor(
                    size=size, dtype=dtype, device=device
                )
            ).fill_(1.0)
            return ones

        self.allocate_ones_tensor = allocate_ones_tensor

        # Due to the lack of `torch.eye`-like support for DTensor, we need to manually construct the identity matrix.
        def allocate_eye_tensor(
            n: int, dtype: torch.dtype | None = None, device: torch.device | None = None
        ) -> Tensor:
            self.get_tensor(
                eye := self.allocate_zeros_tensor(
                    size=(n, n), dtype=dtype, device=device
                )
            ).fill_diagonal_(1.0)
            return eye

        self.allocate_eye_tensor = allocate_eye_tensor

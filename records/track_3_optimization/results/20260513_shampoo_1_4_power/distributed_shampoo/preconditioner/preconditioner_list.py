"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import torch
from torch import Tensor
from torch.autograd import profiler


_MemberFuncReturnType = TypeVar("_MemberFuncReturnType")


def profile_decorator(
    member_func: Callable[..., _MemberFuncReturnType],
) -> Callable[..., _MemberFuncReturnType]:
    """Decorator that profiles the execution of a class method.

    This decorator wraps a class method with PyTorch's profiler.record_function
    to track its execution time and resource usage. The profiling information
    is recorded with a name that includes the class name and method name.

    Args:
        member_func (Callable[..., _MemberFuncReturnType]): The class method to be profiled.

    Returns:
        wrapper (Callable[..., _MemberFuncReturnType]): A wrapped function that profiles the execution of the original method.
    """

    @wraps(member_func)
    def wrapper(them: object, *args: Any, **kwargs: Any) -> _MemberFuncReturnType:
        with profiler.record_function(
            f"## {them.__class__.__name__}:{member_func.__name__} ##"
        ):
            return member_func(them, *args, **kwargs)

    return wrapper


class PreconditionerList(ABC):
    """Preconditioner base class.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None: ...

    @abstractmethod
    def precondition(
        self, masked_grad_list: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None: ...

    @property
    def numel_list(self) -> tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)

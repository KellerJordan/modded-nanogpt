"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Parallelism strategies for distributed training.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from distributed_shampoo import DistributedConfig
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class WrappedModel:
    """Result of wrapping a model for distributed training."""

    model: nn.Module
    distributed_config: DistributedConfig | None = None


class ParallelismStrategy(ABC):
    """Base class for parallelism strategies."""

    @property
    def requires_distributed(self) -> bool:
        return False

    @property
    def requires_device_mesh(self) -> bool:
        return False

    @abstractmethod
    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        pass


@dataclass
class SingleGPUStrategy(ParallelismStrategy):
    """Single GPU training (no wrapping)."""

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        return WrappedModel(model=model)


@dataclass
class DDPStrategy(ParallelismStrategy):
    """DDP strategy."""

    distributed_config: Callable[..., DistributedConfig] | None = None

    @property
    def requires_distributed(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        if backend == "nccl":
            wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            wrapped = DDP(model)

        config = self.distributed_config() if self.distributed_config else None
        return WrappedModel(model=wrapped, distributed_config=config)


@dataclass
class FSDPStrategy(ParallelismStrategy):
    """FSDP v1 strategy."""

    distributed_config: Callable[..., DistributedConfig] | None = None

    @property
    def requires_distributed(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        wrapped = FSDP(model, use_orig_params=True)
        config = None
        if self.distributed_config:
            config = self.distributed_config(
                param_to_metadata=compile_fsdp_parameter_metadata(wrapped)
            )
        return WrappedModel(model=wrapped, distributed_config=config)


@dataclass
class HSDPStrategy(ParallelismStrategy):
    """HSDP v1 strategy."""

    distributed_config: Callable[..., DistributedConfig] | None = None

    @property
    def requires_distributed(self) -> bool:
        return True

    @property
    def requires_device_mesh(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        assert device_mesh is not None
        wrapped = FSDP(
            model,
            device_mesh=device_mesh,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
        )
        config = None
        if self.distributed_config:
            config = self.distributed_config(
                param_to_metadata=compile_fsdp_parameter_metadata(wrapped),
                device_mesh=device_mesh,
            )
        return WrappedModel(model=wrapped, distributed_config=config)


@dataclass
class FullyShardStrategy(ParallelismStrategy):
    """FSDP v2 (fully_shard) strategy."""

    distributed_config: Callable[..., DistributedConfig] | None = None

    @property
    def requires_distributed(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        config = self.distributed_config() if self.distributed_config else None
        return WrappedModel(model=fully_shard(model), distributed_config=config)  # type: ignore[arg-type]


@dataclass
class HybridShardStrategy(ParallelismStrategy):
    """HSDP v2 (hybrid_shard) strategy."""

    distributed_config: Callable[..., DistributedConfig] | None = None

    @property
    def requires_distributed(self) -> bool:
        return True

    @property
    def requires_device_mesh(self) -> bool:
        return True

    def wrap_model(
        self,
        model: nn.Module,
        local_rank: int,
        backend: str,
        device_mesh: DeviceMesh | None = None,
    ) -> WrappedModel:
        assert device_mesh is not None
        config = None
        if self.distributed_config:
            config = self.distributed_config(device_mesh=device_mesh)
        return WrappedModel(
            model=fully_shard(model, mesh=device_mesh),  # type: ignore[arg-type]
            distributed_config=config,
        )

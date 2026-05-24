"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Hydra resolvers for complex types in YAML configs.
"""

import torch
from distributed_shampoo import FSDPParamAssignmentStrategy, WeightDecayType
from omegaconf import OmegaConf


def _torch_dtype_resolver(dtype_str: str) -> torch.dtype:
    """Resolve a string to a torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. Valid options: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str]


def _fsdp_param_strategy_resolver(strategy_str: str) -> FSDPParamAssignmentStrategy:
    """Resolve a string to FSDPParamAssignmentStrategy."""
    try:
        return FSDPParamAssignmentStrategy[strategy_str]
    except KeyError:
        valid = [s.name for s in FSDPParamAssignmentStrategy]
        raise ValueError(f"Unknown strategy: {strategy_str}. Valid options: {valid}")


def _weight_decay_type_resolver(type_str: str) -> WeightDecayType:
    """Resolve a string to WeightDecayType."""
    try:
        return WeightDecayType[type_str]
    except KeyError:
        valid = [t.name for t in WeightDecayType]
        raise ValueError(
            f"Unknown weight decay type: {type_str}. Valid options: {valid}"
        )


def register_resolvers() -> None:
    """Register all custom Hydra resolvers. Call once at startup."""
    if not OmegaConf.has_resolver("torch_dtype"):
        OmegaConf.register_new_resolver("torch_dtype", _torch_dtype_resolver)
    if not OmegaConf.has_resolver("fsdp_param_strategy"):
        OmegaConf.register_new_resolver(
            "fsdp_param_strategy", _fsdp_param_strategy_resolver
        )
    if not OmegaConf.has_resolver("weight_decay_type"):
        OmegaConf.register_new_resolver(
            "weight_decay_type", _weight_decay_type_resolver
        )

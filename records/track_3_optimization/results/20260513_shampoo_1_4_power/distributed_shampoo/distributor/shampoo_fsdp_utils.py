"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
from collections.abc import Callable, Iterable

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT


def compile_fsdp_parameter_metadata(
    module: torch.nn.Module,
) -> dict[torch.nn.Parameter, FSDPParameterMetadata]:
    """Compiles parameter metadata necessary for FSDP Shampoo.

    Args:
        module (nn.Module): Module to compile metadata for.

    Returns:
        param_metadata (dict[torch.nn.Parameter, FSDPParameterMetadata]): Dictionary mapping each parameter to its FSDP metadata.

    """
    param_metadata: dict[torch.nn.Parameter, FSDPParameterMetadata] = {}

    for fsdp_module in FSDP.fsdp_modules(module):
        if (flat_param := fsdp_module._flat_param) is None:
            continue

        fqns = flat_param._fqns
        shapes = flat_param._shapes
        numels = flat_param._numels
        shard_param_infos = flat_param._shard_param_infos
        sharding_strategy = fsdp_module.sharding_strategy

        assert flat_param._params is not None, (
            "flat_param._params should not be None! Set the value of `use_orig_params` in FSDP module to True "
        )
        "would populate flat_param._params."
        params = flat_param._params

        param_metadata |= {
            param: FSDPParameterMetadata(
                fqn=fqn,
                shape=shape,
                numel=numel,
                start_idx=shard_param_info.intra_param_start_idx or 0,
                end_idx=(
                    shard_param_info.intra_param_end_idx + 1
                    if shard_param_info.intra_param_end_idx is not None
                    else 0
                ),
                sharding_strategy=sharding_strategy,
            )
            for param, fqn, shape, numel, shard_param_info in zip(
                params,
                fqns,
                shapes,
                numels,
                shard_param_infos,
                strict=True,
            )
        }

    return param_metadata


def _partition_params(
    params: ParamsT,
    fsdp_criteria: Callable[[torch.Tensor], bool],
    hsdp_criteria: Callable[[torch.Tensor], bool],
) -> tuple[ParamsT, ParamsT, ParamsT]:
    """Partitions parameters into FSDP, HSDP, and other parameters.

    NOTE: The output partitions are guaranteed to cover all input `params`.

    Args:
        params (ParamsT): Iterable of parameters, parameter groups, or dicts/tuples of parameters.
        fsdp_criteria (Callable[[torch.Tensor], bool]): Function to determine if a parameter is FSDP.
        hsdp_criteria (Callable[[torch.Tensor], bool]): Function to determine if a parameter is HSDP.
    Returns:
        fsdp_params (ParamsT): Partition of FSDP parameters in the same format as input.
        hsdp_params (ParamsT): Partition of HSDP parameters in the same format as input.
        other_params (ParamsT): Partition of non-FSDP and non-HSDP parameters in the same format as input.

    """

    # Create a tuple of criteria functions for FSDP, HSDP, and other parameters
    fsdp_hsdp_other_criterions: tuple[Callable[[torch.Tensor], bool], ...] = (
        fsdp_criteria,
        hsdp_criteria,
        lambda param: not fsdp_criteria(param) and not hsdp_criteria(param),
    )

    params_to_peek, original_params = itertools.tee(iter(params))
    peek_param = next(params_to_peek)

    def partition_param_list(
        original_params: Iterable[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Partitions a list of parameters into FSDP, HSDP, and other parameters.

        Args:
            original_params (Iterable[torch.Tensor]): Iterable of parameters to partition.

        Returns:
            fsdp_params (list[torch.Tensor]): Partition of FSDP parameters in a list.
            hsdp_params (list[torch.Tensor]): Partition of HSDP parameters in a list.
            other_params (list[torch.Tensor]): Partition of non-FSDP and non-HSDP parameters in a list.
        """
        original_params = list(original_params)
        fsdp_params, hsdp_params, other_params = (
            list(filter(lambda p: criteria(p), original_params))
            for criteria in fsdp_hsdp_other_criterions
        )
        return fsdp_params, hsdp_params, other_params

    match peek_param:
        # Case 1: The original params is a Iterable[torch.Tensor]
        case torch.Tensor():
            return partition_param_list(
                original_params=original_params  # type: ignore
            )
        # Case 2: The original params is a Iterable[dict[str, torch.Tensor]]
        case dict():
            fsdp_params_list, hsdp_params_list, other_params_list = [], [], []
            for cur_dict in original_params:
                cur_fsdp_params, cur_hsdp_params, cur_other_params = (
                    partition_param_list(
                        original_params=cur_dict[
                            "params"  # type: ignore
                        ]
                    )
                )
                fsdp_params_list.append({"params": cur_fsdp_params})
                hsdp_params_list.append({"params": cur_hsdp_params})
                other_params_list.append({"params": cur_other_params})
            return fsdp_params_list, hsdp_params_list, other_params_list

        # Case 3: The original params is a Iterable[tuple[str, torch.Tensor]]
        case tuple():
            original_params_dict: dict[str, torch.Tensor] = dict(
                original_params  # type: ignore
            )
            fsdp_params_dict, hsdp_params_dict, other_params_dict = (
                {k: v for k, v in original_params_dict.items() if criteria(v)}
                for criteria in fsdp_hsdp_other_criterions
            )

            assert (
                unioned_keys := fsdp_params_dict.keys()
                | hsdp_params_dict.keys()
                | other_params_dict.keys()
            ) == original_params_dict.keys(), (
                f"{unioned_keys - original_params_dict.keys()=} {original_params_dict.keys() - unioned_keys=}"
            )
            for (name1, dict1), (name2, dict2) in itertools.combinations(
                (
                    ("fsdp_params_dict", fsdp_params_dict),
                    ("hsdp_params_dict", hsdp_params_dict),
                    ("other_params_dict", other_params_dict),
                ),
                2,
            ):
                assert not (common_keys := dict1.keys() & dict2.keys()), (
                    f"{common_keys} exist in both {name1} and {name2}!"
                )

            return (
                list(fsdp_params_dict.items()),
                list(hsdp_params_dict.items()),
                list(other_params_dict.items()),
            )
        # Default case for unsupported types
        case _:
            raise ValueError(
                f"Unsupported params type: {type(peek_param)}. "
                "Please use a list of parameters, parameter groups, or tuples of named parameters."
            )


def parse_fsdp_params(
    params: ParamsT,
    param_metadata: dict[torch.nn.Parameter, FSDPParameterMetadata],
) -> tuple[ParamsT, ParamsT, ParamsT]:
    """Splits parameters into FSDP, HSDP, and the rest of parameters.

    This is useful for parsing the parameters when FSDP and HSDP are wrapping a subset of modules within a model.

    NOTE: The output partitions are guaranteed to cover all input `params`.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts/tuples of parameters.
        param_metadata (dict[torch.nn.Parameter, FSDPParameterMetadata]): Dictionary mapping each parameter to its FSDP metadata.

    Returns:
        fsdp_params (ParamsT): Partition of FSDP parameters in the same format as input.
        hsdp_params (ParamsT): Partition of HSDP parameters in the same format as input.
        other_params (ParamsT): Partition of non-FSDP and non-HSDP parameters in the same format as input.
    """
    return _partition_params(
        params=params,
        fsdp_criteria=lambda param: isinstance(param, torch.nn.Parameter)
        and param in param_metadata
        and param_metadata[param].sharding_strategy
        in [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
        hsdp_criteria=lambda param: isinstance(param, torch.nn.Parameter)
        and param in param_metadata
        and param_metadata[param].sharding_strategy
        in [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2],
        # The implied other_criteria is that param not in param_metadata or
        # param_metadata[param].sharding_strategy == ShardingStrategy.NO_SHARD.
    )


def parse_fully_shard_params(
    params: ParamsT,
) -> tuple[ParamsT, ParamsT, ParamsT]:
    """Splits parameters into fully shard, hybrid shard, and other parameters.

    This is useful for parsing the parameters when fully shard or hybrid shard are wrapping a subset of modules within a model.

    NOTE: The output partitions are guaranteed to cover all input `params`.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts/tuples of parameters.

    Returns:
        fully_shard_params (ParamsT): Partition of fully shard parameters in the same format as input.
        hybrid_shard_params (ParamsT): Partition of hybrid shard parameters in the same format as input.
        other_params (ParamsT): Partition of parameters that do not use FSDP or HSDP in the same format as input.
    """
    return _partition_params(
        params=params,
        fsdp_criteria=lambda param: isinstance(param, DTensor)
        and len(param.device_mesh.shape) == 1,
        hsdp_criteria=lambda param: isinstance(param, DTensor)
        and len(param.device_mesh.shape) == 2,
        # The implied other_criteria is that the parameter is not a DTensor.
    )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest
from collections.abc import Callable, Iterator

import torch
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
    parse_fsdp_params,
    parse_fully_shard_params,
)
from distributed_shampoo.shampoo_types import FSDPParameterMetadata
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from torch import distributed as dist, nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parameter import Parameter
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


# Note: Ideally this function should be resided inside Test as part of setUp() but FSDPTest
#       only calls setUp() on one device; as a result, every device has to call this function
#       separately.
def _create_model_and_params(
    model_linear_layers_dims: tuple[int, ...] = (2, 5, 3),
) -> tuple[nn.Module, list[Parameter]]:
    model, _, _, _ = construct_training_problem(
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layers_dims=None,
        enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
        fill=(1.0, 2.0),
    )
    return model, list(model.parameters())


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class CompileFSDPParameterMetadataTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_compile_fsdp_parameter_metadata(self) -> None:
        model, params = _create_model_and_params()
        fsdp_model = FSDP(model, use_orig_params=True)
        actual_fsdp_parameter_metadata = compile_fsdp_parameter_metadata(fsdp_model)

        expected_fsdp_parameter_metadata = (
            {
                params[0]: FSDPParameterMetadata(
                    fqn="linear_layers.0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=10,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="linear_layers.1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=0,
                    end_idx=2,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
            }
            if dist.get_rank() == 0
            else {
                params[0]: FSDPParameterMetadata(
                    fqn="linear_layers.0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=0,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="linear_layers.1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=2,
                    end_idx=15,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
            }
        )

        self.assertEqual(
            actual_fsdp_parameter_metadata, expected_fsdp_parameter_metadata
        )

    @skip_if_lt_x_gpu(2)
    def test_compile_fsdp_parameter_metadata_with_no_flat_param(self) -> None:
        model, params = _create_model_and_params()
        # Ignored all params in FSDP so there is no flat_param field in FSDP module.
        fsdp_model = FSDP(model, use_orig_params=True, ignored_states=params)
        actual_fsdp_parameter_metadata = compile_fsdp_parameter_metadata(fsdp_model)

        expected_fsdp_parameter_metadata: dict[Parameter, FSDPParameterMetadata] = {}

        self.assertEqual(
            actual_fsdp_parameter_metadata, expected_fsdp_parameter_metadata
        )


@instantiate_parametrized_tests
@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ParseFSDPParamsTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @parametrize(
        "model_to_params, named_params_to_expected_parsed_params",
        (
            # Case 1: model.parameters() - returns iterator of parameters
            (
                lambda model: model.parameters(),
                lambda named_parameters, filter_condition: [
                    param for name, param in named_parameters if filter_condition(name)
                ],
            ),
            # Case 2: optimizer-style param groups - returns list with dict containing params
            (
                lambda model: [{"params": model.parameters()}],
                lambda named_parameters, filter_condition: [
                    {
                        "params": [
                            param
                            for name, param in named_parameters
                            if filter_condition(name)
                        ]
                    }
                ],
            ),
            # Case 3: model.named_parameters() - returns iterator of (name, param) tuples
            (
                lambda model: model.named_parameters(),
                lambda named_parameters, filter_condition: [
                    (name, param)
                    for name, param in named_parameters
                    if filter_condition(name)
                ],
            ),
        ),
    )
    def test_parse_fsdp_params(
        self,
        model_to_params: Callable[[nn.Module], Iterator[tuple[str, Parameter]]],
        named_params_to_expected_parsed_params: Callable[
            [Iterator[tuple[str, Parameter]], Callable[[str], bool]], list[object]
        ],
    ) -> None:
        # Create modules with different sharding strategies
        # FULL_SHARD strategy - parameters fully sharded across devices
        fsdp_module_0 = FSDP(
            _create_model_and_params()[0],
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
        # SHARD_GRAD_OP strategy - only gradients and optimizer states are sharded
        fsdp_module_1 = FSDP(
            _create_model_and_params()[0],
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            use_orig_params=True,
        )
        # HYBRID_SHARD strategy - uses 2D mesh for sharding
        hsdp_module_2 = FSDP(
            _create_model_and_params()[0],
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=init_device_mesh("cuda", (2, 2)),  # 2x2 device mesh
            use_orig_params=True,
        )
        # _HYBRID_SHARD_ZERO2 strategy - another hybrid sharding approach
        hsdp_module_3 = FSDP(
            _create_model_and_params()[0],
            sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
            device_mesh=init_device_mesh("cuda", (2, 2)),
            use_orig_params=True,
        )
        # NO_SHARD strategy
        other_module_4 = FSDP(
            _create_model_and_params()[0],
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=True,
        )
        # Regular nn.Linear module - not FSDP
        other_module_5 = nn.Linear(3, 2, bias=False)

        model = nn.Sequential(
            fsdp_module_0,
            fsdp_module_1,
            hsdp_module_2,
            hsdp_module_3,
            other_module_4,
            other_module_5,
        )

        fsdp_parameter_metadata = compile_fsdp_parameter_metadata(module=model)

        # Parse parameters into three categories based on sharding strategy
        actual_fsdp_params, actual_hsdp_params, actual_other_params = parse_fsdp_params(
            params=model_to_params(model), param_metadata=fsdp_parameter_metadata
        )

        # Create expected parameter lists for each category
        # FSDP params: modules 0 and 1
        expected_fsdp_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("0.") or name.startswith("1."),
        )
        # HSDP params: modules 2 and 3
        expected_hsdp_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("2.") or name.startswith("3."),
        )
        # Other params: modules 4 and 5
        expected_other_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("4.") or name.startswith("5."),
        )

        self.assertEqual(actual_fsdp_params, expected_fsdp_params)
        self.assertEqual(actual_hsdp_params, expected_hsdp_params)
        self.assertEqual(actual_other_params, expected_other_params)

    @skip_if_lt_x_gpu(4)
    @parametrize(
        "model_to_params, named_params_to_expected_parsed_params",
        (
            # Case 1: model.parameters() - returns iterator of parameters
            (
                lambda model: model.parameters(),
                lambda named_parameters, filter_condition: [
                    param for name, param in named_parameters if filter_condition(name)
                ],
            ),
            # Case 2: optimizer-style param groups - returns list with dict containing params
            (
                lambda model: [{"params": model.parameters()}],
                lambda named_parameters, filter_condition: [
                    {
                        "params": [
                            param
                            for name, param in named_parameters
                            if filter_condition(name)
                        ]
                    }
                ],
            ),
            # Case 3: model.named_parameters() - returns iterator of (name, param) tuples
            (
                lambda model: model.named_parameters(),
                lambda named_parameters, filter_condition: [
                    (name, param)
                    for name, param in named_parameters
                    if filter_condition(name)
                ],
            ),
        ),
    )
    def test_parse_fully_shard_params(
        self,
        model_to_params: Callable[[nn.Module], Iterator[tuple[str, Parameter]]],
        named_params_to_expected_parsed_params: Callable[
            [Iterator[tuple[str, Parameter]], Callable[[str], bool]], list[object]
        ],
    ) -> None:
        # Create 1D mesh for fully sharded module
        mesh_1d = init_device_mesh("cuda", (self.world_size,))
        fully_shard_module, _ = _create_model_and_params((16, 8, 1))
        fully_shard(fully_shard_module, mesh=mesh_1d)

        # Create 2D mesh for hybrid sharded module
        mesh_2d = init_device_mesh(
            "cuda",
            (2, self.world_size // 2),  # 2 x 2 mesh for 4 GPUs
            mesh_dim_names=("dp_replicate", "dp_shard"),  # Named dimensions
        )
        hybrid_shard_module, _ = _create_model_and_params((16, 8, 1))
        fully_shard(hybrid_shard_module, mesh=mesh_2d)

        # Combine modules into a sequential model
        model = nn.Sequential(
            fully_shard_module, hybrid_shard_module, nn.Linear(3, 2, bias=False)
        )

        # Parse parameters into three categories based on sharding type
        actual_fully_shard_params, actual_hybrid_shard_params, actual_other_params = (
            parse_fully_shard_params(params=model_to_params(model))
        )

        # Create expected parameter lists for each category
        # Fully sharded params: module 0
        expected_fully_shard_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("0."),
        )
        # Hybrid sharded params: module 1
        expected_hybrid_shard_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("1."),
        )
        # Other params: module 2
        expected_other_params = named_params_to_expected_parsed_params(
            model.named_parameters(),
            lambda name: name.startswith("2."),
        )

        self.assertEqual(actual_fully_shard_params, expected_fully_shard_params)
        self.assertEqual(actual_hybrid_shard_params, expected_hybrid_shard_params)
        self.assertEqual(actual_other_params, expected_other_params)

    def test_parse_fully_shard_params_invalid_paramsT(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Please use a list of parameters, parameter groups, or tuples of named parameters."
            ),
            parse_fully_shard_params,
            [0, 1],
        )

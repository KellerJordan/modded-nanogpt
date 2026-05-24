"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from collections.abc import Callable
from functools import partial
from itertools import filterfalse
from typing import cast, overload

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.gpu_tests.distributor_test_utils import (
    DistributorOnEmptyParamTest,
)
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_fully_shard_distributor import (
    FullyShardDistributor,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    FullyShardDistributedConfig,
    GeneralizedPrimalAveragingConfig,
    HybridShardDistributedConfig,
    SingleDeviceDistributedConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)
from torch import distributed as dist, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

PRECONDITIONER_DIM = 4


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class ShampooFullyShardDistributorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @overload
    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]: ...

    @overload
    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[
            [nn.Module], FSDPModule
        ] = lambda x: fully_shard(x),
    ) -> tuple[FSDPModule, nn.Module, torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[
            [nn.Module], nn.Module | FSDPModule
        ] = lambda x: x,
    ) -> tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]:
        # NOTE: We construct the model here specifically in order to ensure that
        #       FullyShard Shampoo and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FullyShard will split the
        #       parameters such that the preconditioners created between the FullyShard
        #       and default Shampoo are equivalent.
        #      +----------------+
        #      |     [4, 16]    |
        #      |      GPU0      |
        #     --------------------     +------+
        #      |     [4, 16]    |      |[4, 4]|
        #      |      GPU1      |      |      |
        #      +----------------+      +------+
        #      For the first linear layer, each GPU has a [4, 16] parameter. The blocked
        #      parameters are of size [4, 4] and each GPU has four local blocks (eight
        #      blocks in total). In comparison, with default shampoo, the eight blocks
        #      are replicated on two GPUs.
        #      Similarly, the second linear layer has a [1, 8] parameter and is split
        #      into two [4] chunks.

        model_linear_layers_dims = (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1)
        # model dead layers won't participate in the training and thus don't have grads.
        model_dead_layers_dims = (PRECONDITIONER_DIM, 1)
        # Using partial here to prevent Pyre complain on incompatible parameter type.
        return partial(
            construct_training_problem, post_model_decoration=post_model_decoration
        )(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.1,
        )

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: FullyShardDistributedConfig | SingleDeviceDistributedConfig,
    ) -> Callable[[ParamsT], torch.optim.Optimizer]:
        return partial(
            DistributedShampoo,
            lr=0.001,
            betas=(0.9, 1.0),
            epsilon=1e-8,
            weight_decay=0.0,
            max_preconditioner_dim=PRECONDITIONER_DIM,
            precondition_frequency=1,
            start_preconditioning_step=2,
            weight_decay_type=WeightDecayType.DECOUPLED,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
            distributed_config=distributed_config,
            iterate_averaging_config=GeneralizedPrimalAveragingConfig(),
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_all_ranks_with_no_grads(self) -> None:
        fully_shard_config = FullyShardDistributedConfig()

        steps_without_gradients = 2
        with unittest.mock.patch.object(torch.Tensor, "backward") as mock_backward:
            # By mocking the backward() method, we're intercepting gradient calculation.
            # This effectively simulates running forward passes without computing gradients.
            train_model(
                optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                    distributed_config=fully_shard_config,
                ),
                model_factory=partial(
                    ShampooFullyShardDistributorTest._construct_model,
                    post_model_decoration=partial(fully_shard),
                ),
                num_steps=steps_without_gradients,
            )

        # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
        self.assertEqual(mock_backward.call_count, steps_without_gradients)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_shampoo_against_default_shampoo(self) -> None:
        fully_shard_config = FullyShardDistributedConfig()
        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=DefaultSingleDeviceDistributedConfig,
            ),
            control_model_factory=ShampooFullyShardDistributorTest._construct_model,
            experimental_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config,
            ),
            experimental_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard),
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize("communicate_params", (True, False))
    def test_hybrid_shard_distributed_config_against_fully_shard_distributed_config_bitwise_identical(
        self, communicate_params: bool
    ) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("replicate", "shard")
        )
        fully_shard_config = FullyShardDistributedConfig()
        hybrid_shard_config = HybridShardDistributedConfig(
            device_mesh=mesh_2d, communicate_params=communicate_params
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config
            ),
            control_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
            experimental_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config
            ),
            experimental_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
            total_steps=100,
            rtol=0.0,
            atol=0.0,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_shampoo_block_index(self) -> None:
        model, _, _, _, optimizer = train_model(
            optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=FullyShardDistributedConfig()
            ),
            model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard),
            ),
        )
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, DistributedShampoo)
        osd_state = optimizer.state_dict()["state"]
        # We don't care about the parameter index, and we just want to get
        # the keys of each inner state of the parameter.
        flatten_keys = [
            key for inner_dict in osd_state.values() for key in inner_dict.keys()
        ]

        rank: int = dist.get_rank()

        def expected_key_criterion(key: str) -> bool:
            return f"rank_{rank}-block_" in key

        # Verify the only keys that are not the fqn of a block are "step",
        # "train_mode", and "lr_sum".
        block_keys, non_block_keys = (
            list(filter(expected_key_criterion, flatten_keys)),
            list(filterfalse(expected_key_criterion, flatten_keys)),
        )
        filtered_flatten_keys = [
            key for key in flatten_keys if key not in {"step", "train_mode", "lr_sum"}
        ]

        self.assertEqual(
            non_block_keys,
            ["step", "train_mode", "lr_sum"],
            msg=f"find unexpected non-block key in {non_block_keys=}.",
        )
        self.assertEqual(block_keys, filtered_flatten_keys)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class FullyShardDistributorOnEmptyParamTest(
    DTensorTestBase, DistributorOnEmptyParamTest.Interface
):
    @property
    def world_size(self) -> int:
        return 2

    def _construct_model_and_distributor(
        self,
    ) -> tuple[nn.Module, FullyShardDistributor]:
        # Create a model with specific configuration:
        # - linear_layers are empty params (second dimension is 0)
        # - dead_layers will be partitioned into two split params for each rank
        # For rank 0: One split param with torch.Size((PRECONDITIONER_DIM, PRECONDITIONER_DIM)) and another with torch.Size((PRECONDITIONER_DIM // 2, PRECONDITIONER_DIM))
        # For rank 1: One split param with torch.Size((PRECONDITIONER_DIM // 2, PRECONDITIONER_DIM)) and another with torch.Size((PRECONDITIONER_DIM, PRECONDITIONER_DIM))
        assert isinstance(
            model := construct_training_problem(
                model_linear_layers_dims=(PRECONDITIONER_DIM, 0),
                model_dead_layers_dims=(PRECONDITIONER_DIM, 3 * PRECONDITIONER_DIM),
                enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
                device=torch.device("cuda"),
                fill=0.01,
                post_model_decoration=partial(fully_shard),
            )[0],
            nn.Module,
        )
        distributor = FullyShardDistributor(
            param_group=DistributedShampoo(
                model.parameters(),
                lr=0.001,
                betas=(0.9, 1.0),
                epsilon=1e-8,
                weight_decay=0.0,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                max_preconditioner_dim=PRECONDITIONER_DIM,
                distributed_config=FullyShardDistributedConfig(),
            ).param_groups[0],
        )

        # Get the weight of the linear layers (which is empty) and set its gradient
        linear_layers: nn.ModuleList = cast(nn.ModuleList, model.linear_layers)
        first_linear_layer_weight: torch.Tensor = cast(
            torch.Tensor, linear_layers[0].weight
        )
        assert first_linear_layer_weight.numel() == 0
        first_linear_layer_weight.grad = torch.ones_like(first_linear_layer_weight)

        # Get the weight of the dead layers and set its gradient to None to make sure this is a dead layer
        dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
        first_dead_layer_weight: torch.Tensor = cast(
            torch.Tensor, dead_layers[0].weight
        )
        first_dead_layer_weight.grad = None

        return model, distributor

    @with_comms
    @parametrize("use_masked_tensors", [True, False])
    def test_update_params(self, use_masked_tensors: bool) -> None:
        DistributorOnEmptyParamTest.Interface._test_update_params_impl(
            self, use_masked_tensors
        )

    @property
    def _expected_local_grad_selector(self) -> tuple[bool, ...]:
        return (False, False)

    @with_comms
    def test_local_grad_selector(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_grad_selector(self)

    @property
    def _expected_local_blocked_params(self) -> tuple[torch.Tensor, ...]:
        # Define expected parameters for each rank
        return (
            torch.zeros(
                (PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                dtype=torch.float,
                device="cuda",
            ),
            torch.zeros(
                (PRECONDITIONER_DIM // 2, PRECONDITIONER_DIM),
                dtype=torch.float,
                device="cuda",
            ),
        )

    @with_comms
    def test_local_blocked_params(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_blocked_params(self)

    def _expected_local_block_info_list(
        self, model: nn.Module
    ) -> tuple[BlockInfo, ...]:
        # Get the weight parameter from the first dead layer
        dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
        first_dead_layer_weight: torch.Tensor = cast(
            torch.Tensor, dead_layers[0].weight
        )

        # Define expected BlockInfo objects for each rank
        return (
            BlockInfo(
                param=first_dead_layer_weight,
                composable_block_ids=(0, f"rank_{dist.get_rank()}-block_0"),
            ),
            BlockInfo(
                param=first_dead_layer_weight,
                composable_block_ids=(0, f"rank_{dist.get_rank()}-block_1"),
            ),
        )

    @with_comms
    def test_local_block_info_list(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_block_info_list(self)

    @property
    def _expected_local_masked_block_grads(self) -> tuple[torch.Tensor, ...]:
        return ()

    @with_comms
    def test_merge_and_block_gradients(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_merge_and_block_gradients(self)

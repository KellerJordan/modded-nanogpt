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
from itertools import filterfalse, pairwise
from typing import cast

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.gpu_tests.distributor_test_utils import (
    DistributorOnEmptyParamTest,
)
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_fsdp_distributor import FSDPDistributor
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    FSDPDistributedConfig,
    GeneralizedPrimalAveragingConfig,
    HSDPDistributedConfig,
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1, ShardingStrategy
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


PRECONDITIONER_DIM = 4


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class ShampooFSDPDistributorTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
        distributed_config: FSDPDistributedConfig | None = None,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        # NOTE: We construct the model here specifically in order to ensure that
        #       FSDP1 Shampoo and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FSDP1 will split the
        #       parameters such that the resulting blocks from tensor block recovery
        #       and merging/blocking are equivalent to what would be obtained by
        #       merging/blocking on the original parameters.
        #
        #       An additional constraint imposed by FSDP1 is from PT2; the split must be
        #       16-byte aligned. With FP32 elements, this corresponds to 4 elements.
        #
        #       Based on the design of the model below, the model has 512 + 72 + 576 + 64 =
        #       1224 elements, which means that the model will be split at index 612 across
        #       the flattened param in FSDP1.
        #       This corresponds to index 612 - 512 - 72 = 28 in the third parameter. Note
        #       that splitting at this index is equivalent to the standard blocking with a
        #       block size of 4.
        model_linear_layers_dims = (
            4 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
            2 * PRECONDITIONER_DIM,
            9,
            4 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
            1,
        )
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=None,
            enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.01,
            post_model_decoration=post_model_decoration,
        )
        if isinstance(distributed_config, FSDPDistributedConfig):
            assert (
                sum(param.numel() for param in model.parameters())
                == sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2
            ), (
                f"{sum(param.numel() for param in model.parameters())=}, {sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2=}"
            )
            distributed_config.param_to_metadata = compile_fsdp_parameter_metadata(
                model
            )

        return model, loss, data, target

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: FSDPDistributedConfig | SingleDeviceDistributedConfig,
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

    @skip_if_lt_x_gpu(2)
    def test_all_ranks_with_no_grads(self) -> None:
        fsdp_config = FSDPDistributedConfig(param_to_metadata={})

        steps_without_gradients = 2
        with unittest.mock.patch.object(torch.Tensor, "backward") as mock_backward:
            # By mocking the backward() method, we're intercepting gradient calculation.
            # This effectively simulates running forward passes without computing gradients.
            train_model(
                optim_factory=ShampooFSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=fsdp_config,
                ),
                model_factory=partial(
                    ShampooFSDPDistributorTest._construct_model,
                    post_model_decoration=partial(FSDP1, use_orig_params=True),
                    distributed_config=fsdp_config,
                ),
                num_steps=steps_without_gradients,
            )

        # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
        self.assertEqual(mock_backward.call_count, steps_without_gradients)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_shampoo_against_default_shampoo(self) -> None:
        fsdp_config = FSDPDistributedConfig(param_to_metadata={})
        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFSDPDistributorTest._shampoo_optim_factory(
                distributed_config=DefaultSingleDeviceDistributedConfig,
            ),
            control_model_factory=ShampooFSDPDistributorTest._construct_model,
            experimental_optim_factory=ShampooFSDPDistributorTest._shampoo_optim_factory(
                distributed_config=fsdp_config,
            ),
            experimental_model_factory=partial(
                ShampooFSDPDistributorTest._construct_model,
                post_model_decoration=partial(FSDP1, use_orig_params=True),
                distributed_config=fsdp_config,
            ),
        )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_shampoo_block_index(self) -> None:
        model = ShampooFSDPDistributorTest._construct_model(
            post_model_decoration=partial(FSDP1, use_orig_params=True)
        )[0]
        fsdp_config = FSDPDistributedConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model)
        )
        assert isinstance(
            optimizer := ShampooFSDPDistributorTest._shampoo_optim_factory(fsdp_config)(
                model.parameters()
            ),
            DistributedShampoo,
        )

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

    @skip_if_lt_x_gpu(2)
    @parametrize("communicate_params", (True, False))
    def test_fsdp_distributed_config_against_hsdp_distributed_config_bitwise_identical(
        self, communicate_params: bool
    ) -> None:
        fsdp_config = FSDPDistributedConfig(param_to_metadata={})
        hsdp_config = HSDPDistributedConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (1, self.world_size)),
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFSDPDistributorTest._shampoo_optim_factory(
                distributed_config=fsdp_config,
            ),
            control_model_factory=partial(
                ShampooFSDPDistributorTest._construct_model,
                post_model_decoration=partial(FSDP1, use_orig_params=True),
                distributed_config=fsdp_config,
            ),
            experimental_optim_factory=ShampooFSDPDistributorTest._shampoo_optim_factory(
                distributed_config=hsdp_config,
            ),
            experimental_model_factory=partial(
                ShampooFSDPDistributorTest._construct_model,
                post_model_decoration=partial(
                    FSDP1,
                    device_mesh=hsdp_config.device_mesh,
                    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    use_orig_params=True,
                ),
                distributed_config=hsdp_config,
            ),
            total_steps=100,
            rtol=0.0,
            atol=0.0,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class FSDPDistributorOnEmptyParamTest(FSDPTest, DistributorOnEmptyParamTest.Interface):
    @property
    def world_size(self) -> int:
        return 2

    def _construct_model_and_distributor(self) -> tuple[nn.Module, FSDPDistributor]:
        # Create a model with specific configuration:
        # - linear_layers are empty params (second dimension is 0)
        # - dead_layers will be partitioned into two split params for each rank
        # For rank 0: One split param with torch.Size((PRECONDITIONER_DIM,)) and another with torch.Size((PRECONDITIONER_DIM // 2,))
        # For rank 1: One split param with torch.Size((PRECONDITIONER_DIM // 2,)) and another with torch.Size((PRECONDITIONER_DIM,))
        model = construct_training_problem(
            model_linear_layers_dims=(PRECONDITIONER_DIM, 0),
            model_dead_layers_dims=(PRECONDITIONER_DIM, 3),
            enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.01,
            post_model_decoration=partial(FSDP1, use_orig_params=True),
        )[0]
        distributed_config = FSDPDistributedConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model)
        )
        distributor = FSDPDistributor(
            param_group=DistributedShampoo(
                model.parameters(),
                lr=0.001,
                betas=(0.9, 1.0),
                epsilon=1e-8,
                weight_decay=0.0,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                max_preconditioner_dim=PRECONDITIONER_DIM,
                distributed_config=distributed_config,
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

    @skip_if_lt_x_gpu(2)
    @parametrize("use_masked_tensors", [True, False])
    def test_update_params(self, use_masked_tensors: bool) -> None:
        DistributorOnEmptyParamTest.Interface._test_update_params_impl(
            self, use_masked_tensors
        )

    @property
    def _expected_local_grad_selector(self) -> tuple[bool, ...]:
        return (False, False)

    @property
    def _expected_local_blocked_params(self) -> tuple[torch.Tensor, ...]:
        # Define expected parameters for each rank
        return {
            0: (  # For rank 0
                torch.zeros((PRECONDITIONER_DIM,), dtype=torch.float, device="cuda:0"),
                torch.zeros(
                    (PRECONDITIONER_DIM // 2,), dtype=torch.float, device="cuda:0"
                ),
            ),
            1: (  # For rank 1
                torch.zeros(
                    (PRECONDITIONER_DIM // 2,), dtype=torch.float, device="cuda:1"
                ),
                torch.zeros((PRECONDITIONER_DIM,), dtype=torch.float, device="cuda:1"),
            ),
        }[dist.get_rank()]

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
                composable_block_ids=(1, f"rank_{dist.get_rank()}-block_0"),
            ),
            BlockInfo(
                param=first_dead_layer_weight,
                composable_block_ids=(1, f"rank_{dist.get_rank()}-block_1"),
            ),
        )

    @property
    def _expected_local_masked_block_grads(self) -> tuple[torch.Tensor, ...]:
        return ()

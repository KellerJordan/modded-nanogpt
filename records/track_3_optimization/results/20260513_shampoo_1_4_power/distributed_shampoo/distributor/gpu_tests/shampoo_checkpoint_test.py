"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import copy
from collections.abc import Callable
from functools import partial
from operator import attrgetter
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from distributed_shampoo.distributed_shampoo import (
    DDPDistributedConfig,
    DistributedShampoo,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    HybridShardDistributedConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
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
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


PRECONDITIONER_DIM = 4

# Test configuration constants for lossless distributor checkpoint test
NUM_INITIAL_STEPS = 10
NUM_CONTINUATION_STEPS = 10
NUM_BATCH = 8

# Model layer dimensions for lossless distributor checkpoint test
TEST_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1),
    (3 * PRECONDITIONER_DIM - 1, PRECONDITIONER_DIM + 1, PRECONDITIONER_DIM - 1),
    (2, 2 * PRECONDITIONER_DIM - 1),
    (PRECONDITIONER_DIM, 1),
)

# TODO (irisz): Add dead layer dims once lossless distributor supports it.
DEAD_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = ((),)


@torch.no_grad()
def compare_optimizer_state_dict(
    ref_optim_state: dict[str, Any],
    optim_state: dict[str, Any],
    assert_fn: Callable[..., None],
) -> None:
    """
    Recursively compares two optimizer state dicts for bitwise equivalence, skipping 'param_groups' keys.

    Args:
        ref_optim_state (dict[str, Any]): The reference optimizer state dict.
        optim_state (dict[str, Any]): The optimizer state dict to compare.
        assert_fn (Callable[..., None]): Assertion function to use for comparing values (e.g., self.assertTrue or self.assertFalse).

    This function is useful for verifying that optimizer state dicts are equal (or not equal) after checkpoint save/load operations,
    while ignoring 'param_groups', as 'param_groups' is unchanged.
    """

    def recursive_compare(d1: dict[str, Any], d2: dict[str, Any]) -> None:
        assert d1.keys() == d2.keys(), f"Keys mismatch: {d1.keys()} != {d2.keys()}"
        for k in d1:
            if isinstance(k, str) and "param_groups" in k:
                # skip param_groups comparison
                continue
            v1 = d1[k]
            assert k in d2.keys(), f"Key {k} not found in {d2.keys()}"
            v2 = d2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                recursive_compare(v1, v2)
            else:
                if isinstance(v1, DTensor) and isinstance(v2, DTensor):
                    v1 = v1.to_local()
                    v2 = v2.to_local()
                assert_fn(
                    torch.equal(v1, v2),
                    f"Param {k} mismatch: v1 param {v1} v2 param {v2}",
                )

    recursive_compare(ref_optim_state, optim_state)


@instantiate_parametrized_tests
class DistributedShampooDistributedCheckpointTest(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    @parametrize(
        "flatten_optimizer_state_dict",
        [
            True,
            False,
        ],
    )
    def test_ddp_shampoo_checkpoint(self, flatten_optimizer_state_dict: bool) -> None:
        """
        This test is intended to make sure that Shampoo `state_dict` and `load_state_dict`
        are compatible with PyTorch's API, including `torch.distributed.checkpoint.get_optimizer_state_dict`,
        `torch.distributed.checkpoint.set_optimizer_state_dict`, `torch.distributed.checkpoint.save`,
        and `torch.distributed.checkpoint.load`. `get_optimizer_state_dict` calls `state_dict` and
        `set_optimizer_state_dict` calls `load_state_dict` in Shampoo under the hood.

        Note: It tests both scenarios where `flatten_optimizer_state_dict` is True and False.
        """

        CHECKPOINT_DIR = attrgetter("temp_dir")(self)
        device = torch.device(self.device_type)

        # create a DDP model
        model, _, _, _ = construct_training_problem(
            model_linear_layers_dims=(PRECONDITIONER_DIM * 2, PRECONDITIONER_DIM * 4),
            model_dead_layers_dims=None,
            device=device,
            post_model_decoration=torch.nn.parallel.DistributedDataParallel,
        )
        optim = DistributedShampoo(
            model.parameters(),
            max_preconditioner_dim=PRECONDITIONER_DIM,
            distributed_config=DDPDistributedConfig(num_trainers_per_group=-1),
        )

        # step ahead to initialize the optimizer
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()

        # deep copy step 1's optimizer state for comparison later.
        ref_optim_state_dict: dict[str, Any] = copy.deepcopy(
            get_optimizer_state_dict(
                model,
                optim,
                options=StateDictOptions(
                    flatten_optimizer_state_dict=flatten_optimizer_state_dict
                ),
            )
        )

        # get the current model and optimizer state at step 1
        # `get_model_state_dict` and `get_optimizer_state_dict` call `model.state_dict()`
        # and `optim.state_dict()` under the hood.
        ref_state_dict = {
            "model": get_model_state_dict(model),
            "optim": ref_optim_state_dict,
        }

        # save model's state and optimizer's state to disk
        dcp.save(
            state_dict=ref_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
        )

        # step forward to step 2
        # so both the model and optimizer are different from previous step.
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()

        # get the current model and optimizer state at step 2 for dcp to load into.
        model_state_dict: dict[str, Any] = get_model_state_dict(model)
        optim_state_dict: dict[str, Any] = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )

        # compare to make sure optimizer state is different between step 1 and 2.
        compare_optimizer_state_dict(
            ref_optim_state=ref_optim_state_dict,
            optim_state=optim_state_dict,
            assert_fn=self.assertFalse,
        )

        state_dict = {
            "model": model_state_dict,
            "optim": optim_state_dict,
        }

        # load from disk to memory
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
        )
        # load from memory to model and optimizer
        # `set_model_state_dict` and `set_optimizer_state_dict` call `model.load_state_dict()`
        # and `optim.load_state_dict()` under the hood.
        set_model_state_dict(
            model=model,
            model_state_dict=state_dict["model"],
        )
        set_optimizer_state_dict(
            model=model,
            optimizers=optim,
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )
        osd_after_load: dict[str, Any] = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )

        # compare to make sure the current optimizer state is the same as step 1.
        compare_optimizer_state_dict(
            ref_optim_state=ref_optim_state_dict,
            optim_state=osd_after_load,
            assert_fn=self.assertTrue,
        )

        # step ahead to make sure training can continue.
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()


@instantiate_parametrized_tests
class LosslessDistributorCheckpointPreemptionTest(DTensorTestBase):
    """Test to validate that checkpoint preserves exact training dynamics for lossless distributors."""

    @property
    def world_size(self) -> int:
        return 4

    @staticmethod
    def _construct_model(
        model_linear_layers_dims: tuple[int, ...],
        model_dead_layers_dims: tuple[int, ...],
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> nn.Module:
        # Using partial here to prevent Pyre complain on incompatible parameter type.
        model, _, _, _ = partial(
            construct_training_problem, post_model_decoration=post_model_decoration
        )(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.1,
        )
        return model

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: DistributedConfig,
        lr: float = 0.001,
    ) -> Callable[[ParamsT], DistributedShampoo]:
        return partial(
            DistributedShampoo,
            lr=lr,
            betas=(0.9, 1.0),
            epsilon=1e-8,
            weight_decay=0.0,
            max_preconditioner_dim=PRECONDITIONER_DIM,
            precondition_frequency=1,
            start_preconditioning_step=2,
            weight_decay_type=WeightDecayType.DECOUPLED,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
            distributed_config=distributed_config,
        )

    @staticmethod
    def _train_step(
        model: nn.Module,
        optimizer: DistributedShampoo,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single training step and return the loss (detached)."""
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.detach()

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    @parametrize("model_linear_layers_dims", TEST_MODEL_LAYER_DIMS)
    @parametrize("model_dead_layers_dims", DEAD_MODEL_LAYER_DIMS)
    @parametrize(
        "param_assignment_strategy",
        [
            FSDPParamAssignmentStrategy.REPLICATE,
            FSDPParamAssignmentStrategy.ROUND_ROBIN,
        ],
    )
    @parametrize(
        "distributed_config_type",
        [
            FullyShardDistributedConfig,
            HybridShardDistributedConfig,
        ],
    )
    def test_checkpoint_preemption_bitwise_equivalence(
        self,
        model_linear_layers_dims: tuple[int, ...],
        model_dead_layers_dims: tuple[int, ...],
        param_assignment_strategy: FSDPParamAssignmentStrategy,
        distributed_config_type: type[
            FullyShardDistributedConfig | HybridShardDistributedConfig
        ],
    ) -> None:
        """
        This test validates that when using fully_shard/hybrid_shard lossless distributor,
        resuming training after preemption produces **bitwise equivalent** results.

        Specifically, we verify that after loading a checkpoint:
        - Losses are bitwise equivalent (using torch.equal)
        - Optimizer states are bitwise equivalent (using torch.equal)

        Test flow:
        1. model1/optimizer1:
            Train NUM_INITIAL_STEPS steps → save checkpoint → continue NUM_CONTINUATION_STEPS more steps
        2. model2/optimizer2:
            Load checkpoint → train NUM_CONTINUATION_STEPS steps
        3. Verify bitwise equivalence:
            - torch.equal(loss1, loss2) for each continuation step
            - torch.equal for all optimizer state tensors
        """
        # Set up config and model decoration based on distributed config type
        if distributed_config_type is HybridShardDistributedConfig:
            mesh_2d = init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            )
            config: DistributedConfig = HybridShardDistributedConfig(
                device_mesh=mesh_2d,
                param_assignment_strategy=param_assignment_strategy,
            )
            post_model_decoration = partial(fully_shard, mesh=mesh_2d)
        else:  # FullyShardDistributedConfig
            config = FullyShardDistributedConfig(
                param_assignment_strategy=param_assignment_strategy
            )
            post_model_decoration = partial(fully_shard)

        # Phase 1: Create model1/optimizer1 and train for initial steps
        model1 = self._construct_model(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            post_model_decoration=post_model_decoration,  # type: ignore[arg-type]
        )
        input_dim = model_linear_layers_dims[0]
        optimizer1 = self._shampoo_optim_factory(distributed_config=config)(
            model1.parameters()
        )

        # Use rank-specific seed for different input data per rank
        torch.manual_seed(42 + self.rank)

        for _ in range(NUM_INITIAL_STEPS):
            input_data = torch.randn(NUM_BATCH, input_dim, device="cuda")
            self._train_step(model1, optimizer1, input_data)

        self.assertEqual(
            optimizer1._per_group_state_lists[0]["step"].item(),
            NUM_INITIAL_STEPS,
        )

        # Phase 2: Save checkpoint (dcp.save is a collective operation)
        checkpoint_dir: str = attrgetter("temp_dir")(self)
        model_state_dict = get_model_state_dict(model1, options=StateDictOptions())
        optim_state_dict = get_optimizer_state_dict(
            model1, optimizer1, options=StateDictOptions()
        )

        dcp.save(
            state_dict={"model": model_state_dict, "optimizer": optim_state_dict},
            storage_writer=dcp.FileSystemWriter(checkpoint_dir),
        )

        # Phase 3: Create model2/optimizer2 and initialize
        model2 = self._construct_model(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            post_model_decoration=post_model_decoration,  # type: ignore[arg-type]
        )
        # Use tiny lr for init step to minimize impact
        optimizer2 = self._shampoo_optim_factory(distributed_config=config, lr=1e-10)(
            model2.parameters()
        )

        # Do one step to initialize internal structures
        dummy_input = torch.randn(NUM_BATCH, input_dim, device="cuda")
        self._train_step(model2, optimizer2, dummy_input)

        # Phase 4: Load checkpoint
        loaded_state_dict: dict[str, Any] = {
            "model": get_model_state_dict(model2, options=StateDictOptions()),
            "optimizer": get_optimizer_state_dict(
                model2, optimizer2, options=StateDictOptions()
            ),
        }

        dcp.load(
            state_dict=loaded_state_dict,
            storage_reader=dcp.FileSystemReader(checkpoint_dir),
        )

        set_model_state_dict(
            model=model2,
            model_state_dict=loaded_state_dict["model"],
            options=StateDictOptions(),
        )

        set_optimizer_state_dict(
            model=model2,
            optimizers=optimizer2,
            optim_state_dict=loaded_state_dict["optimizer"],
            options=StateDictOptions(),
        )

        self.assertEqual(
            optimizer2._per_group_state_lists[0]["step"].item(),
            NUM_INITIAL_STEPS,
        )

        # Phase 5: Continue training both models and verify bitwise equivalence
        # We check that losses and optimizer states are EXACTLY equal (torch.equal)
        # This confirms that FSDP/HSDP collective operations are deterministic.
        for step in range(NUM_CONTINUATION_STEPS):
            input_data = torch.randn(NUM_BATCH, input_dim, device="cuda")

            loss1 = self._train_step(model1, optimizer1, input_data)
            loss2 = self._train_step(model2, optimizer2, input_data)

            # Assert optimizer states are bitwise equivalent
            compare_optimizer_state_dict(
                ref_optim_state=optimizer1.state_dict(),
                optim_state=optimizer2.state_dict(),
                assert_fn=self.assertTrue,
            )

            # Assert losses are bitwise equivalent
            self.assertTrue(
                torch.equal(loss1, loss2),
                msg=f"Step {NUM_INITIAL_STEPS + step}: "
                f"Model1 loss {loss1:.8f} != Model2 loss {loss2:.8f}",
            )

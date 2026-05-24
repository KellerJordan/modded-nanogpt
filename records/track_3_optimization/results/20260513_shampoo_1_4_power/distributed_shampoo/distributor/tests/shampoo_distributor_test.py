"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from typing import cast

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_distributor import Distributor
from distributed_shampoo.shampoo_types import DISTRIBUTOR, ShampooRuntimeConfig
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from torch import nn

PRECONDITIONER_DIM = 5


class DistributorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model: nn.Module = construct_training_problem(
            (2 * PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            model_dead_layers_dims=None,
            bias=True,
            fill=0.0,
        )[0]
        self._distributor = Distributor(
            param_group=DistributedShampoo(
                self._model.parameters(),
                max_preconditioner_dim=PRECONDITIONER_DIM,
            ).param_groups[0]
        )

    def test_update_params(self) -> None:
        # Explicitly disable the gradient of the scalar parameter and call merge_and_block_gradients()
        # to update the local gradient selector for the scalar parameter (i.e., 1st block) and bias layer (i.e., 4th block).
        cast(torch.Tensor, self._model.scalar).grad = None
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)
        linear_layers[0].bias.grad = None
        self._distributor.merge_and_block_gradients()

        actual_masked_blocked_params = self._distributor.local_masked_blocked_params

        masked_blocked_search_directions = (
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM, dtype=torch.float
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                2 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                dtype=torch.float,
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
        )
        self._distributor.update_params(
            blocked_search_directions=masked_blocked_search_directions
        )

        expected_masked_blocked_params = (
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM, dtype=torch.float
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                2 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                dtype=torch.float,
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
        )
        torch.testing.assert_close(
            actual_masked_blocked_params, expected_masked_blocked_params
        )

    def test_local_grad_selector(self) -> None:
        # Explicitly disable the gradient of the scalar parameter and call merge_and_block_gradients()
        # to update the local gradient selector for the scalar parameter (i.e., 1st block) and bias layer (i.e., 4th block).
        cast(torch.Tensor, self._model.scalar).grad = None
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)
        linear_layers[0].bias.grad = None
        self._distributor.merge_and_block_gradients()

        expected_local_grad_selector = (False, True, True, False)
        self.assertEqual(
            self._distributor.local_grad_selector,
            expected_local_grad_selector,
        )

    def test_local_blocked_params(self) -> None:
        expected_local_params = (
            torch.zeros((1,), dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, PRECONDITIONER_DIM, dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, PRECONDITIONER_DIM, dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, dtype=torch.float),
        )
        torch.testing.assert_close(
            self._distributor.local_blocked_params,
            expected_local_params,
        )

    def test_local_block_info_list(self) -> None:
        def block_info_equality(
            a: BlockInfo, b: BlockInfo, msg: str | None = None
        ) -> None:
            # Only comparing param and composable_block_ids fields but not others like get_tensor()
            # because function objects are not comparable in BlockInfo.
            torch.testing.assert_close(a.param, b.param, msg=msg)
            self.assertEqual(a.composable_block_ids, b.composable_block_ids, msg=msg)

        self.addTypeEqualityFunc(BlockInfo, block_info_equality)

        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_bias: torch.Tensor = cast(torch.Tensor, linear_layers[0].bias)
        expected_local_block_info_list = (
            BlockInfo(
                param=cast(torch.Tensor, self._model.scalar),
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(param=layer_weight, composable_block_ids=(1, "block_0")),
            BlockInfo(param=layer_weight, composable_block_ids=(1, "block_1")),
            BlockInfo(param=layer_bias, composable_block_ids=(2, "block_0")),
        )
        for index, (a, b) in enumerate(
            zip(
                self._distributor.local_block_info_list,
                expected_local_block_info_list,
                strict=True,
            )
        ):
            self.assertEqual(
                a,
                b,
                f"Difference found at {index=}: {self._distributor.local_block_info_list[index]=} != {expected_local_block_info_list[index]=}",
            )

    def test_merge_and_block_gradients(self) -> None:
        scalar_tensor: torch.Tensor = cast(torch.Tensor, self._model.scalar)
        scalar_tensor.grad = torch.ones_like(scalar_tensor)
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)
        linear_layers[0].bias.grad = None
        actual_local_masked_block_grads = self._distributor.merge_and_block_gradients()
        expected_local_masked_block_grads = (
            torch.ones((1,)),
            torch.ones((PRECONDITIONER_DIM, PRECONDITIONER_DIM)),
            torch.ones((PRECONDITIONER_DIM, PRECONDITIONER_DIM)),
        )
        torch.testing.assert_close(
            actual_local_masked_block_grads, expected_local_masked_block_grads
        )

    def test_enable_eager_nan_check(self) -> None:
        self._distributor = DistributedShampoo(
            self._model.parameters(),
            max_preconditioner_dim=PRECONDITIONER_DIM,
            shampoo_runtime_config=ShampooRuntimeConfig(eager_nan_check=True),
        )._per_group_state_lists[0][DISTRIBUTOR]

        cast(torch.Tensor, self._model.scalar).grad = torch.tensor(1.0)
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)
        linear_layers[0].bias.grad = None

        with unittest.mock.patch.object(torch, "isfinite") as mock_isfinite:
            self._distributor.merge_and_block_gradients()

        self.assertEqual(mock_isfinite.call_count, 2)

    def test_disable_eager_nan_check(self) -> None:
        self._distributor = DistributedShampoo(
            self._model.parameters(),
            max_preconditioner_dim=PRECONDITIONER_DIM,
        )._per_group_state_lists[0][DISTRIBUTOR]

        cast(torch.Tensor, self._model.scalar).grad = torch.tensor(1.0)
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)
        layer_weight: torch.Tensor = cast(torch.Tensor, linear_layers[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)
        linear_layers[0].bias.grad = None

        with unittest.mock.patch.object(torch, "isfinite") as mock_isfinite:
            self._distributor.merge_and_block_gradients()

        self.assertEqual(mock_isfinite.call_count, 0)


class DistributorOnEmptyParamTest(unittest.TestCase):
    def setUp(self) -> None:
        # Note that this model has one empty parameter, and one scalar parameter.
        self._model: nn.Module = construct_training_problem(
            (PRECONDITIONER_DIM, 0),
            model_dead_layers_dims=None,
            bias=False,
            fill=0.0,
        )[0]
        self._distributor = Distributor(
            param_group=DistributedShampoo(
                self._model.parameters(),
                max_preconditioner_dim=PRECONDITIONER_DIM,
            ).param_groups[0]
        )

        # Set up the scalar tensor with a gradient
        self._scalar_tensor: torch.Tensor = cast(torch.Tensor, self._model.scalar)
        self._scalar_tensor.grad = torch.ones_like(self._scalar_tensor)

        # Get the linear layers module list - note that this contains an empty parameter
        linear_layers: nn.ModuleList = cast(nn.ModuleList, self._model.linear_layers)

        # Get the weight of the first layer (which is empty) and set its gradient
        self._first_linear_layer_weight: torch.Tensor = cast(
            torch.Tensor, linear_layers[0].weight
        )
        assert self._first_linear_layer_weight.numel() == 0
        self._first_linear_layer_weight.grad = torch.ones_like(
            self._first_linear_layer_weight
        )

    def test_update_params(self) -> None:
        # Process gradients and update the local gradient selector
        # Since layer_weight is empty, it won't produce block params
        self._distributor.merge_and_block_gradients()

        # Get the current masked blocked parameters
        actual_masked_blocked_params = self._distributor.local_masked_blocked_params

        # Create search directions only for the scalar tensor
        # No search directions for the empty layer_weight
        masked_blocked_search_directions = (
            torch.ones_like(self._scalar_tensor, dtype=torch.float),
        )

        # Update parameters using the search directions
        self._distributor.update_params(
            blocked_search_directions=masked_blocked_search_directions
        )

        # Define expected masked blocked parameters
        # Only contains the scalar tensor (unsqueezed to match dimensions)
        expected_masked_blocked_params = (
            torch.ones_like(self._scalar_tensor, dtype=torch.float).unsqueeze(0),
        )

        # Verify that the actual parameters match the expected ones
        torch.testing.assert_close(
            actual_masked_blocked_params, expected_masked_blocked_params
        )

    def test_local_grad_selector(self) -> None:
        # Process gradients and update the local gradient selector
        self._distributor.merge_and_block_gradients()

        # Since layer_weight is empty (shape contains 0), it won't produce block params
        # The gradient selector will be True for scalar_tensor and False for the empty layer_weight
        expected_local_grad_selector = (True, False)
        self.assertEqual(
            self._distributor.local_grad_selector,
            expected_local_grad_selector,
        )

    def test_local_blocked_params(self) -> None:
        # Expected local parameters:
        # - First element: a scalar tensor (shape 1)
        # - Second element: an empty tensor (shape 0) since layer_weight is empty
        expected_local_params = (
            torch.zeros((1,), dtype=torch.float),
            torch.zeros((0,), dtype=torch.float),
        )
        torch.testing.assert_close(
            self._distributor.local_blocked_params,
            expected_local_params,
        )

    def test_local_block_info_list(self) -> None:
        # Custom equality function for BlockInfo objects
        # We only compare param and composable_block_ids fields
        # The msg parameter is required by unittest but not used
        def block_info_equality(
            a: BlockInfo, b: BlockInfo, msg: str | None = None
        ) -> None:
            # Only comparing param and composable_block_ids fields but not others like get_tensor()
            # because function objects are not comparable in BlockInfo.
            torch.testing.assert_close(a.param, b.param, msg=msg)
            self.assertEqual(a.composable_block_ids, b.composable_block_ids, msg=msg)

        # Register the custom equality function for BlockInfo type
        self.addTypeEqualityFunc(BlockInfo, block_info_equality)

        # Expected block info list:
        # - First element: the scalar tensor with block ID (0, "block_0")
        # - Second element: the empty layer weight with block ID (1, "block_0")
        expected_local_block_info_list = (
            BlockInfo(
                param=self._scalar_tensor,
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(
                param=self._first_linear_layer_weight,
                composable_block_ids=(1, "block_0"),
            ),
        )

        # Compare each element in the block info lists
        for index, (a, b) in enumerate(
            zip(
                self._distributor.local_block_info_list,
                expected_local_block_info_list,
                strict=True,
            )
        ):
            self.assertEqual(
                a,
                b,
                f"Difference found at {index=}: {self._distributor.local_block_info_list[index]=} != {expected_local_block_info_list[index]=}",
            )

    def test_merge_and_block_gradients(self) -> None:
        # Process gradients - since layer_weight is empty, it won't produce block gradients
        actual_local_masked_block_grads = self._distributor.merge_and_block_gradients()

        # Only the scalar tensor produces a block gradient (shape 1)
        # The empty layer_weight doesn't contribute to the masked block gradients
        expected_local_masked_block_grads = (torch.ones((1,)),)
        torch.testing.assert_close(
            actual_local_masked_block_grads, expected_local_masked_block_grads
        )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

from distributed_shampoo.examples.parallelism import (
    DDPStrategy,
    FSDPStrategy,
    FullyShardStrategy,
    HSDPStrategy,
    HybridShardStrategy,
    SingleGPUStrategy,
    WrappedModel,
)
from torch import nn


class WrappedModelTest(unittest.TestCase):
    def test_wrapped_model_with_model_only(self) -> None:
        """Test WrappedModel with only model, no distributed_config."""
        model = nn.Linear(10, 5)
        wrapped = WrappedModel(model=model)

        self.assertIs(wrapped.model, model)
        self.assertIsNone(wrapped.distributed_config)

    def test_wrapped_model_with_distributed_config(self) -> None:
        """Test WrappedModel with both model and distributed_config."""
        model = nn.Linear(10, 5)
        mock_config = MagicMock()
        wrapped = WrappedModel(model=model, distributed_config=mock_config)

        self.assertIs(wrapped.model, model)
        self.assertIs(wrapped.distributed_config, mock_config)


class SingleGPUStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_false(self) -> None:
        """Test that SingleGPUStrategy does not require distributed."""
        strategy = SingleGPUStrategy()
        self.assertFalse(strategy.requires_distributed)

    def test_requires_device_mesh_is_false(self) -> None:
        """Test that SingleGPUStrategy does not require device mesh."""
        strategy = SingleGPUStrategy()
        self.assertFalse(strategy.requires_device_mesh)

    def test_wrap_model_returns_unwrapped_model(self) -> None:
        """Test that wrap_model returns the model unchanged."""
        strategy = SingleGPUStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl")

        self.assertIsInstance(result, WrappedModel)
        self.assertIs(result.model, model)
        self.assertIsNone(result.distributed_config)


class DDPStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_true(self) -> None:
        """Test that DDPStrategy requires distributed."""
        strategy = DDPStrategy()
        self.assertTrue(strategy.requires_distributed)

    def test_requires_device_mesh_is_false(self) -> None:
        """Test that DDPStrategy does not require device mesh."""
        strategy = DDPStrategy()
        self.assertFalse(strategy.requires_device_mesh)

    @patch("distributed_shampoo.examples.parallelism.DDP")
    def test_wrap_model_nccl_backend(self, mock_ddp: MagicMock) -> None:
        """Test that wrap_model uses device_ids for nccl backend."""
        mock_wrapped = MagicMock()
        mock_ddp.return_value = mock_wrapped
        strategy = DDPStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 2, "nccl")

        mock_ddp.assert_called_once_with(model, device_ids=[2], output_device=2)
        self.assertIs(result.model, mock_wrapped)

    @patch("distributed_shampoo.examples.parallelism.DDP")
    def test_wrap_model_gloo_backend(self, mock_ddp: MagicMock) -> None:
        """Test that wrap_model does not use device_ids for gloo backend."""
        mock_wrapped = MagicMock()
        mock_ddp.return_value = mock_wrapped
        strategy = DDPStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "gloo")

        mock_ddp.assert_called_once_with(model)
        self.assertIs(result.model, mock_wrapped)

    @patch("distributed_shampoo.examples.parallelism.DDP")
    def test_wrap_model_with_distributed_config(self, mock_ddp: MagicMock) -> None:
        """Test that wrap_model calls distributed_config partial."""
        mock_ddp.return_value = MagicMock()
        mock_config = MagicMock()
        mock_partial = MagicMock(return_value=mock_config)
        strategy = DDPStrategy(distributed_config=mock_partial)
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "gloo")

        mock_partial.assert_called_once()
        self.assertIs(result.distributed_config, mock_config)


class FSDPStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_true(self) -> None:
        """Test that FSDPStrategy requires distributed."""
        strategy = FSDPStrategy()
        self.assertTrue(strategy.requires_distributed)

    def test_requires_device_mesh_is_false(self) -> None:
        """Test that FSDPStrategy does not require device mesh."""
        strategy = FSDPStrategy()
        self.assertFalse(strategy.requires_device_mesh)

    @patch(
        "distributed_shampoo.examples.parallelism.compile_fsdp_parameter_metadata",
        return_value={},
    )
    @patch("distributed_shampoo.examples.parallelism.FSDP")
    def test_wrap_model_uses_orig_params(
        self, mock_fsdp: MagicMock, mock_compile: MagicMock
    ) -> None:
        """Test that FSDP is called with use_orig_params=True."""
        mock_wrapped = MagicMock()
        mock_fsdp.return_value = mock_wrapped
        strategy = FSDPStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl")

        mock_fsdp.assert_called_once_with(model, use_orig_params=True)
        self.assertIs(result.model, mock_wrapped)

    @patch(
        "distributed_shampoo.examples.parallelism.compile_fsdp_parameter_metadata",
        return_value={"key": "value"},
    )
    @patch("distributed_shampoo.examples.parallelism.FSDP")
    def test_wrap_model_with_distributed_config(
        self, mock_fsdp: MagicMock, mock_compile: MagicMock
    ) -> None:
        """Test that distributed_config receives param_to_metadata."""
        mock_fsdp.return_value = MagicMock()
        mock_config = MagicMock()
        mock_partial = MagicMock(return_value=mock_config)
        strategy = FSDPStrategy(distributed_config=mock_partial)
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl")

        mock_partial.assert_called_once()
        call_kwargs = mock_partial.call_args[1]
        self.assertIn("param_to_metadata", call_kwargs)
        self.assertIs(result.distributed_config, mock_config)


class HSDPStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_true(self) -> None:
        """Test that HSDPStrategy requires distributed."""
        strategy = HSDPStrategy()
        self.assertTrue(strategy.requires_distributed)

    def test_requires_device_mesh_is_true(self) -> None:
        """Test that HSDPStrategy requires device mesh."""
        strategy = HSDPStrategy()
        self.assertTrue(strategy.requires_device_mesh)

    @patch(
        "distributed_shampoo.examples.parallelism.compile_fsdp_parameter_metadata",
        return_value={},
    )
    @patch("distributed_shampoo.examples.parallelism.FSDP")
    def test_wrap_model_uses_hybrid_shard_strategy(
        self, mock_fsdp: MagicMock, mock_compile: MagicMock
    ) -> None:
        """Test that FSDP is called with HYBRID_SHARD strategy."""
        from torch.distributed.fsdp import ShardingStrategy

        mock_wrapped = MagicMock()
        mock_fsdp.return_value = mock_wrapped
        mock_device_mesh = MagicMock()
        strategy = HSDPStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl", mock_device_mesh)

        mock_fsdp.assert_called_once_with(
            model,
            device_mesh=mock_device_mesh,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
        )
        self.assertIs(result.model, mock_wrapped)

    def test_wrap_model_asserts_device_mesh_not_none(self) -> None:
        """Test that wrap_model raises assertion error if device_mesh is None."""
        strategy = HSDPStrategy()
        model = nn.Linear(10, 5)

        with self.assertRaises(AssertionError):
            strategy.wrap_model(model, 0, "nccl", None)


class FullyShardStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_true(self) -> None:
        """Test that FullyShardStrategy requires distributed."""
        strategy = FullyShardStrategy()
        self.assertTrue(strategy.requires_distributed)

    def test_requires_device_mesh_is_false(self) -> None:
        """Test that FullyShardStrategy does not require device mesh."""
        strategy = FullyShardStrategy()
        self.assertFalse(strategy.requires_device_mesh)

    @patch("distributed_shampoo.examples.parallelism.fully_shard")
    def test_wrap_model_calls_fully_shard(self, mock_fully_shard: MagicMock) -> None:
        """Test that wrap_model calls fully_shard."""
        mock_wrapped = MagicMock()
        mock_fully_shard.return_value = mock_wrapped
        strategy = FullyShardStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl")

        mock_fully_shard.assert_called_once_with(model)
        self.assertIs(result.model, mock_wrapped)


class HybridShardStrategyTest(unittest.TestCase):
    def test_requires_distributed_is_true(self) -> None:
        """Test that HybridShardStrategy requires distributed."""
        strategy = HybridShardStrategy()
        self.assertTrue(strategy.requires_distributed)

    def test_requires_device_mesh_is_true(self) -> None:
        """Test that HybridShardStrategy requires device mesh."""
        strategy = HybridShardStrategy()
        self.assertTrue(strategy.requires_device_mesh)

    @patch("distributed_shampoo.examples.parallelism.fully_shard")
    def test_wrap_model_passes_mesh(self, mock_fully_shard: MagicMock) -> None:
        """Test that wrap_model passes device mesh to fully_shard."""
        mock_wrapped = MagicMock()
        mock_fully_shard.return_value = mock_wrapped
        mock_device_mesh = MagicMock()
        strategy = HybridShardStrategy()
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl", mock_device_mesh)

        mock_fully_shard.assert_called_once_with(model, mesh=mock_device_mesh)
        self.assertIs(result.model, mock_wrapped)

    def test_wrap_model_asserts_device_mesh_not_none(self) -> None:
        """Test that wrap_model raises assertion error if device_mesh is None."""
        strategy = HybridShardStrategy()
        model = nn.Linear(10, 5)

        with self.assertRaises(AssertionError):
            strategy.wrap_model(model, 0, "nccl", None)

    @patch("distributed_shampoo.examples.parallelism.fully_shard")
    def test_wrap_model_with_distributed_config(
        self, mock_fully_shard: MagicMock
    ) -> None:
        """Test that distributed_config receives device_mesh."""
        mock_fully_shard.return_value = MagicMock()
        mock_config = MagicMock()
        mock_partial = MagicMock(return_value=mock_config)
        mock_device_mesh = MagicMock()
        strategy = HybridShardStrategy(distributed_config=mock_partial)
        model = nn.Linear(10, 5)

        result = strategy.wrap_model(model, 0, "nccl", mock_device_mesh)

        mock_partial.assert_called_once_with(device_mesh=mock_device_mesh)
        self.assertIs(result.distributed_config, mock_config)

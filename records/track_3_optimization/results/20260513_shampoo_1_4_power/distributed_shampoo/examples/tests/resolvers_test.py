"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest

import torch
from distributed_shampoo import FSDPParamAssignmentStrategy, WeightDecayType
from distributed_shampoo.examples.resolvers import (
    _fsdp_param_strategy_resolver,
    _torch_dtype_resolver,
    _weight_decay_type_resolver,
    register_resolvers,
)
from omegaconf import OmegaConf


class TorchDtypeResolverTest(unittest.TestCase):
    def test_float32_resolution(self) -> None:
        """Test that float32 is resolved correctly."""
        result = _torch_dtype_resolver("float32")
        self.assertEqual(result, torch.float32)

    def test_float16_resolution(self) -> None:
        """Test that float16 is resolved correctly."""
        result = _torch_dtype_resolver("float16")
        self.assertEqual(result, torch.float16)

    def test_bfloat16_resolution(self) -> None:
        """Test that bfloat16 is resolved correctly."""
        result = _torch_dtype_resolver("bfloat16")
        self.assertEqual(result, torch.bfloat16)

    def test_float64_resolution(self) -> None:
        """Test that float64 is resolved correctly."""
        result = _torch_dtype_resolver("float64")
        self.assertEqual(result, torch.float64)

    def test_invalid_dtype_int32_raises_error(self) -> None:
        """Test that int32 dtype string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _torch_dtype_resolver("int32")
        self.assertIn("Unknown dtype", str(context.exception))

    def test_invalid_dtype_float_raises_error(self) -> None:
        """Test that 'float' dtype string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _torch_dtype_resolver("float")
        self.assertIn("Unknown dtype", str(context.exception))

    def test_invalid_dtype_empty_raises_error(self) -> None:
        """Test that empty dtype string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _torch_dtype_resolver("")
        self.assertIn("Unknown dtype", str(context.exception))


class FSDPParamStrategyResolverTest(unittest.TestCase):
    def test_default_strategy_resolution(self) -> None:
        """Test that DEFAULT strategy is resolved correctly."""
        result = _fsdp_param_strategy_resolver("DEFAULT")
        self.assertEqual(result, FSDPParamAssignmentStrategy.DEFAULT)

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy strings raise ValueError."""
        with self.assertRaises(ValueError) as context:
            _fsdp_param_strategy_resolver("INVALID_STRATEGY")
        self.assertIn("Unknown strategy", str(context.exception))


class WeightDecayTypeResolverTest(unittest.TestCase):
    def test_decoupled_resolution(self) -> None:
        """Test that DECOUPLED weight decay type is resolved correctly."""
        result = _weight_decay_type_resolver("DECOUPLED")
        self.assertEqual(result, WeightDecayType.DECOUPLED)

    def test_l2_resolution(self) -> None:
        """Test that L2 weight decay type is resolved correctly."""
        result = _weight_decay_type_resolver("L2")
        self.assertEqual(result, WeightDecayType.L2)

    def test_invalid_weight_decay_type_raises_error(self) -> None:
        """Test that invalid weight decay type strings raise ValueError."""
        with self.assertRaises(ValueError) as context:
            _weight_decay_type_resolver("INVALID_TYPE")
        self.assertIn("Unknown weight decay type", str(context.exception))


class RegisterResolversTest(unittest.TestCase):
    def test_register_resolvers_idempotent(self) -> None:
        """Test that register_resolvers can be called multiple times safely."""
        # Should not raise even if called multiple times
        register_resolvers()
        register_resolvers()

        # Verify resolvers are registered
        self.assertTrue(OmegaConf.has_resolver("torch_dtype"))
        self.assertTrue(OmegaConf.has_resolver("fsdp_param_strategy"))
        self.assertTrue(OmegaConf.has_resolver("weight_decay_type"))

    def test_torch_dtype_resolver_via_omegaconf(self) -> None:
        """Test torch_dtype resolver works via OmegaConf interpolation."""
        register_resolvers()
        cfg = OmegaConf.create({"dtype": "${torch_dtype:float32}"})
        self.assertEqual(cfg.dtype, torch.float32)

    def test_fsdp_param_strategy_resolver_via_omegaconf(self) -> None:
        """Test fsdp_param_strategy resolver works via OmegaConf interpolation."""
        register_resolvers()
        cfg = OmegaConf.create({"strategy": "${fsdp_param_strategy:DEFAULT}"})
        self.assertEqual(cfg.strategy, FSDPParamAssignmentStrategy.DEFAULT)

    def test_weight_decay_type_resolver_via_omegaconf(self) -> None:
        """Test weight_decay_type resolver works via OmegaConf interpolation."""
        register_resolvers()
        cfg = OmegaConf.create({"wdt": "${weight_decay_type:DECOUPLED}"})
        self.assertEqual(cfg.wdt, WeightDecayType.DECOUPLED)

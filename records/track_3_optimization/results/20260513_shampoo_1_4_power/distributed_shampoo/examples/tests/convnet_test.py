"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest

import torch
from distributed_shampoo.examples.convnet import ConvNet
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class ConvNetTest(unittest.TestCase):
    @parametrize("batch_size", [1, 2, 4, 8])
    @parametrize(
        "height, width",
        [
            (28, 28),  # MNIST-like
            (32, 32),  # CIFAR-10-like
            (64, 64),  # Larger image
            (16, 16),  # Smaller image
            (48, 32),  # Rectangular image
        ],
    )
    def test_forward_pass_different_sizes(
        self, batch_size: int, height: int, width: int
    ) -> None:
        """Test forward pass with different input sizes - indirectly validates _infer_conv_output_shape."""
        model = ConvNet(height=height, width=width)
        input_tensor = torch.randn(batch_size, 3, height, width)

        # If _infer_conv_output_shape calculated sizes correctly, this should work
        output = model(input_tensor)
        self.assertEqual(output.shape, (batch_size, 10))

    @parametrize(
        "model_height, model_width, input_height, input_width",
        [
            (32, 32, 28, 28),  # Model expects 32x32, input is 28x28
            (28, 28, 32, 32),  # Model expects 28x28, input is 32x32
            (64, 64, 32, 32),  # Model expects 64x64, input is 32x32
        ],
    )
    def test_forward_pass_mismatched_input_size(
        self, model_height: int, model_width: int, input_height: int, input_width: int
    ) -> None:
        """Test that forward pass fails with mismatched input size."""
        # Create model expecting specific input size
        model = ConvNet(height=model_height, width=model_width)

        # Try to pass different input size - this should fail because the linear layer
        # was sized for the expected conv output, not the actual conv output
        input_tensor = torch.randn(2, 3, input_height, input_width)

        with self.assertRaises(RuntimeError):
            model(input_tensor)

    @parametrize(
        "height, width",
        [
            (16, 16),
            (32, 32),
            (48, 24),
            (64, 64),
        ],
    )
    def test_model_parameters(self, height: int, width: int) -> None:
        """Test that model parameters can be accessed and have correct shapes."""
        model = ConvNet(height=height, width=width)

        parameters = list(model.parameters())

        # Should have 3 parameters: conv weight, linear weight, linear bias
        # (conv has bias=False)
        self.assertEqual(len(parameters), 3)

        # Check conv weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = parameters[0]
        self.assertEqual(conv_weight.shape, (64, 3, 3, 3))

        # Check linear weight and bias shapes
        linear_weight = parameters[1]
        linear_bias = parameters[2]
        expected_linear_input_size = height * width * 64
        self.assertEqual(linear_weight.shape, (10, expected_linear_input_size))
        self.assertEqual(linear_bias.shape, (10,))

    @parametrize(
        "height, width",
        [
            (3, 3),  # Very small image
            (5, 5),  # Small image
            (7, 9),  # Small rectangular image
        ],
    )
    def test_model_with_edge_case_dimensions(self, height: int, width: int) -> None:
        """Test model with edge case dimensions."""
        model = ConvNet(height=height, width=width)
        input_tensor = torch.randn(1, 3, height, width)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))

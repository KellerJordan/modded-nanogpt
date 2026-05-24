"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import math
import re
import unittest
from operator import methodcaller

import torch
from distributed_shampoo.shampoo_types import LoadBalancingConfig
from distributed_shampoo.utils.load_balancing_utils import (
    PolynomialComputationalCostModel,
)
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    distribute_buffer_sizes,
    generate_pairwise_indices,
    get_dtype_size,
    merge_small_dims,
    multi_dim_split,
    ParameterizeEnterExitContext,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class MergeSmallDimsTest(unittest.TestCase):
    @parametrize("threshold, expected_new_tensor_shape", ((10, (10,)), (1, (2, 5))))
    def test_merge_all_small_dims(
        self, threshold: int, expected_new_tensor_shape: tuple[int, ...]
    ) -> None:
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 2, 5, 1),
                threshold=threshold,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    def test_merge_small_dims_for_single_dim(self) -> None:
        expected_new_tensor_shape = (2,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=torch.Size([2]),
                threshold=10,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    @parametrize("threshold", (10, 1))
    def test_merge_small_dims_all_ones(self, threshold: int) -> None:
        expected_new_tensor_shape = (1,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 1, 1, 1),
                threshold=threshold,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    @parametrize(
        "tensor_shape", ((0,), (0, 1, 5, 10, 20), (1, 5, 0, 10, 20), (1, 5, 10, 20, 0))
    )
    def test_merge_small_dims_empty(self, tensor_shape: tuple[int, ...]) -> None:
        expected_new_tensor_shape = (0,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=tensor_shape, threshold=10, target_tensor_dimensionality=1
            ),
            expected_new_tensor_shape,
        )

    @parametrize("threshold", (10, 1))
    def test_empty_dims(self, threshold: int) -> None:
        expected_new_tensor_shape = (1,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(), threshold=threshold, target_tensor_dimensionality=1
            ),
            expected_new_tensor_shape,
        )

    def test_target_tensor_dimensionality_is_inf(self) -> None:
        expected_new_tensor_shape = (1, 2, 5, 1)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 2, 5, 1),
                threshold=10,
                target_tensor_dimensionality=math.inf,
            ),
            expected_new_tensor_shape,
        )

    @parametrize(
        "threshold, target_tensor_dimensionality, expected_new_tensor_shape",
        [
            (10, 1, (32, 3, 64, 64)),
            (200, 1, (32, 192, 64)),
            (8192, 1, (96, 4096)),
            (1_000_000, 1, (96 * 4096,)),
            (10, 2, (32, 3, 64, 64)),
            (200, 2, (32, 192, 64)),
            (8192, 2, (96, 4096)),
            (
                1_000_000,
                2,
                (32, 3 * 4096),
            ),
            (8192, 1, (96, 4096)),
            (8192, 2, (96, 4096)),
            (8192, 3, (32, 3, 4096)),
            (8192, 4, (32, 3, 64, 64)),
            (8192, math.inf, (32, 3, 64, 64)),
        ],
    )
    def test_convolution_like_dims(
        self,
        threshold: int,
        target_tensor_dimensionality: int,
        expected_new_tensor_shape: tuple[int, ...],
    ) -> None:
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(32, 3, 64, 64),
                threshold=threshold,
                target_tensor_dimensionality=target_tensor_dimensionality,
            ),
            expected_new_tensor_shape,
        )


class MultiDimSplitTest(unittest.TestCase):
    def test_multi_dim_split_for_one_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        expected_split_grad = (
            torch.arange(6).reshape(3, 2),
            torch.arange(6, 10).reshape(2, 2),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=3), expected_split_grad
        )

    def test_multi_dim_split_for_two_dim(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1], [3, 4]]),
            torch.tensor([[2], [5]]),
            torch.tensor([[6, 7], [9, 10]]),
            torch.tensor([[8], [11]]),
            torch.tensor([[12, 13]]),
            torch.tensor([[14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=2), expected_split_grad
        )

    def test_multi_dim_split_without_spliting(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=5), expected_split_grad
        )

    def test_split_size_is_inf(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (grad,)
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=math.inf), expected_split_grad
        )


@instantiate_parametrized_tests
class CompressListTest(unittest.TestCase):
    @parametrize(
        "selector, compressed_tuple",
        (
            ((True, True, False), (1, 2)),
            ((False, True, True), (2, 3)),
            ((True, False, True), (1, 3)),
        ),
    )
    def test_compress_list(
        self, selector: tuple[bool, ...], compressed_tuple: tuple[int,]
    ) -> None:
        self.assertEqual(
            compress_list(complete_list=[1, 2, 3], selector=selector), compressed_tuple
        )

    def test_compress_list_with_different_size(self) -> None:
        self.assertRaisesRegex(
            AssertionError,
            re.escape("Inconsistent lengths"),
            compress_list,
            complete_list=[1, 2, 3],
            selector=(True, False),
        )


@instantiate_parametrized_tests
class GetDTypeSizeTest(unittest.TestCase):
    @parametrize(
        "dtype, size",
        ((torch.int64, 8), (torch.float32, 4), (torch.bfloat16, 2), (torch.bool, 1)),
    )
    def test_get_dtype_size(self, dtype: torch.dtype, size: int) -> None:
        self.assertEqual(get_dtype_size(dtype), size)


class GeneratePairwiseIndicesTest(unittest.TestCase):
    def test_generate_pairwise_indices(self) -> None:
        input_tuple = (1, 3, 2)
        expected_pairwise_indices = [(0, 1), (1, 4), (4, 6)]
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )

    def test_generate_pairwise_indices_with_empty_list(self) -> None:
        input_tuple = ()
        expected_pairwise_indices: list[int] = []
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )


class ParameterizeEnterExitContextTest(unittest.TestCase):
    """Test suite for the ParameterizeEnterExitContext class.

    This test case verifies the functionality of the ParameterizeEnterExitContext
    class, ensuring that the enter and exit methods are called correctly on the
    input object, and that the object's state is modified as expected.
    """

    def test_parameterize_enter_exit_context(self) -> None:
        """Test the enter and exit context management.

        This test creates an instance of a TestClass, which has enter and exit
        methods that modify an internal variable. It then uses the
        ParameterizeEnterExitContext to ensure that the enter method is called
        upon entering the context and the exit method is called upon exiting,
        verifying the changes in the internal state of the TestClass instance.
        """

        class TestClass:
            def __init__(self) -> None:
                self._test_var = 0

            def enter(self) -> None:
                self._test_var = 1

            def exit(self) -> None:
                self._test_var = -1

            @property
            def test_var(self) -> int:
                return self._test_var

        test_class = TestClass()
        with ParameterizeEnterExitContext(
            input_with_enter_exit_context=test_class,
            enter_method_caller=methodcaller("enter"),
            exit_method_caller=methodcaller("exit"),
        ):
            # Due to the invocation of test_class.enter(), the state of test_class.test_var should be 1.
            self.assertEqual(test_class.test_var, 1)

        # Due to the invocation of test_class.exit(), the state of test_class.test_var should be -1.
        self.assertEqual(test_class.test_var, -1)


@instantiate_parametrized_tests
class DistributeBufferSizesTest(unittest.TestCase):
    @staticmethod
    def empty_tensor_list(sizes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(size, device="meta", dtype=torch.bool) for size in sizes
        )

    @parametrize(
        "blocked_params, group_size, load_balancing_config, expected_result",
        (
            # Test case 1: Even distribution of buffer sizes
            (
                empty_tensor_list((128, 64, 500, 256)),
                2,
                LoadBalancingConfig(),
                (
                    (128, 1),
                    (64, 1),
                    (512, 0),
                    (256, 1),
                ),
            ),
            # Test case 2: Single group
            (
                empty_tensor_list((128, 64, 500, 256)),
                1,
                LoadBalancingConfig(),
                (
                    (128, 0),
                    (64, 0),
                    (512, 0),
                    (256, 0),
                ),
            ),
            # Test case 3: More groups than buffers
            (
                empty_tensor_list((128, 64)),
                4,
                LoadBalancingConfig(),
                ((128, 0), (64, 1)),
            ),
            # Test case 4: Empty buffer sizes
            ((), 2, LoadBalancingConfig(), ()),
            # Test case 5: Linear Compute Cost Model
            (
                empty_tensor_list((128, 64, 512, 256)),
                2,
                LoadBalancingConfig(
                    cost_model=PolynomialComputationalCostModel(coefficients=(0, 2))
                ),
                (
                    (128, 1),
                    (64, 1),
                    (512, 0),
                    (256, 1),
                ),
            ),
            # Test case 6: Quadratic Compute Cost Model
            # Rank 0 gets tensors of sizes 256, 256, and 128, Rank 1 gets a tensor of size 384.
            # The quadratic computational cost of Rank 0 equals that of Rank 1: 384² = 256² + 256² + 128²
            (
                empty_tensor_list((256, 128, 384, 256)),
                2,
                LoadBalancingConfig(
                    cost_model=PolynomialComputationalCostModel(coefficients=(0, 0, 1))
                ),
                (
                    (256, 1),
                    (128, 1),
                    (384, 0),
                    (256, 1),
                ),
            ),
        ),
    )
    def test_distribute_buffer_sizes(
        self,
        blocked_params: tuple[torch.Tensor, ...],
        group_size: int,
        load_balancing_config: LoadBalancingConfig,
        expected_result: tuple[tuple[int, int], ...],
    ) -> None:
        self.assertEqual(
            distribute_buffer_sizes(
                blocked_params=blocked_params,
                group_size=group_size,
                load_balancing_config=load_balancing_config,
            ),
            expected_result,
        )

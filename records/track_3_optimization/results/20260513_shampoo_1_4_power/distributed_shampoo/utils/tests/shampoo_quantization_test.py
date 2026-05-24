"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import copy
import re
import unittest
from unittest import mock

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.utils import shampoo_quantization
from distributed_shampoo.utils.shampoo_quantization import (
    _FLOAT_DTYPES,
    DequantizeQuantizedTensorListContext,
    QuantizedTensor,
    QuantizedTensorList,
)
from torch.testing._comparison import default_tolerances


class QuantizedTensorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._base_tensor = torch.rand(10)
        self._quantized_tensor = QuantizedTensor.init_from_dequantized_tensor(
            dequantized_values=self._base_tensor,
            quantized_dtype=torch.float16,
            block_info=BlockInfo(torch.zeros(10), (0, "dummy")),
        )

    def test_init_float_from_dequantized(self) -> None:
        torch.testing.assert_close(
            self._quantized_tensor.quantized_values,
            self._base_tensor,
            check_dtype=False,
        )
        self.assertIsNone(self._quantized_tensor.min_value)
        self.assertIsNone(self._quantized_tensor.max_value)

    def test_dequantize_quantize(self) -> None:
        for dequantized_dtype in _FLOAT_DTYPES:
            # Set max_rtol and max_atol for the given dequantized_dtype and quantized dtype.
            max_rtol, max_atol = default_tolerances(
                self._quantized_tensor.quantized_values.dtype, dequantized_dtype
            )

            # Test dequantization.
            deq_tensor = self._quantized_tensor.dequantize(
                dequantized_dtype=dequantized_dtype
            )
            with self.subTest(dequantized_dtype=dequantized_dtype):
                torch.testing.assert_close(
                    deq_tensor,
                    self._base_tensor,
                    check_dtype=False,
                    atol=max_atol,
                    rtol=max_rtol,
                )

            # Copy self._quantized_tensor to revert the effect of the quantization below.
            quantized_tensor_copy = copy.deepcopy(self._quantized_tensor)

            # Test quantization with delta.
            delta = 1.0
            deq_tensor.add_(delta)

            self._quantized_tensor.quantize(dequantized_tensor=deq_tensor)
            with self.subTest(dequantized_dtype=dequantized_dtype):
                torch.testing.assert_close(
                    self._quantized_tensor.quantized_values,
                    self._base_tensor + delta,
                    check_dtype=False,
                    atol=max_atol,
                    rtol=max_rtol,
                )

            # Restore self._quantized_tensor before the next test.
            self._quantized_tensor = quantized_tensor_copy

    def test_invalid_quantized_data_type(self) -> None:
        self.assertRaisesRegex(
            NotImplementedError,
            re.escape("Quantization for torch.int64 is not yet supported!"),
            QuantizedTensor.init_from_dequantized_tensor,
            dequantized_values=torch.rand(10),
            quantized_dtype=torch.int64,
            block_info=BlockInfo(torch.zeros(10), (0, "dummy")),
        )

        quantized_tensor = QuantizedTensor(
            quantized_values=torch.zeros(10, dtype=torch.int64),
            block_info=BlockInfo(torch.zeros(10), (0, "dummy")),
        )
        self.assertRaisesRegex(
            NotImplementedError,
            re.escape("Quantization for torch.int64 is not yet supported!"),
            quantized_tensor.dequantize,
            dequantized_dtype=torch.float16,
        )


class QuantizedTensorListInitTest(unittest.TestCase):
    def test_invalid_quantized_data_type(self) -> None:
        with mock.patch.object(
            shampoo_quantization,
            "isinstance",
            side_effect=lambda object, classinfo: False,
        ):
            self.assertRaisesRegex(
                TypeError,
                re.escape(
                    "quantized_data must be collections.abc.Sequence[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] | collections.abc.Sequence[distributed_shampoo.utils.shampoo_quantization.QuantizedTensor] but get <class 'list'>"
                ),
                QuantizedTensorList,
                quantized_data=[
                    (torch.randn(2, 2, dtype=torch.float16), None, None)
                    for _ in range(5)
                ],
                quantized_dtype=torch.float16,
                computation_dtype=torch.float64,
            )

    def test_invalid_computation_dtype(self) -> None:
        self.assertRaisesRegex(
            AssertionError,
            re.escape(
                "computation_dtype=torch.int64 is not supported! It must be one of (torch.float16, torch.bfloat16, torch.float32, torch.float64)!"
            ),
            QuantizedTensorList,
            quantized_data=[
                (torch.randn(2, 2, dtype=torch.float16), None, None) for _ in range(5)
            ],
            quantized_dtype=torch.float16,
            computation_dtype=torch.int64,
        )


class QuantizedTensorListTest(unittest.TestCase):
    def setUp(self) -> None:
        self._base_tensors = tuple(
            torch.randn(10, 10, dtype=torch.float16) for _ in range(5)
        )
        self._quantized_tensors = QuantizedTensorList(
            tuple((torch.clone(tensor), None, None) for tensor in self._base_tensors),
            quantized_dtype=torch.float16,
            computation_dtype=torch.float64,
        )

    def test_init_from_quantized_tensors(self) -> None:
        quantized_tensors = [
            QuantizedTensor(
                torch.ones(10, dtype=torch.float16) * i,
                BlockInfo(torch.zeros(10), (i, "dummy")),
            )
            for i in range(5)
        ]
        quantized_tensor_list = QuantizedTensorList(
            quantized_tensors, torch.float16, torch.float64
        )
        for i, tensor in enumerate(quantized_tensor_list.quantized_value):
            self.assertFalse(torch.any(torch.nonzero(tensor - i)))

    def test_len(self) -> None:
        self.assertEqual(len(self._quantized_tensors), 5)

    def test_compress(self) -> None:
        selector = (True, False, True, False, True)
        compressed_quantized_tensors = self._quantized_tensors.compress(selector)
        self.assertEqual(
            len(compressed_quantized_tensors.quantized_value), selector.count(True)
        )

    def test_dequantize_with_same_quantize_dtype_and_computation_dtype(self) -> None:
        quantized_tensors = QuantizedTensorList(
            tuple((torch.clone(tensor), None, None) for tensor in self._base_tensors),
            # Explicitly set quantized_dtype and computation_dtype to be the same.
            quantized_dtype=torch.float16,
            computation_dtype=torch.float16,
        )

        # Value should be the same as the original tensor.
        torch.testing.assert_close(
            quantized_tensors.dequantize(), quantized_tensors.quantized_value
        )

        # Because the quantized_dtype and computation_dtype are the same, it never creates dequantized_value_list.
        self.assertFalse(quantized_tensors.is_dequantized_stored())

    def test_dequantize_quantize(self) -> None:
        deq_tensors = self._quantized_tensors.dequantize()
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(deq_tensors, self._base_tensors, strict=True):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        delta = 1.0
        torch._foreach_add_(deq_tensors, delta)

        self._quantized_tensors.quantize(deq_tensors)
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_close(quantized_tensor, tensor + delta)

        # Calling quantize() while dequantize_value_list exists should trigger warning message.
        # Leverage DequantizeQuantizedTensorListContext to call dequantize_() and quantize_() automatically.
        with DequantizeQuantizedTensorListContext(
            quantized_tensor_list=self._quantized_tensors
        ):
            with self.assertLogs(
                level="WARNING",
            ) as cm:
                self._quantized_tensors.quantize(deq_tensors)
                self.assertIn(
                    "Existing stored dequantized values.\nWriting quantized values with input tensor_list without using these stored dequantized values...",
                    [r.msg for r in cm.records],
                )

    def test_inplace_dequantize_quantize(self) -> None:
        # Calling quantize_() before dequantize_() should trigger warning message.
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            self._quantized_tensors.quantize_()
            self.assertIn(
                "No stored dequantized values self.dequantized_value_list=None. Must first call dequantize_().",
                [r.msg for r in cm.records],
            )

        self._quantized_tensors.dequantize_()
        self.assertTrue(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(
            self._quantized_tensors.dequantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        # Calling dequantize_() before consuming already stored dequantized value should trigger warning message.
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            self._quantized_tensors.dequantize_()
            self.assertIn(
                "Dequantized values are already stored; overwriting these values...",
                [r.msg for r in cm.records],
            )
        # All dequantized values should be there without any change because no changes in quantized values.
        self.assertTrue(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(
            self._quantized_tensors.dequantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        delta = 1.0
        torch._foreach_add_(self._quantized_tensors.dequantized_value, delta)

        self._quantized_tensors.quantize_()
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_close(quantized_tensor, tensor + delta)

    def test_invalid_quantized_data_type(self) -> None:
        quantized_tensors = QuantizedTensorList(
            quantized_data=[
                (torch.zeros(5, dtype=torch.int64), None, None) for _ in range(5)
            ],
            quantized_dtype=torch.int64,
            computation_dtype=torch.float64,
        )
        self.assertRaisesRegex(
            NotImplementedError,
            re.escape("Quantization for torch.int64 is not yet supported!"),
            quantized_tensors.dequantize,
        )

        self.assertRaisesRegex(
            NotImplementedError,
            re.escape("Quantization for torch.int64 is not yet supported!"),
            quantized_tensors.quantize,
            tensor_list=tuple(torch.rand(5) for _ in range(5)),
        )

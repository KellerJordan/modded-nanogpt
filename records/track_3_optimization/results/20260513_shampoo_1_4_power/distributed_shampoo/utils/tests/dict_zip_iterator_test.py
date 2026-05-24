"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from collections.abc import Iterator

from distributed_shampoo.utils.dict_zip_iterator import DictZipIterator
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class DictZipIteratorTest(unittest.TestCase):
    @parametrize(
        "data, expected_results",
        (
            (
                {
                    "a": [1, 2, 3],
                    "b": ["x", "y", "z"],
                    "c": [True, False, True],
                },
                [
                    {"a": 1, "b": "x", "c": True},
                    {"a": 2, "b": "y", "c": False},
                    {"a": 3, "b": "z", "c": True},
                ],
            ),
        ),
    )
    def test_dict_zip_iterator_same_length(
        self,
        data: dict[str, Iterator[object]],
        expected_results: list[dict[str, object]],
    ) -> None:
        iterator = DictZipIterator(data=data)

        self.assertEqual(expected_results, list(iterator))

        # Test StopIteration when all iterators are exhausted
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_dict_zip_iterator_different_lengths_raises_error(self) -> None:
        """Test that ValueError is raised when iterators have different lengths."""
        data = {
            "a": iter([1, 2, 3]),
            "b": iter(["x", "y", "z"]),
            "c": iter([True, False]),  # Shorter list
        }
        iterator = DictZipIterator(data=data)

        # First iteration should work
        result1 = next(iterator)
        self.assertEqual(result1, {"a": 1, "b": "x", "c": True})

        # Second iteration should work
        result2 = next(iterator)
        self.assertEqual(result2, {"a": 2, "b": "y", "c": False})

        # Third iteration should raise ValueError because 'c' is exhausted but 'a' and 'b' still have values
        with self.assertRaises(ValueError) as context:
            next(iterator)

        error_message = str(context.exception)
        self.assertIn("Iterators have different lengths", error_message)
        self.assertIn("Exhausted: ['c']", error_message)
        self.assertIn("Still active: ['a', 'b']", error_message)

    def test_dict_zip_iterator_empty_iterators(self) -> None:
        """Test that empty iterators raise StopIteration immediately."""
        data: dict[str, Iterator[object]] = {
            "a": iter([]),
            "b": iter([]),
        }
        iterator = DictZipIterator(data=data)

        # Should raise StopIteration immediately
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_dict_zip_iterator_single_element(self) -> None:
        """Test with single element iterators."""
        data = {
            "a": iter([1]),
            "b": iter(["x"]),
            "c": iter([True]),
        }
        iterator = DictZipIterator(data=data)

        # First iteration should work
        result = next(iterator)
        self.assertEqual(result, {"a": 1, "b": "x", "c": True})

        # Second iteration should raise StopIteration
        with self.assertRaises(StopIteration):
            next(iterator)

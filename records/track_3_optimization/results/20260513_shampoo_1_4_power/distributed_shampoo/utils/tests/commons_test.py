"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest
from abc import ABC, abstractmethod

from distributed_shampoo.utils.commons import batched, get_all_non_abstract_subclasses
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class BatchedTest(unittest.TestCase):
    def test_normal_batching(self) -> None:
        """Test batching an iterable with size divisible by batch size."""
        data = [1, 2, 3, 4, 5, 6]
        result = list(batched(data, n=2))
        self.assertEqual(result, [(1, 2), (3, 4), (5, 6)])

    def test_uneven_batching(self) -> None:
        """Test batching an iterable with size not divisible by batch size."""
        data = [1, 2, 3, 4, 5]
        result = list(batched(data, n=2))
        self.assertEqual(result, [(1, 2), (3, 4), (5,)])

    def test_empty_iterable(self) -> None:
        """Test batching an empty iterable."""
        data: list[int] = []
        result = list(batched(data, n=3))
        self.assertEqual(result, [])

    def test_batch_size_one(self) -> None:
        """Test batching with batch size of 1."""
        data = [1, 2, 3]
        result = list(batched(data, n=1))
        self.assertEqual(result, [(1,), (2,), (3,)])

    @parametrize("n", (-1, 0))
    def test_invalid_batch_size(self, n: int) -> None:
        """Test that batched raises ValueError for batch size < 1."""
        data = [1, 2, 3]
        with self.assertRaisesRegex(
            ValueError, re.escape(f"{n=} must be at least one")
        ):
            list(batched(data, n=n))


class DummyRootClass:
    """Dummy root class for GetAllNonAbstractSubclassesTest."""


class DummyFirstSubclass(DummyRootClass):
    """First dummy subclass for GetAllNonAbstractSubclassesTest."""


class DummySecondSubclass(DummyFirstSubclass):
    """Second dummy subclass for GetAllNonAbstractSubclassesTest."""


class DummySecondRootClass:
    """Second dummy root class for GetAllNonAbstractSubclassesTest."""


class DummyMixedSubclass(DummySecondRootClass, DummySecondSubclass):
    """Dummy subclass with mixed inheritance for GetAllNonAbstractSubclassesTest."""


class DummyAbstractSubclass(DummyMixedSubclass, ABC):
    """Dummy abstract subclass for GetAllNonAbstractSubclassesTest."""

    @abstractmethod
    def __init__(self) -> None:
        """An abstract method that must be implemented by all subclasses. This abstract method will be ignored by get_all_non_abstract_subclasses()."""


class DummyLeafClass(DummyAbstractSubclass):
    """Dummy leaf class for GetAllNonAbstractSubclassesTest."""

    def __init__(self) -> None:
        """This is a non-abstract method, so get_all_non_abstract_subclasses() will include this class."""


@instantiate_parametrized_tests
class GetAllNonAbstractSubclassesTest(unittest.TestCase):
    @parametrize(
        "cls, expected_subclasses",
        (
            (
                DummyRootClass,
                [
                    DummyRootClass,
                    DummyFirstSubclass,
                    DummySecondSubclass,
                    DummyMixedSubclass,
                    DummyLeafClass,
                ],
            ),
            (
                DummySecondRootClass,
                [DummySecondRootClass, DummyMixedSubclass, DummyLeafClass],
            ),
            (
                DummySecondSubclass,
                [DummySecondSubclass, DummyMixedSubclass, DummyLeafClass],
            ),
            (DummyAbstractSubclass, [DummyLeafClass]),
            (DummyLeafClass, [DummyLeafClass]),
        ),
    )
    def test_all_non_abstract_subclasses(
        self, cls: object, expected_subclasses: list[object]
    ) -> None:
        self.assertCountEqual(
            get_all_non_abstract_subclasses(cls=cls), expected_subclasses
        )

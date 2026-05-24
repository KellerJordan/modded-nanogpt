"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Iterable
from functools import reduce
from itertools import islice
from operator import methodcaller, or_
from typing import TypeVar


_SubclassesType = TypeVar("_SubclassesType")


def get_all_non_abstract_subclasses(cls: _SubclassesType) -> Iterable[_SubclassesType]:
    """
    Retrieves all non-abstract (instantiable) subclasses of a given class.

    This function uses a helper function to recursively find all unique subclasses
    of the specified class, and then filters out any abstract classes.

    Args:
        cls (_SubclassesType): The class for which to find subclasses.

    Returns:
        non_abstract_subclasses (Iterable[_SubclassesType]): An iterable of all unique non-abstract subclasses of the given class.
    """

    def get_all_unique_subclasses(cls: _SubclassesType) -> set[_SubclassesType]:
        """Gets all unique subclasses of a given class recursively."""
        return reduce(
            or_,
            map(get_all_unique_subclasses, methodcaller("__subclasses__")(cls)),
            {cls},
        )

    return filter(
        # Filters out abstract classes by checking if '__abstractmethods__' is an empty set or not present.
        lambda sub_cls: not getattr(sub_cls, "__abstractmethods__", frozenset()),
        get_all_unique_subclasses(cls),
    )


_BatchedInputType = TypeVar("_BatchedInputType")


def batched(
    iterable: Iterable[_BatchedInputType], n: int
) -> Iterable[tuple[_BatchedInputType, ...]]:
    """
    Batches an iterable into chunks of size n.

    Note: This is a re-implementation of itertools.batched which is available in Python 3.12+.
    Consider replacing usages with itertools.batched since Python 3.12 is the minimum supported version.

    Args:
        iterable (Iterable[_BatchedInputType]): The iterable to be batched.
        n (int): The size of each batch.

    Yields:
        batched_tuple (tuple[_BatchedInputType, ...]): A generator that yields batches of size n.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError(f"{n=} must be at least one")

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

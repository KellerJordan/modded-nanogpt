"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Iterator
from typing import Generic, TypeVar

_DictValType = TypeVar("_DictValType")


class DictZipIterator(Generic[_DictValType], Iterator[dict[str, _DictValType]]):
    """
    Iterator that yields dictionaries by zipping values from iterators in a dictionary.

    Given a dictionary mapping from strings to iterators,
    each iteration yields a new dictionary with the same keys but with values
    taken from the corresponding position in each iterator.

    All iterators must have the same length. If iterators have different lengths,
    a ValueError will be raised when the length mismatch is detected.

    Example:
        # Same length iterators - works correctly
        data = {
            "a": iter([1, 2, 3]),
            "b": iter(["x", "y", "z"]),
            "c": iter([True, False, True])
        }

        iterator = DictZipIterator(data)

        # First iteration: {"a": 1, "b": "x", "c": True}
        # Second iteration: {"a": 2, "b": "y", "c": False}
        # Third iteration: {"a": 3, "b": "z", "c": True}
        # StopIteration is raised after all iterators are exhausted

        # Different length iterators - raises ValueError
        data_mismatched = {
            "a": iter([1, 2, 3]),
            "b": iter(["x", "y"]),  # Shorter iterator
        }

        iterator = DictZipIterator(data_mismatched)
        next(iterator)  # {"a": 1, "b": "x"}
        next(iterator)  # {"a": 2, "b": "y"}
        next(iterator)  # Raises ValueError: Iterators have different lengths
    """

    def __init__(self, data: dict[str, Iterator[_DictValType]]) -> None:
        """
        Initialize the iterator with a dictionary mapping from strings to iterators.

        Args:
            data: Dictionary mapping from strings to iterators

        Returns:
            None
        """
        self._keys: tuple[str, ...] = tuple(data.keys())
        # Create an iterator for each iterable in the dictionary
        self._iterators: dict[str, Iterator[_DictValType]] = {
            key: iter(value) for key, value in data.items()
        }

    def __iter__(self) -> Iterator[dict[str, _DictValType]]:
        """Return self as iterator."""
        return self

    def __next__(self) -> dict[str, _DictValType]:
        """
        Return the next dictionary with values from the current position.

        Returns:
            Dictionary with the same keys as the input dictionary, but with values
            from the current position in each iterable

        Raises:
            StopIteration: When all iterables are exhausted at the same time
            ValueError: When iterables have different lengths (some exhausted while others have values)
        """
        result = {}
        exhausted_keys = []
        active_keys = []

        # Try to get next value from each iterator
        for key in self._keys:
            try:
                result[key] = next(self._iterators[key])
                active_keys.append(key)
            except StopIteration:
                exhausted_keys.append(key)

        # If some iterators are exhausted but not all, raise an error
        if exhausted_keys and active_keys:
            raise ValueError(
                f"Iterators have different lengths. "
                f"Exhausted: {exhausted_keys}, Still active: {active_keys}"
            )

        # If all iterators are exhausted, stop iteration
        if exhausted_keys:
            raise StopIteration

        return result

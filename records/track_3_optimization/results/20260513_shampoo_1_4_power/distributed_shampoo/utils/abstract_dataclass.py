"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(init=False)
class AbstractDataclass(ABC):
    """
    An abstract base class for creating dataclasses.
    This class provides a basic structure for creating dataclasses, ensuring that
    all subclasses implement their own `__init__` method. It does not generate an
    `__init__` method by default to allow for custom initialization logic in
    subclasses.

    Note that the `init=False` parameter is explicitly set here, which is the
    opposite of the default behavior of the `@dataclass` decorator, where
    `init=True` by default. By setting `init=False`, we prevent the automatic
    generation of an `__init__` method in the subclass, allowing the subclass
    to define its own `__init__` method. The abstract `__init__` method defined here
    must be implemented by all subclasses, either by allowing the `@dataclass` decorator
    to auto-generate it (by not setting `init=False`) or by providing a manual implementation.

    If you want to keep this abstract property in your subclasses, make sure to set
    `init=False` in your subclass definition as well; otherwise, `@dataclass`
    automatically generates an `__init__` method to make it a concrete dataclass.

    Following is the example usage:
    ```
    @dataclass(init=False)
    class ChildAbstractDataclass(AbstractDataclass):
    # Not able to instantiate this dataclass.

    @dataclass(init=False)
    class GrandchildAbstractDataclass(ChildAbstractDataclass):
    # Still not able to instantiate this dataclass.

    @dataclass
    class EmptyConcreteDataclass(AbstractDataclass):
    # A dataclass with no field.
    ```

    """

    @abstractmethod
    def __init__(self, *args: object, **kwargs: object) -> None:
        """An abstract method that must be implemented by all subclasses."""

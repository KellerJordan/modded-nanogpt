"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest
from dataclasses import dataclass

from distributed_shampoo.utils.abstract_dataclass import AbstractDataclass
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class InvalidAbstractDataclassInitTest(unittest.TestCase):
    @dataclass(init=False)
    class DummyOptimizerConfig(AbstractDataclass):
        """Dummy abstract dataclass for testing. Instantiation should fail."""

    @parametrize("abstract_cls", (AbstractDataclass, DummyOptimizerConfig))
    def test_invalid_init(self, abstract_cls: type[AbstractDataclass]) -> None:
        self.assertRaisesRegex(
            TypeError,
            re.escape(f"Can't instantiate abstract class {abstract_cls.__name__} "),
            abstract_cls,
        )

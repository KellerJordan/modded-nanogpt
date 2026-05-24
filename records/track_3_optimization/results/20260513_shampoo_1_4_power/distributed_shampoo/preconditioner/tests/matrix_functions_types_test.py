"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest

from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigenConfig,
    EigendecompositionConfig,
)
from distributed_shampoo.utils.commons import get_all_non_abstract_subclasses
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class EigendecompositionConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[EigendecompositionConfig]] = list(
        get_all_non_abstract_subclasses(
            EigendecompositionConfig  # type: ignore[type-abstract]
        )
    )

    # tolerance has to be in the interval [0.0, 1.0].
    @parametrize("tolerance", (-1.0, 1.1))
    @parametrize("cls", subclasses_types)
    def test_illegal_tolerance(
        self, cls: type[EigendecompositionConfig], tolerance: float
    ) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid tolerance value: {tolerance}. Must be in the interval [0.0, 1.0]."
            ),
            cls,
            tolerance=tolerance,
        )


@instantiate_parametrized_tests
class EigenConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[EigenConfig]] = list(
        get_all_non_abstract_subclasses(EigenConfig)
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_tolerance(self, cls: type[EigenConfig]) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid tolerance value: 0.01. Must be 0.0 for {cls.__name__}."
            ),
            cls,
            tolerance=0.01,
        )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest

import torch
from distributed_shampoo.utils.optimizer_modules import OptimizerModule
from distributed_shampoo.utils.shampoo_state_dict_utils import (
    extract_state_dict_content,
    update_param_state_dict_object,
)
from torch import Tensor


class DummyOptimizerModule(OptimizerModule):
    def __init__(
        self,
        field: Tensor,
        thl: list[Tensor],
    ) -> None:
        super().__init__()
        self._field: Tensor = field
        self._thl: list[Tensor] = thl

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DummyOptimizerModule)
            and bool((self._field == other._field).item())
            and self._thl == other._thl
        )


class ExtractStateDictContentTest(unittest.TestCase):
    def test_extract_state_dict_content(self) -> None:
        state_dict = {
            "inner_dict": {
                "foo": torch.tensor(0),
                "bar": DummyOptimizerModule(
                    field=torch.tensor(1.0), thl=[torch.tensor(2.0), torch.tensor(3.0)]
                ),
            },
        }
        self.assertEqual(
            extract_state_dict_content(state_dict),
            {
                "inner_dict": {
                    "foo": torch.tensor(0),
                    "bar": {
                        "_field": torch.tensor(1.0),
                        "_thl": {
                            0: torch.tensor(2.0),
                            1: torch.tensor(3.0),
                        },
                    },
                },
            },
        )


class UpdateParamStateDictObjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._current_state_dict = {
            "123": {
                123: {
                    0.1: {
                        True: {
                            None: torch.tensor(1.2),
                        },
                        False: {
                            "foo": torch.tensor(-0.3),
                        },
                    },
                },
            },
            "foo": torch.tensor(-3),
            "bar": DummyOptimizerModule(
                field=torch.tensor(1.0),
                thl=[torch.tensor(2.0), torch.tensor(3.0)],
            ),
            "deepcopy_key": [],
        }
        self._extracted_state_dict = {
            "123": {
                123: {
                    0.1: {
                        True: {
                            None: torch.tensor(0.0),
                        },
                        False: {
                            "foo": torch.tensor(0.1),
                        },
                    },
                },
            },
            "foo": torch.tensor(0),
            "bar": {
                "_field": torch.tensor(3.0),
                "_thl": {
                    0: torch.tensor(4.0),
                    1: torch.tensor(5.0),
                },
            },
            "deepcopy_key": ["deepcopy_value"],
        }

    def test_update_param_state_dict_object(self) -> None:
        update_param_state_dict_object(
            current_param_state_dict=self._current_state_dict,
            param_state_dict_to_load=self._extracted_state_dict,
        )

        self.assertEqual(
            self._current_state_dict,
            {
                "123": {
                    123: {
                        0.1: {
                            True: {
                                None: torch.tensor(0.0),
                            },
                            False: {
                                "foo": torch.tensor(0.1),
                            },
                        },
                    },
                },
                "foo": torch.tensor(0),
                "bar": DummyOptimizerModule(
                    field=torch.tensor(3.0),
                    thl=[torch.tensor(4.0), torch.tensor(5.0)],
                ),
                "deepcopy_key": ["deepcopy_value"],
            },
        )

    def test_update_param_state_dict_object_with_missing_key(self) -> None:
        # Explicitly delete 'bar' from self._extracted_state_dict to simulate a missing key.
        self._extracted_state_dict.pop("bar")
        self.assertRaisesRegex(
            KeyError,
            re.escape("Key bar not found in state dict to load."),
            update_param_state_dict_object,
            current_param_state_dict=self._current_state_dict,
            param_state_dict_to_load=self._extracted_state_dict,
        )

        # Disable enable_missing_key_check so the exception will be suppressed, and the value of 'bar' key will not be updated.
        update_param_state_dict_object(
            current_param_state_dict=self._current_state_dict,
            param_state_dict_to_load=self._extracted_state_dict,
            enable_missing_key_check=False,
        )
        self.assertEqual(
            self._current_state_dict,
            {
                "123": {
                    123: {
                        0.1: {
                            True: {
                                None: torch.tensor(0.0),
                            },
                            False: {
                                "foo": torch.tensor(0.1),
                            },
                        },
                    },
                },
                "foo": torch.tensor(0),
                # Following field will not be updated because it does not exist in self._extracted_state_dict.
                "bar": DummyOptimizerModule(
                    field=torch.tensor(1.0),
                    thl=[torch.tensor(2.0), torch.tensor(3.0)],
                ),
                "deepcopy_key": ["deepcopy_value"],
            },
        )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from copy import deepcopy
from typing import Any

import torch
from distributed_shampoo.utils.optimizer_modules import OptimizerModule


logger: logging.Logger = logging.getLogger(__name__)


def update_param_state_dict_object(
    current_param_state_dict: dict[str, Any],
    param_state_dict_to_load: dict[str, Any],
    enable_missing_key_check: bool = True,
) -> None:
    for k, v in current_param_state_dict.items():
        if k not in param_state_dict_to_load:
            if enable_missing_key_check:
                raise KeyError(f"Key {k} not found in state dict to load.")
            else:
                logger.warning(f"Key {k} not found in state dict to load.")
                continue

        if isinstance(v, dict):
            update_param_state_dict_object(
                v,
                param_state_dict_to_load[k],
                enable_missing_key_check,
            )
        elif hasattr(v, "load_state_dict") and callable(v.load_state_dict):
            v.load_state_dict(param_state_dict_to_load[k])
        elif isinstance(v, torch.Tensor):
            v.detach().copy_(param_state_dict_to_load[k])
        else:
            current_param_state_dict[k] = deepcopy(param_state_dict_to_load[k])


def extract_state_dict_content(
    input_dict: dict[str, Any],
) -> dict[str, Any]:
    """Converts nested dictionary with objects with state dict functionality.

    Args:
        input_dict (dict[str, Any]): Nested dictionary containing objects with
            state dict functionality.

    Output:
        output_dict (dict[str, Any]): Nested dictionary where the terminal values
            cannot have state dict functionality.

    """

    def parse_value(
        value: dict[str, Any] | torch.Tensor | OptimizerModule,
    ) -> dict[str, Any] | torch.Tensor:
        if isinstance(value, dict):
            return extract_state_dict_content(value)
        elif isinstance(value, OptimizerModule):
            return value.state_dict()
        else:
            return value

    return {k: parse_value(v) for k, v in input_dict.items()}

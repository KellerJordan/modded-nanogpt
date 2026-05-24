"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, TypeVar

import torch
from torch.optim.optimizer import StateDict

logger: logging.Logger = logging.getLogger(__name__)

_StateType = TypeVar("_StateType")


class OptimizerModule:
    r"""
    Optimizer module that supports state_dict and load_state_dict functions that recursively
    construct the state dictionary by examining other OptimizerModule objects. Similar to
    nn.Module but "trims the fat" by removing unnecessary functions for more general optimizer
    modules.

    When generating the state_dict, looks at the internal dictionary and recursively calls state_dict
    on other optimizer modules.

    """

    def state_dict(
        self,
        destination: StateDict | None = None,
        keep_vars: bool = False,
        store_non_tensors: bool = False,
    ) -> StateDict:
        r"""Returns a nested state dictionary containing the whole internal
        dict of the module. OptimizerModules and other common data structures
        are represented by a dictionary within the dict.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (StateDict | None): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            keep_vars (bool): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            store_non_tensors (bool): flag for storing non-tensor
                objects. Default: ``False``.

        Returns:
            dict:
                a dictionary containing the whole state of the module

        """

        def remove_empty_entry(d: dict[str, Any], key: str) -> None:
            if not d[key]:
                del d[key]

        def save_to_state_dict(
            states: Iterable[tuple[str, _StateType]], destination: StateDict
        ) -> None:
            r"""Saves module state to `destination` dictionary, containing a state
            of the module, but not its descendants. This is called on every
            submodule in :meth:`~OptimizerModule.state_dict`.

            In rare cases, subclasses can achieve class-specific behavior by
            overriding this method with custom logic.

            Args:
                states (Iterable[tuple[str, StateType]]): iterable that gives tuples of values to be stored
                    in destination dict
                destination (StateDict): a dict where state will be stored

            """

            for key, value in states:
                if isinstance(value, torch.Tensor):
                    destination[key] = value if keep_vars else value.detach()
                elif isinstance(value, OptimizerModule):
                    destination[key] = {}
                    value.state_dict(
                        destination=destination[key],
                        keep_vars=keep_vars,
                        store_non_tensors=store_non_tensors,
                    )
                elif isinstance(value, dict):
                    destination[key] = {}
                    save_to_state_dict(
                        states=value.items(),
                        destination=destination[key],
                    )
                    remove_empty_entry(destination, key)
                elif isinstance(value, (list, tuple, set)):
                    destination[key] = {}
                    save_to_state_dict(
                        # Note: mypy is right on this typing error but it is impossible to flatten one more level of codes to eliminate this.
                        states=enumerate(value),  # type: ignore[arg-type]
                        destination=destination[key],
                    )
                    remove_empty_entry(destination, key)
                elif store_non_tensors:
                    destination[key] = value

        if destination is None:
            destination = {}

        save_to_state_dict(self.__dict__.items(), destination)
        return destination

    def load_state_dict(
        self, state_dict: StateDict, store_non_tensors: bool = False
    ) -> None:
        """
        This implementation requires the stored and loaded states to be fully initialized.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (StateDict): State dictionary to load
            store_non_tensors (bool): Load non-tensor objects

        """

        def _convert_state_key_from_str_to_int(
            old_state: list[Any], new_state: dict[int, Any] | dict[str, Any]
        ) -> dict[int, Any]:
            """
            Convert new_state dictionary keys to integers to match old_state collection indices.

            When optimizer state dictionaries are flattened (e.g., through PyTorch state_dict API
            `get_optimizer_state_dict` with `flatten_optimizer_state_dict` set to  `True`), integer
            keys used for indexing collections get converted to strings. This function converts
            those string keys back to integers so they can be properly matched with the integer
            indices from enumerate(old_state).

            Args:
                old_state (list[Any]): The original collection (list/tuple/set) whose length
                    determines the expected integer key range
                new_state (dict[str, Any] | dict[int, Any]): Dictionary that may have string
                    keys (from flattening) or integer keys that need to be normalized

            Returns:
                dict[int, Any]: Dictionary with integer keys corresponding to old_state indices

            Raises:
                KeyError: If any index from old_state doesn't have a corresponding key in new_state
            """
            # Check if all keys are already integers - if so, return as is
            if new_state and all(isinstance(key, int) for key in new_state.keys()):
                return new_state  # type: ignore[return-value]

            # Create a new dictionary to store the converted state
            converted_new_state = {}

            # Iterate through each index in the old_state collection
            for i in range(len(old_state)):
                # Convert string keys to integers (common case after state dict flattening)
                if str(i) in new_state:
                    converted_new_state[i] = new_state[str(i)]  # type: ignore[index]
                else:
                    # If the string representation doesn't exist, raise an error
                    raise KeyError(
                        f"Index {i} not found in new_state. new_state keys: {new_state.keys()}"
                    )

            return converted_new_state

        def load_from_new_state_to_old_state(
            old_state: StateDict, new_state: StateDict
        ) -> StateDict:
            if isinstance(old_state, torch.Tensor):
                if not isinstance(new_state, torch.Tensor):
                    logger.warning(
                        f"Both old state {old_state} and new state {new_state} must be tensors! Continuing..."
                    )
                    return old_state
                old_state.detach().copy_(new_state)
            elif isinstance(old_state, OptimizerModule):
                old_state.load_state_dict(new_state, store_non_tensors)
            elif isinstance(old_state, dict):
                if not isinstance(new_state, dict):
                    logger.warning(
                        f"Both old state {old_state} and new_state {new_state} must be dicts! Continuing..."
                    )
                    return old_state
                old_state |= {
                    key: load_from_new_state_to_old_state(
                        old_state=old_value,
                        new_state=new_state[key],
                    )
                    for key, old_value in old_state.items()
                    if key in new_state
                }
            elif isinstance(old_state, (list, tuple, set)):
                # Handle key type conversion for flatten/unflatten compatibility
                # When state dict is flattened/unflattened, dictionary keys in new_state become strings
                if isinstance(new_state, dict) and len(new_state) > 0:
                    # Convert new_state dict to match the collection structure expected by old_state
                    new_state = _convert_state_key_from_str_to_int(old_state, new_state)

                old_state = type(old_state)(
                    (
                        load_from_new_state_to_old_state(
                            old_state=old_value,
                            new_state=new_state[i],
                        )
                        if store_non_tensors
                        or isinstance(
                            old_value,
                            (torch.Tensor, dict, list, tuple, set, OptimizerModule),
                        )
                        else old_value
                    )
                    for i, old_value in enumerate(old_state)
                )
            elif store_non_tensors:
                if type(old_state) is not type(new_state):
                    logger.warning(
                        f"Types of old value {type(old_state)} and new value {type(new_state)} do not match! Continuing..."
                    )
                    return old_state
                old_state = deepcopy(new_state)

            return old_state

        # load state
        load_from_new_state_to_old_state(old_state=self.__dict__, new_state=state_dict)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import unittest
from typing import Any
from unittest import mock

import torch
from distributed_shampoo.preconditioner import (
    adagrad_preconditioner_list,
    shampoo_preconditioner_list,
)
from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import (
    BaseShampooPreconditionerList,
)
from torch import Tensor


class AbstractPreconditionerListTest:
    class Interface(abc.ABC, unittest.TestCase):
        """AbstractPreconditionerListTest.Interface is the base class for testing all PreconditionerList implementations.

        This abstract class provides a standardized testing framework for various preconditioner list implementations.
        It defines a set of abstract methods that subclasses must implement, as well as concrete test methods that
        verify the behavior of preconditioner lists.

        Note that all the subclasses may not implement setUp() and only need to implement _instantiate_block_list() and
        _instantiate_preconditioner_list() to enable the the usages of self._block_list and self._preconditioner_list.

        Subclasses might override the following test cases for their specific needs:
            1. _expected_numel_list() - Expected number of elements in each preconditioner
            2. _expected_dims_list() - Expected dimensions of each preconditioner
            3. _expected_num_bytes_list() - Expected memory usage of each preconditioner
            4. _expected_numel() - Expected total number of elements across all preconditioners
            5. _expected_num_bytes() - Expected total memory usage across all preconditioners
            6. _expected_compress_list_call_count() - Expected number of calls to compress_list

        Subclasses should leverage the _verify_preconditioner_updates() helper method to test their
        preconditioner_list implementations with specific gradient inputs and expected outputs.
        """

        def setUp(self) -> None:
            self._block_list: tuple[Tensor, ...] = self._instantiate_block_list()
            self._preconditioner_list: PreconditionerList = (
                self._instantiate_preconditioner_list()
            )

        @abc.abstractmethod
        def _instantiate_block_list(self) -> tuple[Tensor, ...]: ...

        @abc.abstractmethod
        def _instantiate_preconditioner_list(
            self, **kwargs: Any
        ) -> PreconditionerList: ...

        @property
        @abc.abstractmethod
        def _expected_numel_list(self) -> tuple[int, ...]: ...

        def test_numel_list(self) -> None:
            self.assertEqual(
                self._preconditioner_list.numel_list, self._expected_numel_list
            )

        @property
        @abc.abstractmethod
        def _expected_dims_list(self) -> tuple[torch.Size, ...]: ...

        def test_dims_list(self) -> None:
            self.assertEqual(
                self._preconditioner_list.dims_list, self._expected_dims_list
            )

        @property
        @abc.abstractmethod
        def _expected_num_bytes_list(self) -> tuple[int, ...]: ...

        def test_num_bytes_list(self) -> None:
            self.assertEqual(
                self._preconditioner_list.num_bytes_list, self._expected_num_bytes_list
            )

        @property
        @abc.abstractmethod
        def _expected_numel(self) -> int: ...

        def test_numel(self) -> None:
            self.assertEqual(self._preconditioner_list.numel(), self._expected_numel)

        @property
        @abc.abstractmethod
        def _expected_num_bytes(self) -> int: ...

        def test_num_bytes(self) -> None:
            self.assertEqual(
                self._preconditioner_list.num_bytes(), self._expected_num_bytes
            )

        @property
        @abc.abstractmethod
        def _expected_compress_list_call_count(self) -> int: ...

        def test_compress_preconditioner_list(self) -> None:
            # Note: New PreconditionerList implementations might need to add mocks here
            # to make this test case work properly
            with (
                mock.patch.object(
                    adagrad_preconditioner_list, "compress_list"
                ) as mock_adagrad_compress_list,
                mock.patch.object(
                    shampoo_preconditioner_list, "compress_list"
                ) as mock_shampoo_compress_list,
            ):
                self.assertIsNone(
                    self._preconditioner_list.compress_preconditioner_list(
                        local_grad_selector=(True,) * len(self._block_list)
                    )
                )
            self.assertEqual(
                mock_adagrad_compress_list.call_count
                + mock_shampoo_compress_list.call_count,
                self._expected_compress_list_call_count,
            )

        def _verify_preconditioner_updates(
            self,
            preconditioner_list: PreconditionerList,
            masked_grad_lists: list[tuple[Tensor, ...]],
            masked_expected_preconditioned_grad_list: tuple[Tensor, ...],
        ) -> None:
            """
            Test helper function that verifies preconditioner updates and preconditioning.

            This function takes a preconditioner_list and updates it using gradients from masked_grad_lists.
            It performs amortized computation at the end and verifies the preconditioned gradients of the last step.

            Subclasses should use this method to test their preconditioner_list implementations with specific
            gradient inputs and expected outputs. This provides a standardized way to verify that preconditioners
            are correctly updating and producing the expected preconditioned gradients.

            Args:
                preconditioner_list (PreconditionerList): The list of preconditioners to be updated.
                masked_grad_lists (list[tuple[Tensor, ...]]): A list of gradient tuples for each update step.
                masked_expected_preconditioned_grad_list (tuple[Tensor, ...]): The expected preconditioned gradients after all updates.
            """
            for step, masked_grad_list in enumerate(masked_grad_lists, start=1):
                preconditioner_list.update_preconditioners(
                    masked_grad_list=masked_grad_list,
                    step=torch.tensor(step),
                    # Only update the complete preconditioner during the last call to update_preconditioners().
                    perform_amortized_computation=isinstance(
                        preconditioner_list, BaseShampooPreconditionerList
                    )
                    and step == len(masked_grad_lists),
                )

            masked_actual_preconditioned_grad_list = preconditioner_list.precondition(
                masked_grad_list=masked_grad_lists[-1]
            )
            torch.testing.assert_close(
                masked_actual_preconditioned_grad_list,
                masked_expected_preconditioned_grad_list,
            )

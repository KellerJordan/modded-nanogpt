"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import tempfile
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch
from distributed_shampoo.examples import loss_metrics
from distributed_shampoo.examples.loss_metrics import LossMetrics


class TestLossMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self._device = torch.device("cpu")
        self._window_size = 3
        self._world_size = 1

    loss_metrics_module: ModuleType = loss_metrics

    @patch.object(loss_metrics_module, "logger")
    def test_multiple_updates_logging(self, mock_logger: MagicMock) -> None:
        """Test logging behavior with multiple updates."""
        metrics = LossMetrics(window_size=self._window_size, device=self._device)

        # Add multiple losses
        for i in range(5):
            loss = torch.tensor(float(i + 1), device=self._device)
            metrics.update(loss)
            metrics.log()

        # Should have logged 5 times
        self.assertEqual(mock_logger.info.call_count, 5)

        # Check the last log message shows iteration 5
        last_call_message = mock_logger.info.call_args[0][0]
        self.assertIn("Iteration: 5", last_call_message)

    @patch.object(loss_metrics_module, "SummaryWriter", return_value=MagicMock())
    def test_tensorboard_logging(self, mock_summary_writer_class: MagicMock) -> None:
        """Test that tensorboard logging works when metrics_dir is provided."""
        mock_writer_instance = mock_summary_writer_class.return_value

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = LossMetrics(
                window_size=self._window_size,
                device=self._device,
                metrics_dir=tmp_dir,
            )

            loss = torch.tensor(2.0, device=self._device)
            metrics.update(loss)
            metrics.log()

            # Verify SummaryWriter was instantiated with correct log_dir
            mock_summary_writer_class.assert_called_once_with(log_dir=tmp_dir)

            # Verify add_scalars was called with correct structure
            mock_writer_instance.add_scalars.assert_called_once()
            call_kwargs = mock_writer_instance.add_scalars.call_args[1]
            self.assertEqual(call_kwargs["main_tag"], "Local Loss")
            self.assertEqual(call_kwargs["global_step"], 1)
            self.assertIn("Lifetime", call_kwargs["tag_scalar_dict"])
            self.assertIn("Window", call_kwargs["tag_scalar_dict"])

            metrics.flush()
            mock_writer_instance.flush.assert_called_once()

    torch_distributed_module: ModuleType = torch.distributed

    @patch.object(torch_distributed_module, "all_reduce")
    def test_global_metrics_distributed_behavior(
        self, mock_all_reduce: MagicMock
    ) -> None:
        """Test global metrics behavior in distributed setting."""
        world_size = 4
        metrics = LossMetrics(
            window_size=self._window_size,
            device=self._device,
            world_size=world_size,
        )

        loss = torch.tensor(4.0, device=self._device)
        metrics.update(loss)

        with patch.object(torch.distributed, "is_initialized", return_value=True):
            # Call update_global_metrics and verify distributed calls
            metrics.update_global_metrics()

        # Should call all_reduce twice (window and lifetime loss)
        self.assertEqual(mock_all_reduce.call_count, 2)

        # Verify it was called with correct reduce operation
        for call in mock_all_reduce.call_args_list:
            self.assertEqual(call[1]["op"], torch.distributed.ReduceOp.SUM)

    @patch.object(loss_metrics_module, "logger")
    def test_global_metrics_logging(self, mock_logger: MagicMock) -> None:
        """Test global metrics logging behavior."""
        world_size = 2
        metrics = LossMetrics(
            window_size=self._window_size,
            device=self._device,
            world_size=world_size,
        )

        loss = torch.tensor(2.0, device=self._device)
        metrics.update(loss)

        # Call log_global_metrics
        metrics.log_global_metrics()

        # Should log global metrics for multi-worker setup
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("Global Lifetime Loss:", log_message)
        self.assertIn("Global Window Loss:", log_message)

    @patch.object(loss_metrics_module, "logger")
    def test_global_metrics_logging_single_worker(self, mock_logger: MagicMock) -> None:
        """Test that global metrics are not logged for single worker."""
        metrics = LossMetrics(
            window_size=self._window_size,
            device=self._device,
            world_size=1,  # Single worker
        )

        metrics.log_global_metrics()

        # Should not log anything for single worker
        mock_logger.info.assert_not_called()

    @patch.object(loss_metrics_module, "SummaryWriter", return_value=MagicMock())
    def test_tensorboard_global_metrics(
        self, mock_summary_writer_class: MagicMock
    ) -> None:
        """Test global metrics tensorboard logging."""
        mock_writer_instance = mock_summary_writer_class.return_value

        with tempfile.TemporaryDirectory() as tmp_dir:
            world_size = 2
            metrics = LossMetrics(
                window_size=self._window_size,
                device=self._device,
                world_size=world_size,
                metrics_dir=tmp_dir,
            )

            loss = torch.tensor(2.0, device=self._device)
            metrics.update(loss)
            metrics.log_global_metrics()

            # Verify SummaryWriter was instantiated
            mock_summary_writer_class.assert_called_once_with(log_dir=tmp_dir)

            # Should call add_scalars for global metrics
            mock_writer_instance.add_scalars.assert_called_once()
            call_kwargs = mock_writer_instance.add_scalars.call_args[1]
            self.assertEqual(call_kwargs["main_tag"], "Global Loss")
            self.assertIn("Lifetime", call_kwargs["tag_scalar_dict"])
            self.assertIn("Window", call_kwargs["tag_scalar_dict"])

            metrics.flush()
            mock_writer_instance.flush.assert_called_once()

    @patch.object(loss_metrics_module, "SummaryWriter", return_value=MagicMock())
    def test_flush_behavior_with_tensorboard(
        self, mock_summary_writer_class: MagicMock
    ) -> None:
        """Test flush behavior with tensorboard writer."""
        mock_writer_instance = mock_summary_writer_class.return_value

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = LossMetrics(
                window_size=self._window_size,
                device=self._device,
                metrics_dir=tmp_dir,
            )

            metrics.flush()

            # Verify SummaryWriter was instantiated
            mock_summary_writer_class.assert_called_once_with(log_dir=tmp_dir)

            # Verify flush was called on the writer instance
            mock_writer_instance.flush.assert_called_once()

    @patch.object(loss_metrics_module, "logger")
    def test_window_size_behavior(self, mock_logger: MagicMock) -> None:
        """Test that window size affects logging behavior correctly."""
        small_window = 2
        metrics = LossMetrics(window_size=small_window, device=self._device)

        # Add more losses than window size
        losses = [1.0, 2.0, 3.0, 4.0]  # 4 losses, window size 2
        for loss_val in losses:
            loss = torch.tensor(loss_val, device=self._device)
            metrics.update(loss)
            metrics.log()

        # Should have logged 4 times
        self.assertEqual(mock_logger.info.call_count, len(losses))

        # The method should handle window overflow correctly (no crashes)
        last_log = mock_logger.info.call_args[0][0]
        self.assertIn(f"Iteration: {len(losses)}", last_log)
        self.assertIn(f"Local Lifetime Loss: {sum(losses) / len(losses)}", last_log)
        self.assertIn(
            f"Local Window Loss: {sum(losses[-small_window:]) / small_window}", last_log
        )

    def test_edge_case_zero_window_size(self) -> None:
        """Test behavior with edge case of zero window size."""
        metrics = LossMetrics(window_size=0, device=self._device)
        loss = torch.tensor(1.0, device=self._device)

        # Should raise RuntimeError when trying to update with empty window
        with self.assertRaises(RuntimeError):
            metrics.update(loss)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import logging

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

# create default device
DEFAULT_DEVICE = torch.device("cpu")


class LossMetrics:
    def __init__(
        self,
        window_size: int = 100,
        device: torch.device = DEFAULT_DEVICE,
        world_size: int = 1,
        metrics_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._world_size = world_size
        self._window_size = window_size
        self._device = device
        self._iteration = 0
        self._window_losses: list[torch.Tensor] = []
        self._window_loss: torch.Tensor = torch.tensor(0.0, device=device)
        self._accumulated_loss: torch.Tensor = torch.tensor(0.0, device=device)
        self._lifetime_loss: torch.Tensor = torch.tensor(0.0, device=device)

        if self._world_size > 1:
            self._global_window_loss: torch.Tensor = torch.tensor(0.0, device=device)
            self._global_lifetime_loss: torch.Tensor = torch.tensor(0.0, device=device)

        self._metrics_writer: SummaryWriter | None = (
            SummaryWriter(log_dir=metrics_dir) if metrics_dir else None
        )

    def update(self, loss: torch.Tensor) -> None:
        self._iteration += 1
        self._window_losses.append(loss)
        if len(self._window_losses) > self._window_size:
            self._window_losses.pop(0)
        self._window_loss = torch.mean(torch.stack(self._window_losses))
        self._accumulated_loss += loss
        self._lifetime_loss = self._accumulated_loss / self._iteration

    def log(self) -> None:
        logger.info(
            f"Iteration: {self._iteration} | Local Lifetime Loss: {self._lifetime_loss} | Local Window Loss: {self._window_loss}"
        )
        if self._metrics_writer is not None:
            self._metrics_writer.add_scalars(
                main_tag="Local Loss",
                tag_scalar_dict={
                    "Lifetime": self._lifetime_loss,
                    "Window": self._window_loss,
                },
                global_step=self._iteration,
            )

    def update_global_metrics(self) -> None:
        if dist.is_initialized() and self._world_size > 1:
            self._global_window_loss = self._window_loss / self._world_size
            self._global_lifetime_loss = self._lifetime_loss / self._world_size
            dist.all_reduce(self._global_window_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._global_lifetime_loss, op=dist.ReduceOp.SUM)

    def log_global_metrics(self) -> None:
        if self._world_size > 1:
            logger.info(
                f"Iteration: {self._iteration} | Global Lifetime Loss: {self._global_lifetime_loss} | Global Window Loss: {self._global_window_loss}"
            )
            if self._metrics_writer is not None:
                self._metrics_writer.add_scalars(
                    main_tag="Global Loss",
                    tag_scalar_dict={
                        "Lifetime": self._global_lifetime_loss,
                        "Window": self._global_window_loss,
                    },
                    global_step=self._iteration,
                )

    def flush(self) -> None:
        if self._metrics_writer is not None:
            self._metrics_writer.flush()

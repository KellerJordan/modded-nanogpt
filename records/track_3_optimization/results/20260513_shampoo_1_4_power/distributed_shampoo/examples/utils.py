"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Utility functions for CIFAR-10 training examples.
"""

import importlib
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from distributed_shampoo import DistributedConfig, DistributedShampoo
from distributed_shampoo.examples.convnet import ConvNet
from distributed_shampoo.examples.loss_metrics import LossMetrics
from distributed_shampoo.examples.parallelism import ParallelismStrategy
from omegaconf import DictConfig
from torch import nn
from torch.distributed import checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torchvision import datasets, transforms

logger: logging.Logger = logging.getLogger(__name__)

CIFAR_10_DATASET_FILENAME = "cifar-10-python.tar.gz"


class PerRankLoggingFormatter(logging.Formatter):
    """Formatter that adds rank to the log message."""

    def __init__(self) -> None:
        if dist.is_initialized():
            fmt = f"[RANK {dist.get_rank()}] %(levelname)s - %(name)s - %(message)s"
        else:
            fmt = None
        super().__init__(fmt=fmt)


def setup_per_rank_logging(verbose: bool) -> None:
    """Configure per-rank logging with rank prefix and optional debug level."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(PerRankLoggingFormatter())
    root_logger.addHandler(handler)

    if verbose:
        root_logger.setLevel(logging.DEBUG)


def setup_environment() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_distributed_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)


def setup_distribution(
    backend: str, rank: int, world_size: int, local_rank: int
) -> torch.device:
    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def get_model_and_loss_fn(
    device: torch.device,
    out_channels: int = 64,
    disable_linear_bias: bool = False,
) -> tuple[nn.Module, nn.Module]:
    return (
        ConvNet(
            height=32,
            width=32,
            out_channels=out_channels,
            disable_bias=disable_linear_bias,
        ).to(device),
        nn.CrossEntropyLoss(),
    )


def create_device_mesh(
    dp_replicate_degree: int, world_size: int
) -> torch.distributed.device_mesh.DeviceMesh:
    mesh_shape = (dp_replicate_degree, world_size // dp_replicate_degree)
    mesh_dim_names = ("dp_replicate", "dp_shard")
    return init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)


def instantiate_parallelism_strategy(cfg: DictConfig) -> ParallelismStrategy:
    return hydra.utils.instantiate(cfg.parallelism)


def instantiate_optimizer(
    cfg: DictConfig,
    params: Any,
    distributed_config: DistributedConfig | None = None,
) -> torch.optim.Optimizer:
    is_shampoo = "DistributedShampoo" in cfg.optimizer.get("_target_", "")

    optimizer_partial = hydra.utils.instantiate(cfg.optimizer)

    if is_shampoo and distributed_config is not None:
        return optimizer_partial(params=params, distributed_config=distributed_config)

    return optimizer_partial(params=params)


def get_data_loader_and_sampler(
    data_path: Path | str, world_size: int, rank: int, batch_size: int
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset],
]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_path = Path(data_path) / str(rank)

    with importlib.resources.path(
        __package__, CIFAR_10_DATASET_FILENAME
    ) as resource_path:
        if resource_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(resource_path, data_path)

    dataset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform
    )
    sampler: torch.utils.data.distributed.DistributedSampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    )
    return (
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=2
        ),
        sampler,
    )


def load_checkpoint(
    checkpoint_dir: str | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Load model and optimizer state from checkpoint if available."""
    if checkpoint_dir is None:
        return

    if not isinstance(optimizer, DistributedShampoo):
        return

    metadata_path = checkpoint_dir + "/.metadata"
    if not os.path.exists(metadata_path):
        return

    # Since we store the optimizer in eval mode, we need to load it while it is in eval mode.
    optimizer.eval()

    state_dict: dict[str, Any] = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir,
    )

    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optim"])

    # Ensure optimizer is in train mode after loading checkpoint.
    # This is necessary for iterate averaging (GPA/Schedule-Free) to
    # properly resume training with the Y (training) sequence.
    optimizer.train()

    logger.info(f"Loaded checkpoint from {checkpoint_dir}")


def train_model(
    model: nn.Module,
    world_size: int,
    loss_fn: nn.Module,
    sampler: torch.utils.data.Sampler | None,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    window_size: int = 100,
    local_rank: int = 0,
    checkpoint_dir: str | None = None,
    metrics_dir: str | None = None,
) -> tuple[float, float, int]:
    metrics = LossMetrics(
        window_size=window_size,
        device=device,
        world_size=world_size,
        metrics_dir=metrics_dir,
    )

    # Ensure optimizer is in train mode for iterate averaging (GPA/Schedule-Free).
    # This is a no-op for optimizers without iterate averaging.
    if isinstance(optimizer, DistributedShampoo):
        optimizer.train()

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            metrics.update(loss.detach())
            metrics.log()
            metrics.update_global_metrics()
            if local_rank == 0:
                metrics.log_global_metrics()

    # checkpoint optimizer and model using distributed checkpointing solution
    if checkpoint_dir is not None and isinstance(optimizer, DistributedShampoo):
        # Switch optimizer to eval mode before saving checkpoint.
        # This ensures the averaged parameters (X sequence) are saved.
        optimizer.eval()

        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        }
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )

        # Switch optimizer back to train mode after saving checkpoint.
        optimizer.train()

    metrics.flush()
    return (
        metrics._lifetime_loss.item(),
        metrics._window_loss.item(),
        metrics._iteration,
    )

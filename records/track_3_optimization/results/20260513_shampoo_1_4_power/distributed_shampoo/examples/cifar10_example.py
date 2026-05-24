#!/usr/bin/env python3

"""CIFAR-10 training example.

Supports single GPU and distributed training with various parallelism strategies.
Optimizers (sgd, adam, adamw, shampoo) can be combined with any parallelism strategy.

Examples:

    # Single GPU
    python -m distributed_shampoo.examples.cifar10_example optimizer=shampoo
    python -m distributed_shampoo.examples.cifar10_example optimizer=adam
    python -m distributed_shampoo.examples.cifar10_example optimizer=sgd

    # DDP (Data Distributed Parallel)
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=shampoo parallelism=ddp
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=adam parallelism=ddp

    # FSDP (Fully Sharded Data Parallel v1)
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=shampoo parallelism=fsdp
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=sgd parallelism=fsdp

    # HSDP (Hybrid Sharded Data Parallel v1)
    torchrun --nnodes=1 --nproc_per_node=8 -m distributed_shampoo.examples.cifar10_example \
        optimizer=shampoo parallelism=hsdp dp_replicate_degree=2

    # FSDP2 (fully_shard composable API)
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=shampoo parallelism=fully_shard
    torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
        optimizer=adamw parallelism=fully_shard

    # HSDP2 (hybrid_shard composable API)
    torchrun --nnodes=1 --nproc_per_node=8 -m distributed_shampoo.examples.cifar10_example \
        optimizer=shampoo parallelism=hybrid_shard dp_replicate_degree=2
"""

import hydra
import torch
import torch.distributed as dist
from distributed_shampoo.examples.resolvers import register_resolvers
from distributed_shampoo.examples.utils import (
    create_device_mesh,
    get_data_loader_and_sampler,
    get_distributed_env,
    get_model_and_loss_fn,
    instantiate_optimizer,
    instantiate_parallelism_strategy,
    load_checkpoint,
    set_seed,
    setup_distribution,
    setup_environment,
    setup_per_rank_logging,
    train_model,
)
from omegaconf import DictConfig

# Register custom Hydra resolvers for complex types (torch.dtype, enums)
register_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="cifar10")
def main(cfg: DictConfig) -> None:
    setup_environment()
    set_seed(cfg.seed)

    local_rank, rank, world_size = get_distributed_env()

    parallelism = instantiate_parallelism_strategy(cfg)

    if parallelism.requires_distributed:
        device = setup_distribution(cfg.backend, rank, world_size, local_rank)
        setup_per_rank_logging(cfg.verbose)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, loss_fn = get_model_and_loss_fn(
        device,
        out_channels=cfg.out_channels,
        disable_linear_bias=cfg.disable_linear_bias,
    )

    device_mesh = None
    if parallelism.requires_device_mesh:
        device_mesh = create_device_mesh(cfg.dp_replicate_degree, world_size)

    wrapped_model = parallelism.wrap_model(model, local_rank, cfg.backend, device_mesh)

    optimizer = instantiate_optimizer(
        cfg, wrapped_model.model.parameters(), wrapped_model.distributed_config
    )

    load_checkpoint(cfg.checkpoint_dir, wrapped_model.model, optimizer)

    batch_size = (
        cfg.local_batch_size if parallelism.requires_distributed else cfg.batch_size
    )
    data_loader, sampler = get_data_loader_and_sampler(
        cfg.data_path, world_size, rank, batch_size
    )

    train_model(
        model=wrapped_model.model,
        world_size=world_size,
        loss_fn=loss_fn,
        sampler=sampler if parallelism.requires_distributed else None,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.epochs,
        window_size=cfg.window_size,
        local_rank=local_rank,
        checkpoint_dir=cfg.checkpoint_dir,
        metrics_dir=cfg.metrics_dir if rank == 0 else None,
    )

    if parallelism.requires_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

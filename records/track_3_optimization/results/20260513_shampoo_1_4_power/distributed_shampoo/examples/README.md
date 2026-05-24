# CIFAR-10 Training Example

Unified training example supporting single GPU and distributed training with various parallelism strategies.

## Structure

```
examples/
├── cifar10_example.py      # Main entry point
├── parallelism.py          # Parallelism strategy classes
├── resolvers.py            # Custom Hydra resolvers for complex types
├── utils.py                # Utility functions
└── configs/
    ├── cifar10.yaml        # Main config
    ├── optimizer/          # sgd, adam, adamw, shampoo
    └── parallelism/        # none, ddp, fsdp, hsdp, fully_shard, hybrid_shard
```

## Usage

```bash
# Single GPU
python -m distributed_shampoo.examples.cifar10_example optimizer=shampoo

# DDP
torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
    optimizer=shampoo parallelism=ddp

# FSDP
torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
    optimizer=shampoo parallelism=fsdp

# HSDP
torchrun --nnodes=1 --nproc_per_node=8 -m distributed_shampoo.examples.cifar10_example \
    optimizer=shampoo parallelism=hsdp dp_replicate_degree=2

# FSDP2 (fully_shard)
torchrun --nnodes=1 --nproc_per_node=4 -m distributed_shampoo.examples.cifar10_example \
    optimizer=shampoo parallelism=fully_shard

# HSDP2 (hybrid_shard)
torchrun --nnodes=1 --nproc_per_node=8 -m distributed_shampoo.examples.cifar10_example \
    optimizer=shampoo parallelism=hybrid_shard dp_replicate_degree=2
```

Any optimizer can be combined with any parallelism strategy.

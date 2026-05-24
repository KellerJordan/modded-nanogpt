# PyTorch Distributed Shampoo

[![arXiv](https://img.shields.io/badge/arXiv-2309.06497-b31b1b.svg)](https://arxiv.org/abs/2309.06497)


Distributed Shampoo is a preconditioned stochastic gradient optimizer in the adaptive gradient (Adagrad) family of methods [1, 2]. It converges faster by leveraging neural network-specific structures to achieve comparable model quality/accuracy in fewer iterations or epochs at the cost of additional FLOPs and memory, or achieve higher model quality in the same number of iterations or epochs. Our implementation offers specialized support for serial, [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Hybrid Sharding Data Parallel (HSDP)](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp), [Per-parameter Fully Sharded Data Parallel (FSDP2)](#fsdp2-training-support), and [Per-parameter Hybrid Sharded Data Parallel (HSDP2)](#hsdp2-training-support) training.

Distributed Shampoo currently only supports dense parameters.

The key to tuning this optimizer is to balance accuracy, performance, and memory. This is discussed in the Step-by-Step Guide below.

Developers:
- Hao-Jun Michael Shi (Meta Platforms, Inc.)
- Tsung-Hsien Lee
- Anna Cai (Meta Platforms, Inc.)
- Runa Eschenhagen (University of Cambridge)
- Shintaro Iwasaki (Meta Platforms, Inc.)
- Ke Sang (Meta Platforms, Inc.)
- Wang Zhou (Meta Platforms, Inc.)
- Iris Zhang (Meta Platforms, Inc.)

with contributions and support from:

Ganesh Ajjanagadde (Meta), Rohan Anil (Google), Adnan Aziz (Meta), Pavan Balaji (Meta), Shuo Chang (Meta), Weiwei Chu (Meta), Assaf Eisenman (Meta), Will Feng (Meta), Zhuobo Feng (Meta), Jose Gallego-Posada (Mila / Meta Platforms, Inc.), Avirup Ghosh (Meta), Yizi Gu (Meta), Vineet Gupta (Google), Yuchen Hao (Meta), Brian Hirsh (Meta), Yusuo Hu (Meta), Yuxi Hu (Meta), Minhui Huang (Meta), Guna Lakshminarayanan (Meta), Michael Lazos (Meta), Zhijing Li (Meta), Ming Liang (Meta), Wanchao Liang (Meta), Ying Liu (Meta), Wenguang Mao (Meta), Dheevatsa Mudigere (NVIDIA), Maxim Naumov (Meta), Jongsoo Park (Meta), Mike Rabbat (Meta), Kaushik Rangadurai (Meta), Dennis van der Staay (Meta), Fei Tian (Meta), Rohan Varma (Meta), Sanjay Vishwakarma (Meta), Xunnan (Shawn) Xu (Meta), Jiyan Yang (Meta), Chunxing Yin (Meta), Gavin Zhang (Meta), Haoran Zhang (Meta), Haoyu Zhang (Meta), Chuanhao Zhuge (Meta), and Will Zou (Meta).

## 🏆 Competition Winner 🏆

**Shampoo won the [MLCommons AlgoPerf: Training Algorithms Benchmark Competition](https://mlcommons.org/2024/08/mlc-algoperf-benchmark-competition/)!** 🥇

The external tuning ruleset saw four submissions beating the challenging prize-qualification baseline, improving over the state-of-the-art training algorithm. The "Distributed Shampoo" submission provides an impressive **28% faster model training** compared to the baseline, establishing it as a leading optimizer in the field.

This achievement was recognized by major AI organizations:
- 🐦 [AI at Meta announcement](https://x.com/AIatMeta/status/1819128535002538016)
- 🐦 [Google AI announcement](https://x.com/GoogleAI/status/1819138806647316504)

## Features

Key distinctives of this implementation include:
- Homogeneous multi-node multi-GPU support in PyTorch.
- Learning rate grafting [3]. Our version of grafting only grafts the second moment/diagonal preconditioner. Momentum/first moment updates are performed separate from grafting.
- Supports both normal and AdamW (decoupled) weight decay.
- Incorporates exponential moving averaging (with or without bias correction) to the estimate the first moment (akin to Adam).
- Incorporates iterate averaging methods (Generalized Primal Averaging and Schedule-Free) that provide momentum-equivalent behavior with improved theoretical properties [13,14].
- Offers multiple approaches for computing the root inverse, including:
    - Using symmetric eigendecomposition (used by default).
    - Using the QR algorithm to compute an approximate eigendecomposition.
    - Coupled inverse Newton iteration [4].
    - Higher-order coupled iterations with relative epsilon based on estimate of largest eigenvalue.
- Choice of precision for preconditioner accumulation and root inverse computation.
- Ability to cache split parameters.
- Merging of small dimensions.
- Option to (approximately) correct the eigenvalues/run Adam in the eigenbasis of Shampoo's preconditioner (SOAP) [2,6,7].
- Option to use an adaptive preconditioner update frequency when symmetric eigendecompositions or the QR algorithm is used [8].
- Spectral descent via reduced SVD or Newton-Schulz iteration for 2D gradients, or gradients that have been reshaped to 2D [9,10]. This can be used to implement Muon [11], see [Example 6](#example-6-muon).
- KL-Shampoo (without per-factor matrix eigenvalue correction) [12].

## Requirements

We have tested this implementation on the following versions of PyTorch:

- PyTorch >= 2.8;
- Python >= 3.12;
- CUDA 11.3-11.4; 12.2+;

Note: We have observed known instabilities with the torch.linalg.eigh operator on CUDA 11.6-12.1, specifically for low-rank matrices, which may appear with using a small `start_preconditioning_step`. Please avoid these versions of CUDA if possible. See: https://github.com/pytorch/pytorch/issues/94772.

## How to Use

**Given a learning rate schedule for your previous base optimizer, we can replace the optimizer with Shampoo and "graft" from the learning rate schedule of the base method. Alternatively, you can consider replacing Adam(W) by eigenvalue-corrected Shampoo (SOAP).**

A few notes on hyperparameters:

- Notice that Shampoo contains some new hyperparameters (`max_preconditioner_dim` and `precondition_frequency`) that are important for performance. We describe how to tune these below in the section on Hyperparameter Tuning.

- Here, `betas` refer to the hyperparameters used for the exponential moving average of the gradients and Shampoo preconditioners, while `grafting_beta2` corresponds to the `beta2` used specifically for exponential moving averaging of the grafted method. This is similar for `epsilon` and `grafting_epsilon`. As a first choice, we recommend setting `betas` equal to the previous `betas` and additionally setting `grafting_beta2` equal to `betas[1]`, and set `epsilon = 1e-12` and `grafting_epsilon` equal to the previous `epsilon`.

- We also distinguish between `beta1` and iterate averaging. `beta1` (via `betas[0]`) corresponds to the EMA of the gradients (or gradient filtering), while iterate averaging (via `iterate_averaging_config`) provides momentum-like behavior through primal averaging. See [Example 7](#example-7-iterate-averaging-gpa-and-schedule-free) for details on configuring iterate averaging to achieve SGD momentum equivalence.

- We allow for decoupled and coupled weight decay. If one sets `use_decoupled_weight_decay=True`, then you are enabling AdamW-style weight decay, while `use_decoupled_weight_decay=False` corresponds to the normal L2-regularization style weight decay.

- When setting `preconditioner_config` as an instance of `EigenvalueCorrectedShampooPreconditionerConfig` (see Example 5), there is typically no need to use learning rate grafting from Adam (`grafting_config=None`) and, when they are available, Adam's optimal `lr`, `betas`, and `weight_decay` should be a good starting point for further tuning. However, the case of `beta2=1.0`, i.e. an AdaGrad-like accumulation, has not been explored yet.  Also, in settings where Shampoo would usually graft its learning rate from SGD, grafting might still be beneficial.

### Example 1: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) with Momentum

If we previously used the optimizer:
```python
import torch
from torch.optim import SGD

model = instantiate_model()

optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-05,
)
```
we would instead use:
```python
import torch
from distributed_shampoo import (
    DistributedShampoo,
    GeneralizedPrimalAveragingConfig,
    SGDPreconditionerConfig,
)

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.1,  # = 0.01 / (1 - 0.9) to account for primal averaging formulation
    betas=(0., 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    grafting_config=SGDPreconditionerConfig(),
    iterate_averaging_config=GeneralizedPrimalAveragingConfig(
        eval_interp_coeff=0.9,   # = momentum
        train_interp_coeff=1.0,  # 1.0 for heavy-ball momentum
    ),
)
```


### Example 2: [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

If we previously used the optimizer:
```python
import torch
from torch.optim import Adam

model = instantiate_model()

optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
)
```
we would instead use:
```python
import torch
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=False,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-08,
    ),
)
```

### Example 3: [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)

If we previously used the optimizer:
```python
import torch
from torch.optim import Adagrad

model = instantiate_model()

optimizer = Adagrad(
    model.parameters(),
    lr=0.01,
    eps=1e-10,
    weight_decay=1e-05,
)
```
we would instead use:
```python
import torch
from distributed_shampoo import AdaGradPreconditionerConfig, DistributedShampoo

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    betas=(0., 1.0),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=False,
    grafting_config=AdaGradPreconditionerConfig(
        epsilon=1e-10,
    ),
)
```

### Example 4: [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

If we previously used the optimizer:
```python
import torch
from torch.optim import AdamW

model = instantiate_model()

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
)
```
we would instead use:
```python
import torch
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-08,
    ),
)
```

### Example 5: eigenvalue-corrected Shampoo/SOAP

If we previously used the optimizer:
```python
import torch
from torch.optim import AdamW

model = instantiate_model()

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
)
```
we would instead use:
```python
import torch
from distributed_shampoo import (
    DistributedShampoo,
    DefaultEigenvalueCorrectedShampooConfig,
)

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    # This can also be set to `DefaultSOAPConfig` which uses QR decompositions, hence is
    # less expensive and might thereby allow for a smaller `precondition_frequency`.
    preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
)
```

### Example 6: Muon

```python
import math

from distributed_shampoo import (
    DistributedShampoo,
    NewtonSchulzOrthogonalizationConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
)


model = instantiate_model()
# Separate parameters into hidden layers (only 2D) and other parameters (first layer, biases and other 1D parameters, and last layer).
hidden_layer_params = ...
other_params = ...

optimizer = DistributedShampoo(
    [
        # Use spectral descent with Newton-Schulz semi-orthogonalization for hidden layer parameters.
        {
            "params": hidden_layer_params,
            "lr": 0.02,
            "preconditioner_config": SpectralDescentPreconditionerConfig(
                orthogonalization_config=NewtonSchulzOrthogonalizationConfig(
                    scale_by_dims_fn=lambda d_in, d_out: max(1, d_out / d_in)**0.5,
                ),
            ),
            # The two settings below guarantee that the >2D parameters are reshaped to 2D by flattening all but the first dimension (after squeezing dimensions of size 1).
            "max_preconditioner_dim": math.inf,
            "distributed_config": SingleDeviceDistributedConfig(
                target_parameter_dimensionality=2,
            ),
        },
        # Use AdamW for other parameters.
        {
            "params": other_params,
            "lr": 3e-4,
            "start_preconditioning_step": math.inf,
            "grafting_config": AdamPreconditionerConfig(
                beta2=0.95,
                epsilon=1e-10,
            ),
        },
    ],
    weight_decay=1e-05,
    use_decoupled_weight_decay=True,
)
```

`SpectralDescentPreconditionerConfig` can also be used to implement other variations of spectral descent.

### Example 7: Iterate Averaging (GPA and Schedule-Free)

Distributed Shampoo supports iterate averaging methods including Generalized Primal Averaging (GPA) and Schedule-Free optimization. These methods provide an alternative to traditional momentum with improved theoretical properties.

#### Generalized Primal Averaging (GPA)

GPA maintains two sequences of iterates and can reproduce momentum behavior:

```python
from distributed_shampoo import (
    DistributedShampoo,
    GeneralizedPrimalAveragingConfig,
    SGDPreconditionerConfig,
)

model = instantiate_model()

# Example: Shampoo with GPA equivalent to SGD momentum=0.9
optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.1,  # = original_lr / (1 - momentum) = 0.01 / (1 - 0.9)
    betas=(0.0, 1.0),
    epsilon=1e-12,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    preconditioner_config=SGDPreconditionerConfig(),
    iterate_averaging_config=GeneralizedPrimalAveragingConfig(
        eval_interp_coeff=0.9,   # = momentum
        train_interp_coeff=1.0,  # 1.0 for heavy-ball, momentum for Nesterov
    ),
)
```

#### Schedule-Free Optimization

Schedule-Free eliminates the need for learning rate schedules:

```python
from distributed_shampoo import (
    DistributedShampoo,
    ScheduleFreeConfig,
    AdamPreconditionerConfig,
)

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-08,
    ),
    iterate_averaging_config=ScheduleFreeConfig(
        train_interp_coeff=0.9,
    ),
)

# Important: Call train() before training and eval() before evaluation
optimizer.train()
# ... training loop ...
optimizer.eval()
# ... evaluation ...
```

#### Migration from Previous Momentum Parameters

The previous `momentum`, `dampening`, and `use_nesterov` parameters have been replaced by iterate averaging configs. Here are the equivalences:

| Previous Configuration | New Configuration |
|------------------------|-------------------|
| `momentum=β, dampening=0, use_nesterov=False` | `GeneralizedPrimalAveragingConfig(eval_interp_coeff=β, train_interp_coeff=1.0)` with `lr = lr / (1-β)` |
| `momentum=β, dampening=0, use_nesterov=True` | `GeneralizedPrimalAveragingConfig(eval_interp_coeff=β, train_interp_coeff=β)` with `lr = lr / (1-β)` |
| `momentum=β, dampening=d` (d≠0) | No direct equivalent. Dampening is not supported in iterate averaging. |

**Note on LaProp:** The previous momentum implementation (sometimes called LaProp) is mathematically equivalent to the heavy-ball/primal averaging formulation when `dampening=0`. Use the heavy-ball configuration above.

**Note on LaPropW:** The original LaPropW from the paper includes additional weight decay handling that may differ slightly from the iterate averaging formulation. For most practical purposes, the heavy-ball configuration provides equivalent behavior.

## Distributed Training Support

Our implementation offers specialized compatibility and performance optimizations for different distributed training paradigms, including Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (including FSDP and per-parameter FSDP, a.k.a. FSDP2) training. Note that Distributed Shampoo will work out of the box for DDP training, but not for FSDP training.

### DDP Training Support

In order to support fast DDP training, our implementation offers ZeRO-1 support, which distributes the computation and memory (via `DTensor`) in order to lower both Shampoo's memory requirements and its per-iteration wall-clock time at the cost of additional (`AllGather`) communication. Our DDP Shampoo implementation can either: (1) communicate the updated parameters; or (2) communicate the parameter updates.

We support:
- Quantized (or low-precision) communications using BF16, FP16, or FP32 communications.
- Specification of the number of trainers within each process group to distribute compute and memory. This trades off the amount of communication and compute each trainer is responsible for.
- Option to communicate updated parameters.

To use DDP Shampoo, simply configure the `distributed_config` as `DDPDistributedConfig`:
```python
import os

import torch
import torch.distributed as dist

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DDPDistributedConfig,
    DistributedShampoo,
)
from torch import nn

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))
torch.cuda.set_device(LOCAL_RANK)

model = instantiate_model().to(device)
model = nn.parallel.DistributedDataParallel(
    model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=DDPDistributedConfig(
        communication_dtype=torch.float32,
        num_trainers_per_group=8,
        communicate_params=False,
    ),
)
```
Please see [`ddp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/ddp_cifar10_example.py) as an example.

### FSDP1 (FullyShardedDataParallel)

#### FSDP1 Training Support

FSDP training will create flattened parameters by flattening and concatenating all parameters within each FSDP module. By default, this removes all information about each parameter's tensor shape that Shampoo aims to exploit. Therefore, in order to support FSDP training, we have to use additional FSDP metadata in order to recover valid tensor blocks of the original parameters.

Note that we only support PyTorch FSDP with the `use_orig_params=True` option.
```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from distributed_shampoo import (
    AdamPreconditionerConfig,
    compile_fsdp_parameter_metadata,
    DistributedShampoo,
    FSDPDistributedConfig,
)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))

model = instantiate_model().to(device)
model = FSDP(model, use_orig_params=True)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=FSDPDistributedConfig(
        param_to_metadata=compile_fsdp_parameter_metadata(model),
    ),
)
```
Please see [`fsdp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/fsdp_cifar10_example.py) as an example.

#### HSDP1 Training Support

Note that we only support PyTorch HSDP with `sharding_strategy=ShardingStrategy.HYBRID_SHARD` and the `use_orig_params=True` option.
```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from distributed_shampoo import (
    AdamPreconditionerConfig,
    compile_fsdp_parameter_metadata,
    DistributedShampoo,
    HSDPDistributedConfig,
)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))

# Instantiate device mesh for HSDP Shampoo.
# Assuming 8 GPUs, a 2 x 4 mesh will be initialized.
# This means we shard model into four shards, and each sub-model has two replicas.
# [0, 1, 2, 3] and [4, 5, 6, 7] are the two shard groups.
# [0, 4], [1, 5], [2, 6], [3, 7] are the four replicate groups.
device_mesh = init_device_mesh("cuda", (2, 4))

model = instantiate_model().to(device)
model = FSDP(model, device_mesh=device_mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD, use_orig_params=True)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=HSDPDistributedConfig(
        param_to_metadata=compile_fsdp_parameter_metadata(model),
        device_mesh=device_mesh,
    ),
)
```
Please see [`hsdp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/hsdp_cifar10_example.py) as an example.

### FSDP2 (fully_shard)

#### FSDP2 Training Support

Per-parameter sharding FSDP, also known as FSDP2, is the new fully sharded data parallelism implementation, which uses ``DTensor``-based dim-0 per-parameter sharding for a simpler sharding representation compared to FSDP1's flat-parameter sharding, while preserving similar throughput performance. In short, FSDP2 chunks each parameter on dim-0 across the data parallel workers (using ``torch.chunk(dim=0)``). To support Shampoo with FSDP2, we implement a new distributor that creates Shampoo preconditioner tensor blocks based on the rank local tensors of the dim-0 sharded ``DTensor`` parameters. One simplification brought by FSDP2 to Shampoo is that tensor blocks are local to each rank, so we don't need the ``tensor block recovery`` algorithm implemented for FSDP1 (where parameters are flatten and then sharded).

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DistributedShampoo,
    FullyShardDistributedConfig,
)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))

model = instantiate_model().to(device)
model = fully_shard(model)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=FullyShardDistributedConfig(),
)
```

Please see [`fully_shard_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/fully_shard_cifar10_example.py) as an example.

#### HSDP2 Training Support

We support PyTorch HSDP for FSDP2 (fully_shard).

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DistributedShampoo,
    HybridShardDistributedConfig,
)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))

# Instantiate device mesh for HSDP Shampoo.
# Assuming 8 GPUs, a 2 x 4 mesh will be initialized.
# This means we shard model into four shards, and each sub-model has two replicas.
# [0, 1, 2, 3] and [4, 5, 6, 7] are the two shard groups.
# [0, 4], [1, 5], [2, 6], [3, 7] are the four replicate groups.
device_mesh = init_device_mesh("cuda", (2, WORLD_SIZE // 2))

model = instantiate_model().to(device)
model = fully_shard(model, mesh=device_mesh)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamPreconditionerConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=HybridShardDistributedConfig(device_mesh=device_mesh),
)
```
Please see [`hybrid_shard_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/hybrid_shard_cifar10_example.py) as an example.

## Checkpointing Support

Distributed Shampoo supports PyTorch standard state dict API via `state_dict()` and `load_state_dict()`. For saving and loading checkpoints, it is compatible with both:
- Standard PyTorch serialization: torch.save() / torch.load()
- Distributed checkpointing: dcp.save() / dcp.load()

Given a `CHECKPOINT_DIR`, to store the checkpoint with PyTorch's `torch.distributed.checkpoint`:
```python
import torch.distributed.checkpoint as dcp

state_dict = {
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
}
dcp.save(
    state_dict=state_dict,
    checkpoint_id=CHECKPOINT_DIR,
)
```

To load the checkpoint:
```python
dcp.load(
    state_dict=state_dict,
    checkpoint_id=CHECKPOINT_DIR,
)
model.load_state_dict(state_dict["model"])
optimizer.load_state_dict(state_dict["optim"])
```

You can also refer to [`ddp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/ddp_cifar10_example.py) as an example.

PyTorch Distributed Checkpoint will save your sharded checkpoint in a folder named `step-{STEP}`. To convert the sharded checkpoints from DCP format to `torch.save` format (`.pt` file), you can use the following offline conversion command provided by PyTorch:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch {YOUR_DCP_CHECKPOINT_PATH}/step-{STEP} {YOUR_TORCH_SAVE_FILE_NAME}.pt
```

For more information, please refer to [Distributed Checkpoint - torch.distributed.checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html).

## Hyperparameter Tuning

**We want to tune Shampoo to balance model quality, memory, and efficiency/performance by applying approximations to a "pure" version of Shampoo.**

This requires adjusting the hyperparameters `max_preconditioner_dim`, `precondition_frequency`, and `start_preconditioning_step`. The general approach is to start by using as close to a “pure” version of Shampoo as possible, then incorporate approximations to ensure that one obtains fast performance. A pure version of Shampoo would set `max_preconditioner_dim = 8192` and `precondition_frequency = 1`.

With the inclusion of learning rate grafting, we can extract a good learning rate schedule from your existing scheduler. Other techniques for preventing divergence (gradient clipping) may also be removed.

### Step-by-Step Guide

1. Start with a reasonable `max_preconditioner_dim` (i.e., 8192) and reduce the block size as necessary for memory and performance.

    * The maximum effective value of this hyperparameter is the maximum value of the products of each layer’s dimensions. For example, if we have a model with three layers where the first layer is 5x5x3x6, the second layer is 3x3x3x8, and the third layer is 216x5; the products of the first, second, and third layers’ dimensions are 5x5x3x6=450, 3x3x3x8=216, and 216x10=1080, respectively. In this example, 1080 is the maximum effective value of this hyperparameter, and any value greater than 1080 will perform the same as 1080.

    * The higher this value is, the better the model quality we expect.

    * There is a sweet spot in terms of performance - if the number is too small, the algorithm will slow down due to kernel latency. On the other hand, using too large of a value leads to slow matrix computations (i.e., matrix root inverses), which scale as $O(n^3)$ if $n$ is the dimension of the matrix, as well as poor load-balancing. In our experience, using a `max_preconditioner_dim` between 1024 and 8192 is ideal for performance.

    * Memory varies depending on the order of the tensor. For vectors, increasing `max_preconditioner_dim` leads to increased memory costs, but for 3rd-order tensors (or higher), increasing `max_preconditioner_dim` leads to decreased memory costs. Blocked matrices yield a fixed memory cost regardless of `max_preconditioner_dim`.

    * For efficiency purposes, it is best to set this value as a multiple of 2.

    * The following is an example of setting `max_preconditioner_dim = 4096` with SGD grafting:
    ```python
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        weight_decay=0.01,
        max_preconditioner_dim=4096,
        grafting_config=SGDPreconditionerConfig(),
    )
    ```

2. Use the smallest `precondition_frequency` (i.e., 1) and increase the precondition frequency.

    * This hyperparameter determines how frequently the preconditioner is computed. The smaller the value, the slower Shampoo becomes but with faster convergence. The goal is to find a value that balances convergence and speed.

    * It is normal to eventually set this hyperparameter on the order of hundreds or thousands. This is based primarily on the size of the network and the effective ratio between the cost of a single forward-backward pass + standard optimizer step to the cost of computing a series of matrix root inverses.

    * In practice, we have found that an upper bound to `precondition_frequency` is on the order of thousands. This approach will offer diminishing performance gains if the bottleneck is due to preconditioning, which is performed at every iteration.

    * The following is an example of setting `precondition_frequency = 100`:
    ```python
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        weight_decay=0.01,
        precondition_frequency=100,
        grafting_config=SGDPreconditionerConfig(),
    )
    ```

3. Set `start_preconditioning_step` to be consistent with the precondition frequency.

    * This hyperparameter determines when to start using Shampoo. Prior to this, the optimizer will use the grafted method. This value should generally be set larger than or equal to `precondition_frequency` except when the precondition frequency is 1. By default, `start_preconditioning_step` is set equal to `precondition_frequency`.

    * If the `precondition_frequency = 1`, then set `start_preconditioning_step = -1` in order to use Shampoo from the start.

    * Following is an example of setting `start_preconditioning_step = 300`:
    ```python
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        weight_decay=0.01,
        start_preconditioning_step=300,
        grafting_config=SGDPreconditionerConfig(),
    )
    ```

4. To tune for better model quality, one can tune:

    * **Learning Rate** (`lr`): One can change the learning rate schedule, and potentially use a larger learning rate.
    * **Epsilon Regularization** (`epsilon`): One should typically search for a value in $\{10^{−12},10^{−11},...,10^{−2},10^{−1}\}$.
    * **Exponential Moving Average Parameters** (`betas`): One can tune the `betas = (beta1, beta2)` parameters as is typical for Adam(W).
    * **Preconditioner Data Type** (`factor_matrix_dtype`): For certain models, it is necessary to use higher precision to accumulate the Shampoo factor matrices and compute its eigendecomposition to obtain high enough numerical accuracy. In those cases, one can specify this as `torch.float64`. (Note that this will use more memory.)
    * **MTML Task Weights**: Task weights may need to be re-tuned as Distributed Shampoo will better exploit certain imbalances between different task losses.

5. If enabling DDP Shampoo, you can tune for performance:

    * **Process Group Size** (`num_trainers_per_group`): For large-scale distributed jobs, this hyperparameter allows us to trade off computational and communication costs. Assuming the number of GPUs per node is 8, one should search for a value in $\{8,16,32,64\}$. This hyperparameter has no impact on model quality.
    * **Quantized Communications** (`communication_dtype`): One can enable quantized communications by setting the `communication_dtype`. We have found that using `torch.float16` works well in practice (with `communicate_params = False`).
    * **Communicate Updated Parameters** (`communicate_params`): If one does not enable quantized communications, one can possibly obtain better performance by communicating the updated parameters by setting this to `True`.

## Common Questions

### Encountering `NaN/Inf` numerical error:

When gradients are `NaN/Inf`, most optimizers still proceed smoothly and modify model weights with those `NaN/Inf` values but Shampoo reacts with error messages like "Encountered nan values ...".

When encountering those errors, following are things you could try:

1. Decrease the learning rate.
2. Adjust the learning rate scheduler.
3. Increase `start_preconditioning_step`.
4. Consider applying gradient clipping.

## Citing PyTorch Distributed Shampoo

If you use PyTorch Distributed Shampoo in your work, please use the following BibTeX entry.

```BibTeX
@misc{shi2023pytorchshampoo,
    title={A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale},
    author={Hao-Jun Michael Shi and Tsung-Hsien Lee and Shintaro Iwasaki and Jose Gallego-Posada and Zhijing Li and Kaushik Rangadurai and Dheevatsa Mudigere and Michael Rabbat},
    howpublished={\url{https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo}},
    year ={2023},
    eprint={2309.06497},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## References
1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf). Vineet Gupta, Tomer Koren, and Yoram Singer. ICML, 2018.
2. [Scalable Second-Order Optimization for Deep Learning](https://arxiv.org/abs/2002.09018). Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Tech report, 2021.
3. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/pdf?id=FpKgG31Z_i9). Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Tech report, 2021.
4. [Functions of Matrices: Theory and Computation](https://epubs.siam.org/doi/book/10.1137/1.9780898717778). Nicholas J. Higham. SIAM, 2008.
5. [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale](https://arxiv.org/abs/2309.06497). Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat. Tech report, 2023.
6. [Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](https://arxiv.org/abs/1806.03884). Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent. NeurIPS, 2018.
7. [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade. ICLR, 2025.
8. [Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](https://arxiv.org/abs/2506.03595). Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner, Hao-Jun Michael Shi. NeurIPS, 2025.
9. [Preconditioned Spectral Descent for Deep Learning](https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html). David E. Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, Volkan Cevher. NeurIPS, 2015.
10. [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325). Jeremy Bernstein, Laker Newhouse. Tech report, 2024.
11. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/). Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein. Blog post, 2024.
12. [Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](https://arxiv.org/abs/2509.03378). Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, Roger B. Grosse. Tech report, 2025.
13. [The Road Less Scheduled](https://arxiv.org/abs/2405.15682). Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky. NeurIPS, 2025.
14. [Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs](https://arxiv.org/abs/2512.17131). Aaron Defazio, Konstantin Mishchenko, Parameswaran Raman, Hao-Jun Michael Shi, Lin Xiao. Tech report, 2025.

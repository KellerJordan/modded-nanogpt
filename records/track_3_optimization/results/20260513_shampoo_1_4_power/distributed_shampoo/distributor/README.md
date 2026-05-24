# Distributed Shampoo Distributor Module

This module implements the distributor components for the Distributed Shampoo optimizer, responsible for managing parameter distribution, gradient communication, and coordination across multiple devices in distributed training environments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Distributor Implementations](#distributor-implementations)
- [Block Management](#block-management)
- [Distributed Utilities](#distributed-utilities)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

The distributor module is the distributed training engine of Distributed Shampoo. It handles parameter partitioning, gradient synchronization, and memory management across different distributed training paradigms, enabling efficient scaling from single-GPU to multi-node training.

### Key Features

- **Multiple Distribution Strategies**: Support for DDP, FSDP1, HSDP1, FSDP2, and HSDP2 approaches
- **Automatic Parameter Blocking**: Intelligent partitioning of parameters for optimal memory usage
- **Communication Optimization**: Efficient gradient synchronization with minimal overhead
- **Memory Management**: Smart tensor allocation and buffer management for distributed environments
- **Fault Tolerance**: Robust handling of gradient sparsity and device failures
- **Flexible Configuration**: Configurable group sizes and communication patterns

## Quick Start

### Basic Usage (Single GPU)

```python
# Set up parameter group
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: SingleDeviceDistributedConfig()
}

# Create a basic distributor for single-GPU training
distributor = Distributor(param_group=param_group)

# Training step
local_masked_blocked_grads = distributor.merge_and_block_gradients()
# ... compute search directions with preconditioner ...
distributor.update_params(blocked_search_directions=search_directions)
```

### Distributed Data Parallel (DDP)

```python
# Set up parameter group with DDP configuration
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: DDPDistributedConfig(
        num_trainers_per_group=4,  # Use 4 GPUs in group
        communicate_params=False,  # Communicate search directions
        communication_dtype=torch.float16
    )
}

# Create DDP distributor
ddp_distributor = DDPDistributor(param_group=param_group)

# Training step
local_masked_blocked_grads = ddp_distributor.merge_and_block_gradients()
ddp_distributor.update_params(blocked_search_directions=search_directions)
```

### Fully Sharded Data Parallel (FSDP1)

```python
# Prepare parameter metadata for FSDP
param_to_metadata = {
    param: FSDPParameterMetadata(
        fqn=param_name,
        shape=param.shape,
        numel=param.numel(),
        start_idx=start_idx,
        end_idx=end_idx,
        sharding_strategy=ShardingStrategy.FULL_SHARD
    )
    for param, param_name, start_idx, end_idx in param_metadata_list
}

# Set up parameter group with FSDP configuration
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: FSDPDistributedConfig(
        param_to_metadata=param_to_metadata
    )
}

# Create FSDP distributor
fsdp_distributor = FSDPDistributor(param_group=param_group)
```

### Hybrid Sharded Data Parallel (HSDP1)

```python
# Create device mesh for HSDP (e.g., 2 nodes × 4 GPUs = 8 total)
device_mesh = init_device_mesh("cuda", mesh=(2, 4))

# Set up parameter group with HSDP configuration
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: HSDPShampooConfig(
        param_to_metadata=param_to_metadata,
        device_mesh=device_mesh,
        num_trainers_per_group=4,
        communicate_params=False,
        communication_dtype=torch.bfloat16
    )
}

# Create HSDP distributor
hsdp_distributor = HSDPDistributor(param_group=param_group)
```

### Fully Shard Distributor (FSDP2)

```python
# Set up parameter group for fully sharded training
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: FullyShardShampooConfig(
        param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
    )
}

# Create fully shard distributor
fully_shard_distributor = FullyShardDistributor(param_group=param_group)

# For lossless precision variant
lossless_distributor = FullyShardLosslessDistributor(param_group=param_group)
```

### Hybrid Shard Distributor (HSDP2)

```python
# Create device mesh for hybrid sharding
device_mesh = init_device_mesh("cuda", mesh=(4,))

# Set up parameter group with hybrid shard configuration
param_group = {
    PARAMS: model.parameters(),
    DISTRIBUTED_CONFIG: HybridShardShampooConfig(
        device_mesh=device_mesh,
        communication_dtype=torch.bfloat16,
        num_trainers_per_group=4,
        communicate_params=False,
        param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
    )
}

# Create hybrid shard distributor
hybrid_distributor = HybridShardDistributor(param_group=param_group)
```

## Architecture

### Base Interface: `DistributorInterface`

All distributors inherit from the abstract base class `DistributorInterface` defined in [`shampoo_distributor.py`](shampoo_distributor.py):

```python
class DistributorInterface(ABC):
    @abstractmethod
    def update_params(
        self,
        blocked_search_directions: tuple[Tensor, ...],
    ) -> None:
        """Update parameters with computed search directions."""

    @abstractmethod
    def merge_and_block_gradients(self) -> tuple[Tensor, ...]:
        """Process gradients into blocked format."""

    @property
    def local_blocked_params(self) -> tuple[Tensor, ...]:
        """Parameters assigned to local device."""

    @property
    def local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Metadata for each local parameter block."""
```

### Key Design Principles

1. **Distributed-First**: Designed for multi-device training from the ground up
2. **Memory Efficiency**: Optimal memory usage through intelligent parameter blocking
3. **Communication Minimization**: Reduce network overhead through efficient synchronization
4. **Modularity**: Easy to extend with new distribution strategies
5. **Fault Tolerance**: Robust handling of sparse gradients and device failures

## Distributor Implementations

### 1. Base Distributor

**File**: [`shampoo_distributor.py`](shampoo_distributor.py)

#### `Distributor`
- **Purpose**: Single-GPU training with no communication overhead
- **Best for**: Development, debugging, and small-scale training
- **Algorithm**: Local parameter blocking and gradient processing
- **Memory**: Stores all parameters and gradients on single device
- **Key Features**: Parameter blocking, gradient processing, local optimization

### 2. DDP Distributor

**File**: [`shampoo_ddp_distributor.py`](shampoo_ddp_distributor.py)

#### `DDPDistributor`
- **Purpose**: Distribute computation across multiple GPUs with model replication
- **Best for**: Multi-GPU training with moderate model sizes
- **Algorithm**: AllGather communication for gradient/parameter synchronization
- **Memory**: Full model replica on each GPU, distributed computation
- **Key Features**: Flexible group configurations, configurable communication patterns

### 3. FSDP Distributor

**File**: [`shampoo_fsdp_distributor.py`](shampoo_fsdp_distributor.py)

#### `FSDPDistributor`
- **Purpose**: Shard model parameters across devices for memory efficiency
- **Best for**: Large models that don't fit on single GPU
- **Algorithm**: Tensor block recovery and resharding for gradient processing
- **Memory**: Each GPU holds only a subset of model parameters
- **Key Features**: Split tensor block recovery, memory-efficient parameter sharding

### 4. HSDP Distributor

**File**: [`shampoo_hsdp_distributor.py`](shampoo_hsdp_distributor.py)

#### `HSDPDistributor`
- **Purpose**: Balance memory efficiency and communication cost using hybrid approach
- **Best for**: Large-scale multi-node training
- **Algorithm**: FSDP within nodes, DDP across nodes
- **Memory**: Sharded parameters within nodes, replicated across nodes
- **Key Features**: Combines FSDP and DDP strategies, optimized for multi-node setups

### 5. Fully Shard Distributor

**File**: [`shampoo_fully_shard_distributor.py`](shampoo_fully_shard_distributor.py)

#### `FullyShardDistributor`
- **Purpose**: Complete parameter and gradient sharding across all devices
- **Best for**: Maximum memory efficiency scenarios
- **Algorithm**: Full sharding with minimal memory footprint per device
- **Memory**: Minimal parameter storage per device
- **Key Features**: Complete sharding, advanced memory optimization

### 6. Fully Shard Lossless Distributor

**File**: [`_shampoo_fully_shard_lossless_distributor.py`](_shampoo_fully_shard_lossless_distributor.py)

#### `FullyShardLosslessDistributor`
- **Note**: This is an experimental feature and subject to change.
- **Purpose**: Fully sharded training while maintaining numerical precision
- **Best for**: Scenarios requiring maximum precision with memory efficiency
- **Algorithm**: Lossless precision maintenance during sharding operations
- **Memory**: Optimized sharding with precision guarantees
- **Key Features**: Precision preservation, advanced numerical stability

### 7. Hybrid Shard Distributor

**File**: [`shampoo_hybrid_shard_distributor.py`](shampoo_hybrid_shard_distributor.py)

#### `HybridShardDistributor`
- **Purpose**: Flexible hybrid approach combining multiple sharding strategies
- **Best for**: Complex distributed setups requiring custom sharding patterns
- **Algorithm**: Configurable sharding strategies per parameter group
- **Memory**: Variable memory usage based on sharding configuration
- **Key Features**: Flexible sharding patterns, customizable distribution strategies

## Block Management

### Block Info System ([`shampoo_block_info.py`](shampoo_block_info.py))

The block info system provides comprehensive metadata management for parameter blocks, enabling efficient distributed operations across different sharding strategies.

#### Key Components

- **`BlockInfo`**: Core metadata class that stores parameter references, block identification, and tensor allocation functions
- **`DTensorBlockInfo`**: Extended block info for distributed tensors with specialized operations for distributed environments

#### Features

- **Flexible Allocation**: Support for different tensor allocation patterns (zeros, ones, identity matrices)
- **Distributed Compatibility**: Automatic handling of DTensor-specific operations and type conversions
- **Memory Optimization**: Efficient tensor allocation and buffer management

### Parameter Blocking Process

1. **Dimension Analysis**: Analyze parameter tensor dimensions and merge small dimensions to meet minimum size requirements
2. **Block Partitioning**: Split large tensors into manageable chunks based on memory constraints and distributed strategy
3. **Ownership Tracking**: Create boolean masks and selectors to track parameter ownership across devices
4. **Metadata Generation**: Generate block info objects with appropriate allocation functions and access patterns

## Distributed Utilities

### Core Utilities ([`shampoo_dist_utils.py`](shampoo_dist_utils.py))

Essential distributed training utilities:

#### Device Mesh Management
```python
def get_device_mesh(device_type: str, mesh: tuple[int, ...]) -> DeviceMesh:
    """Create device mesh for distributed tensor operations."""
```

#### Communication Primitives
- Process group management
- Collective operation helpers
- Device topology utilities

### FSDP Utilities ([`shampoo_fsdp_utils.py`](shampoo_fsdp_utils.py))

Specialized utilities for FSDP operations:
- Tensor sharding algorithms
- Parameter reconstruction functions
- Memory optimization helpers

## Examples

### Single Device Configuration

```python
config = SingleDeviceDistributedConfig()
```

### DDP Configuration

```python
config = DDPDistributedConfig(
    num_trainers_per_group=-1,  # Use all GPUs
    communicate_params=False,   # Communicate search directions
    communication_dtype=torch.float16
)
```

### FSDP Configuration

```python
# Prepare parameter metadata
param_to_metadata = {
    param: FSDPParameterMetadata(
        fqn=param_name,
        shape=param.shape,
        numel=param.numel(),
        start_idx=start_idx,
        end_idx=end_idx,
        sharding_strategy=ShardingStrategy.FULL_SHARD
    )
    for param, param_name, start_idx, end_idx in param_metadata_list
}

config = FSDPDistributedConfig(
    param_to_metadata=param_to_metadata
)
```

### HSDP Configuration

```python
# Create device mesh for HSDP (e.g., 2 nodes × 4 GPUs = 8 total)
device_mesh = init_device_mesh("cuda", mesh=(2, 4))  # (replicate_dim, shard_dim)

config = HSDPDistributedConfig(
    param_to_metadata=param_to_metadata,
    device_mesh=device_mesh,
    num_trainers_per_group=4,
    communicate_params=False,
    communication_dtype=torch.bfloat16
)
```

### Fully Shard Configuration

```python
config = FullyShardDistributedConfig(
    param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
)

# For lossless precision variant (if available)
config_lossless = FullyShardLosslessDistributedConfig(
    param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
)
```

### Hybrid Shard Configuration

```python
# Create device mesh for hybrid sharding (e.g., 4 GPUs total)
device_mesh = init_device_mesh("cuda", mesh=(4,))

config = HybridShardDistributedConfig(
    device_mesh=device_mesh,
    communication_dtype=torch.bfloat16,
    num_trainers_per_group=4,
    communicate_params=False,
    param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
)
```

## Contributing

When contributing to the distributor module:

1. **Test Distributed Scenarios**: Ensure functionality works across multiple GPUs/nodes
2. **Memory Efficiency**: Optimize memory usage for large-scale training
3. **Communication Patterns**: Minimize network overhead in distributed operations
4. **Backward Compatibility**: Maintain compatibility with existing distributed configurations
5. **Documentation**: Update README with new distributor types and configuration options
6. **Error Handling**: Robust handling of distributed training failures and
recovery

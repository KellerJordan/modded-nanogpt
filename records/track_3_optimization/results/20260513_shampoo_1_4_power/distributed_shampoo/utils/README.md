# Distributed Shampoo Utilities Module

This module provides essential utility components for the Distributed Shampoo optimizer, including abstract base classes, mathematical functions, optimization modules, checkpoint utilities, model utilities, quantization support, and general-purpose helper functions.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Utilities](#core-utilities)
- [Optimizer Infrastructure](#optimizer-infrastructure)
- [Mathematical Utilities](#mathematical-utilities)
- [Model Utilities](#model-utilities)
- [Checkpoint Management](#checkpoint-management)
- [Quantization Support](#quantization-support)
- [Load Balancing Utilities](#load-balancing-utilities)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

The utilities module serves as the foundational layer for Distributed Shampoo, providing reusable components that handle common patterns, mathematical operations, and infrastructure requirements. These utilities enable clean separation of concerns, improve code reusability, and provide robust building blocks for the optimizer's advanced features.

### Key Features

- **Abstract Base Classes**: Type-safe dataclass patterns for configuration objects
- **Mathematical Operations**: Efficient tensor manipulation and mathematical utilities
- **Optimizer Infrastructure**: Base classes for modular optimizer components with state management
- **Checkpoint Support**: Robust serialization and deserialization for distributed training
- **Model Utilities**: Specialized neural network layers optimized for Shampoo
- **Quantization Framework**: Memory-efficient tensor compression and decompression
- **Common Utilities**: General-purpose helper functions and iterator patterns

## Quick Start

Each utility in this module is designed to be independent and can be used standalone. Here are examples for each major component:

### Abstract Configuration Classes

```python
# Create type-safe configuration classes
@dataclass
class MyOptimizerConfig(AbstractDataclass):
    learning_rate: float = 0.001
    momentum: float = 0.9

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum

# Usage
config = MyOptimizerConfig(learning_rate=0.01, momentum=0.95)
```

### Mathematical Tensor Utilities

```python
# Optimize tensor shapes for better performance
tensor_shape = (32, 3, 64, 64)  # Typical convolution tensor
merged_shape = merge_small_dims(
    tensor_shape=tensor_shape,
    threshold=1024,
    target_tensor_dimensionality=2
)
print(f"Original: {tensor_shape}, Merged: {merged_shape}")
# Output: Original: (32, 3, 64, 64), Merged: (96, 4096)

# Split large tensors into smaller blocks
large_tensor = torch.randn(100, 200)
tensor_blocks = multi_dim_split(large_tensor, split_size=50)
print(f"Split into {len(tensor_blocks)} blocks")
```

### Optimizer Module Infrastructure

```python
# Create stateful optimizer components
class AdaptivePreconditioner(OptimizerModule):
    def __init__(self, dim: int):
        self.squared_gradients = torch.zeros(dim)
        self.step_count = 0

    def update(self, grad: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        self.squared_gradients += grad ** 2
        return grad / (self.squared_gradients.sqrt() + 1e-8)

# Automatic state management
preconditioner = AdaptivePreconditioner(100)
state = preconditioner.state_dict()
preconditioner.load_state_dict(state)
```

### Memory-Efficient Quantization

```python
# Compress tensors for memory efficiency
tensor_list = [torch.randn(500, 500, dtype=torch.float32) for _ in range(3)]
quantized_list = QuantizedTensorList(
    quantized_data=[(t, None, None) for t in tensor_list],
    quantized_dtype=torch.bfloat16,
    computation_dtype=torch.float32
)

# Automatic precision management
with DequantizeQuantizedTensorListContext(quantized_list):
    # Work with full precision
    results = [t.sum() for t in quantized_list.dequantized_value]
# Automatically compressed back to bfloat16
```

### Model Utilities

```python
# Memory-efficient linear layers
combined_layer = CombinedLinear(in_features=256, out_features=128, bias=True)
input_data = torch.randn(32, 256)
output = combined_layer(input_data)  # Shape: (32, 128)

# Access combined weight and bias parameter
combined_param = combined_layer.combined_weight  # Shape: (128, 257)
```

### Dictionary Synchronization

```python
# Synchronize iteration over multiple data sources
training_data = {
    "samples": [sample1, sample2, sample3],
    "labels": [label1, label2, label3],
    "weights": [1.0, 0.8, 1.2]
}

for batch in DictZipIterator(training_data):
    # batch = {"samples": sample1, "labels": label1, "weights": 1.0}
    process_batch(batch)
```

## Core Utilities

### Abstract Dataclass ([`abstract_dataclass.py`](abstract_dataclass.py))

Provides a robust foundation for creating abstract configuration classes with proper inheritance patterns.

#### `AbstractDataclass`

```python
@dataclass(init=False)
class MyBaseConfig(AbstractDataclass):
    """Base configuration class."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

@dataclass
class ConcreteConfig(MyBaseConfig):
    """Concrete implementation."""
    learning_rate: float = 0.001

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
```

**Key Features**:
- Enforces proper abstract dataclass patterns
- Prevents instantiation of abstract classes
- Supports nested inheritance hierarchies
- Type-safe configuration objects

### Common Utilities ([`commons.py`](commons.py))

General-purpose utility functions for class introspection, iteration, and data processing.

#### Class Introspection
```python
# Get all concrete implementations of an abstract class
class BaseOptimizer(ABC):
    pass

class SGDOptimizer(BaseOptimizer):
    pass

class AdamOptimizer(BaseOptimizer):
    pass

concrete_optimizers = list(get_all_non_abstract_subclasses(BaseOptimizer))
# Returns: [SGDOptimizer, AdamOptimizer]
```

#### Iteration Utilities
```python
data = range(10)
for batch in batched(data, 3):
    print(batch)
# Output: (0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)
```

### Dictionary Iterator ([`dict_zip_iterator.py`](dict_zip_iterator.py))

Provides synchronized iteration over dictionary values with length validation.

#### `DictZipIterator`

```python
# Synchronize iteration over multiple sequences
data = {
    "gradients": [grad1, grad2, grad3],
    "parameters": [param1, param2, param3],
    "learning_rates": [0.1, 0.01, 0.001]
}

iterator = DictZipIterator(data)
for batch in iterator:
    # batch = {"gradients": grad1, "parameters": param1, "learning_rates": 0.1}
    update_parameter(batch["parameters"], batch["gradients"], batch["learning_rates"])
```

**Features**:
- Length validation across all iterators
- Type-safe iteration with generic support
- Clear error messages for mismatched lengths

## Optimizer Infrastructure

### Optimizer Modules ([`optimizer_modules.py`](optimizer_modules.py))

Provides base infrastructure for creating modular optimizer components with state management capabilities.

#### `OptimizerModule`

A lightweight base class similar to `nn.Module` but specialized for optimizer components:

```python
class PreconditionerModule(OptimizerModule):
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.accumulator = torch.zeros(shape)
        self.step_count = 0

    def update(self, gradient: torch.Tensor):
        self.accumulator += gradient ** 2
        self.step_count += 1

    def precondition(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient / (self.accumulator.sqrt() + 1e-8)

# Usage
preconditioner = PreconditionerModule((100, 50))

# State management
state_dict = preconditioner.state_dict()
preconditioner.load_state_dict(state_dict)
```

**Key Features**:
- Recursive state dictionary construction
- Support for nested `OptimizerModule` objects
- Tensor-aware serialization with `keep_vars` option
- Efficient state loading with direct tensor copying

## Mathematical Utilities

### Shampoo Utilities ([`shampoo_utils.py`](shampoo_utils.py))

Core mathematical functions for tensor manipulation, dimension merging, and distributed computing support.

#### Dimension Merging

```python
# Optimize tensor shapes for better performance
original_shape = (1, 2, 5, 1)  # Small dimensions
merged_shape = merge_small_dims(
    tensor_shape=original_shape,
    threshold=10,
    target_tensor_dimensionality=1
)
print(merged_shape)  # (10,) - all dimensions merged

# Convolution-like tensors
conv_shape = (32, 3, 64, 64)
merged_conv = merge_small_dims(
    tensor_shape=conv_shape,
    threshold=8192,
    target_tensor_dimensionality=2
)
print(merged_conv)  # (96, 4096) - optimal for Muon-style optimizers
```

#### Multi-Dimensional Splitting

```python
# Split tensors across all dimensions
tensor = torch.randn(5, 3)
split_tensors = multi_dim_split(tensor, split_size=2)
# Returns tuple of smaller tensors after splitting along each dimension

# No splitting when size exceeds dimensions
large_split = multi_dim_split(tensor, split_size=math.inf)
# Returns (tensor,) unchanged
```

#### Utility Functions

```python
# Compress sequences based on boolean selector
data = ['a', 'b', 'c', 'd']
selector = [True, False, True, False]
compressed = compress_list(data, selector)
print(compressed)  # ('a', 'c')

# Get memory footprint of data types
float32_size = get_dtype_size(torch.float32)  # 4 bytes
bool_size = get_dtype_size(torch.bool)  # 1 byte

# Generate index ranges for partitioning
partitions = [2, 3, 1]  # Partition sizes
indices = list(generate_pairwise_indices(partitions))
print(indices)  # [(0, 2), (2, 5), (5, 6)]

# Distribute buffer sizes across ranks
buffer_sizes = (128, 64, 500, 256)
distribution = distribute_buffer_sizes(buffer_sizes, group_size=2)
# Balances memory allocation across 2 ranks
```

#### Context Managers

```python
class StatefulObject:
    def __init__(self):
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

obj = StatefulObject()
with ParameterizeEnterExitContext(
    input_with_enter_exit_context=obj,
    enter_method_caller=lambda x: x.activate(),
    exit_method_caller=lambda x: x.deactivate()
):
    assert obj.active  # True inside context
assert not obj.active  # False after context
```

## Model Utilities

### Combined Linear Layer ([`shampoo_model_utils.py`](shampoo_model_utils.py))

Specialized linear layer that combines weight and bias parameters for optimizers that exploit parameter structure.

#### `CombinedLinear`

```python
# Create combined linear layer
layer = CombinedLinear(
    in_features=512,
    out_features=256,
    bias=True
)

# Forward pass
input_tensor = torch.randn(32, 512)  # (batch_size, in_features)
output = layer(input_tensor)  # (32, 256)

# Access combined parameter (weight + bias concatenated)
combined_param = layer.combined_weight  # Shape: (256, 513) when bias=True
weight_part = layer.combined_weight[:, :-1]  # (256, 512)
bias_part = layer.combined_weight[:, -1]     # (256,)
```

**Benefits**:
- Reduces parameter count for structure-exploiting optimizers
- Standard PyTorch initialization (Kaiming uniform)
- Drop-in replacement for `nn.Linear`
- Separates weight and bias during forward pass

## State Dict Management

### State Dict Utilities ([`shampoo_state_dict_utils.py`](shampoo_state_dict_utils.py))

Robust checkpointing support for complex nested state dictionaries with proper handling of `OptimizerModule` objects.

#### State Dictionary Management

```python
# Extract state dictionaries from nested objects
nested_objects = {
    "preconditioner": some_optimizer_module,
    "parameters": {"weight": tensor}
}
state_dict = extract_state_dict_content(nested_objects)

# Update state dictionary objects in place
current_state = {"param1": tensor1, "param2": {"nested": tensor2}}
new_state = {"param1": new_tensor1, "param2": {"nested": new_tensor2}}

update_param_state_dict_object(current_state, new_state)
# current_state is updated in place
```

## Quantization Support

### Quantization Framework ([`shampoo_quantization.py`](shampoo_quantization.py))

Memory-efficient tensor compression and decompression with support for different data types and automatic context management.

#### `QuantizedTensor`

```python
# Create quantized tensor from full precision
full_precision_tensor = torch.randn(1000, 1000, dtype=torch.float32)
quantized = QuantizedTensor.init_from_dequantized_tensor(
    dequantized_values=full_precision_tensor,
    quantized_dtype=torch.bfloat16,
    block_info=block_info
)

# Dequantize for computation
result = quantized.dequantize(torch.float32)
```

#### `QuantizedTensorList`

```python
# Create list of quantized tensors
tensor_list = [torch.randn(100, 100) for _ in range(5)]
quantized_list = QuantizedTensorList(
    quantized_data=[(t, None, None) for t in tensor_list],
    quantized_dtype=torch.bfloat16,
    computation_dtype=torch.float32
)

# Automatic dequantization context
with DequantizeQuantizedTensorListContext(quantized_list):
    # Access dequantized tensors for computation
    dequantized_tensors = quantized_list.dequantized_value
    # Perform computations...
    results = [torch.matmul(t, t.T) for t in dequantized_tensors]
# Automatically requantized when exiting context

# Manual control
quantized_list.dequantize_()  # Store dequantized version
# ... perform computations ...
quantized_list.quantize_()  # Convert back to quantized format
```

**Features**:
- Support for multiple quantization formats (BF16, FP16, FP32)
- Automatic context management for dequantization/quantization cycles
- Memory-efficient storage and computation
- Integration with distributed training through `BlockInfo`

#### Compression and Selection

```python
# Compress based on boolean selector
selector = (True, False, True, False, True)
compressed_list = quantized_list.compress(selector)
# Returns new QuantizedTensorList with only selected tensors
```

## Load Balancing Utilities

### Cost Models ([`load_balancing_utils.py`](load_balancing_utils.py))

Provides cost models for estimating computational and memory costs of tensors, enabling load balancing in distributed training scenarios.

#### Abstract Base Class

##### `CostModel`

```python
from abc import abstractmethod

class CostModel(AbstractDataclass):
    """Abstract base class for computing tensor cost metrics."""

    @abstractmethod
    def cost(self, tensor: torch.Tensor) -> float:
        """Compute cost for a tensor."""
        pass
```

#### Computational Cost Models

##### `PolynomialComputationalCostModel`

Estimates computational costs using polynomial functions based on tensor dimensions:

```python
# Create a quadratic cost model: cost = a + b*x + c*x²
cost_model = PolynomialComputationalCostModel(
    coefficients=(1.0, 0.1, 0.01),  # a=1.0, b=0.1, c=0.01
    min_cost=10.0  # Minimum cost threshold
)

# Example tensor computation cost
tensor = torch.randn(100, 200)
total_cost = cost_model.cost(tensor)
# Cost = max(10.0, (1.0 + 0.1*100 + 0.01*100²)) + max(10.0, (1.0 + 0.1*200 + 0.01*200²))
```

**Features**:
- Polynomial degree determined by coefficient count
- Applies to each tensor dimension separately
- Minimum cost thresholding for stability
- Suitable for computational complexity estimation

#### Memory Cost Models

##### `AlignedMemoryCostModel`

Calculates memory costs with alignment padding considerations:

```python
# Default 64-byte alignment
memory_model = AlignedMemoryCostModel(alignment_bytes=64)

# Memory cost calculation
tensor = torch.randn(100, 100, dtype=torch.float32)
memory_cost = memory_model.cost(tensor)
# Calculates: aligned_size = ceil(100*100*4 / 64) * 64 bytes

# Custom alignment
cache_aligned_model = AlignedMemoryCostModel(alignment_bytes=128)
cost_128 = cache_aligned_model.cost(tensor)
```

**Features**:
- Accounts for memory alignment padding
- Considers tensor data type size (`element_size()`)
- Configurable alignment boundaries
- Returns aligned buffer size in bytes

#### Usage in Load Balancing

```python
# Distribute tensors based on computational cost
tensors = [torch.randn(size) for size in [(100, 50), (200, 100), (50, 200)]]
comp_model = PolynomialComputationalCostModel(coefficients=(0, 1, 0))  # Linear in dimension

# Calculate costs for load balancing
costs = [comp_model.cost(t) for t in tensors]
# Use costs to distribute across devices/processes

# Memory-aware distribution
mem_model = AlignedMemoryCostModel(alignment_bytes=64)
memory_costs = [mem_model.cost(t) for t in tensors]
# Balance memory usage across devices
```

**Applications**:
- Distributed preconditioner assignment
- Memory-aware tensor partitioning
- Computational load balancing
- Resource allocation optimization

## Examples

### Custom Configuration Pattern

```python
from dataclasses import dataclass

@dataclass(init=False)
class BaseOptimizerConfig(AbstractDataclass):
    """Abstract base for optimizer configurations."""

    @abstractmethod
    def __init__(self) -> None:
        pass

@dataclass
class ClassicShampooPreconditionerConfig(BaseOptimizerConfig):
    """Concrete Shampoo configuration."""
    max_preconditioner_dim: int = 8192
    precondition_frequency: int = 100
    epsilon: float = 1e-8

    def __init__(
        self,
        max_preconditioner_dim: int = 8192,
        precondition_frequency: int = 100,
        epsilon: float = 1e-8
    ):
        self.max_preconditioner_dim = max_preconditioner_dim
        self.precondition_frequency = precondition_frequency
        self.epsilon = epsilon
```

### Optimizer Module with State Management

```python
class AdaptivePreconditioner(OptimizerModule):
    def __init__(self, param_shape: tuple[int, ...], beta2: float = 0.999):
        self.param_shape = param_shape
        self.beta2 = beta2
        self.step = 0
        self.squared_avg = torch.zeros(param_shape)

    def update(self, gradient: torch.Tensor) -> None:
        self.step += 1
        self.squared_avg.mul_(self.beta2).addcmul_(
            gradient, gradient, value=1 - self.beta2
        )

    def precondition(self, gradient: torch.Tensor) -> torch.Tensor:
        bias_correction = 1 - self.beta2 ** self.step
        corrected_avg = self.squared_avg / bias_correction
        return gradient / (corrected_avg.sqrt() + 1e-8)

# Usage with automatic state management
preconditioner = AdaptivePreconditioner((1000, 500))
state = preconditioner.state_dict()  # Includes step, squared_avg, etc.
preconditioner.load_state_dict(state)  # Restore state
```

### Mathematical Utilities for Tensor Processing

```python
# Process convolution parameters for Shampoo
def process_conv_params(param_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # Merge small dimensions for better numerical properties
    original_shape = param_tensor.shape
    merged_shape = merge_small_dims(
        tensor_shape=original_shape,
        threshold=1024,
        target_tensor_dimensionality=2
    )

    # Reshape parameter
    reshaped_param = param_tensor.view(merged_shape)

    # Split into manageable blocks if needed
    if max(merged_shape) > 8192:
        blocks = multi_dim_split(reshaped_param, split_size=4096)
    else:
        blocks = (reshaped_param,)

    return blocks

# Distribute computation across devices
def setup_distributed_computation(
    param_blocks: list[torch.Tensor],
    num_devices: int
) -> dict[int, list[torch.Tensor]]:
    # Calculate memory requirements
    buffer_sizes = tuple(block.numel() * 4 for block in param_blocks)  # 4 bytes per float32

    # Distribute across devices
    assignments = distribute_buffer_sizes(buffer_sizes, num_devices)

    # Group by device
    device_assignments = {}
    for i, (size, device_id) in enumerate(assignments):
        if device_id not in device_assignments:
            device_assignments[device_id] = []
        device_assignments[device_id].append(param_blocks[i])

    return device_assignments
```

## Contributing

When contributing to the utilities module:

1. **Maintain Backward Compatibility**: Utilities are used throughout the codebase
2. **Add Comprehensive Tests**: All utilities must include thorough test coverage
3. **Document Edge Cases**: Clearly document behavior for edge cases and error conditions
4. **Performance Considerations**: Optimize for common use cases, especially in mathematical utilities
5. **Type Safety**: Use proper type hints and ensure compatibility with `pyre-strict`
6. **Memory Efficiency**: Consider memory usage patterns, especially for quantization utilities
7. **Cross-Platform Compatibility**: Ensure utilities work across different hardware configurations

### Testing Guidelines

- Add test files in the [`tests/`](tests/) directory
- Follow the naming convention `{module_name}_test.py`
- Include edge cases, error conditions, and performance tests
- Use parameterized tests for comprehensive coverage
- Test distributed scenarios when applicable

### Documentation Standards

- Include docstrings for all public functions and classes
- Provide usage examples in docstrings
- Document parameter constraints and return value formats
- Update this README when adding new utilities or modules

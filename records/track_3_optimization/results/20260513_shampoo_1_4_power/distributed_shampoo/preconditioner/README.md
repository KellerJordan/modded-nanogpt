# Distributed Shampoo Preconditioner Module

This module implements the preconditioner components for the Distributed Shampoo optimizer, a second-order optimization algorithm that uses curvature information to achieve superior convergence properties compared to first-order methods like SGD and Adam.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Preconditioner Implementations](#preconditioner-implementations)
- [Matrix Functions Library](#matrix-functions-library)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

The preconditioner module is the core computational engine of Distributed Shampoo. It provides multiple preconditioning strategies that transform gradients using second-order information, enabling faster and more stable convergence for deep learning models.

### Key Features

- **Multiple Preconditioner Types**: From simple diagonal scaling to sophisticated Kronecker-factored approximations
- **Distributed Computing**: Built for large-scale distributed training with memory-efficient implementations
- **Numerical Stability**: Advanced matrix functions with multiple algorithms and fallback mechanisms
- **Vectorized Operations**: Optimized in-place computations for performance
- **Configurable Precision**: Support for different numerical precisions and stability configurations

## Quick Start

### Basic Usage

```python
# Create a Shampoo preconditioner directly (advanced usage)
preconditioner = RootInvShampooPreconditionerList(
    block_list=blocked_params,
    preconditioner_config=RootInvShampooPreconditionerConfig(),
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=0.999,
    weighting_factor=1.0 - 0.999,
    epsilon=1e-8,
    use_bias_correction=True
)

# Update preconditioners with gradients
preconditioner.update_preconditioners(
    masked_grad_list=gradients,
    step=step_tensor,
    perform_amortized_computation=True
)

# Apply preconditioning to gradients
preconditioned_gradients = preconditioner.precondition(masked_grad_list=gradients)
```

### Alternative Preconditioners

```python
# AdaGrad-style diagonal preconditioning
adagrad_preconditioner = AdagradPreconditionerList(
    block_list=blocked_params,
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=0.999,  # Use 1.0 for pure AdaGrad
    weighting_factor=1.0 - 0.999,  # Use 1.0 for pure AdaGrad
    epsilon=1e-8,
    use_bias_correction=True
)

# SGD (no preconditioning)
sgd_preconditioner = SGDPreconditionerList(block_list=blocked_params)
```

## Architecture

### Base Class: `PreconditionerList`

All preconditioners inherit from the abstract base class `PreconditionerList` defined in [`preconditioner_list.py`](preconditioner_list.py):

```python
class PreconditionerList(ABC):
    @abstractmethod
    def update_preconditioners(self) -> None:
        """Update preconditioner states with current gradient information."""

    @abstractmethod
    def precondition(self) -> None:
        """Apply preconditioning transformation to gradients."""

    def compress_preconditioner_list(self) -> None:
        """Compress preconditioner list for memory efficiency."""
```

### Key Design Principles

1. **Modularity**: Each preconditioner type is self-contained and interchangeable
2. **Distributed Training**: Optimized for large-scale training with minimal communication overhead
3. **Numerical Stability**: Robust implementations with multiple fallback algorithms
4. **Extensibility**: Easy to add new preconditioner types and matrix function algorithms

## Preconditioner Implementations

### 1. Shampoo Preconditioners

**File**: [`shampoo_preconditioner_list.py`](shampoo_preconditioner_list.py)

The flagship implementation providing three sophisticated variants for second-order optimization:

#### `RootInvShampooPreconditionerList`
- **Purpose**: Direct computation of matrix inverse roots
- **Best for**: Medium-sized models where direct computation is feasible
- **Algorithm**: Computes G^(-1/4) directly using eigendecomposition or Newton methods
- **Memory**: Most memory-efficient Shampoo variant

```python
# Adagrad-like Shampoo (cumulative sum of gradient outer products)
preconditioner_adagrad = RootInvShampooPreconditionerList(
    block_list=blocked_params,
    preconditioner_config=RootInvShampooPreconditionerConfig(),
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=1.0,
    weighting_factor=1.0,
    epsilon=1e-8,
    use_bias_correction=False
)

# RMSprop-like Shampoo (exponential moving average of gradient outer products)
beta2 = 0.999
preconditioner_rmsprop = RootInvShampooPreconditionerList(
    block_list=blocked_params,
    preconditioner_config=RootInvShampooPreconditionerConfig(),
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=beta2,
    weighting_factor=1.0 - beta2,
    epsilon=1e-8,
    use_bias_correction=False
)

# Adam-like Shampoo (EMA with bias correction)
beta2 = 0.999
preconditioner_adam = RootInvShampooPreconditionerList(
    block_list=blocked_params,
    preconditioner_config=RootInvShampooPreconditionerConfig(),
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=beta2,
    weighting_factor=1.0 - beta2,
    epsilon=1e-8,
    use_bias_correction=True
)
```

#### `EigendecomposedShampooPreconditionerList`
- **Purpose**: Maintains explicit eigendecomposition for numerical stability
- **Best for**: Ill-conditioned problems requiring maximum stability
- **Algorithm**: Stores Q, Λ from G = QΛQ^T and computes preconditioners in eigenspace
- **Memory**: Higher memory usage but superior numerical properties

#### `EigenvalueCorrectedShampooPreconditionerList`
- **Purpose**: Separate tracking of eigenvectors and eigenvalues with corrections
- **Best for**: Advanced scenarios requiring eigenvalue manipulation
- **Algorithm**: Updates eigenvalues directly in eigenspace for improved conditioning

### 2. AdaGrad Preconditioner

**File**: [`adagrad_preconditioner_list.py`](adagrad_preconditioner_list.py)

Diagonal preconditioning using accumulated squared gradients:

```python
# Pure AdaGrad (cumulative)
adagrad = AdagradPreconditionerList(
    block_list=blocked_params,
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=1.0,
    weighting_factor=1.0,
    epsilon=1e-8
)

# RMSprop-style (exponential moving average)
rmsprop = AdagradPreconditionerList(
    block_list=blocked_params,
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=0.999,
    weighting_factor=1.0 - 0.999,
    epsilon=1e-8
)

# Adam-style with bias correction
adam_style = AdagradPreconditionerList(
    block_list=blocked_params,
    state=optimizer_state,
    block_info_list=block_info_list,
    beta2=0.999,
    weighting_factor=1.0 - 0.999,
    epsilon=1e-8,
    use_bias_correction=True
)
```

**Features**:
- Memory-efficient diagonal preconditioning
- Vectorized operations using `torch._foreach_*` for performance
- In-place updates to minimize memory allocation

### 3. SGD Preconditioner

**File**: [`sgd_preconditioner_list.py`](sgd_preconditioner_list.py)

Identity transformation (no preconditioning):
```python
sgd_preconditioner = SGDPreconditionerList(block_list=blocked_params)
```

- **Use case**: Baseline comparison or fallback when second-order methods fail
- **Performance**: Minimal computational overhead
- **Compatibility**: Works with any parameter configuration

### 4. Specialized Preconditioners

#### Sign Descent Preconditioner
**File**: [`sign_descent_preconditioner_list.py`](sign_descent_preconditioner_list.py)

Uses only gradient signs for direction:
```python
sign_preconditioner = SignDescentPreconditionerList(
    block_list=blocked_params,
    preconditioner_config=SignDescentPreconditionerConfig(
        scale_fn=lambda grad: 1.0  # Constant scaling
    )
)
```

#### Spectral Descent Preconditioner
**File**: [`spectral_descent_preconditioner_list.py`](spectral_descent_preconditioner_list.py)

Advanced spectral analysis-based preconditioning using matrix orthogonalization for specialized optimization landscapes. This method transforms gradients by computing their orthogonalized versions, which can provide better optimization dynamics for certain problems.

**Key Features**:
- **Matrix Orthogonalization**: Computes the closest orthogonal matrix to the gradient matrix
- **2D Parameter Requirement**: Only works with 2D parameters or parameters reshaped to 2D
- **Multiple Algorithms**: Supports SVD-based and Newton-Schulz iterative orthogonalization
- **Configurable Scaling**: Dimension-aware scaling functions for improved convergence

**Basic Usage**:
```python
# Direct instantiation (advanced usage)
spectral_preconditioner = SpectralDescentPreconditionerList(
    block_list=blocked_2d_params,  # Must be 2D tensors
    preconditioner_config=SpectralDescentPreconditionerConfig()
)

# Apply spectral descent preconditioning
preconditioned_grads = spectral_preconditioner.precondition(masked_grad_list=gradients)
```

**SVD-Based Orthogonalization**:
```python
# Exact orthogonalization using SVD
svd_config = SpectralDescentPreconditionerConfig(
    orthogonalization_config=SVDOrthogonalizationConfig(
        scale_by_nuclear_norm=False,  # Don't scale by nuclear norm
        scale_by_dims_fn=lambda d_in, d_out: 1.0  # No dimension scaling
    )
)

# With nuclear norm scaling for better conditioning
nuclear_norm_config = SpectralDescentPreconditionerConfig(
    orthogonalization_config=SVDOrthogonalizationConfig(
        scale_by_nuclear_norm=True,  # Scale by nuclear norm
        scale_by_dims_fn=lambda d_in, d_out: (d_out / d_in) ** 0.5
    )
)
```

**Newton-Schulz Iterative Orthogonalization**:
```python
# Fast iterative semi-orthogonalization
newton_config = SpectralDescentPreconditionerConfig(
    orthogonalization_config=NewtonSchulzOrthogonalizationConfig(
        num_iterations=5,  # Number of iterations
        coefficients=(3.4445, -4.7750, 2.0315),  # Quintic Newton-Schulz coefficients
        scale_by_dims_fn=lambda d_in, d_out: max(1, d_out / d_in) ** 0.5
    )
)

# Custom scaling for wide matrices
wide_matrix_config = SpectralDescentPreconditionerConfig(
    orthogonalization_config=NewtonSchulzOrthogonalizationConfig(
        num_iterations=3,  # Fewer iterations for speed
        scale_by_dims_fn=lambda d_in, d_out: min(2.0, d_out / d_in)  # Cap scaling
    )
)
```

**Important Considerations**:
- **2D Requirement**: All parameters must be 2D matrices. Use `max_preconditioner_dim=math.inf` and `target_parameter_dimensionality=2` to automatically reshape higher-dimensional parameters
- **Memory Usage**: SVD orthogonalization requires full SVD computation; Newton-Schulz is more memory-efficient
- **Convergence**: Newton-Schulz provides approximate orthogonalization but converges faster
- **Scaling Functions**: Dimension-based scaling can significantly impact convergence behavior
- **Computational Cost**: More expensive than diagonal preconditioning but can provide superior optimization dynamics

**Performance Tips**:
- Use Newton-Schulz with 3-5 iterations for a good speed/accuracy tradeoff
- Apply spectral descent only to large 2D parameters (e.g., hidden layer weights)
- Combine with standard Shampoo or AdaGrad for non-2D parameters
- Consider dimension-adaptive scaling for networks with varying layer sizes

## Matrix Functions Library

### Core Functions ([`matrix_functions.py`](matrix_functions.py))

The numerical engine providing robust implementations of advanced matrix operations:

#### Matrix Inverse Roots
```python
def matrix_inverse_root(
    A: Tensor,
    root: Fraction,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
    epsilon: float = 0.0,
) -> Tensor:
    """Compute A^(-1/root) using various algorithms."""
```

**Supported Algorithms**:
- **Eigendecomposition**: Most stable, best for symmetric positive definite matrices
- **Newton Iteration**: Fast convergence for well-conditioned matrices
- **Higher-Order Coupled**: Advanced methods for fractional powers

#### Eigendecomposition
```python
def matrix_eigendecomposition(
    A: Tensor,
    epsilon: float = 0.0,
    eigendecomposition_config: EigendecompositionConfig = DefaultEigendecompositionConfig,
    eigenvectors_estimate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Robust eigendecomposition with multiple algorithms."""
```

**Features**:
- Automatic fallback between `torch.linalg.eigh` and QR algorithm
- Configurable convergence tolerances
- Memory-efficient implementations

#### Matrix Orthogonalization
```python
def matrix_orthogonalization(
    A: Tensor,
    orthogonalization_config: OrthogonalizationConfig = DefaultNewtonSchulzOrthogonalizationConfig,
) -> Tensor:
    """Compute the orthogonalization of a matrix."""
```

**Supported Algorithms**:
- **SVD Orthogonalization**: Uses singular value decomposition for exact orthogonalization
- **Newton-Schulz Iteration**: Quintic Newton-Schulz iteration for semi-orthogonalization with faster convergence

**Features**:
- Configurable scaling based on matrix dimensions
- Iterative refinement for improved numerical properties
- Optimized for different matrix aspect ratios

### Configuration System ([`matrix_functions_types.py`](matrix_functions_types.py))

Comprehensive configuration classes for customizing numerical behavior.

## Examples

### Custom Preconditioner Configuration

```python
# Advanced configuration for DistributedShampoo optimizer
optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.95),
    epsilon=1e-10,
    max_preconditioner_dim=512,
    precondition_frequency=50,
    use_bias_correction=True,
    use_decoupled_weight_decay=True,
    grafting_config=RMSpropPreconditionerConfig(beta2=0.95, weighting_factor=1.0 - 0.95, epsilon=1e-8),
    preconditioner_config=RootInvShampooPreconditionerConfig(
        amortized_computation_config=EigenConfig(
            max_iterations=1000,
            tolerance=1e-8
        )
    )
)
```

## Contributing

When contributing to the preconditioner module:

1. **Add Tests**: All new functionality must include comprehensive tests
2. **Documentation**: Update this README and add inline documentation
3. **Numerical Stability**: Ensure robust handling of edge cases and ill-conditioned matrices
4. **Performance**: Optimize critical computational paths
5. **Backward Compatibility**: Maintain API compatibility when possible

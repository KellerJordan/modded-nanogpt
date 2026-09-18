# Polar Express Optimization

## Optimizations

All changes are confined to the **Polar Express orthogonalization** step and the **sparse communication** routines used during distributed training. The model architecture, hyperparameters, training schedule, and all other code remain identical.

### 1. Gram-Trace Normalization with Diagonal Fold

**Before:** The spectral norm of the input matrix is bounded by dividing by the Frobenius norm — a full-matrix operation that materializes a normalized copy every iteration.

```python
X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
```

**After:** Normalization uses the Gram matrix diagonal (trace of X^T X), which is already computed for the iteration. The scale factor is folded into the first polynomial coefficient via diagonal addition, avoiding a separate full-matrix division.

```python
scale = A.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True, dtype=torch.float32)
scale.sqrt_().mul_(_POLAR_SAFETY_SCALE).add_(_POLAR_EPS).reciprocal_()
A.mul_((scale * scale)[..., None].to(dtype=A.dtype))
# ... first iteration ...
B.diagonal(dim1=-2, dim2=-1).add_(_P0A)  # fold aI into polynomial
B.mul_(scale[..., None].to(dtype=B.dtype))
```

This eliminates the `aX + X @ B` fused addmm/baddbmm pattern (which required referencing X twice and triggered defensive copies in PyTorch) and replaces it with a single `bmm` per iteration.

### 2. Specialized Tall-Matrix Paths (MLP Weights)

The MLP weight bank has shape (3072, 768) per matrix — a 4:1 tall aspect ratio. The baseline uses the same iteration loop for all shapes, performing 5 tall-matrix multiplications per step.

**After:** Two dedicated tall-matrix paths are introduced:

- **3-step accumulated transform** (`_polar_express_tall_transform_batched`, used for steps < 500): Propagates the Gram and accumulated transform entirely on the small n×n factor, touching the tall m×n matrix only twice (initial XTX, final bmm). This cuts 3 expensive tall-matrix bmm operations per step.

- **5-step standard path** (`_polar_express_tall_standard_batched`, used for steps ≥ 500): Uses the full 5-step self-correcting iteration with Gram-trace normalization for higher quality convergence in later training.

The transition at step 500 (`_POLAR_SWITCH_STEP`) balances speed in early training against quality in later training.

### 3. Fully Unrolled Iterations

**Before:** Polar Express iterations are written as a Python `for` loop over the coefficient list, which prevents `torch.compile` from fully optimizing memory reuse across iterations.

**After:** Each specialized function has all 5 (or 3) iterations fully unrolled with pre-extracted scalar coefficients (`_P0A`, `_P0B`, ..., `_P4C`). Combined with `fullgraph=True`, this allows the compiler to fuse operations across iteration boundaries and optimize buffer reuse.

### 4. Sparse Communication Improvements

Several micro-optimizations to the distributed sparse gradient communication:

- **Cached partition boundaries** (`_sparse_partition_boundaries`): The `searchsorted` boundary array is computed once per (N, world_size) pair and reused, avoiding repeated allocation.
- **In-place index slicing**: Local rank's rows are excluded via `narrow` + conditional `copy_` instead of `torch.cat` with two slices, reducing CPU-side allocations.
- **Local gradient merge**: `sparse_comms_merge_gradients` now operates on a `narrow`ed local slice with adjusted indices (`recv_idx.sub(start)`), avoiding the final full-tensor slice.

## Results

All experiments run on 8×H100 GPUs with identical seeds, hyperparameters, and training schedule. Each configuration was run multiple times to assess variance.

| Configuration | Runs | Mean Time (ms) | Mean Val Loss | Time Saved |
|---|---|---|---|---|
| **Baseline (03/22/26)** | 5 | 85,658 | 3.2786 | — |
| **Baseline + Polar Express Optimization** | 5 | 85,219 | 3.2777 | **439 ms (0.51%)** |
| **PR#251** | 4 | 84,930 | 3.2788 | — |
| **PR#251 + Polar Express Optimization** | 5 | 84,545 | 3.2795 | **385 ms (0.45%)** |

All configurations achieve val_loss < 3.28.

## Applicability

The Polar Express optimizations are independent of other code changes and can be applied separately:

- **On the baseline codebase**: The polar version saves ~439 ms on average while producing equivalent or slightly better val_loss.

- **On top of existing PR#251**: The polar version saves ~385 ms on average, demonstrating that the optimizations compose cleanly with the independent changes in PR#251.

## Log Files

Training logs for each configuration are stored as text files named by UUID:

```
./baseline/     # 5 runs of baseline (03/22/26)
./baseline_plus_polar/        # 5 runs of official + polar optimizations
./pr251/           # 4 runs of PR#251
./pr251_plus_polar/     # 5 runs of PR#251 + polar optimizations
```
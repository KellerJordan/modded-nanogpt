# FP8 Quantization Attempts Log

## Goal
Speed up training by using FP8 tensor cores for attention projections (QKV, O) and MLP matmuls (c_fc, c_proj). The lm_head already uses FP8 via `mm_t_op` custom_op.

## Microbenchmark Results (1x H100 80GB)
Raw FP8 `_scaled_mm` vs BF16 matmul at actual model sizes (M=16384, dim=768, mlp_dim=3072):

| Operation | BF16 (ms) | FP8 (ms) | Speedup |
|-----------|-----------|----------|---------|
| attn_qkv (768→3072) | 0.794 | 0.112 | **7.1x** |
| attn_out (768→768) | 0.055 | 0.024 | 2.3x |
| mlp_fc (768→3072) | 0.118 | 0.070 | 1.7x |
| mlp_proj (3072→768) | 0.103 | 0.058 | 1.8x |

0-dim vs 1-dim scale tensors: identical performance (0.057ms both).

## Baseline
- 1x H100, `torchrun --nproc_per_node=1`: **237ms/step**

---

## Attempt 1: Dynamic per-tensor scaling (from prior session)
- Quantize weights AND activations to FP8 on every call using `amax` reduction
- Result: **580ms/step (2.4x slower)**
- Cause: per-call `amax` reduction + quantization overhead dominates at small matrix sizes

## Attempt 2: Pre-quantized weights + fixed activation scale
- Pre-quantize `attn_bank` and `mlp_bank` to FP8 once per optimizer step (amortized over 8 micro-batches)
- Fixed activation scale `FP8_ACT_SCALE = 16.0 / 448 ≈ 0.036` (no amax reduction)
- Fixed grad scale `FP8_GRAD_SCALE = 1024.0 / 57344`
- Attention: used `linear_fp8_op` custom_op (with inner `@torch.compile`) — worked
- MLP: used `_scaled_mm` directly inside `FusedLinearReLUSquareFunction` (autograd.Function)

### Attempt 2a: MLP with direct `_scaled_mm` + 0-dim scales from `.clone()`
- Scale extraction: `self._mlp_bank_scales[i, 0].clone()` (2D tensor, 0-dim result)
- Result: **Inductor LoweringException**
- Error location: `template_heuristics/triton.py:1882` in `adjust_kernel_inputs`
- Bug: `L[aten.unsqueeze](scale_b, 0)` → `view()` → `resolve_negative_size` assertion failure
- The inductor tries to unsqueeze 0-dim scale to [1,1] but symbolic size check fails

### Attempt 2b: MLP with `scaled_mm_op` custom_op wrapper
- Created generic `nanogpt::scaled_mm` custom_op wrapping `_scaled_mm` with inner `@torch.compile`
- This provides a compile boundary like the working `mm_t_op` and `linear_fp8_op`
- Result: **965ms/step (4.1x slower)**
- Cause: ~88 custom_op calls per step (12 MLP × 4 + 10 attn × 4), each breaking fusion

### Attempt 2c: 1D scale storage to avoid 2D indexing
- Changed `_mlp_bank_scales` from shape `(12, 2)` to `(24,)` — flat 1D
- Scale extraction: `self._mlp_bank_scales[i * 2]` (1D indexing)
- Direct `_scaled_mm` in autograd.Function (no custom_op wrapper)
- Result: **Same inductor LoweringException** — the bug is not specific to 2D indexing

### Attempt 2d: 1-dim `[1]` scale tensors (bypass unsqueeze path)
- Scale extraction: `self._mlp_bank_scales[i*2 : i*2+1]` (slice gives shape `[1]`)
- `act_scale = post.new_tensor([act_s], dtype=torch.float32)` (also shape `[1]`)
- This skips the buggy `unsqueeze` code path (line 1878: `if len(scale.get_size()) == 0`)
- Result: **557ms/step (2.35x slower)** — compiles and runs!
- But still way slower than baseline due to fusion breakage from `_scaled_mm` itself

### Attempt 2e: Attention-only FP8 (MLP FP8 disabled)
- Only attention uses FP8 via `linear_fp8_op` custom_op
- MLP uses standard BF16 `post @ W2`
- Result: **542ms/step (2.29x slower)**
- Shows that even attention-only FP8 custom_ops break fusion badly

## Root Cause Analysis

### The inductor `_scaled_mm` bug
Location: `torch/_inductor/template_heuristics/triton.py:1878-1882`
```python
if len(scale_a.get_size()) == 0 or len(scale_b.get_size()) == 0:
    assert len(scale_a.get_size()) == len(scale_b.get_size())
    scale_a = L[aten.unsqueeze](L[aten.unsqueeze](scale_a, 0), 1)
    scale_b = L[aten.unsqueeze](L[aten.unsqueeze](scale_b, 0), 1)
```
The `unsqueeze` on a 0-dim tensor calls `view(x, [1])` which triggers `resolve_negative_size` → `check_equals(sympy_product([]), sympy_product([1]))`. This fails when the 0-dim tensor comes from buffer indexing (the symbolic size representation doesn't simplify properly).

### The fusion problem
Even when `_scaled_mm` compiles (with 1-dim scales), it generates a separate CUTLASS/cuBLAS kernel that can't be fused with surrounding operations. In the baseline, inductor fuses `post @ W2` with norm, residual adds, and lambda multiplies into combined kernels. `_scaled_mm` breaks this fusion.

The existing `mm_t_op` (lm_head FP8) works because the lm_head is a single large matmul at the end — fusion losses there are minimal.

## What would fix this
1. **Fix the inductor unsqueeze bug** for 0-dim scale tensors in `_scaled_mm` lowering — would allow `_scaled_mm` directly inside compiled autograd.Functions without custom_op wrappers
2. **If inductor could fuse `_scaled_mm`** with surrounding pointwise ops (like it does for regular matmul), we'd get FP8 compute benefits without fusion losses
3. Both fixes together would likely yield a speedup given the 1.7-7x raw matmul improvements

## Attempt 2f: MLP-only FP8 with direct `_scaled_mm` in autograd.Function (no custom_ops)
- Pre-quantized weights (once per optimizer step), fixed activation scale
- `_scaled_mm` directly inside `FusedLinearReLUSquareFunction` autograd.Function
- 1-dim scales to bypass inductor unsqueeze bug
- Attention stays BF16
- Result: **263ms/step (11% slower than 237ms baseline)**
- Overhead breakdown:
  - Activation quantization (div + cast): ~2.2ms
  - Activation transposes (.T.contiguous()): ~0.7ms
  - **Fusion breakage: ~23ms** (dominant)
  - FP8 compute savings: only ~2.2ms
- The inductor can't fuse `_scaled_mm` with surrounding norm/residual/lambda ops, causing each MLP to generate separate kernel launches

## Previous Conclusion: REVERTED (approaches using `_scaled_mm`)
All `_scaled_mm`-based approaches reverted. Fusion breakage costs ~23ms, far exceeding compute savings.

---

## Attempt 3: FP8 inside Triton kernel (no `_scaled_mm`, no fusion breakage)

### Key Insight
Instead of `_scaled_mm` (opaque CUDA kernel), use FP8 `tl.dot` **directly inside the existing `linear_relu_square` Triton kernel**. Both inputs as FP8 → FP8 tensor cores, FP32 accumulation. No fusion breakage since it's the same kernel.

### Microbenchmark: `tl.dot` FP8 vs BF16 (pure matmul, 16384×768 @ 768×3072)
| Variant | Time (ms) | Notes |
|---------|-----------|-------|
| BF16×BF16 | 0.121 | Baseline `tl.dot` |
| Cast BF16→FP8 in-kernel + FP8 | 0.130 | **Slower** — in-kernel cast overhead kills it |
| FP8×FP8 (both pre-cast) | 0.101 | **1.2x faster** — real FP8 benefit |

### Critical finding: in-kernel cast kills performance
Casting `a.to(tl.float8e4nv)` inside the kernel's inner loop adds per-tile instructions that negate FP8 compute gains. Solution: cast activations to FP8 **outside** the kernel (separate CUDA kernel, 0.014ms), pass both FP8 tensors via TMA.

### BLOCK_K advantage for FP8
FP8 elements are 1 byte vs 2 bytes for BF16. This allows BLOCK_K=128 (vs 64 for BF16) within the same shared memory budget. BF16 with BLOCK_K=128 fails with "out of resource: shared memory" (426KB > 232KB limit).

| Config | Kernel Time (ms) | Speedup |
|--------|------------------|---------|
| BF16 BLOCK_K=64 | 0.149 | baseline |
| FP8 BLOCK_K=64 (pre-cast) | 0.119 | 1.25x |
| FP8 BLOCK_K=128 (pre-cast) | 0.109 | **1.37x** |

### Attempt 3a: In-kernel cast approach
- Cast `a.to(tl.float8e4nv)` inside the Triton kernel
- Weight loaded as FP8 via TMA
- Result: **242ms/step** (vs 237ms baseline — 5ms slower)
- Cause: in-kernel cast overhead exceeds FP8 compute savings

### Attempt 3b: Pre-cast activation + BLOCK_K=64
- Cast x to FP8 outside kernel: `x_f8 = x_flat.to(torch.float8_e4m3fn)`
- Both `a_f8` and `b_f8` loaded as FP8 via TMA
- Result: **236.5ms/step** (vs 237.2ms baseline — 0.7ms faster)

### Attempt 3c: Pre-cast activation (no scaling) + BLOCK_K=128
- Same as 3b but with BLOCK_K=128 for FP8 (only possible due to half element size)
- Result: **233.6ms/step** (vs 237.2ms baseline — **3.6ms faster, 1.5% speedup**)
- **Convergence FAILED**: val_loss=3.5176 vs baseline 3.2790. Direct FP8 cast without scaling loses too much precision (FP8 E4M3 has only 3 mantissa bits).

### Attempt 3d: Pre-cast activation WITH fixed scaling + BLOCK_K=128 + forward-only FP8
- Scale activations before FP8 cast: `x_f8 = (x / FP8_ACT_SCALE).to(fp8)` where `FP8_ACT_SCALE = 16/448 ≈ 0.036`
- Pre-multiply combined scale (`w_scale * act_scale`) in `quantize_weights_fp8` to avoid symbolic float × float in compiled graph (Triton launcher error)
- Forward-only FP8: backward `linear_relu_square` stays BF16 (no grad FP8 cast needed)
- Result: **235.2ms/step** (vs 237.2ms baseline — **2.0ms faster, 0.8% speedup**)
- **Convergence MATCHED**: val_loss=3.2832 vs baseline 3.2790 (+0.004, within noise)
- Total train time: 655.0s vs baseline 663.9s = **8.9 seconds faster**
- Scaling adds ~1.5ms overhead (div by constant before cast), but saves convergence

### Implementation details (current, Attempt 3d)
- Pre-quantize weights: `quantize_weights_fp8(model)` after each optimizer step
- Vectorized: `flat.abs().amax(dim=1)` for all 24 weights at once
- Forward: `x_f8 = (x / act_scale).to(fp8)` then `linear_relu_square(x, W1, a_f8=x_f8, b_f8=W1_f8, w_scale=combined_scale)`
- Backward: stays fully BF16 (no FP8 in backward)
- Weight grads (dW1, dW2) and dx: all BF16
- No `_scaled_mm` anywhere — pure Triton `tl.dot`
- Combined scale pre-computed in `quantize_weights_fp8`: `scales * FP8_ACT_SCALE`
- On 8×H100 (40ms/step): estimated ~0.25ms/step savings → ~0.4s over 1500 steps

### Bugs encountered
- **Triton launcher TypeError**: `w_scale=W1_s * _FP8_ACT_SCALE` inside compiled graph creates symbolic expression `0.0357*zuf0` that Triton can't accept. Fix: pre-multiply in `quantize_weights_fp8`.
- **In-kernel cast kills perf**: `a.to(tl.float8e4nv)` inside tight loop is slower than a separate cast kernel outside
- **Direct FP8 cast hurts convergence**: Must scale activations (÷ FP8_ACT_SCALE) before casting to preserve precision

### Follow-up: optimize the activation scaling overhead
The scaling div+cast (`x / act_scale` then `.to(fp8)`) adds ~1.5ms/step vs the unscaled version (233.6ms unscaled vs 235.2ms scaled). This is two separate kernel launches (divide, then cast) on a (16384, 768) tensor, 96 times per step (12 layers × 8 micro-batches on 1×H100).

Options to reduce:
1. **Fused scale+cast Triton kernel** — tiny pointwise kernel: load BF16, multiply by `1/act_scale`, write FP8. One kernel launch + one read instead of two. Saves ~1ms/step.
2. **Move cast into compiled graph** — do `x_f8 = (norm(lane0) / scale).to(fp8)` in the compiled model forward (before calling `ReLUSqrdMLP`), so inductor can fuse div+cast with the preceding norm. Pass x_f8 as arg to the autograd function. Riskier but could eliminate the overhead entirely.
3. **Multiply instead of divide** — `x * (1/act_scale)` = `x * 28.0`. Trivial change, small gain.
4. **Skip scaling + add ~5 steps** — unscaled is 1.6ms/step faster but val_loss is 0.004 worse. 5 extra steps at 635ms/step = 3.2s. Net vs scaled on 1×H100: saves 2.4s more from speed but costs 3.2s from extra steps = -0.8s. Not worth it on 1×H100. On 8×H100 with cheaper steps it might break even.

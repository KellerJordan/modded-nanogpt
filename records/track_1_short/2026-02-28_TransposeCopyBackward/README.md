# Triton Transpose Copy in FusedSoftcappedCrossEntropy Backward

Replace `.T.contiguous()` with a custom Triton tiled transpose kernel (`transpose_copy`) in `FusedSoftcappedCrossEntropy.backward` for the FP8 weight-gradient matmul. This avoids the
generic elementwise copy kernel on large tensors.

Additional changes:
- `_transpose_copy_kernel` offsets cast to int64 to handle stage 3 matrices (49152 x 50304 exceeds int32 address space)
- `transpose_copy` tile size updated from 32x32 to 64x128, num_warps from 4 to 8

```
(modded-nanogpt) sam@8xh100 modded-nanogpt % python compare_logs.py
          Runs  Steps    Time μ     Time σ   Time +/-   Loss μ    Loss σ   Loss +/-   p
Baseline  6     1490     88.3765    0.0974   +0.0000    3.2791    0.0006   +0.0000    0.0091
This PR   6     1490     87.9355    0.1931   -0.4410    3.2789    0.0006   -0.0002    0.0026

Baseline:
  losses = [3.2799, 3.2789, 3.2783, 3.2787, 3.2798, 3.2791]
  times  = [88.3510, 88.3980, 88.2810, 88.5590, 88.3200, 88.3500]

This PR:
  losses = [3.2781, 3.2784, 3.2795, 3.2789, 3.2793, 3.2793]
  times  = [88.3240, 87.8240, 87.8740, 87.9120, 87.8530, 87.8260]
```

## Microbenchmark (1xH100, per-call)

| Path | Shape | Compiled | Triton 64x128 w=8 | Speedup |
|------|-------|----------|-------------------|---------|
| CE backward stage 1 | (16384, 50304) fp8e5m2 | 0.769 ms | 0.588 ms | 1.31x |
| CE backward stage 2 | (32768, 50304) fp8e5m2 | 1.451 ms | 1.179 ms | 1.23x |
| CE backward stage 3 | (49152, 50304) fp8e5m2 | 2.170 ms | 1.768 ms | 1.23x |


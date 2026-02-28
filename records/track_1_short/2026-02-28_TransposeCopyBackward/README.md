# Triton Transpose Copy in FusedSoftcappedCrossEntropy Backward

Replace `.T.contiguous()` with a custom Triton tiled transpose kernel (`transpose_copy`) in `FusedSoftcappedCrossEntropy.backward` for the FP8 weight-gradient matmul. This avoids the
generic elementwise copy kernel on large tensors.

Additional changes:
- `_transpose_copy_kernel` offsets cast to int64 to handle stage 3 matrices (49152 x 50304 exceeds int32 address space)
- `transpose_copy` tile size updated from 32x32 to 64x128, num_warps from 4 to 8

```
(modded-nanogpt) sam@8xh100$ python compare_logs.py 
                   Runs  Steps    Time μ   Time σ  Time +/-   Loss μ   Loss σ  Loss +/-        p
Baseline                2   1490   88.5470   0.0198    0.0000   3.2797   0.0003    0.0000   0.1872
This PR                 2   1490   88.1965   0.0262   -0.3505   3.2797   0.0001    0.0000   0.1024

Baseline:
  losses = [3.2799, 3.2795]
  times  = [88.5330, 88.5610]

This PR:
  losses = [3.2798, 3.2796]
  times  = [88.2150, 88.1780]
```

## Microbenchmark (1xH100, per-call)

| Path | Shape | Compiled | Triton 64x128 w=8 | Speedup |
|------|-------|----------|-------------------|---------|
| CE backward stage 1 | (16384, 50304) fp8e5m2 | 0.769 ms | 0.588 ms | 1.31x |
| CE backward stage 2 | (32768, 50304) fp8e5m2 | 1.451 ms | 1.179 ms | 1.23x |
| CE backward stage 3 | (49152, 50304) fp8e5m2 | 2.170 ms | 1.768 ms | 1.23x |


# FP8 MLP Input Gradient and Kernel Tuning Statistics

## Summary

This PR makes three changes.

The `linear_relu_square_kernel` FP8 forward drops `BLOCK_SIZE_K` from 128 to 64,
which frees shared memory for seven pipeline stages instead of three under the
232 448-byte sm_90 ceiling. The BF16 backward keeps three stages, where seven
would not fit.

The `ce_fwd_bwd_kernel` raises its minimum blocks per SM from 2 to 3 and unrolls
the softmax-sum loop 25 times.

The MLP up-projection input gradient runs in FP8 through `torch._scaled_mm` with
delayed per-tensor scaling. The Triton kernel emits the FP8 pre-activation
gradient and its partial amax in the same pass, so the scale costs no extra
traversal of the tensor. `dW1` stays BF16, since it needs the gradient
transposed and a transposed FP8 copy costs more than the GEMM saves.

- GPUs: 8x H100 SXM
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- Driver: `580.126.16`
- runs: `13`

## Statistics

| metric                |           value |
| --------------------- | --------------: |
| mean val loss         |    3.2778153846 |
| val loss sample std   |    0.0028483463 |
| one-sided p vs 3.28   |    0.0085539733 |
| mean train time       |     73.250077 s |
| train time sample std |      0.038718 s |
| median train time     |        73.238 s |
| train time range      | 73.205-73.316 s |

## Comparison

Both arms ran back to back on the same 8x H100 node.

|                 |     baseline |      this PR |
| --------------- | -----------: | -----------: |
| mean train time |  73.703538 s |  73.250077 s |
| mean val loss   | 3.2784538462 | 3.2778153846 |

Mean train time falls by 0.4535 s, 19.9 sigma on a Welch two-sample test. The
two arms do not overlap: the fastest baseline run is 73.590 s and the slowest
run of this PR is 73.316 s. Mean val loss falls by 0.00064.

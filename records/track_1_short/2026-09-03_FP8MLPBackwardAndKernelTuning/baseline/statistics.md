# Baseline Statistics

## Summary

Baseline for the FP8 MLP input gradient and kernel tuning PR, run at
`ecbb586296d3dac36fd206211f25d63bad4a6b35`, on the same 8x H100 node and in the
same session as the PR arm.

- GPUs: 8x H100 SXM
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- Driver: `580.126.16`
- runs: `13`

## Statistics

| metric                |           value |
| --------------------- | --------------: |
| mean val loss         |    3.2784538462 |
| val loss sample std   |    0.0015532844 |
| one-sided p vs 3.28   |    0.0018601384 |
| mean train time       |     73.703538 s |
| train time sample std |      0.072523 s |
| median train time     |        73.707 s |
| train time range      | 73.590-73.834 s |

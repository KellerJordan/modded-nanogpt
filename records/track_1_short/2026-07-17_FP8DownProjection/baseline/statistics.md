# Baseline Statistics

## Summary

Baseline run for the FP8 MLP down-projection comparison.

- GPUs: 8x H100
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- runs: `13`

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2782461538 |
| val loss sample std | 0.0020164389 |
| one-sided p vs 3.28 | 0.0042976481 |
| mean train time | 80.728615 s |
| train time sample std | 0.065165 s |
| median train time | 80.720 s |
| train time range | 80.610-80.840 s |

Losses:

```python
[3.2817, 3.2796, 3.2778, 3.2802, 3.2775, 3.2767, 3.2758, 3.2821, 3.2780, 3.2768, 3.2769, 3.2775, 3.2766]
```

Times:

```python
[80.610, 80.710, 80.678, 80.720, 80.747, 80.693, 80.659, 80.694, 80.763, 80.795, 80.818, 80.840, 80.745]
```

Calculation snippet:

```python
import scipy.stats
import torch

losses = [3.2817, 3.2796, 3.2778, 3.2802, 3.2775, 3.2767, 3.2758, 3.2821, 3.2780, 3.2768, 3.2769, 3.2775, 3.2766]
times = [80.610, 80.710, 80.678, 80.720, 80.747, 80.693, 80.659, 80.694, 80.763, 80.795, 80.818, 80.840, 80.745]

print("p=%.10f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(losses)))
print("time:", torch.std_mean(torch.tensor(times)))
```

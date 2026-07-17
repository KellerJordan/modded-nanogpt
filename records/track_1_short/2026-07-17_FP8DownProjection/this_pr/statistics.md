# FP8 MLP Down Projection Statistics

## Summary

This PR runs the MLP down-projection forward path in FP8 while keeping the
backward path in BF16.

The up-projection/ReLU-squared Triton kernel also emits FP8 post-activation
values and partial activation amax values. The MLP down-projection weights are
quantized and transposed to FP8 with delayed scales, then the forward down
projection runs through `torch._scaled_mm`.

- GPUs: 8x H100
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- runs: `13`

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2783692308 |
| val loss sample std | 0.0014156180 |
| one-sided p vs 3.28 | 0.0006690604 |
| mean train time | 79.684923 s |
| train time sample std | 0.182767 s |
| median train time | 79.653 s |
| train time range | 79.532-80.231 s |

Losses:

```python
[3.2791, 3.2773, 3.2783, 3.2775, 3.2781, 3.2775, 3.2793, 3.2772, 3.2769, 3.2775, 3.2806, 3.2779, 3.2816]
```

Times:

```python
[79.536, 79.740, 79.814, 79.658, 79.593, 79.629, 80.231, 79.653, 79.641, 79.532, 79.539, 79.682, 79.656]
```

Calculation snippet:

```python
import scipy.stats
import torch

losses = [3.2791, 3.2773, 3.2783, 3.2775, 3.2781, 3.2775, 3.2793, 3.2772, 3.2769, 3.2775, 3.2806, 3.2779, 3.2816]
times = [79.536, 79.740, 79.814, 79.658, 79.593, 79.629, 80.231, 79.653, 79.641, 79.532, 79.539, 79.682, 79.656]

print("p=%.10f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(losses)))
print("time:", torch.std_mean(torch.tensor(times)))
```

## Vs Baseline

Baseline:

- mean val loss: `3.2782461538`
- one-sided p vs `3.28`: `0.0042976481`
- mean train time: `80.728615 s`

This PR:

- mean val loss: `3.2783692308`
- one-sided p vs `3.28`: `0.0006690604`
- mean train time: `79.684923 s`

Delta:

- train time: `-1.043692 s`
- relative train time: `-1.292841%`
- speedup: `1.01309774x`
- val loss: `+0.0001230769`

Statistical comparison:

- Welch p-value candidate loss vs baseline loss: `0.8587465`
- Welch t-statistic candidate time vs baseline time: `19.394`
- Welch p-value candidate time vs baseline time: `4.9e-12`
- slowest candidate run: `80.231 s`
- fastest baseline run: `80.610 s`

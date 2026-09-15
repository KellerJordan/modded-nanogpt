# Engram Host Table Statistics

- GPUs: 8x H100
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- runs: `12`

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2785083333 |
| val loss sample std | 0.0016423145 |
| one-sided p vs 3.28 | 0.0046513102 |
| mean train time | 67.868750 s |
| train time sample std | 0.122511 s |
| median train time | 67.852 s |
| train time range | 67.714-68.089 s |

Losses:

```python
[3.2766, 3.2787, 3.2776, 3.2785, 3.2796, 3.2782, 3.276, 3.2817, 3.2781, 3.2786, 3.2775, 3.281]
```

Times:

```python
[67.965, 68.062, 67.918, 67.827, 67.738, 67.815, 67.723, 67.846, 67.858, 68.089, 67.714, 67.87]
```

Calculation snippet:

```python
import scipy.stats
import torch

losses = [3.2766, 3.2787, 3.2776, 3.2785, 3.2796, 3.2782, 3.276, 3.2817, 3.2781, 3.2786, 3.2775, 3.281]
times = [67.965, 68.062, 67.918, 67.827, 67.738, 67.815, 67.723, 67.846, 67.858, 68.089, 67.714, 67.87]

print("p=%.10f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(losses)))
print("time:", torch.std_mean(torch.tensor(times)))
```

## Vs Baseline

Baseline (upstream master ecbb586, unmodified, 1,285 steps):

- mean val loss: `3.2776500000`
- one-sided p vs `3.28`: `0.0010204949`
- mean train time: `74.027333 s`

This PR (1,140 steps):

- mean val loss: `3.2785083333`
- one-sided p vs `3.28`: `0.0046513102`
- mean train time: `67.868750 s`

Delta:

- train time: `-6.158583 s`
- relative train time: `-8.319337%`
- speedup: `1.09074255x`
- val loss: `+0.0008583333`

Statistical comparison:

- Welch p-value candidate loss vs baseline loss: `0.2674088`
- Welch t-statistic candidate time vs baseline time: `-155.048`
- Welch p-value candidate time vs baseline time: `2.22e-27`
- slowest candidate run: `68.089 s`
- fastest baseline run: `73.944 s`

Both sets ran interleaved (baseline, candidate, baseline, ...) on the same machine in the same session,
each a plain `torchrun --standalone --nproc_per_node=8 train_gpt.py` in its own checkout.

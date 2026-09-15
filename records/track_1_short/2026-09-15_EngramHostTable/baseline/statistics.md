# Baseline Statistics

- GPUs: 8x H100
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- runs: `12`

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2776500000 |
| val loss sample std | 0.0020286583 |
| one-sided p vs 3.28 | 0.0010204949 |
| mean train time | 74.027333 s |
| train time sample std | 0.062639 s |
| median train time | 74.008 s |
| train time range | 73.944-74.109 s |

Losses:

```python
[3.2769, 3.2756, 3.2759, 3.2768, 3.2801, 3.282, 3.276, 3.2781, 3.2771, 3.2792, 3.2787, 3.2754]
```

Times:

```python
[73.944, 73.981, 74.109, 74.102, 74.101, 74.016, 73.98, 74.094, 74.001, 74.065, 73.98, 73.955]
```

Calculation snippet:

```python
import scipy.stats
import torch

losses = [3.2769, 3.2756, 3.2759, 3.2768, 3.2801, 3.282, 3.276, 3.2781, 3.2771, 3.2792, 3.2787, 3.2754]
times = [73.944, 73.981, 74.109, 74.102, 74.101, 74.016, 73.98, 74.094, 74.001, 74.065, 73.98, 73.955]

print("p=%.10f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
print("losses:", torch.std_mean(torch.tensor(losses)))
print("time:", torch.std_mean(torch.tensor(times)))
```

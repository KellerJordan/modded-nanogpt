# Learnable Attention Scale

This record, by [Franz Cesista](@leloykun), makes the attention scale learnable instead of just being fixed at $1/\sqrt{d}$.

Originally recommended by the [OG paper on QK-Normalization](https://arxiv.org/abs/2010.04245) and more recently by the [Cosmos team at NVidia](https://arxiv.org/abs/2501.03575v1).

---

The Jan 4 2025 record takes 205.5 secs on my 8xH100 machine (vs. the 204.4 secs as reported), but this new record takes 203.7 secs on my machine--at most a 2 sec improvement.

```python
acc = [3.2807, 3.279, 3.2801, 3.2784, 3.279, 3.2783, 3.2785, 3.2796, 3.2793, 3.2795]

import scipy.stats
print('p=%.4f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0061
```

NOTE: current record on PyTorch 2.7.0 nightly v.250110:
step:1390/1390 val_loss:3.2782 train_time:202898ms step_avg:147.03ms

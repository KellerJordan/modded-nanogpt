## New Record: FP8 lm_head, YaRN fix, LR tuning (-30s)

Three changes from the prior record (#18):

**1. Enable FP8 on lm_head**

The prior PR dropped FP8 on `lm_head` due to the effort required to tune the scaling coefficients.
We calibrated the coefficients for the medium track (`x_s=32/448`, `w_s=2.25/448`, `grad_s=1.5/448`)
and switched `CastedLinear` → `CastedLinearT` (transposed weight storage) to match the short track.
This saves ~4% per step by running the largest matmul in the model in FP8.

**2. Fix YaRN cap**

The prior PR stopped applying YaRN once the long attention window exceeded 13 blocks, citing uncertainty
about its behaviour at larger windows. Removing this cap and applying YaRN at every window-size transition
improves validation loss by ~0.001.

**3. Tune learning rate**

Reduced `muon_lr` from 0.015 → 0.012. The prior LR was selected conservatively while chasing an
attention bug; the lower value gives a small but consistent improvement.

## Timing and Validation

```
import scipy.stats
import torch

losses = [2.9174, 2.9182, 2.9183, 2.9179, 2.9177]
times = [1011.438, 1011.512, 1011.451, 1011.359, 1011.098]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 2.92, alternative="less").pvalue)
# p=0.0001

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0004), tensor(2.9179))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1624), tensor(1011.3716))
```

timing prior record: 1041.277s

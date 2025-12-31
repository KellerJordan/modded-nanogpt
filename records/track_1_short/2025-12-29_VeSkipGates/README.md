## New Record: Value Embed and Skip Gates (-35 steps, -1.3s)

Updates in PR:
* Add gates to value embeds, using same structure as attention gates (first 12 dims of residual stream)
* Add gate to skip connection, using same structure as smear gate
* Update w_s from 1.5/448 to 1.6/448
* Add wd_mul 5 to value embeds, wd_mul 150x to embed and lm_head.
* Drop 35 steps

When I tested re-adding the frozen scalars for 40 steps (that got removed last PR) I got worse loss, so I kept it off for now. May be worthwhile to re-add for just the first window transition, not sure. The biggest change here is the value_embed gates. Loss is around 3.2830 without them. The skip_gate is pretty marginal, but it only costs ~300ms to add and seems to slightly pay for itself. I haven't heavily ablated the weight decay parameters, nonzero chance one of those is actually detrimental. At some point we should update the notation of how weight decay is calculated to be lr/init_lr * lr instead of lr^2. lr^2 is making the wd_mul rather unintuitive. The w_s update is surprisingly important for controlling variance.

I also tested adding gates for the x0 contributions but saw no improvement and higher runtime.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.279, 3.2795, 3.2774, 3.2779, 3.2784, 3.2792, 3.28]
times = [114.508, 114.464, 114.373, 114.343, 114.22, 114.401, 114.369]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0061

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0009), tensor(3.2788))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0922), tensor(114.3826))
```

retiming prior record (faster machine this time): 115.686 [115.669, 115.717, 115.674]

If no changes, will merge at 115.1s to be consistent with 1.3s improvement.
## New Record: Logit Rescale (-40 steps, -2.9s)

Old code:
```
logits = self.lm_head(x)
logits = 30 * torch.sigmoid(logits / 7.5)
```

New code:
```
logits = self.lm_head(x)
logits = 23 * torch.sigmoid((logits+5) / 7.5)
```

Also adding some training refactoring. Happy to merge in anything that improves code clarity.
I would expect only 40*.062 = 2.5s from this record. The remaining 0.4s may be from a more thorough warmup.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2794, 3.2777, 3.2793, 3.2799, 3.2784, 3.277, 3.2778, 3.2803, 3.2793]
times = [116.405, 116.313, 116.483, 116.359, 116.521, 116.3, 116.536, 116.402, 116.36]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0058

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0011), tensor(3.2788))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0868), tensor(116.4088))
```

retiming prior record matches official time: 119.3: [119.432, 119.138, 119.329]
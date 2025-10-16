## New record 146.8s 09/30/25 (-0.6s): CustomBatching, only update Adam Params every other step

This submission reflects all recent WR changes up to [PR#134](https://github.com/KellerJordan/modded-nanogpt/pull/134).

The main contribution of this PR is to introduce the concept of variable batch size by parameter group, by having different gradient accumulation strategies for different param groups. I drop the batch size by 1/3, and then allow gradients to accumlate in Adam params over 2 steps before updating, whereras Muon params update every step.
Also, for a minor improvement, I add a cooldown to the momentum term in Muon for the last 50 steps.

* num_iterations: 1630->2380
* cooldown_frac: 0.5->0.4
* adam beta1: 0.8->0.7

The mean loss of 3.2765 is well below the 3.28 cutoff. There may be some hyperparameter tuning that can be performed, even as simple as decreasing the step count. I only tried 1 batch size, 1 cooldown frac, and only minimally looked at parameter learning rates or momentum terms. There is a consistent hiccup of 300ms around step 2385 that may be fixable. Overall I encourage hyperparameter tuning as that was left out of scope for this PR.

```
# only step Adam every other step
if step%2==0:
    optimizer2.step()
    optimizer2.zero_grad(set_to_none=True)
else:
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
```

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2745, 3.2747, 3.2771, 3.2794, 3.2767]
times = [146.933, 146.893, 146.943, 146.553, 146.739]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0086

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0020), tensor(3.2765))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1664), tensor(146.8122))
```

retiming prior record: 147.4: [147.451, 147.336, 147.508]

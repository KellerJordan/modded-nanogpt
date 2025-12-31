# Faster Muon step, corrected learning rates

This record improves the step time of Muon and halves the learning rate.

## Timing and Validation

This record improves the final training by 30 steps and decreases time per step by around 1%.

This PR:

```
import scipy.stats
import torch

losses = [3.2804, 3.2754, 3.2753, 3.2800, 3.2780, 3.2813, 3.2778, 3.2771, 3.2780, 3.2783]
times = [139.441, 139.780, 139.889, 139.464, 139.761, 139.411, 139.555, 139.570, 139.804, 139.847]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0083

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0020, mean=3.2782)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.1825, mean=139.6522)
```

Previous PR (timed on same machine):

```
import scipy.stats
import torch

times = [143.018, 142.641, 142.789, 143.072, 143.241]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.2375, mean=142.9522)
```

In total, this corresponds to a $3.30$ second decrease in training time. Roughly half of this is from the decreased number of steps, and the other half should be from the increasing of Muon efficiency. I expect that the timing improvements from the Muon vectorization will vary moderately by machine.

Thank you to Prime Intellect for sponsoring my research.

## Changes


### (1) LR Adjustment

I found that the Muon learning rate was ~twice as high as it should be, so I've decreased it to `0.03`. The lower LR may be in part due to the second order impacts of Normuon.

Following the theory that effective learning rate is proportional to `sqrt(output_dim)` I have increased `lr_mul` on the MLP up-projection to `2.0`. I have removed the logic that requires all parameters in the same group the share the same learning rate and weight decay.

### (2) Muon step update

#### Vectorization

I vectorized several loops inside the Muon `step`, which slightly decreases step time. I am guessing we can apply `torch.compile` to a subpart of `step` for further gains, as well. I moved the momentum buffers to being properties of groups, not of states, though this requires that we add a `reset()` (similar to `Yarn`).

#### Moved attention reshape

Moving the attention parameter reshape (from `[dim, 4 * dim] -> [4, dim, dim]`) to an earlier state ensures that Normuon gets applied columnwise to each parameter instead of rowwise. Empirical testing seems to indicate that Normuon is more effective on the output dim (columnwise) than the input dim (rowwise).


### (3) Corrections

As noted [here](https://github.com/KellerJordan/modded-nanogpt/pull/144#issuecomment-3463464783), the current logic for `get_lr` does not flatten out during the iteration extension. I've corrected this issue, as well as a similar issue in `get_ws`.

Additionally, I corrected a subtle bug where gradients were being summed in `grad_accum_steps` but averaged over ranks. In practice this is mostly irrelevant due to magnitude invariance, however it causes minor precision issues for $<8$ devices.

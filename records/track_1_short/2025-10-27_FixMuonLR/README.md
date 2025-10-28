# Faster Muon step, corrected learning rates

This record improves the step time of Muon and addresses some bugs in our current effective learning rate calculation. It incorporates the results from [PR#144](https://github.com/KellerJordan/modded-nanogpt/pull/144).

## Timing and Validation

This record improves the final training by 30 steps and decreases by step by around 1%.

This PR:

```
import scipy.stats
import torch

losses = [3.2766, 3.2794, 3.2770, 3.2776, 3.2760, 3.2802, 3.2757]
times = [138.986, 138.838, 138.877, 138.905, 138.916, 138.846, 138.937]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0041

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.001706, mean=3.277500)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.052171, mean=138.900711)
```

Previous PR (timed on same machine):

```
import scipy.stats
import torch

times = [142.379, 142.156, 141.391, 142.374, 142.316]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.419136, mean=142.123200)
```

In total, this corresponds to roughly a $3.22$ second decrease in training time. On a faster machine (the one used for official timing), this will probably be $\approx 3.17$ seconds.

Thank you to Prime Intellect for sponsoring my research.

## Changes


### (1) Vectorized Muon Step

I vectorized several loops inside the Muon `step`, which slightly decreases step time. I am guessing we can apply `torch.compile` to a subpart of `step` for further gains, as well. I moved the momentum buffers to being properties of groups, not of states, though this requires that we add a `reset()` (similar to `Yarn`).

### (2) Corrected learning rate

In the previous Muon step, the `eff_lr_val` was scaling the learning rate on the attention parameters by `1/2`, since they were treated as `[dim, 4 * dim]`-shaped parameters.  However, in practice, they are square parameters, so we should not do this. Moving the attention reshape in the step corrects this issue.

Similarly, the MLP up-projection is also scaled down. Following the theory that effective learning rate is proportional to $\sqrt{\text{output_dim}}$ I have increased `lr_mul` on the MLP up-projection to `2.0`. I have removed the logic that requires all parameters in the same group the share the same learning rate and weight decay.

Both of these changes meant that our previous Muon learning rate was ~twice as high as it should be, so I've decreased it to `0.03`. I have not further tuned this value.

### (3) LR refactoring + WS Schedule tweak

I removed the logic for iteration extension and instead changed `get_lr` to account for a "flat" section at the end. The hyperparameters for learning rate have been changed to instead to be fractional breakpoints, which helps in testing out lower step accounts. I believe that the LR schedule can be further improved.

Since the WS schedule was also impacted by the iteration extension, I updated the schedule from being 3 parts to 6 parts. This schedule is different than the previous three part schedule though it performs essentially the same as the version with iteration extension.

Additionally, I corrected a subtle bug where gradients were being summed in `grad_accum_steps` but averaged over ranks. In practice this is mostly irrelevant due to magnitude invariance, however it causes minor precision issues for $<8$ devices.

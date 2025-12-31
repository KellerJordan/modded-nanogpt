# Cautious Weight Decay

This record implements [Cautious Weight Decay](https://arxiv.org/abs/2510.12402).

## Timing and Validation

This record improves the final training 40 steps, with a slight increase in step time.

This PR:

```
import scipy.stats
import torch

losses = [3.2784, 3.2771, 3.2777, 3.2790, 3.2794, 3.2813, 3.2772, 3.2772, 3.2785, 3.2783]
times = [137.582, 137.753, 137.636, 137.507, 137.639, 137.708, 137.722, 137.677, 137.456, 137.705]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0018

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0013, mean=3.2784)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0970, mean=137.6385)
```

Previous PR (timed on same machine):

```
import scipy.stats
import torch

times = [139.813, 139.832, 139.877, 139.839, 139.939]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0499, mean=139.8600)
```

These timings show an improvement of ~2.22 seconds.

Thank you to Prime Intellect for sponsoring my research.

## "Cautious" weight decay

I found that weight decay leads to stable training dynamics, but performance seems to suffer. I stumbled upon the paper [Cautious Weight Decay](https://arxiv.org/pdf/2510.12402) which proposes only applying weight decay on the parameters that are growing in magnitude, and this proved to be effective.

Based on suggestion from @classiclaryd, I kept weight decay on a schedule. After trying various combinations, I found that the same schedule as learning rate is quite good, so I kept the previous calculation of `effective_weight_decay = learning_rate * weight_decay`. Scheduled weight decay improves performance on CWD by 10-15 steps.

The choice of `wd=1.2` is well tuned. In practice, it actually corresponds to starting `effective_weight_decay = 1.2 x 0.03 = 0.036`.

Cautious weight decay might be better called "masked decoupled weight decay". While it should be an unbiased estimator, I noticed that this weight decay has a very different training dynamic than the baseline:

<img src="./assets/validation_loss.jpg" alt="val-loss" width="600"/>


In particular, we find that CWD has higher validation loss for the majority of the training run. There is an inflection point when the learning rate decreases, and CWD only "catches up" to the baseline in the final steps of training. I noticed this dynamic irrespective of whether WD is placed on a schedule.

Parameters under CWD have mean square magnitude `<20%` of the magnitude under the baseline. I found this pattern consistently for both MLP and ATTN parameters.

I found that the condition number after CWD is virtually identical the the condition number after NorMuon:

<img src="./assets/cwd_condition_numbers.jpg" alt="cond-numbers" width="1200"/>

I believe this PR opens the door for rich future work, including tuning the WD schedule and CWD for Adam.

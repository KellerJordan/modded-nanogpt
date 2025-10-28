# Cautious Weight Decay

This record implements [Cautious Weight Decay](https://arxiv.org/abs/2510.12402). It builds on top of [PR#146](https://github.com/KellerJordan/modded-nanogpt/pull/146).

## Timing and Validation


This record improves the final training 15 steps, with increase in step time below noise levels.

This PR:

```
import scipy.stats
import torch

losses = [3.2768, 3.2747, 3.2774, 3.2793, 3.2764, 3.2774, 3.2769]
times = [138.118, 137.977, 137.925, 137.762, 137.816, 138.127, 138.326]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0006

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0014, mean=3.2770)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.1969, mean=138.0073)
```

Previous PR (timed on same machine):

```
import scipy.stats
import torch

times = [138.986, 138.838, 138.877, 138.905, 138.916, 138.846, 138.937]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.052171, mean=138.900711)
```

In total, this corresponds to roughly a $0.9$ second decrease in training time. On a faster machine (the one used for official timing), this will probably be $\approx 0.88$ seconds.

Thank you to Prime Intellect for sponsoring my research.

## "Cautious" weight decay

I found that weight decay leads to stable training dynamics, but performance seems to suffer. I stumbled upon the paper [Cautious Weight Decay](https://arxiv.org/pdf/2510.12402) which proposes only applying weight decay on the parameters that are growing in magnitude, and this proved to be effective. The choise of `wd=0.01` is somewhat tuned. It is certainly within the right order of magnitude.

Cautious weight decay might be better called "Masked weight decay", and can be implemented cheaply and in just one or two lines of code.

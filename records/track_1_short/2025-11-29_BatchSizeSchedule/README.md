This record implements a batch size schedule.

## Timing and Validation

This record has 65 fewer steps at slightly higher step time than the current record.

```
import scipy.stats
import torch

losses = [3.2772, 3.2772, 3.2761, 3.2788, 3.2795, 3.2808, 3.2783, 3.2795, 3.2767, 3.2794]
times = [132.754, 132.869, 132.706, 132.803, 132.627, 132.642, 132.683, 132.654, 132.913, 132.706]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0035

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0015, mean=3.2784)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0977, mean=132.7357)
```

Previous record (timed on same machine):

```
import scipy.stats
import torch

times = [134.778, 134.789, 134.823]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0234, mean=134.7967)
```

These timings show an improvement of $\approx 2$ seconds.

Thank you to Prime Intellect for sponsoring my research.

## Batch Size Schedule

Choosing a batch size involves a tradeoff:
1) Smaller batch sizes are more token-efficient. For a fixed token budget, above a certain point, increasing the batch size will worsen final model performance.
2) Training at higher batch sizes is faster for various reasons: GPUs parallelize over the batch dimension, DDP is easy, every step incurs overhead (optimizers, comms, etc.)

A critical batch size balances both of these considerations: low enough to be token-efficient, and high enough to be speed-efficient.

Since the critical batch size increases as the total token budget increases, it is safe to increase the batch size throughout the course of training. A batch size schedule allows us to benefit from the token-efficiency of low batch sizes when they are most critical.

This record introduced quite a few hyperparameters. I chose to follow the tripartite window size schedule in order to somewhat minimize the number of hyperparameters. Additionally, I constrained myself to keep the avg tokens per step the same (though there are a bit more in the PR due to the extension). Therefore, the main factors that needed tuning are
- the rate of inrease for the batch size
- the amount to increase the learning rate when we increase batch size

For the former, I went with a linear ramp. For the latter, instead of power-law scaling, I chose to multiply by $p$ since this is cheaper. I went with $p=0.8$, which is better than $p=1$ and $p=0.5$ (I did not check other values).

There are certainly much richer sweeps of hyperparameter sweep that could drive down the results in this record.

If you are interested in some experiments I conducted for this record, as well as surronding literature, please do read Section 2 of my recent blog post [Muon in Modded NanoGPT](https://varunneal.github.io/Essays/Muon-in-Modded-NanoGPT).

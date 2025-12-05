This record implements a batch size schedule.

## Timing and Validation

This record has 65 fewer steps at slightly higher average step time than the current record.

```
import scipy.stats
import torch

losses = [3.2772, 3.2781, 3.2780, 3.2784, 3.2773, 3.2806, 3.2799, 3.2765]
times = [132.259, 132.152, 132.194, 132.179, 132.249, 132.190, 132.217, 132.275]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0045

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0014, mean=3.2782)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0431, mean=132.2144)
```

Previous record (timed on same machine):

```
import scipy.stats
import torch

times = [134.069, 134.013, 134.044]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.00281, mean=134.042)
```

These timings show an improvement of $\approx 1.8$ seconds.

Thank you to Prime Intellect for sponsoring my research with GPU credits.

## Batch Size Schedule

Choosing a batch size involves a tradeoff:
1) Token efficiency. Smaller batch sizes are more token-efficient. For a fixed token budget, once the batch size is above some threshold, increasing it further tends to hurt final model performance. Beyond this point, the gradient signal from a single batch is saturated, so larger batches just waste tokens.
2) Token speed: Training at higher batch sizes is faster *per-token* than training with a lower batch size for many steps. This is because GPUs parallelize over the batch dimension, DDP is easy, and every step incurs overhead (optimizers, comms, etc).

A critical batch size balances both of these considerations: low enough to be token-efficient, but high enough to be speed-efficient.

Batch size can often be safely increased over the course of training. Conceptually, this may be because early training focuses on common patterns, so even small batches provide strong gradient signals. Later training involves learning rarer patterns. These sparse signals remain noisy even at high batch sizes, so the batch size can be increased without saturating the gradient. Notably, two strong models trained with Muon, Kimi K2 and GLM 4.5, both increase batch size mid-training.

Batch size schedules have been unsuccessfully tried in the speedrun for some time now. It is a multifaceted problem, requiring consideration of token budgets, optimal learning rates, and token efficiency.

In order to simplify the problem, I assumed that the total token budget should be approximately the same, but the batch size should be linearly ramped over three phrases of training (this is also for convenience -- to match the window size schedule). Within these constraints, I only needed to optimize (a) the slope of the linear ramp and (b) the learning rate schedule.

There are certainly much richer sweeps of hyperparameter sweep that could drive down the results in this record.

If you are interested in some experiments I conducted for this record, as well as surronding literature, please do read Section 2 of my recent blog post [Muon in Modded NanoGPT](https://varunneal.github.io/essays/muon).

Additionally, I have attached logs with validation every 5 steps in this repo for more fine-grained analysis of the batch size schedule. Here is a before/after plot:

![](val_loss_five_step.png)

## PyTorch version

The previous nightly (Torch 2.10, 09/26/2025) used for most records has expired. Unfortunately, both Torch 2.9.0 and Torch 2.9.1 seem to slow down the record. As such, I traveled significantly forward to nightly 12/04/2025, which seems to be very slightly faster than the prevously nightly.

# New record 09/29/25

This submission reflects all recent WR changes up to [PR#133](https://github.com/KellerJordan/modded-nanogpt/pull/133).

The main improvement in this PR is using the [Polar Express](https://arxiv.org/pdf/2505.16932)
sign method in Muon instead of Newton-Schulz. This paper was designed with reference to ModdedNanoGPT so it was very easy to implement, 
and I direct the reader to this paper directly for details. Using Polar Express, I've reduced the train steps by 10. 

The next change in this PR is packaging Flash Attention 3 via [Huggingface's Kernels](https://huggingface.co/docs/kernels/en/index).
This does not impact timing but should increase ease of development for anyone working on this project.

## Timing and Validation

This PR improves the final training by 10 steps, with no change in the time per step.

```
import scipy.stats
import torch

losses = [3.2789, 3.2792, 3.2796, 3.2776, 3.2797, 3.2787, 3.2792]
times = [148.617, 148.580, 148.569, 148.653, 148.578, 148.542, 148.587]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0045

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0007057, mean=3.2789857)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0358076, mean=148.5894318)
```

You may notice that this PR shows a 0.2 second mean *increase* in timing over the result in PR#133.
However, that PR was timed on very fast machine. To demonstrate that this PR accurately represents
a decrease in train time, I timed PR#133 on the same machine as above:

```
import scipy.stats
import torch

times = [149.714, 149.676, 149.659, 149.716, 149.649, 149.569, 149.521]

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0732, mean=149.6434)
```

Therefore, I believe that this PR represents at least a 1 second improvement. 

Thank you to Prime Intellect for compute credits, which made this PR possible. 

## Polar Express

All credit to the original authors (Noah Amsel, David Persson, Christopher Musco, Robert M. Gower) 
for discovery and implementation of this method. I adapted their code from https://github.com/NoahAmsel/PolarExpress/tree/main.

I found optimal parameters with
- `num_iters=5`: each iteration adds about a second to train time
- `muon_lr=0.06`: I found bumping the Muon LR seems to perform slightly better
- `safety_factor=1.02`: hyperparameter for Polar Express coefficients

Despite the paper explicitly referencing and showing improvements on Modded NanoGPT,
I was unable to replicate the level of success shown in this paper. However, it may 
be possible to further tune parameters to achieve a better result.
Additionally, like [Cesista 2025](https://leloykun.github.io/ponder/muon-opt-coeffs/) I believe it may be more promising on the GPT Medium track. 

## Flash Attention 3 Huggingface Kernel

A couple weeks ago, Flash Attention merged [ABI-stability](https://github.com/Dao-AILab/flash-attention/pull/1791)
into the main FA3 repo. This allows builds of Flash Attention on PyTorch nightlies after 08/30 to be compatible with each other. 
Since [PR#118](https://github.com/KellerJordan/modded-nanogpt/pull/118), we have been using 
[a variant](https://github.com/Dao-AILab/flash-attention/pull/1769) of FA3 by @Guilhermeleobas that is compatible with `torch.compile`.  
I have written a [fork](https://github.com/varunneal/flash-attention/tree/stable) that combines
these changes and uploaded its build to Huggingface at https://huggingface.co/varunneal/flash-attention-3.

I have modified training script to fetch these builds via Hugginface's `get_kernel`.
Therefore, it will no longer be needed for developers to manually build Flash Attention.

I have packaged this kernel for both CUDA 12.6 and 12.8 for the following PyTorch versions:
- `2.8.0`
- `2.9` nightlies after 8/30
- `2.10` nightlies

Note that the actual build `.so` is identical for all Torch Nightly versions. 

This most recent record uses the same `2.10` nightly as PR#133. 
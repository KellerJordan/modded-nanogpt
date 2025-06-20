# Optimization leaderboard for the NanoGPT-medium speedrun

The goal of this leaderboard is to collaboratively/competitively find good optimizers for use with the NanoGPT-medium speedrun record. Unlike the main speedrun which seeks to minimize wallclock, here we will only care about minimizing step count by modifying only the optimizer.

## Rules

- Each optimizer will have its own history of records. We are interested in finding the best setting for each optimizer, in order to help the community make a fair comparisons and find good new optimizers.
- The foundation of all runs in this leaderboard will be the NanoGPT-medium record from 04/22/25 which is held by @jadenj3o. New runs must not modify the main speedrun's architecture or data pipeline; in particular the batch size should stay the same.
- New runs can use any optimizer. They will be considered a new record for their respective optimizer if they attain a faster steps-to-target than the previous record.

## Best currently known hyperparameters for each optimizer

| Optimizer | Steps to 2.92 | Hparam summary | Log | Contributors |
| - | - | - | - | - |
| Muon w/ 2x MLP wd | 5960 | lr=.025, wd=.01, double wd for MLP | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj30 et al. |
| [Muon](https://kellerjordan.github.io/posts/muon/) | 6125 | lr=.025, wd=.01 | ? | @kellerjordan0 |
| [AdamW](https://arxiv.org/abs/1711.05101) | 9500 | lr=.0015, wd=.125, warmup_steps=500 | ? | @kellerjordan0 |
| [PSGD Kron](https://github.com/evanatyourservice/kron_torch) | 7875 | lr=.0005, wd=.625 | ? | @kellerjordan0 |
| [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ? | ? | ? |
| Sophia | ? | ? | ? | ? |
| Lion | ? | ? | ? | ? |
| ? | ? | ? | ? | ? |


## Record histories for each optimizer

### Muon w/ 2x MLP wd

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 5960 | lr=0.025, wd=0.01, double wd for MLP | 04/22/25 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj3o |

Comment: This is just the NanoGPT-medium speedrun record as of 06/15/25.

### [Muon](https://kellerjordan.github.io/posts/muon/)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 6125 | lr=0.025, wd=0.01 | 06/19/25 | ? | @kellerjordan0 |

Comment: We lose a ~165 steps by removing the custom weight decay settings from the speedrun.

### [AdamW](https://arxiv.org/abs/1711.05101)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 10500 | lr=0.0015, wd=0.125, warmup_steps=500, bf16 weights | 06/15/25 | ? | @kellerjordan0 |
| 2 | 9500 | lr=0.0015, wd=0.125, warmup_steps=500 | 06/19/25 | ? | @kellerjordan0 |

Precise steps to reproduce:
* #1: In the main `train_gpt_medium.py`, replace `optimizer2` with `AdamW(hidden_matrix_params, lr=0.0015, weight_decay=0.125, betas=(0.9, 0.95), eps=1e-10)`
and add a warmup using `if step < 500: return step / 500` in `get_lr()`.
* #2: In the `train_gpt_medium.py` contained in this folder, which enables fp32 weights for any optimizer, do the same thing as above.

### [PSGD Kron](https://github.com/evanatyourservice/kron_torch)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 9000 | lr=.0005, wd=.625, bf16 weights | 06/19/25 | ? | @kellerjordan0 |
| 2 | 9000 | lr=.0005, wd=.625 | 06/19/25 | ? | @kellerjordan0 |

Precise steps to reproduce:
* #1: Install and import `Kron`, then replace `optimizer2` with `Kron(hidden_matrix_params, lr=.0005, weight_decay=.625)` in the `train_gpt_medium.py` of the main folder. Note: Adding lr warmup does not seem to be needed.
* #2: Same thing, but instead on top of the `train_gpt_medium.py` that's contained in this folder instead of the main folder. That way we get fp32 weights.

## Response to a possible critique

Critique: "The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms." (this is quoted from a researcher on Twitter)

Response: Firstly, I'm simply skeptical about whether this is true. Muon was originally empirically determined for the CIFAR-10 speedrun, where it lowered the record from 3.09 to 2.59 seconds, and then it was transferred to NanoGPT. These two settings are about as different as you can get in deep learning research, which indicates that optimizers can't actually be dramatically "overfit" to architectures. Secondly, if this critique is true, then it should be possible to procure evidence proving it. The critic could describe an archiectural modification to the speedrun which causes the relative strength of a pair of optimizers to significantly change. I would like to see this evidence before taking the critique too seriously. Thirdly, even if this critique is true and that evidence can indeed be procured, then there's still not really anything we can do about it: The community would still need open-source leaderboards; it would just need more of them in order to cover additional architecture cases. So overall, I don't believe this critique significantly detracts from the value of a leaderboard like this.

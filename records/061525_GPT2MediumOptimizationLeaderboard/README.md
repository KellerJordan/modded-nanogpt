# NanoGPT-Medium speedrun optimization leaderboard

The goal of the NanoGPT-Medium speedrun is to minimize the amount of wallclock time required to train a model that attains 2.92 validation loss on FineWeb.
The competitive process of improving this speedrun has generated a high-quality baseline training algorithm.

In this sub-leaderboard we will use this trainer as a foundation to compare various optimizers.
In particular, we freeze the NanoGPT-Medium speedrun at its 04/22/25 record (held by @jadenj3o) and then collaboratively/competitively search for the best hyperparameters for each optimizer of interest.

The target remains the same as in the speedrun: 2.92 validation loss on the FineWeb validation set. However, here we will mainly focus on the steps required to reach that goal rather than the wallclock time.

## Table of best currently known hyperparameters

| Optimizer | Steps to 2.92 | Wallclock | Hparam summary | Log | Contributors |
| - | - | - | - | - | - |
| [Muon](https://kellerjordan.github.io/posts/muon/) | 5960 | 24.48 minutes | lr=0.025, wd=0.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj30 |
| [AdamW](https://arxiv.org/abs/1711.05101) | 10500 | 41.78 minutes | lr=0.0015, wd=0.125, warmup_steps=500 | ? | @kellerjordan0 |
| [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ? | ? | ? |
| PSGD Kron | ? | ? | ? | ? | ? |
| Sophia | ? | ? | ? | ? | ? |
| Lion | ? | ? | ? | ? | ? |
| ? | ? | ? | ? | ? | ? |


## Record histories for each optimizer

### [Muon](https://kellerjordan.github.io/posts/muon/)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 5960 | lr=0.025, wd=0.01 | 04/22/25 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj3o |

### [AdamW](https://arxiv.org/abs/1711.05101)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 10500 | lr=0.0015, wd=0.125, warmup_steps=500 | 06/15/25 | ? | @kellerjordan0 |

Precise details of #1: Replace `optimizer2` with `AdamW(hidden_matrix_params, lr=0.0015, weight_decay=0.125, betas=(0.9, 0.95), eps=1e-10)`
and add a warmup using `if step < 500: return step / 500` in `def get_lr()`.

### [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo)

# Optimization leaderboard for the NanoGPT-medium speedrun

The goal of this leaderboard is to collaboratively/competitively search for best optimizer on top of the NanoGPT-medium speedrun record. Unlike the main speedrun which seeks to minimize wallclock by any means necessary, here we will try to minimize the step count by modifying only the optimizer.

## Rules

- Each optimizer will have its own history of records. We are interested in finding the best setting for each optimizer, in order to help the community make a fair comparisons and find good new optimizers.
- Records are not allowed to modify the speedrun's architecture or data pipeline. In particular the batch size should stay the same as in the speedrun.
- Submissions can use any optimizer. They will be considered a new record for their respective optimizer if they attain a faster steps-to-target than the previous best.

## Best currently known hyperparameters for each optimizer

| Optimizer | Steps to 2.92 | Hparam summary | Log | Contributors |
| - | - | - | - | - |
| [Muon](https://kellerjordan.github.io/posts/muon/) | 5960 | lr=0.025, wd=0.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj30 |
| [AdamW](https://arxiv.org/abs/1711.05101) | 10500 | lr=0.0015, wd=0.125, warmup_steps=500 | ? | @kellerjordan0 |
| [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ? | ? | ? |
| PSGD Kron | ? | ? | ? | ? |
| Sophia | ? | ? | ? | ? |
| Lion | ? | ? | ? | ? |
| ? | ? | ? | ? | ? |


## Record histories for each optimizer

### [Muon](https://kellerjordan.github.io/posts/muon/)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 5960 | lr=0.025, wd=0.01 | 04/22/25 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @jadenj3o |

### [AdamW](https://arxiv.org/abs/1711.05101)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 10500 | lr=0.0015, wd=0.125, warmup_steps=500 | 06/15/25 | ? | @kellerjordan0 |

Precise steps to reproduce #1: Replace `optimizer2` with `AdamW(hidden_matrix_params, lr=0.0015, weight_decay=0.125, betas=(0.9, 0.95), eps=1e-10)`
and add a warmup using `if step < 500: return step / 500` in `get_lr()`.

### [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo)



## Response to some possible critiques

Critique: "The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms." (this is quoted from a researcher on Twitter)

Response: Firstly, I'm simply skeptical about whether this is true. Muon was originally empirically determined for the CIFAR-10 speedrun, where it lowered the record from 3.09 to 2.59 seconds, and then it was transferred to NanoGPT. These two settings are about as different as you can get in deep learning research, which indicates that optimizers can't actually be dramatically "overfit" to architectures. Secondly, if this critique is true, then it should be possible to procure evidence proving it. The critic could describe an archiectural modification to the speedrun which causes the relative strength of a pair of optimizers to significantly change. I would like to see this evidence before taking the critique too seriously. Thirdly, even if this critique is true and that evidence can indeed be procured, then there's still not really any solution to this critique. The community would still need open-source leaderboards; it would just need more of them to cover additional architecture cases. So overall, I don't believe this critique significantly detracts from the value of a leaderboard like this.

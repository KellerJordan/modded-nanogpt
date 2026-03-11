# NanoGPT-Small Optimizer Leaderboard

The goal of this leaderboard is to collaboratively|competitively find the optimizer for training to the NanoGPT-medium benchmark perplexity. Unlike the main speedrun which seeks to minimize wallclock, here we will only care about minimizing step count by improving the optimizer.

## Quickstart

You can run the current record via the following command.
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 55  # downloads 55B training tokens
torchrun --standalone --nproc_per_node=8 records/track_3_small_optimization/train_gpt_simple.py
```

## Rules

- Changing the dataset, batch size, or architecture is disallowed. We are only interested in improving the optimizer.
- Each optimizer will have its own history of records. We are interested in finding the best setting for each optimizer, in order to help the community make a fair comparisons and find good new optimizers.
- New runs can use any optimizer. They will be considered a new record for their respective optimizer if they attain a faster steps-to-target than the previous record.

## Overall optimizer leaderboard

| Optimizer | Steps to 3.28 | Hparam summary | Log | Contributors |
| - | - | - | - | - |
| [Muon](https://kellerjordan.github.io/posts/muon/) | 3800 | lr=.02, wd=.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @kellerjordan0 |
| [AdamW](https://arxiv.org/abs/1711.05101) | ? | ? | ? | ? |
| [PSGD Kron](https://github.com/evanatyourservice/kron_torch) | ? | ? | ? | ? |
| [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ? | ? | ? |
| Sophia | ? | ? | ? | ? |
| Lion | ? | ? | ? | ? |
| ? | ? | ? | ? | ? |


## Per-optimizer record histories

### [Muon](https://kellerjordan.github.io/posts/muon/)

| # | Steps to 3.28 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 3800 | lr=0.02, wd=0.01 | 2026/03/10 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | @kellerjordan0 |

### [AdamW](https://arxiv.org/abs/1711.05101)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | ? | ? | ? | ? | ? |

Note: The last time I tried `AdamW` in a similar setting, it was optimal to use `lr=.0015, wd=.125, warmup_tokens=250`.

### [PSGD Kron](https://github.com/evanatyourservice/kron_torch)

| # | Steps to 2.92 | Hparam summary | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | ? | ? | ? | ? | ? |

Note: The last time I tried `Kron` in a similar setting, it was optimal to use `lr=.0005, weight_decay=.625`.

## Discussion

One possible critique of a leaderboard like this is the following (quoted from a post on X):

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

And here's a reply:

First, it's simply not clear whether this is true. Muon was originally empirically determined for the CIFAR-10 speedrun, where it lowered the record from 3.09 to 2.59 seconds, and then it was transferred to NanoGPT, where it was found to "generalize." The fact that these two settings are about as different as you can get in deep learning research indicates that the process of searching for good optimizers doesn't actually tend to produce results that are "overfit" to the choice of architecture. Therefore, before taking this critique seriously, we should ask for evidence: The critic should describe an archiectural modification to the speedrun which causes the relative strength of a pair of optimizers to significantly change.

That being said, even if such evidence could be procured, there would still not really be anything we can do about it. In particular, the community would still need open-source leaderboards to get signal (the alternative would be blindly trusting the claims made by papers, which has never worked very well). It would just need more than one of them in order to cover a variety of architectural cases.

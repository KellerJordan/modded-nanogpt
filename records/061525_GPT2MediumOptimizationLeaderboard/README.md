# NanoGPT-Medium optimizer leaderboard

The goal of the NanoGPT-Medium speedrun is to minimize the amount of wallclock time required to train a model that attains 2.92 validation loss on FineWeb.
The competitive process of improving this speedrun has generated a high-quality training algorithm.

In this sub-leaderboard we will use this high-quality, easily-accessible, and fast-to-run training algorithm as the foundation to compare various optimizers.
We will freeze the NanoGPT-Medium speedrun at its current record (held by @jadenj3o as of 06/15/25), and then collaboratively/competitively search for the best hyperparameters for each optimizer of interest.

The target remains the same as in the speedrun: 2.92 validation loss on the FineWeb validation set. However, here we will mainly focus on the steps required to reach that goal rather than the wallclock time, in order not to create a level playing field to study purely the training speed of different optimizers.

# Table of best currently known hyperparameters

optimizer | hparam summary | log | steps to 2.92
--- | --- | --- | ---
[Muon](https://kellerjordan.github.io/posts/muon/) | lr=0.025, wd=0.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | 5960
[AdamW](https://arxiv.org/abs/1711.05101) | ? | ? | ?
[DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ?



Open money-bounties:
- Find hparams for AdamW that yield at least 85% of the speed of Muon ($XXXX)
- Find any optimizer with any hparams that yields 10% fewer steps-to-2.92 than Muon ($XXXX)


# Per-optimizer record history

## [Muon](https://kellerjordan.github.io/posts/muon/)

# | Steps to 2.92 | hparam summary | log | steps to 2.92
--- | --- | --- | --- | ---
1 | 5960 | lr=0.025, wd=0.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt)

## [AdamW](https://arxiv.org/abs/1711.05101)

## [DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo)



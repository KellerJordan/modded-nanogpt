# NanoGPT-Medium optimizer leaderboard

In which we collaboratively find the best hyperparameters for various optimizers.

For each optimizer, the goal is to find hyperparameters which minimize the number of steps required to attain the target of 2.92 validation loss.

The underlying training algorithm shall be fixed to the NanoGPT-Medium speedrun record from 04/22/25.

Optimizer | Best hparams | log | Steps to 2.92
--- | --- | --- | ---
[Muon](https://kellerjordan.github.io/posts/muon/) | lr=0.025, wd=0.01 | [log](075_640429f2-e726-4e83-aa27-684626239ffc.txt) | 5960
[AdamW](https://arxiv.org/abs/1711.05101) | ? | ? | ?
[DistributedShampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) | ? | ? | ?



Open bounties:
- Find hyperparameters for AdamW that yield at least 85% of the speed of Muon ($XXXX)
- Find any optimizer with any hparams that yields 10% fewer steps-to-2.92 than Muon ($XXXX)

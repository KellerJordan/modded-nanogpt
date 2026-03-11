# NanoGPT-Small Optimizer Leaderboard

The goal of this leaderboard is to collaboratively|competitively find the best optimizer for training NanoGPT-small models to 3.28 val loss.
Unlike the main speedrun which seeks to minimize wallclock time, here we will only care about minimizing step count by improving the optimizer.

We have initialized this leaderboard with a highly simplified variant of the speedrun, which should make experimentation convenient.
Compared to the main speedrun, this variant removes non-standard parameters (value embeddings, skip connection lambdas) and all triton kernels.
We have also switched from the sophisticated local-global pattern of attention used in the speedrun to simple 1024-context causal attention.

## Quickstart

You can run the current record via the following command.
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 40  # downloads 4B training tokens
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_small_optimization/train_gpt_simple.py
```

## Record history

| # | Steps to 3.28 | Description | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 3800 | Muon baseline, lr=.02 wd=.01 | 2026/03/10 | [log](58ec69a8-ebfa-447c-94b6-88855e0139d1.txt) | @kellerjordan0 |


## Rules

To be considered a valid attempt at a new optimization record, an attempt must:
1. Not change the dataset, batch size, or architecture.
2. Attain 3.28 val loss.

To reduce the number of steps needed to reach 3.28, record attempts are free to change the optimizer (algorithm or hyperparameters) as well as model initialization.


## Discussion

One possible critique of a leaderboard like this is the following, quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

And here's a reply:

A natural response is that this claim simply lacks evidence. For example, Muon was originally found empirically in the CIFAR-10 speedrun setting, where it lowered the record from 3.09 to 2.59 seconds.
It was then transferred to NanoGPT, where it also worked well. These two settings are about as different as one can reasonably find within deep learning research, which suggests that the process of searching for good optimizers does not, in practice, usually produce results that are merely overfit to a particular architecture.

So before taking this critique too seriously, we should ask for evidence. A critic should be able to point to an architectural change in the baseline that substantially alters the relative efficiency of two optimizers. Without that, the objection remains speculative.

That said, even if such evidence could be produced, the practical need for leaderboards would still remain.
For the public community, the only alternative to open-source leaderboards is to place naive trust in the claims made by papers, which has never worked out particularly well.
Therefore, at most such evidence would only imply that we need more than one leaderboard, so that multiple architectural regimes can be represented.


## Notes

For future attempts:
* For [AdamW](https://arxiv.org/abs/1711.05101), it seems that reasonable starting hparams are `lr=.0015, wd=.125, warmup_tokens=250`.
* For [PSGD Kron](https://github.com/evanatyourservice/kron_torch), it seems that reasonable starting hparams are `lr=.0005, weight_decay=.625`.

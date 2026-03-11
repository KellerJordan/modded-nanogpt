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
| 1 | 3800 | Muon baseline, lr=.02 wd=.01 | 2026/03/10 | [log](a34ecb11-38b1-463d-8dc1-c692e4dd233d.txt) | @kellerjordan0 |


## Rules

To be considered a valid, a new record attempt must:
1. Not change the dataset, batch size, or architecture.
2. Attain 3.28 val loss.

To reduce the number of steps needed to reach 3.28, record attempts are free to change the optimizer (algorithm or hyperparameters) as well as the model initialization.


## Discussion

### There is a need for some way to filter signal from noise

There are between [hundreds](https://chatgpt.com/c/69b10bd7-f92c-8325-b516-d999b5b2b409) and [thousands](https://claude.ai/share/fb9590de-c4b7-44f8-bfbb-7f80af30d3f9) of neural network optimizer papers floating around the internet.
The typical such paper claims to improve upon standard practice by a wide margin:
Prototypically, the [Sophia](https://arxiv.org/abs/2305.14342) paper claims a 2x speedup over Adam.
With few compelling meta-analyses out there, anyone interested in conducting neural network optimization research must technically
go through and replicate every one of these papers before being able to claim to be truly caught up with the state of the art.
Of course, in practice this is impossible. So instead, the de facto arrangement is that each researcher relies on their network of (human) connections
in order to become informed of what works and doesn't work.
* Industry reserachers in the big corporate labs benefit from being part of a gigantic pool of other human industry researchers, who are mostly not incentivized to fake results (because if a fake result messes up the bigrun, that's not good).
Such researchers therefore typically have a good picture of what really works and what doesn't.
* Academic researchers in prestigious labs are typically well-connected to sources of information, both in terms of learning from other well-connected academics, and sometimes
even finding out through their connections about what's going on inside the ostensibly-closed corporate labs. These elite academics therefore have a decent idea of what really works.
* All the remaining independent researchers and academics at non-prestigous labs are left out to dry.
Currently, their best source of information is the open-source research published by the Chinese industry.

My understanding is that the majority of all humans who are working on AI research fall into this last category.
Certainly, the majority of efforts to develop new optimizers are conducted by non-elite non-corporate academics.
It is therefore unfortunate that they are not being well-served - to use left terminology - by the current structure of
research. Fortunately, there is a relatively easy solution:
If new optimizer papers line themselves up on a competitive leaderboard/benchmark,
then effective new methods will become readily identifiable.

The reasons this hasn't happened already are that (a) creating a benchmark with enough momentum to have a chance at being adopted requires a relatively high level of preexisting clout to pull off, but
(b) generates little excess clout even if successful (which is a problem because clouty individuals are typically looking for ways to get more), and also separately (c)
faces significant friction, because most authors are of course incentivized to avoid such a leaderboard in favor of keeping their own ad-hoc idiosyncratic baselines,
in order to enable continued manipulation/exaggeration of results (intentional or unintentional).



### Response to a potential critique

One possible critique of a leaderboard like this is the following, quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

Here's a reply:

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

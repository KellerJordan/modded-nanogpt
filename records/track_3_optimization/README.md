# NanoGPT-Small Optimization Benchmark

The goal of this benchmark is to collaboratively|competitively find strong optimizers for training small transformers.
Unlike the main NanoGPT speedrun which seeks to minimize *wallclock time* by any means, here we will restrict our aim to minimizing *step count* by improving the optimization algorithm.

Most neural network optimizer research occurs in the public research community, not in the frontier labs.
This benchmark aims to provide a way for the community to filter signal from noise, thereby reducing the burden on individual researchers to test everything themselves
before being caught up to the SOTA.

The architecture for this benchmark is fixed to a simplified variant of the speedrun, which should make experimentation accessible and convenient.
Compared to the main speedrun, the setup used here removes non-standard parameters (value embeddings, skip connection lambdas) and all triton kernels.
We have also switched from the sophisticated local-global pattern of attention used in the speedrun to simple causal attention across contexts of 1024 tokens.

## Quickstart

The baseline can be run using the following command on any {1,2,4,8}x-{A,H}100 machine:
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 40  # downloads 4B training tokens
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/train_gpt_simple.py
```

## Results history

| # | Steps to 3.28 | Description | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 3800 | Muon baseline, lr=.02 wd=.01 | 2026/03/10 | [log](a34ecb11-38b1-463d-8dc1-c692e4dd233d.txt) | @kellerjordan0 |


## Rules

To be considered valid, new results must:
1. Keep the same dataset, batch size, and architecture as the baseline.
2. Not perform multiple forward-backward passes per step. Each step must correspond to a single forward-backward.
3. Attain 3.28 val loss, thereby matching [Andrej Karpathy's GPT-2 replication](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

New results have the freedom to modify:
1. The optimization algorithm, even to something slow in terms of wallclock speed.
2. The optimizer hyperparameters, including schedules.
3. The model initialization.


## Motivation

> [benchmark competitions are the prime mover of AI progress.](https://www.argmin.net/p/too-much-information#:~:text=benchmark%20competitions%20are%20the%20prime%20mover%20of%20AI%20progress.)
> -- Prof. Ben Recht

Since the release of [Muon](https://kellerjordan.github.io/posts/muon/), there have been [40+ papers published citing it that propose a new optimizer of their own](
https://chatgpt.com/share/69ed22e3-0870-83ea-a449-b4ce97d764f3). More broadly, there exist somewhere between [hundreds](https://chatgpt.com/c/69b10bd7-f92c-8325-b516-d999b5b2b409) and [thousands](https://claude.ai/share/fb9590de-c4b7-44f8-bfbb-7f80af30d3f9) of papers on neural network optimization across the internet.

How do these hundreds of optimizers compare -- which ones are able to optimize neural networks in the fewest steps?
The reality is that as a community, we simply don't know. Why not?
Because typically, these papers all use their own unique experimental setups, making it challenging to verify whether their baselines are well-tuned or to make comparisons between papers.

For researchers interested in neural network optimization, this is daunting -- a sea of methods, many of them claiming to be SOTA, and no shared infrastructure to sort signal from noise.


## Addressing a potential critique

Quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

Two replies:

1. This claim lacks evidence. For example, Muon was originally determined empirically for the CIFAR-10 speedrun setting, where it lowered the record from 3.09 to 2.59 seconds.
It was then transferred to NanoGPT, where it continued to work well. These two settings are about as different as one can reasonably find within deep learning research. This suggests that when a properly-tuned baseline is used, the process of searching for good optimizers does not tend to produce methods that are overfit to any particular experimental setup.
2. That being said, even in the world where the best optimizer *does* depend heavily on the choice of experimental setup, the practical need for leaderboards to filter signal from noise would still remain. We would just need to set up more than one leaderboard, in order to effectively cover the space of experimental setups.


## Notes

For future attempts:
* For [AdamW](https://arxiv.org/abs/1711.05101), it seems that reasonable starting hparams are `lr=.0015, wd=.125, warmup_tokens=250`.
* For [PSGD Kron](https://github.com/evanatyourservice/kron_torch), it seems that reasonable starting hparams are `lr=.0005, weight_decay=.625`.

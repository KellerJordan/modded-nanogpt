# NanoGPT-Small Optimization Leaderboard

The goal of this leaderboard/benchmark is to collaboratively|competitively find the strongest optimizer for training small transformers.
Unlike the main speedrun which seeks to minimize wallclock time by any means, here we will only care about minimizing step count, and we will refrain from modifying the transformer architecture.

Most optimizer research occurs in academia and open-source, not in the frontier labs.
This leaderboard aims to help filter signal from noise, reducing the burden on academic labs and independent researchers to test everything themselves
before being caught up to the SOTA.

The architecture for this leaderboard is fixed to a simplified variant of the speedrun, which should make experimentation accessible and convenient.
Compared to the main speedrun, the setup used here removes non-standard parameters (value embeddings, skip connection lambdas) and all triton kernels.
We have also switched from the sophisticated local-global pattern of attention used in the speedrun to simple causal attention across contexts of 1024 tokens.

## Quickstart

The current record can be run using the following command:
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

To be considered valid, new record attempts must:
1. Keep the same dataset, batch size, and architecture as the baseline.
2. Not perform multiple forward-backward passes per step. Each step must correspond to a single forward-backward.
3. Attain 3.28 val loss, thereby matching the performance of [Andrej Karpathy's GPT-2 replication](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

So, the remaining space of competitive freedom is as follows:
* The optimization algorithm can be modified arbitrarily, even to something that is slow in terms of wallclock speed.
* Optimizer hyperparameters can be tuned
* The model initialization can be changed


## Motivation

> [benchmark competitions are the prime mover of AI progress](https://www.argmin.net/p/too-much-information#:~:text=benchmark%20competitions%20are%20the%20prime%20mover%20of%20AI%20progress.)
> -- Prof. Ben Recht

Since the release of [Muon](https://kellerjordan.github.io/posts/muon/), there have been [40+ citing papers published that propose a new optimizer of their own](
https://chatgpt.com/share/69ed22e3-0870-83ea-a449-b4ce97d764f3). More broadly, there exist somewhere between [hundreds](https://chatgpt.com/c/69b10bd7-f92c-8325-b516-d999b5b2b409) and [thousands](https://claude.ai/share/fb9590de-c4b7-44f8-bfbb-7f80af30d3f9) of papers on neural network optimization across the internet.

How do these hundreds of optimizers compare -- which ones are able to optimize neural networks in the fewest steps?
The reality is that as a community, we simply don't know. Why not?
Because typically, each of these papers uses its own unique experimental setup, making it difficult to verify whether their baseline is well-tuned, and impossible to compare between papers.

For researchers interested in neural network optimization, this is daunting -- a sea of methods, many of them claiming to be SOTA, and no shared infrastructure to sort signal from noise.


How do the hundreds of optimizers which have been proposed compare -- which ones optimize can neural networks in the fewest steps?
The reality is that as a community, we simply don't know, even at small scale where such knowledge is in theory cheaply accessible.





### Response to a potential critique

One possible critique of a leaderboard like this is the following, quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

Two replies:

1) This claim lacks evidence. For example, Muon was originally found empirically in the CIFAR-10 speedrun setting, where it lowered the record from 3.09 to 2.59 seconds.
It was then transferred to NanoGPT, where it also worked well. These two settings are about as different as one can reasonably find within deep learning research, which suggests that when a properly-tuned baseline is used, the process of searching for good optimizers does not actually tend to produce results that are overfit to any particular experimental setup.

2) Even in the world where the best optimizer does depend heavily on the choice of experimental setup, the practical need for leaderboards to filter signal from noise would still remain.
What would be different is just that we would need more than one leaderboard, in order to cover the space of experimental setups.


## Notes

For future attempts:
* For [AdamW](https://arxiv.org/abs/1711.05101), it seems that reasonable starting hparams are `lr=.0015, wd=.125, warmup_tokens=250`.
* For [PSGD Kron](https://github.com/evanatyourservice/kron_torch), it seems that reasonable starting hparams are `lr=.0005, weight_decay=.625`.

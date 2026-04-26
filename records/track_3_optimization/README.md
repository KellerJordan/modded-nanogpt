# Modded-NanoGPT Optimizer Benchmark

The goal of this benchmark is to collaboratively|competitively find efficient neural network optimizers.
Unlike the main NanoGPT speedrun which seeks to minimize *wallclock time* by any means, here we aim to minimize *step count* by improving the optimization algorithm (⇒ methods that are slow in terms of wallclock are perfectly OK).

## Quickstart

The baseline can be run using the following command on any {1,2,4,8}x-{A,H}100 machine:
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install torch==2.10 huggingface_hub
python data/cached_fineweb10B.py 40  # downloads 4B training tokens
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/train_gpt_simple.py
```

## Results history

| # | Steps to 3.28 | Description | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | 3600 | [Muon](https://kellerjordan.github.io/posts/muon/) baseline, lr=.02 wd=.01 | 2026/04/26 | [log](03eada03-f26a-42ae-bc7c-795b449472c4.txt) | @kellerjordan0 |


## Rules

To be considered valid, new results must:
1. Keep the same dataset, batch size, and architecture as the baseline.
2. Not perform multiple forward-backward passes per step. Each step must correspond to a single forward-backward.
3. Attain 3.28 val loss, thereby matching [Andrej Karpathy's GPT-2 replication](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

New results have the freedom to modify:
1. The optimization algorithm, even to something slow in terms of wallclock speed.
2. The optimizer hyperparameters, including schedules.
3. The model initialization.

We welcome not only new results which advance the global SOTA, but even just results which advance the per-optimizer SOTA,
e.g., better hyperparameters for AdamW (even if it still isn't beating the baseline).


------

## Motivation

> [benchmark competitions are the prime mover of AI progress.](https://www.argmin.net/p/too-much-information#:~:text=benchmark%20competitions%20are%20the%20prime%20mover%20of%20AI%20progress.)
> -- Prof. Ben Recht

Most research into novel neural network optimizers occurs in the public research community, not in the frontier labs.
For example, since the release of Muon, there have been [40+ papers published citing it that propose a new optimizer of their own](
https://chatgpt.com/share/69ed22e3-0870-83ea-a449-b4ce97d764f3). And more broadly, there exist somewhere between [hundreds](https://chatgpt.com/c/69b10bd7-f92c-8325-b516-d999b5b2b409) and [thousands](https://claude.ai/share/fb9590de-c4b7-44f8-bfbb-7f80af30d3f9) of papers on neural network optimization across the internet.

How do these hundreds of optimizers compare - which ones are able to optimize neural networks in the fewest steps?
The reality is that as a community, we simply don't know. Why not?
Because typically, these papers all use their own unique experimental setups, making it challenging to verify whether their baselines are well-tuned or to make comparisons between papers.

For researchers interested in neural network optimization, this is daunting - a sea of methods, many of them claiming to be SOTA, and no shared infrastructure to sort signal from noise. As it stands, the burden is on the individual researcher to make sense of this madness. Calculating the outcome: If N different researchers publish N optimizer papers claiming SOTA, all of them unverifiable and mutually incomparable, then there are only two possibilities: Either (a) research grinds to a halt due to the Θ(N) growth in experiments that each researcher needs to conduct to get a private sense of the real SOTA, or (b) researchers start simply ignoring each other's papers.

This benchmark aims to provide a simple, easily-accessible way communally shared way to filter signal from noise, thereby surfacing ignored papers and reducing the number of experiments that each researcher must do in order to get an accurate picture of the SOTA.
Prior optimization benchmarks already exist, but often suffer from high barriers to entry due to strenuous requirements or high complexity.
This benchmark aims for maximum convenience in order to make new results as convenient/accessible as possible:
The baseline code should be comprehensible with minimal effort, and experiments should not take more than ~10 minutes or cost more than ~$5.


## Addressing a potential critique

Quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

Two replies:

1. Muon was originally determined empirically for the CIFAR-10 speedrun setting, where it lowered the record from 3.09 to 2.59 seconds.
It was then transferred to NanoGPT, where it continued to work well. These two settings are about as different as one can reasonably find within deep learning research. This anecdote suggests that when a properly-tuned baseline is used, the process of searching for good optimizers does not tend to produce methods that are overfit to any particular experimental setup.
2. That being said, even in the world where the best optimizer *does* depend heavily on the choice of experimental setup, the practical need for benchmarks to filter signal from noise would still remain. We would just need to set up more than one benchmark, in order to effectively cover the space of experimental setups (e.g., a more developed benchmark suite would likely cover multiple batch sizes and multiple scales). 


## Details on relation to the main speedrun

Aiming towards simplicity, for this benchmark we have removed the non-standard neural network parameters (value embeddings, skip connection lambdas) and triton kernels that are used in the main speedrun. We have also added back standard parameters which are wallclock-inefficient at small scale, namely the RMSNorm gains and Linear layer biases.
Finally, we have replaced the sophisticated local-global pattern of attention by simple causal attention across contexts of 1024 tokens.


## Technical notes and tips

For future attempts:
* For [AdamW](https://arxiv.org/abs/1711.05101), it seems that reasonable starting hparams are `lr=.0015, wd=.125, warmup_steps=250`.
* For [PSGD Kron](https://github.com/evanatyourservice/kron_torch), it seems that reasonable starting hparams are `lr=.0005, weight_decay=.625`.

On data: The baseline trains for 3550 * 524288 = ~2B tokens. The quickstart script downloads 4B tokens of FineWeb, allowing trainings up to 7600 steps. If you'd like to train for more steps than that, then you must get more tokens via something like `python data/cached_fineweb10B.py 100`, which will download the maximum 10B tokens.


## Logfile guidelines

Results should be submitted in the form of logfiles, like the ones linked in the [results history](#results-history) section above. Logfiles must include the full code used by the run, such that if we replace `train_gpt_simple.py` by the code, then running the quickstart will reproduce the run (up to random seed variance). In particular, hardcoded hyperparameters are to be preferred as compared to command line arguments.

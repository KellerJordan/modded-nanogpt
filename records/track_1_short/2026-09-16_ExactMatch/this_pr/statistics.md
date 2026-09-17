# Exact-Match Retrieval Statistics

## Summary

688 total steps; input/middle/output retrieval; minimum context 8, maximum 512; four recent training occurrences and top-2 distinct validation continuations. Seeds 0–4, in run order. Full-corpus cache construction, lookup, release, merge, and final cache wait are timed.

- Source commit: `65fb235a2e755b23ccd8edb26581511f57c88721`
- Hardware: Nebius fast instance `computeinstance-e00ycvar9qaqjdcyap`, 8× NVIDIA H100 80GB HBM3
- Host CPU: Intel Xeon Platinum 8468, 128 logical CPUs, 2 NUMA nodes
- Host RAM: approximately 1.5 TiB
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- Runs: `5`
- Environment: `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=4`, no CPU affinity override
- Data: unchanged FineWeb10B token shards; 10,485,760 validation tokens
- Run order: baseline 1, candidate 1, baseline 2, candidate 2, baseline 3–7, candidate 3–5; no concurrent GPU jobs
- Timing: the trainer's final `train_time`, excluding compilation/warmup and model validation

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2655600000 |
| val loss sample std | 0.0010039920 |
| one-sided p vs 3.28 | 2.786374027e-06 |
| mean train time | 47.205600 s |
| train time sample std | 2.501255 s |
| median train time | 48.128 s |
| train time range | 43.544–49.498 s |

The candidate meets the `p < 0.01` loss criterion. All five original runs are included; none were replaced or excluded.

| run | steps | val loss | train time (s) |
| --- | ---: | ---: | ---: |
| [336568bf-44af-4e04-9930-430e47c6f867.txt](336568bf-44af-4e04-9930-430e47c6f867.txt) | 688 | 3.2650 | 43.544 |
| [a1f643d5-e040-4bd6-87f3-641cdc50edd3.txt](a1f643d5-e040-4bd6-87f3-641cdc50edd3.txt) | 688 | 3.2670 | 48.128 |
| [1c47a501-8245-4017-8bb8-e54c51289366.txt](1c47a501-8245-4017-8bb8-e54c51289366.txt) | 688 | 3.2650 | 45.784 |
| [03b8fa8c-7a5b-47d3-ba4a-e3f37f7cb88b.txt](03b8fa8c-7a5b-47d3-ba4a-e3f37f7cb88b.txt) | 688 | 3.2662 | 49.074 |
| [7e52eb6c-82a2-4efc-a263-686198878657.txt](7e52eb6c-82a2-4efc-a263-686198878657.txt) | 688 | 3.2646 | 49.498 |

Losses:

```python
[3.265, 3.267, 3.265, 3.2662, 3.2646]
```

Times (seconds):

```python
[43.544, 48.128, 45.784, 49.074, 49.498]
```

Calculation snippet:

```python
import statistics
import scipy.stats

losses = [3.265, 3.267, 3.265, 3.2662, 3.2646]
times = [43.544, 48.128, 45.784, 49.074, 49.498]
print("p=%.10g" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
print("loss mean/std:", statistics.mean(losses), statistics.stdev(losses))
print("time mean/std:", statistics.mean(times), statistics.stdev(times))
```

Statistics use the losses and times at the precision emitted by the trainer.

## Reproduction

Check out the source commit listed above and run from the repository root
in the environment above, with `DATA_PATH` pointing to the root containing
`data/fineweb10B`. The run logs embed the training source. Both variants use
the same machine and software.

```bash
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=4 DATA_PATH=/path/to/dataset/root torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For the candidate, first install `./exact_match` using a Rust toolchain and add
`SEED=0` through `SEED=4` to the five commands. Leave `TOTAL_TRAIN_STEPS` unset.
The candidate requires all 103 training shards. Baseline runs use no seed override.


## Vs Baseline

| metric | baseline (7 runs) | this PR (5 runs) |
| --- | ---: | ---: |
| mean val loss | 3.2792428571 | 3.2655600000 |
| mean train time | 74.186143 s | 47.205600 s |
| one-sided p vs 3.28 | 0.3180813059 | 2.786374027e-06 |

- Train time delta (candidate − baseline): `-26.980543 s`
- Relative train time: `-36.368710%`
- Speedup: `1.57155386×`
- Val loss delta (candidate − baseline): `-0.0136828571`
- Two-sided Welch p-value, candidate vs baseline loss: `5.533172521e-05`
- Welch t-statistic, candidate vs baseline time: `-24.1152685`
- Two-sided Welch p-value, candidate vs baseline time: `1.742343344e-05`
- Slowest candidate run: `49.498 s`
- Fastest baseline run: `74.086 s`

# Baseline Statistics

## Summary

Unmodified `master` baseline, 1,285 total steps; independent default random initializations.

- Source commit: `ecbb586296d3dac36fd206211f25d63bad4a6b35`
- Hardware: Nebius fast instance `computeinstance-e00ycvar9qaqjdcyap`, 8× NVIDIA H100 80GB HBM3
- Host CPU: Intel Xeon Platinum 8468, 128 logical CPUs, 2 NUMA nodes
- Host RAM: approximately 1.5 TiB
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- CUDA: `12.8`
- Runs: `7`
- Environment: `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=4`, no CPU affinity override
- Data: unchanged FineWeb10B token shards; 10,485,760 validation tokens
- Run order: baseline 1, candidate 1, baseline 2, candidate 2, baseline 3–7, candidate 3–5; no concurrent GPU jobs
- Timing: the trainer's final `train_time`, excluding compilation/warmup and model validation

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2792428571 |
| val loss sample std | 0.0040219635 |
| one-sided p vs 3.28 | 0.3180813059 |
| mean train time | 74.186143 s |
| train time sample std | 0.058792 s |
| median train time | 74.197 s |
| train time range | 74.086–74.270 s |

This seven-run baseline sample does not meet the `p < 0.01` loss criterion.

| run | steps | val loss | train time (s) |
| --- | ---: | ---: | ---: |
| [903c7ebb-a523-4c67-bf41-ff7732eedd8e.txt](903c7ebb-a523-4c67-bf41-ff7732eedd8e.txt) | 1285 | 3.2754 | 74.086 |
| [db1a09d0-b3aa-4d8a-a3b3-e6c6d05d95b8.txt](db1a09d0-b3aa-4d8a-a3b3-e6c6d05d95b8.txt) | 1285 | 3.2809 | 74.217 |
| [d5a21beb-8e38-46bc-8597-3b7417c49aaf.txt](d5a21beb-8e38-46bc-8597-3b7417c49aaf.txt) | 1285 | 3.2875 | 74.152 |
| [62aeea1b-e591-4489-a187-4c2d7b1a414f.txt](62aeea1b-e591-4489-a187-4c2d7b1a414f.txt) | 1285 | 3.2786 | 74.270 |
| [f35d9565-6919-41b8-8101-5273f0034049.txt](f35d9565-6919-41b8-8101-5273f0034049.txt) | 1285 | 3.2767 | 74.197 |
| [2584d039-9077-4493-87a2-04aeccbd89b2.txt](2584d039-9077-4493-87a2-04aeccbd89b2.txt) | 1285 | 3.2774 | 74.164 |
| [b41f15c4-eecd-4923-8600-eaa652be3ef0.txt](b41f15c4-eecd-4923-8600-eaa652be3ef0.txt) | 1285 | 3.2782 | 74.217 |

Losses:

```python
[3.2754, 3.2809, 3.2875, 3.2786, 3.2767, 3.2774, 3.2782]
```

Times (seconds):

```python
[74.086, 74.217, 74.152, 74.27, 74.197, 74.164, 74.217]
```

Calculation snippet:

```python
import statistics
import scipy.stats

losses = [3.2754, 3.2809, 3.2875, 3.2786, 3.2767, 3.2774, 3.2782]
times = [74.086, 74.217, 74.152, 74.27, 74.197, 74.164, 74.217]
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

Install the dependencies from `requirements.txt`. Run the command seven times
with no seed override; each run uses the baseline’s default random initialization.


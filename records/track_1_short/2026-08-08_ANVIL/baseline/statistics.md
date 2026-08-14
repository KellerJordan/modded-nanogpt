# Baseline (current record) Statistics

The current `train_gpt.py` record was re-run unseeded in the same session, on
two of the four machines used for this PR's certification pool, fresh compile caches.
These baseline runs are wall-clock anchors for the same-hardware comparison (Rule 4);
the current record's validation certification is its own merged pool.

- GPUs: 8x H100 80GB (SXM)
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- NVIDIA driver: `580.126.09`
- runs: `8` total (2 unlogged wall anchors on the initial pool's machines; 2 logged
  on machine A; 4 logged on machine B)

## Statistics

| machine | train time | val loss |
| --- | ---: | ---: |
| Xeon 8480+ (fastest, same machine as this PR's wall pool) | 74.38 s | 3.2822 |
| Xeon 8480+ (second) | 75.05 s | 3.2753 |

Same-machine comparison against this PR's pool (see `../this_pr/statistics.md`):
74.38 s -> 64.95 s mean, an improvement of 9.4 s / 12.7%.

## Per-run logs, machine A (same machine as the 2026-08-13 pool)

The current record re-run twice on the same fresh machine as the second
certification pool, logs retained:

| log | train time (s) | val loss |
| --- | ---: | ---: |
| [8314a37a](./8314a37a-ecae-458a-b868-8228744252fd.txt) | 74.670 | 3.2757 |
| [fdb9d056](./fdb9d056-0535-4d2d-a5ed-d84eb94826df.txt) | 74.653 | 3.2853 |

Same-machine comparison: 74.662 s mean baseline vs 65.296 s mean for this PR's
pool on the identical machine and session: 9.37 s / 12.5%.

## Per-run logs, machine B (same machine as the 2026-08-14 pool)

The current record run four times on the same fresh machine as the third
(shipped-source) certification pool — one before the pool and three interleaved
through it, logs retained:

| log | train time (s) | val loss |
| --- | ---: | ---: |
| [2178a3fb](./2178a3fb-d52a-497a-b50f-7218511e4f39.txt) | 74.146 | 3.2758 |
| [55f2e425](./55f2e425-0d02-4445-8f5e-e738067f8e50.txt) | 73.967 | 3.2769 |
| [17f6062d](./17f6062d-7072-4a13-86ca-4375a2f4a479.txt) | 73.926 | 3.2771 |
| [d4e7991f](./d4e7991f-7273-462a-8bd8-9d46077a482e.txt) | 74.048 | 3.2760 |

Same-machine comparison: 74.022 s mean baseline (n=4) vs 64.723 s mean for this
PR's 12-run pool on the identical machine and session: 9.30 s / 12.6%.

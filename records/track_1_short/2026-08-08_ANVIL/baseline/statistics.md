# Baseline (current record) Statistics

The current `train_gpt.py` record was re-run unseeded in the same session, on
two of the four machines used for this PR's certification pool, fresh compile caches.
These baseline runs are wall-clock anchors for the same-hardware comparison (Rule 4);
the current record's validation certification is its own merged pool.

- GPUs: 8x H100 80GB (SXM)
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- NVIDIA driver: `580.126`
- runs: `2` (one on each of two of the three 8480+ hosts)

## Statistics

| machine | train time | val loss |
| --- | ---: | ---: |
| Xeon 8480+ (fastest, same machine as this PR's wall pool) | 74.38 s | 3.2822 |
| Xeon 8480+ (second) | 75.05 s | 3.2753 |

Same-machine comparison against this PR's pool (see `../this_pr/statistics.md`):
74.38 s -> 64.95 s mean, an improvement of 9.4 s / 12.7%.

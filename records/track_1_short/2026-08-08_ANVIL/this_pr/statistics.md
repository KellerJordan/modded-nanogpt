# ANVIL Record Statistics

## Summary

New track-1 record: ANVIL optimizer (twin-rail whitened velocity), full-fp8 MLP
forward and backward, narrow query/key attention (8 of 10 layers at Q/K head
width 64 with a rotating rotary basis), period-4 embedding-channel cadences, a
virtual sequence cap (train-time attention masking only), and a grown schedule
(1218 scheduled + 23 extension = 1241 steps). Full method notes in README.md.

- GPUs: 8x H100 80GB (SXM)
- PyTorch: `2.10.0+cu128`
- Triton: `3.6.0`
- NVIDIA driver: `580.126`
- runs: `16` (unseeded, fresh compile caches every run, all runs counted)
- machines: `4` (three Xeon 8480+ hosts, one Xeon 8468)

## Statistics

| metric | value |
| --- | ---: |
| mean val loss | 3.2760375 |
| val loss sample std | 0.0008709 |
| one-sided t vs 3.28 | 18.20 |
| one-sided p vs 3.28 | < 1e-10 |
| mean train time, all 16 runs | 65.358 s |
| train time sample std, all 16 runs | 0.407 s |
| median train time, all 16 runs | 65.396 s |
| train time range, all 16 runs | 64.839-66.133 s |
| mean train time, fastest machine by mean (n=5) | 64.947 s |
| train time range, fastest machine by mean | 64.871-65.050 s |
| train time range, other 8480+ hosts | 65.426-65.779 s / 65.742-66.133 s |
| train time range, 8468 host | 64.839-65.752 s |

Val losses, grouped by machine:
- 8480+ (fastest by mean): 3.2765, 3.2764, 3.2758, 3.2750, 3.2754 (mean 3.27582)
- 8480+ (second): 3.2757, 3.2755, 3.2773, 3.2762, 3.2750 (mean 3.27594)
- 8480+ (third): 3.2749, 3.2751 (mean 3.27500)
- 8468: 3.2769, 3.2777, 3.2768, 3.2764 (mean 3.27695)

A cluster-conservative test on the four machine means alone (n=4: 3.27582,
3.27594, 3.27500, 3.27695; mean 3.27593, sd 0.00080) still clears the gate:
one-sided t vs 3.2800 = 10.2, p = 0.001, well inside the 0.01 gate.

Times (s), grouped by machine:
- 8480+ (fastest by mean): 64.871, 64.905, 64.922, 64.985, 65.050
- 8480+ (second): 65.426, 65.490, 65.623, 65.718, 65.779
- 8480+ (third): 65.742, 66.133
- 8468: 64.839, 65.132, 65.366, 65.752

Calculation snippet:

```python
import statistics as st
vals = [3.2765, 3.2764, 3.2757, 3.2769, 3.2755, 3.2758, 3.2777, 3.2773,
        3.2750, 3.2768, 3.2762, 3.2754, 3.2764, 3.2749, 3.2751, 3.2750]
times = [64.871, 64.905, 64.922, 64.985, 65.050, 65.426, 65.490, 65.623,
         65.718, 65.779, 65.742, 66.133, 64.839, 65.132, 65.366, 65.752]
n = len(vals)
t = (3.2800 - st.mean(vals)) / (st.stdev(vals) / n ** 0.5)
print(st.mean(vals), st.stdev(vals), t, st.mean(times), st.stdev(times))
```

Individual val losses (all 16 runs, pooled):
3.2765, 3.2764, 3.2757, 3.2769, 3.2755, 3.2758, 3.2777, 3.2773,
3.2750, 3.2768, 3.2762, 3.2754, 3.2764, 3.2749, 3.2751, 3.2750

## Baseline comparison

The current-record `train_gpt.py` was re-run in the same session on two of the
four machines; per-machine numbers in `../baseline/statistics.md`. Same-machine
improvement: 74.38 s -> 64.95 s mean, 9.4 s / 12.7%.

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
- NVIDIA driver: `580.126.09`
- runs: `44` total across three unseeded certification pools (16 initial across 4
  machines; 16 logged on machine A; 12 logged on machine B), fresh compile caches
  every run, all runs counted within each pool
- machines: `6` (five Xeon 8480+ hosts, one Xeon 8468)

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

## Per-run logs, machine B (2026-08-14) — the shipped source, fastest pool

A third certification pool on another fresh 8x H100 machine (Xeon 8480+, driver
580.126.09), running the exact shipped source tree (`train_gpt.py` byte-identical
to this branch head). Twelve unseeded runs, fresh compile caches each, all runs
counted, bracketed by four same-session runs of the current-record `train_gpt.py`
on the same machine (logs in `../baseline/`, mean 74.022 s — see
`../baseline/statistics.md`).

| run | log | train time (s) | val loss |
| ---: | --- | ---: | ---: |
| 1 | [d643be13](./d643be13-5a52-4e0c-976e-51736215cdd5.txt) | 64.699 | 3.2792 |
| 2 | [fcbae175](./fcbae175-71c6-4802-8ead-9fca5f58228c.txt) | 64.651 | 3.2768 |
| 3 | [2855a319](./2855a319-bfb7-4bb6-849d-76a6335c9cf8.txt) | 64.819 | 3.2754 |
| 4 | [896699d4](./896699d4-4905-4736-9bf5-1d01ce5e6275.txt) | 64.717 | 3.2783 |
| 5 | [4f256906](./4f256906-b217-461a-8c7d-e9f6f8a67398.txt) | 64.560 | 3.2787 |
| 6 | [228003ab](./228003ab-4e49-4630-969d-8bbcb51b6e90.txt) | 64.605 | 3.2789 |
| 7 | [972b2ae8](./972b2ae8-934f-4b8f-8400-a28d40c2684a.txt) | 64.842 | 3.2776 |
| 8 | [bf742f29](./bf742f29-ff5e-4e96-829b-7cf4e79e639b.txt) | 64.618 | 3.2784 |
| 9 | [f3647a7e](./f3647a7e-46e8-4d4d-813c-2261713a216b.txt) | 64.815 | 3.2772 |
| 10 | [3ee0482a](./3ee0482a-526c-4d88-965a-8f168c9925ba.txt) | 64.779 | 3.2751 |
| 11 | [4738dff1](./4738dff1-51cf-4783-8807-b6d21a2b8509.txt) | 64.697 | 3.2759 |
| 12 | [707c9615](./707c9615-0118-4315-ac78-1f05afd34daa.txt) | 64.872 | 3.2773 |

| metric | value |
| --- | ---: |
| mean val loss | 3.2774000 |
| val loss sample std | 0.0013810 |
| one-sided t vs 3.28 | 6.52 |
| one-sided p vs 3.28 | 0.00003 |
| mean train time | 64.723 s |
| train time sample std | 0.102 s |
| train time range | 64.560-64.872 s |
| same-machine baseline (n=4) | 74.022 s |
| same-machine improvement | 9.30 s / 12.6% |

One additional run of the immediately-prior source revision on this machine
(before the dead-code pass) recorded 64.761 s / 3.2769; it is excluded from the
pooled statistics above only because its embedded source dump predates the
shipped tree, and it is consistent with the pool in both metrics.

## Per-run logs, machine A (2026-08-13)

A second certification pool, run on a fresh 8x H100 machine (Xeon 8480+, driver
580.126.09) with per-run logs retained, including the fp8 quantizer reset at the
post-warmup boundary. These 16 logs embed the source revision that preceded a
final dead-code-removal pass (comment and unreachable-code deletions only; the
machine-B pool above runs the exact shipped source, and the two revisions were
verified equivalent — same executed op stream at the shipped configuration). Sixteen unseeded runs, fresh
compile caches each, all runs counted; run 0 is the first run on the freshly
provisioned machine and carries first-run compile overhead in its wall time.

| run | log | train time (s) | val loss |
| ---: | --- | ---: | ---: |
| 0 | [de49bd54](./de49bd54-47a6-4815-8cd6-46614ea2af99.txt) | 65.477 | 3.2770 |
| 1 | [a335e723](./a335e723-04f8-439a-b9f9-e06d40c33c47.txt) | 65.200 | 3.2769 |
| 2 | [2e385983](./2e385983-f307-4b85-bae4-6c49a66dbedb.txt) | 65.229 | 3.2753 |
| 3 | [848e472a](./848e472a-5c16-443e-943a-c0c16a7ca28f.txt) | 65.135 | 3.2770 |
| 4 | [e347d093](./e347d093-d603-47ed-9519-651fae8ddebd.txt) | 65.318 | 3.2777 |
| 5 | [53b446a2](./53b446a2-de72-41f5-96e3-85bc37e4f36b.txt) | 65.354 | 3.2781 |
| 6 | [2f75ebd3](./2f75ebd3-5cc0-4ef8-aa3b-03248355f72b.txt) | 65.163 | 3.2790 |
| 7 | [892213c0](./892213c0-7364-499a-9f7e-c4ddf6c54c0a.txt) | 65.325 | 3.2762 |
| 8 | [06de92a0](./06de92a0-be4b-48ab-8f50-8cc08076b192.txt) | 65.585 | 3.2751 |
| 9 | [b4695c31](./b4695c31-90be-49fe-b2c3-2c1671457b8d.txt) | 65.202 | 3.2746 |
| 10 | [fc48f94a](./fc48f94a-99fe-4a92-84b0-9e21239862ce.txt) | 65.240 | 3.2756 |
| 11 | [c3cd2b6b](./c3cd2b6b-5eed-4cd9-aca7-6409804a8712.txt) | 65.372 | 3.2777 |
| 12 | [d5a12961](./d5a12961-897c-4389-8998-5ae6e4f89432.txt) | 65.343 | 3.2844 |
| 13 | [7169c392](./7169c392-c731-4398-ae68-b7eb2b9f883d.txt) | 65.343 | 3.2754 |
| 14 | [a2ac8d4a](./a2ac8d4a-87ac-487b-a0a1-8232ad4be878.txt) | 65.258 | 3.2763 |
| 15 | [6b7c9d20](./6b7c9d20-b17d-41e4-bf13-66e8f5f0a397.txt) | 65.199 | 3.2764 |

| metric | value |
| --- | ---: |
| mean val loss | 3.2770437 |
| val loss sample std | 0.0022978 |
| one-sided t vs 3.28 | 5.15 |
| one-sided p vs 3.28 | 0.00006 |
| mean train time | 65.296 s |
| train time sample std | 0.119 s |
| train time range | 65.135-65.585 s |

One run (run 12, 3.2844) drew above the gate individually; the same-session
baseline pool drew 3.2853 (see `../baseline/statistics.md`), so the tail is a
property of the machine and the benchmark's run-to-run variance, not of this
trainer; both means sit far under the gate and every run is counted.

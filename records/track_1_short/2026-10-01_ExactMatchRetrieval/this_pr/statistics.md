# Exact-match retrieval record statistics (this PR)

- code: commit `9fd046d`, 610 trained steps (schedule default `KX_STEPS=538`)
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb`
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- protocol: one session (2026-10-01 04:02-05:27 UTC), legs interleaved baseline, this PR, this PR (x9); `KX_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, all 103 train shards + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `18`, all counted; all 18 raw logs ship in this folder

| metric | value |
| --- | ---: |
| mean wall (train_time) | 21.555 s |
| wall sample std | 0.026 s |
| wall range | 21.508-21.611 s |
| mean val loss | 3.2748944444 |
| val loss sample std | 0.0026059823 |
| t-statistic vs 3.28 gate (one-sided) | 8.31 |
| one-sided p vs 3.28 gate | 1.08e-07 |

Against the baseline legs of the same session (`../baseline/`, ANVIL2, n=9):

| metric | value |
| --- | ---: |
| mean wall difference | -18.447 s (-46.1%) |
| Welch t-test on wall, two-sided p | 1.3e-30 |
| mean val loss difference | -0.00437 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 0 | `a86cf524-67e6-4329-b9ad-f982f28b25cd.txt` | 610 | 3.2749 | 21.569 s |
| 3 | 1 | `372ac1b0-eac2-41cb-850a-0f55294a017f.txt` | 610 | 3.2768 | 21.532 s |
| 5 | 2 | `b3b13aa2-98a1-4b12-af3a-dd68d7d116ce.txt` | 610 | 3.2740 | 21.508 s |
| 6 | 3 | `2d896dfe-3916-491d-92bb-102b55c91555.txt` | 610 | 3.2763 | 21.525 s |
| 8 | 4 | `cee2c869-4b70-44db-abee-c92d4d128c80.txt` | 610 | 3.2720 | 21.551 s |
| 9 | 5 | `d1008e11-cceb-4996-a35b-7b25fcb87d4c.txt` | 610 | 3.2829 | 21.566 s |
| 11 | 6 | `9a167c05-257e-495a-9f95-d8cc025f88c5.txt` | 610 | 3.2743 | 21.551 s |
| 12 | 7 | `28d2300c-eb28-40b1-bfee-dbaa7ff652ce.txt` | 610 | 3.2727 | 21.569 s |
| 14 | 8 | `26640c36-6c59-4995-8838-3710f9637024.txt` | 610 | 3.2722 | 21.536 s |
| 15 | 9 | `744fc180-e798-4bca-8b2a-044bc8ff7365.txt` | 610 | 3.2741 | 21.578 s |
| 17 | 10 | `1af907a4-98e1-420b-9734-df1f9a861b4e.txt` | 610 | 3.2716 | 21.574 s |
| 18 | 11 | `dc0f810f-495e-43c9-8832-00eb3826f76c.txt` | 610 | 3.2767 | 21.565 s |
| 20 | 12 | `58dd1502-4fcc-49f5-aa26-f359ff0dbafb.txt` | 610 | 3.2747 | 21.530 s |
| 21 | 13 | `02180729-2664-42c9-820b-cdf50d6d6fb4.txt` | 610 | 3.2743 | 21.611 s |
| 23 | 14 | `725b017a-fbd1-4672-81d4-92c8ddf05ea9.txt` | 610 | 3.2737 | 21.552 s |
| 24 | 15 | `62b7100d-76d8-4673-87ae-4654450b186c.txt` | 610 | 3.2741 | 21.574 s |
| 26 | 16 | `1e3b0248-e054-44ee-945a-6a56b7a04a36.txt` | 610 | 3.2753 | 21.520 s |
| 27 | 17 | `62a1034e-2bfa-4a30-baa4-046442ea64f3.txt` | 610 | 3.2775 | 21.571 s |

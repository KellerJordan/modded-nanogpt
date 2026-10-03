# Baseline (PR #367) statistics — same machine, interleaved

- code: PR #367 head `e7c960f`, 610 trained steps
- GPUs: 8x H100 80GB HBM3 (driver 580.178.04), 2x Xeon Platinum 8462Y+ (128 vCPU), 2 TB RAM
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 12.8
- protocol: one session (2026-10-03 09:17-11:12 UTC), legs interleaved baseline, this PR, this PR (x3), then this PR (x10); `TRAIN_SEED` = seed column; before every leg: the previous run's processes gone, the index tables in `/dev/shm` removed; data in `/dev/shm` (warm); Triton compile cache kept, inductor FX graph cache off (compilation is untimed)
- runs: `3`, all counted; all raw logs ship in this folder

| metric | value |
| --- | ---: |
| mean wall (train_time) | 21.749 s |
| wall sample std | 0.023 s |
| wall range | 21.727-21.773 s |
| mean val loss | 3.2746000000 |
| val loss sample std | 0.0036097091 |
| t-statistic vs 3.28 gate (one-sided) | 2.59 |
| one-sided p vs 3.28 gate | 6.11e-02 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0 | `506e8844-9f5e-46ed-a4bf-610ffc8072ec.txt` | 610 | 3.2787 | 21.773 s |
| 4 | 1 | `92449ae2-cf8b-46f6-89fa-1c924c85342e.txt` | 610 | 3.2719 | 21.727 s |
| 7 | 2 | `4065606b-6bd6-4af4-a463-116aad8e07d7.txt` | 610 | 3.2732 | 21.748 s |

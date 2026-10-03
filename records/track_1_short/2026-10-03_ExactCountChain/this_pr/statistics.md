# Record statistics (this PR)

- code: commit `0e0ff35` (this PR), 725 trained steps
- GPUs: 8x H100 80GB HBM3 (driver 580.178.04), 2x Xeon Platinum 8462Y+ (128 vCPU), 2 TB RAM
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 12.8
- protocol: one session (2026-10-03 09:17-11:12 UTC), legs interleaved baseline, this PR, this PR (x3), then this PR (x10); `TRAIN_SEED` = seed column; before every leg: the previous run's processes gone, the index tables in `/dev/shm` removed; data in `/dev/shm` (warm); Triton compile cache kept, inductor FX graph cache off (compilation is untimed)
- runs: `16`, all counted; all raw logs ship in this folder

| metric | value |
| --- | ---: |
| mean wall (train_time) | 9.653 s |
| wall sample std | 0.041 s |
| wall range | 9.593-9.767 s |
| mean val loss | 3.2758812500 |
| val loss sample std | 0.0016448784 |
| t-statistic vs 3.28 gate (one-sided) | 10.02 |
| one-sided p vs 3.28 gate | 2.45e-08 |

Against the baseline legs of the same session (`../baseline/`, PR #367, n=3):

| metric | value |
| --- | ---: |
| mean wall difference | -12.097 s (-55.6%) |
| Welch t on wall | -717.4 |
| mean val loss difference | +0.00128 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 0 | `54de7700-ccdb-48b7-9a40-86a50bd09df6.txt` | 725 | 3.2765 | 9.660 s |
| 3 | 1 | `e353ecfc-d0d1-44c4-979f-dec16f916188.txt` | 725 | 3.2738 | 9.637 s |
| 5 | 2 | `ddf25ff8-ed58-4c44-80b2-2d6d31ff457a.txt` | 725 | 3.2751 | 9.667 s |
| 6 | 3 | `b2959927-5385-480d-9da8-d852cce47095.txt` | 725 | 3.2772 | 9.617 s |
| 8 | 4 | `67cc051f-5730-40c7-8efd-241ddd28332d.txt` | 725 | 3.2767 | 9.652 s |
| 9 | 5 | `f34a3ed8-f8df-4023-b8d5-cc8a8265edc1.txt` | 725 | 3.2754 | 9.693 s |
| 10 | 6 | `0c945209-8ef9-4814-ab1c-adff7a28bdb8.txt` | 725 | 3.2796 | 9.646 s |
| 11 | 7 | `ad5d8901-b10c-40a5-8fbe-4cb3b5623d56.txt` | 725 | 3.2751 | 9.593 s |
| 12 | 8 | `011e4896-d84d-4ad5-b63b-173256f05b05.txt` | 725 | 3.2747 | 9.639 s |
| 13 | 9 | `88769199-32f5-40bf-b5b0-ed1bbe69b8a8.txt` | 725 | 3.2774 | 9.624 s |
| 14 | 10 | `fbbbd6e9-22a6-4cd9-8607-82d43ab4c3f7.txt` | 725 | 3.2749 | 9.602 s |
| 15 | 11 | `843dc71e-0864-43f8-beb1-6c6d4385f146.txt` | 725 | 3.2739 | 9.767 s |
| 16 | 12 | `162e8204-7ed8-4e6d-8320-a771d2b8a3e2.txt` | 725 | 3.2741 | 9.635 s |
| 17 | 13 | `e7b6b340-d425-4580-962e-bd6ca27a4470.txt` | 725 | 3.2752 | 9.652 s |
| 18 | 14 | `1c1141ca-f817-4c48-8e65-cb5f07321a34.txt` | 725 | 3.2762 | 9.693 s |
| 19 | 15 | `002bc57f-6e74-489c-aaf0-68e23f6d95d7.txt` | 725 | 3.2783 | 9.667 s |

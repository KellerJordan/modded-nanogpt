# Baseline (ANVIL2, record #92) statistics — same VM, interleaved

- code: upstream master `4ea6b937` (ANVIL2 record after the `track_1_short` refactor), 1194 trained steps, with only the train and val paths in `track_1_short/config.py` changed from `data/fineweb10B` to `data/fineweb100B`
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb` (Intel Xeon Platinum 8468, 128 threads, 1575 GiB)
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- data: FineWeb-100B (`data/fineweb100B`), validated on its `fineweb_val_000000.bin`
- protocol: one session (2026-10-20 15:54-17:46 UTC), legs interleaved this PR, this PR, baseline (x8); `TRAIN_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, train shards 1-370 + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `8`, all counted

| metric | value |
| --- | ---: |
| mean wall (train_time) | 40.475 s |
| wall sample std | 0.046 s |
| wall range | 40.431-40.582 s |
| mean val loss | 3.2818750000 |
| val loss sample std | 0.0015691217 |
| t-statistic vs 3.28 gate (one-sided) | -3.38 |
| one-sided p vs 3.28 gate | 9.94e-01 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 3 | 0 | `2f31be27-4ac7-4fa4-aebc-f6b2e280c3f4.txt` | 1194 | 3.2826 | 40.582 s |
| 6 | 1 | `dd0315c8-e7ff-49b1-b0c7-ca372378f68b.txt` | 1194 | 3.2819 | 40.450 s |
| 9 | 2 | `500a66e4-81c2-47b9-8388-6935b24eb0c4.txt` | 1194 | 3.2822 | 40.431 s |
| 12 | 3 | `04ea9c92-741c-448e-91e6-c31ee4044151.txt` | 1194 | 3.2843 | 40.449 s |
| 15 | 4 | `c7dfed9f-4e9e-4861-96d5-c7412e0e279d.txt` | 1194 | 3.2790 | 40.478 s |
| 18 | 5 | `181ffaf0-eca7-429e-a8be-b7e739663c10.txt` | 1194 | 3.2812 | 40.462 s |
| 21 | 6 | `49efd224-3b9b-439f-80bc-e08055ca18ab.txt` | 1194 | 3.2829 | 40.482 s |
| 24 | 7 | `827d8a89-46ab-42bd-987a-6505c2d72229.txt` | 1194 | 3.2809 | 40.464 s |

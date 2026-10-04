# Baseline (ANVIL2, record #92) statistics — same VM, interleaved

- code: commit `f9b6266` (ANVIL2 record as merged), 1194 trained steps
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb`
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- protocol: one session (2026-10-01 04:02-05:27 UTC), legs interleaved baseline, this PR, this PR (x9); `KX_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, all 103 train shards + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `9`, all counted

| metric | value |
| --- | ---: |
| mean wall (train_time) | 40.001 s |
| wall sample std | 0.040 s |
| wall range | 39.958-40.062 s |
| mean val loss | 3.2792666667 |
| val loss sample std | 0.0037292761 |
| t-statistic vs 3.28 gate (one-sided) | 0.59 |
| one-sided p vs 3.28 gate | 2.86e-01 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0 | `c5cccc4f-ea9b-4048-8b59-df4f79ee5a10.txt` | 1194 | 3.2769 | 40.048 s |
| 4 | 1 | `61c62a3a-ce15-457a-aec9-5b7c7b7e74b8.txt` | 1194 | 3.2788 | 40.008 s |
| 7 | 2 | `0634c425-89fe-4251-ae84-2055e12bf951.txt` | 1194 | 3.2771 | 39.960 s |
| 10 | 3 | `9f91b11c-3c8f-412f-9fa4-14e8aebe94ca.txt` | 1194 | 3.2805 | 39.959 s |
| 13 | 4 | `2d06ae33-1c45-47bd-9df6-69b7cfb7386b.txt` | 1194 | 3.2759 | 40.062 s |
| 16 | 5 | `be9d04cb-3d87-469d-bc13-35f49d3f053e.txt` | 1194 | 3.2884 | 39.958 s |
| 19 | 6 | `87e960fe-91b3-4ef7-aa35-c42988579ba9.txt` | 1194 | 3.2785 | 40.001 s |
| 22 | 7 | `2b49e575-7e4e-403f-add6-24faf1f11199.txt` | 1194 | 3.2799 | 40.035 s |
| 25 | 8 | `684e9f13-9aa8-43fe-bb48-be0c08c79190.txt` | 1194 | 3.2774 | 39.980 s |

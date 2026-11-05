# Exact-match retrieval record statistics (this PR)

- code: commit `2bed7ce5`, 540 trained steps (530 scheduled + 10 extension); validation index over the first 370 training shards
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb` (Intel Xeon Platinum 8468, 128 threads, 1575 GiB)
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- data: FineWeb-100B (`data/fineweb100B`), validated on its `fineweb_val_000000.bin`
- protocol: one session (2026-10-20 15:54-17:46 UTC), legs interleaved this PR, this PR, baseline (x8); `TRAIN_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, train shards 1-370 + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `16`, all counted; all 16 raw logs ship in this folder.

| metric | value |
| --- | ---: |
| mean wall (train_time) | 6.702 s |
| wall sample std | 0.018 s |
| wall range | 6.682-6.740 s |
| mean val loss | 3.2740625000 |
| val loss sample std | 0.0040039772 |
| t-statistic vs 3.28 gate (one-sided) | 5.93 |
| one-sided p vs 3.28 gate | 1.38e-05 |

Against the baseline legs of the same session (`../baseline/`, ANVIL2, n=8):

| metric | value |
| --- | ---: |
| mean wall difference | -33.773 s (-83.4%) |
| Welch t-test on wall, two-sided p | 3.8e-24 |
| mean val loss difference | -0.00781 |

## Runs (session order)

The last column is when the validation index was ready, from the start of the clock (from the log's `validation index built` line).

| leg | seed | log | steps | final val loss | train_time | val index ready |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0 | `7fb9d51e-43a9-4dea-b31a-b773d7845df0.txt` | 540 | 3.2734 | 6.699 s | 6.26 s |
| 2 | 1 | `7f2fd821-0c72-4359-83aa-989d2a5597fe.txt` | 540 | 3.2726 | 6.701 s | 6.12 s |
| 4 | 2 | `33179439-6b0a-4208-b038-484651f2368d.txt` | 540 | 3.2707 | 6.692 s | 6.59 s |
| 5 | 3 | `fa7bd14a-afe9-4470-a8f6-f8b7b1568b65.txt` | 540 | 3.2798 | 6.703 s | 6.37 s |
| 7 | 4 | `33fd1093-2d71-442c-86f3-60991994e4e1.txt` | 540 | 3.2771 | 6.738 s | 6.18 s |
| 8 | 5 | `5f3e15c3-69c3-4f03-a993-8c155cd3a02c.txt` | 540 | 3.2709 | 6.682 s | 6.23 s |
| 10 | 6 | `9213852a-4e9a-42b7-85df-faad0de50bbc.txt` | 540 | 3.2792 | 6.686 s | 6.32 s |
| 11 | 7 | `2613c1e9-51eb-4ea1-94d7-8d52e0875173.txt` | 540 | 3.2737 | 6.691 s | 6.23 s |
| 13 | 8 | `4ff74588-419c-486b-ac5c-433cc3f6feaa.txt` | 540 | 3.2708 | 6.683 s | 6.27 s |
| 14 | 9 | `eab617c5-49bf-4822-8e5c-cf76384b7fa1.txt` | 540 | 3.2720 | 6.720 s | 6.41 s |
| 16 | 10 | `9e4c60e3-b65d-40bc-867c-b2acdd8c2804.txt` | 540 | 3.2840 | 6.700 s | 6.24 s |
| 17 | 11 | `3ee5b61a-32f3-491d-8235-0f58dfe9050d.txt` | 540 | 3.2698 | 6.699 s | 6.26 s |
| 19 | 12 | `d3eb70b3-bd75-41cf-97b5-6c9225ea8c92.txt` | 540 | 3.2752 | 6.714 s | 6.14 s |
| 20 | 13 | `8543bc2b-00e8-4a22-b6b4-eecfcb961821.txt` | 540 | 3.2713 | 6.695 s | 6.37 s |
| 22 | 14 | `015205e3-3add-481b-bfb9-e36d0e20a84b.txt` | 540 | 3.2717 | 6.740 s | 6.21 s |
| 23 | 15 | `5c04d9fd-c7f2-4746-af20-b972277d8682.txt` | 540 | 3.2728 | 6.690 s | 6.35 s |

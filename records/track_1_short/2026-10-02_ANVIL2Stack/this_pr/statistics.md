# Record candidate statistics (this PR)

- GPUs: 8x H100 80GB HBM3 (driver 580.95.05, CUDA 13.1), Modal `gpu="H100!:8"`, one node per run
- PyTorch `2.10.0+cu128` · triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` (`pip_freeze.txt`, identical in every run)
- runs: `18` of `code/` unchanged, launched with `./run.sh`, warm compile cache; all runs counted

| metric | value |
| --- | ---: |
| mean wall (train_time) | 38.211 s |
| wall sample std | 0.278 s |
| mean val loss | 3.2792460521 |
| val loss sample std | 0.0007705488 |
| runs above 3.28 | 3 of 18 |
| t-statistic vs 3.28 gate (one-sided) | 4.15 |
| one-sided p vs 3.28 gate | 3.3e-04 |

`scipy.stats.ttest_1samp(val_losses, 3.28, alternative='less')`, as in `records/track_1_short/2025-01-04_SoftCap`.

| log | train_time | val loss | steps |
| --- | ---: | ---: | ---: |
| `880904c9-3770-439f-ad6e-6c13e66674fa.txt` | 38.161 s | 3.279011 | 1168 |
| `a4d3507a-838e-49ba-9f3f-022221882af0.txt` | 38.318 s | 3.280984 | 1168 |
| `68a8c8df-5d7a-4955-b71a-4f3426aaec7b.txt` | 38.128 s | 3.279643 | 1168 |
| `81141df8-d040-4c01-a2d4-2bf6d7104418.txt` | 37.638 s | 3.278952 | 1168 |
| `9ff11d12-0336-4268-be6f-396ca9272b10.txt` | 38.021 s | 3.278670 | 1168 |
| `40cf044d-37ff-450e-9c13-1ca484884a3a.txt` | 37.557 s | 3.278343 | 1168 |
| `2ec76336-aa4f-4060-b163-e6da341dd7bb.txt` | 38.159 s | 3.278636 | 1168 |
| `fc73f691-07de-4264-bdb8-5a65475432c4.txt` | 38.280 s | 3.278386 | 1168 |
| `9395c53c-c2d2-469e-8b80-504e325f7d95.txt` | 38.295 s | 3.279379 | 1168 |
| `f096c9a7-0db6-4e8f-abe9-903098b5b5aa.txt` | 38.529 s | 3.280864 | 1168 |
| `f320a809-14a1-4caf-90c9-75bbbafe1ba5.txt` | 38.697 s | 3.278778 | 1168 |
| `895d18a8-fc0f-4c84-b8a6-f18203f5c8c7.txt` | 38.339 s | 3.279213 | 1168 |
| `7284877e-9ff9-4491-812a-4bdae915a171.txt` | 38.207 s | 3.279287 | 1168 |
| `0bace670-31a2-4937-8020-2ff2049c1a65.txt` | 38.550 s | 3.278682 | 1168 |
| `a42f0f3d-6551-4e9d-b7bd-0b003e7575e9.txt` | 38.298 s | 3.279534 | 1168 |
| `252b6dae-2423-4d02-b8cb-703cb66686cf.txt` | 38.262 s | 3.280196 | 1168 |
| `5d2b2d46-445f-4737-852b-6d66a3b1239c.txt` | 38.186 s | 3.278652 | 1168 |
| `956247ef-9c18-4a4d-be26-d0f1294b2441.txt` | 38.172 s | 3.279219 | 1168 |

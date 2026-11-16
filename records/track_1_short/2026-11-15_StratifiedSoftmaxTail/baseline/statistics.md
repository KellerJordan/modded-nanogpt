# Stratified sampled softmax: same-node baseline statistics (record #92)

- GPUs: 8x H100 80GB HBM3 (driver 580.173.02, Nebius `gpu-h100-sxm`), 2026-11-16
- PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.1 image (the root `Dockerfile`)
- runs: `9` — record #92 (commit `4ea6b93`) from a second checkout on the same node, interleaved with this PR's legs

| metric | value |
| --- | ---: |
| mean wall (train_time) | 40.235 s |
| wall sample std | 0.017 s |
| mean val loss | 3.2760777778 |
| val loss sample std | 0.0013553392 |
| t-statistic vs 3.28 gate (one-sided) | -8.68 |
| one-sided p vs 3.28 gate | 1.21e-05 |

| log | val loss | train_time |
| --- | ---: | ---: |
| `2aaac6b3-667d-447b-9da6-33cb60cc6fe3.txt` | 3.2756 | 40.248 s |
| `33050bdc-25c4-4ce6-833a-8805747fa220.txt` | 3.2747 | 40.246 s |
| `4d69a56e-7313-47dc-a058-159d28c0c686.txt` | 3.2762 | 40.225 s |
| `5219628b-5337-4a77-a19b-48a2d48fb088.txt` | 3.2776 | 40.257 s |
| `7279a756-3b29-4dd5-83a3-39cba12a78b0.txt` | 3.2759 | 40.235 s |
| `9842fc59-f902-4f0a-9b31-44b543f13026.txt` | 3.2748 | 40.225 s |
| `9dcf1140-3f51-4975-a874-f535db34833d.txt` | 3.2788 | 40.239 s |
| `df9312cd-5dbf-46eb-a6f1-dac1d8e2cc53.txt` | 3.2761 | 40.238 s |
| `f36dc2c3-93f0-4eeb-b9b7-fdbbe08b7ac4.txt` | 3.2750 | 40.200 s |

Computed from the logs in this folder; `../pool_statistics.py` recomputes the pool comparison.

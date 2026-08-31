# ANVIL2 record statistics (this PR)

- GPUs: 8x H100 80GB HBM3 (driver 580.95.05, vast.ai instance)
- PyTorch `2.10.0+cu128` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.1 image
- runs: `18` — every leg fully cold (compile caches wiped per leg), interleaved with baseline legs

| metric | value |
| --- | ---: |
| mean wall (train_time) | 39.914 s |
| wall sample std | 0.120 s |
| mean val loss | 3.2773111111 |
| val loss sample std | 0.0009904775 |
| t-statistic vs 3.28 gate (one-sided) | 11.5 |
| one-sided p vs 3.28 gate | 9.40e-10 |

Note: `cert_ship_16` — raw log lost to pod expiry (2026-08-31); row recorded from the pod run ledger, witnessed by two independent pulls (identical values). Its metrics are included in the statistics above; 17 of 18 run logs ship in this folder.

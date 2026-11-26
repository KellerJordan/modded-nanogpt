# Centered Bias-Aware SSMax certification statistics (this PR)

- GPUs: 8x NVIDIA H100 80GB HBM3 (NV18 NVSwitch, driver 580.196.07, Nebius `gpu-h100-sxm`)
- PyTorch `2.10.0+cu128` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA 13.0
- runs: `10` interleaved 1:1 with `10` baseline (`HEAD 4ea6b93`) runs on the same machine, plus `5` matched-step `HEAD @ G=40` controls

| metric | `HEAD` (`G=52`, n=10) | `this_pr` (`G=40`, n=10) | `HEAD @ G=40` (n=5) |
| --- | ---: | ---: | ---: |
| mean wall (`train_time`) | 40.454 s | **39.977 s** | 39.905 s |
| wall sample std | 0.024 s | **0.020 s** | 0.011 s |
| mean val loss | 3.27610 | **3.27738** | 3.27998 |
| val loss sample std | 0.00281 | **0.00079** | 0.00389 |
| runs > 3.28 | 1 / 10 | **0 / 10** | 2 / 5 |
| t-statistic vs 3.28 gate (one-sided) | -4.39 | **-10.43** | -0.01 |
| one-sided p vs 3.28 gate | 8.78e-4 | **1.26e-6** | 4.96e-1 (FAIL) |
| paired Δtime vs `HEAD` (`G=52`) | — | **-0.477 ± 0.034 s** (`p = 3.81e-12`) | -0.549 s |

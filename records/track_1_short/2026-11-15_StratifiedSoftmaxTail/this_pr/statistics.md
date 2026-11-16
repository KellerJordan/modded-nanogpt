# Stratified sampled softmax: record statistics (this PR)

- GPUs: 8x H100 80GB HBM3 (driver 580.173.02, Nebius `gpu-h100-sxm`), 2026-11-16
- PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.1 image (the root `Dockerfile`)
- runs: `18` — all runs of this tree after one functional check, interleaved with the baseline legs; compile caches not wiped between legs (see the README's disclosures)

| metric | value |
| --- | ---: |
| mean wall (train_time) | 39.702 s |
| wall sample std | 0.042 s |
| mean val loss | 3.2767000000 |
| val loss sample std | 0.0011386266 |
| t-statistic vs 3.28 gate (one-sided) | -12.30 |
| one-sided p vs 3.28 gate | 3.46e-10 |

| log | val loss | train_time |
| --- | ---: | ---: |
| `09e18acc-3b81-4afd-b513-a72f8857284b.txt` | 3.2775 | 39.678 s |
| `1431d4c8-b25b-4d84-9240-b5975841bcda.txt` | 3.2771 | 39.694 s |
| `1e3ff10b-7a77-4b00-8963-da8db93c0e22.txt` | 3.2753 | 39.677 s |
| `2da96cf6-6bdb-4653-8329-f97308526a53.txt` | 3.2784 | 39.764 s |
| `3167b30e-e39c-4d8a-9443-c1400094b6e7.txt` | 3.2745 | 39.809 s |
| `326cb902-b623-4c9b-a587-ae21c907596e.txt` | 3.2760 | 39.688 s |
| `542d3bf9-66db-43a3-b197-f8e158ed6885.txt` | 3.2766 | 39.776 s |
| `635b8a87-0a32-4e93-9f0a-5f5a674ebc22.txt` | 3.2780 | 39.668 s |
| `646694f2-af83-4f9e-a318-9cb9afb4863d.txt` | 3.2763 | 39.676 s |
| `6bd93f47-0da2-4d2b-886b-edaa1bac75d5.txt` | 3.2773 | 39.740 s |
| `73b3d75a-c8b5-4573-a562-379e109417df.txt` | 3.2756 | 39.664 s |
| `b00916b5-281d-4827-b27f-979e3840f3f4.txt` | 3.2760 | 39.659 s |
| `b5714bde-f4bb-4cf9-b1e1-9b03e633c544.txt` | 3.2769 | 39.706 s |
| `c897ccf7-88cd-4ebe-955b-3c4bcc0ee849.txt` | 3.2773 | 39.680 s |
| `d3199e54-4741-4036-b9cf-bdb4c6648334.txt` | 3.2776 | 39.699 s |
| `d8276be2-9b98-47a3-8099-f57dad7bee24.txt` | 3.2755 | 39.688 s |
| `f5687fa0-769a-4d71-849a-555b325b8749.txt` | 3.2787 | 39.679 s |
| `f8874293-1ef1-43cb-a217-0c6c1d4b83a3.txt` | 3.2760 | 39.690 s |

Computed from the logs in this folder; `../pool_statistics.py` recomputes the pool comparison.

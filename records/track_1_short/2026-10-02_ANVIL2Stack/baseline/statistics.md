# Baseline (current record, master `4ea6b93`) statistics, same host type

- GPUs: 8x H100 80GB HBM3 (driver 580.95.05, CUDA 13.1), Modal `gpu="H100!:8"`, one node per run; same image and `pip freeze` as `this_pr/`
- code: master `4ea6b93` (`train_gpt.py` + `track_1_short/`) unchanged, launched with `./run.sh`; the FA3 kernel is pre-seeded and loaded offline, as master's README describes
- runs: `7`, in a separate session from `this_pr/`, not interleaved. One (`edfa794e`) ran on a degraded host, already at 67 ms/step by step 25 against about 22 ms on the others, and is left out of the timing statistics; its val loss counts

| metric | value |
| --- | ---: |
| mean wall (train_time), 6 runs | 41.836 s |
| wall sample std | 0.638 s |
| mean val loss, 7 runs | 3.27531 |
| val loss sample std | 0.00073 |

| log | train_time | val loss | steps |
| --- | ---: | ---: | ---: |
| `2b362b48-4a25-4323-b1c4-5a63cde5f31c.txt` | 42.203 s | 3.2746 | 1194 |
| `d6ba414d-66a1-40d4-ad88-214b0575ab38.txt` | 41.652 s | 3.2742 | 1194 |
| `fe79c549-3ce4-4236-a15d-5b17c99abfce.txt` | 42.553 s | 3.2755 | 1194 |
| `580f20e2-c253-41e0-9beb-134e970aa3c7.txt` | 42.156 s | 3.2758 | 1194 |
| `edfa794e-acd5-4e9f-8a68-810769d432f8.txt` | 141.439 s (degraded host) | 3.2763 | 1194 |
| `b363ef15-eabe-4515-ac18-d1ee7b7cbeb7.txt` | 41.727 s | 3.2751 | 1194 |
| `eff6ddb6-769d-44ed-a495-a79929fe77ce.txt` | 40.723 s | 3.2757 | 1194 |

# Recursive from-best

Track 1 submission compared against WR #83 re-run on the same 8x H100 Modal setup.

| | #83 official | baseline_pr299 (#83, Modal) | this_pr (Modal) |
|---|---:|---:|---:|
| n | - | 10 | 13 |
| steps | - | 1385 | 1398 |
| mean val_loss | - | 3.27850 | 3.27893 |
| val_loss std | - | 0.00144 | 0.00137 |
| p(mean<3.28) | - | 0.00471 | 0.007803 |
| mean train_time | 79.7s | 80.61s | 77.34s |
| train_time std | - | 0.22s | 0.20s |

Official PrimeIntellect validation is still pending; these are same-hardware Modal numbers.

Main changes:

- FP8 attention projections for QKV and O in the training forward pass.
- Annealed row-RMS-scaled Langevin/SGLD exploration noise in NorMuon, zero by cooldown.
- C-Optim / cautious Adam on the bigram and value-embedding banks.
- Leaner fused ReLU^2 MLP Triton kernel, storing `post = relu(pre)^2` and reconstructing `sqrt(post)` in backward.
- Schedule/architecture retunes: fewer paired-head layers, tied embed throughout, and step-count/window changes.

Notes:

- Train/val token streams remain FineWeb train/val.
- No extra `torch._inductor.config` or compile flags.
- Logs were collected on Modal, 8x NVIDIA H100 80GB, PyTorch 2.10.0+cu128, Triton 3.6.0.

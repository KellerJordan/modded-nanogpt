# KL-SOAP-H LR Power Decay Results

This result uses KL-SOAP-H from [PR #290](https://github.com/KellerJordan/modded-nanogpt/pull/290) with a power learning-rate decay,
nonzero learning-rate floors, and the `shampoo_beta=0.9` tuple from that
submission. It runs for a fixed 3040 training steps while evaluating the
learning-rate schedule over a 3125-step horizon.

The archived script hardcodes the submitted hyperparameter defaults. It reads
`KL_SOAP_SEED` only to select the random seed for a reproducibility run.

## Configuration

| field | value |
|---|---|
| train steps | 3040 |
| schedule steps | 3125 |
| LR power | 1.5 |
| Adam LR floor | 0.04 |
| KL-SOAP LR floor | 0.017 |
| KL-SOAP lr | 0.018 |
| KL-SOAP beta1 | 0.95 |
| KL-SOAP beta2 | 0.9 |
| KL-SOAP shampoo beta | 0.9 |
| KL-SOAP precondition frequency | 1 |

## Reliability

Across 5 non-cherry-picked seeds at the fixed stopping point K=3040, the mean
validation loss is 3.27814600. The Track 3 significance rule is:

`(3.28 - mean) * sqrt(n) >= 0.004`

For these runs:

`(3.28 - 3.27814600) * sqrt(5) = 0.00414567 >= 0.004`

| seed | val loss at K=3040 | logfile |
|---:|---:|---|
| 0 | 3.27865 | `01906576-8b3d-4bd2-a73f-23997f602ec1.txt` |
| 1 | 3.27793 | `e3827ea2-b13a-415f-a0d4-7e11cfd399df.txt` |
| 2 | 3.27937 | `2046e7c2-24c9-4195-9d94-815326bf4705.txt` |
| 3 | 3.27673 | `d7e8e6af-39e5-4a0b-aa43-e5727c445bca.txt` |
| 4 | 3.27805 | `8927196a-1d5c-4a5f-bfdf-3c35b1577d36.txt` |
| **Mean** | **3.27814600** | |

All logs report `step:3040/3040`; no per-run early stopping is used. The logs
are kept under their original UUID filenames.

## Reproduction

Run one seed from the repo root with the Track 3 quickstart environment:

```bash
KL_SOAP_SEED=0 torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/results/20260511_klsoap_h_lr_power_decay/train_gpt_simple_klsoap_h_lr_power_decay.py
```

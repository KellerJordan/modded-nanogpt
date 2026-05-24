# Newton-Muon, 3275 Steps

This directory contains a first valid Track 3 benchmark result for
[Newton-Muon](https://arxiv.org/abs/2604.01472). This is a per-optimizer result
rather than a global SOTA claim.

The submitted runs were launched with `train_steps = 3300` and logged validation
every 25 steps near the end of training. The earliest shared checkpoint that
passes the Track 3 significance rule is step 3275.

## Method

[Newton-Muon](https://arxiv.org/abs/2604.01472) starts from Muon with auxiliary
AdamW. Before the Newton-Schulz Muon update, hidden-matrix gradients are
right-preconditioned with activation covariance statistics collected by forward
hooks. The covariance inverse is refreshed every 64 steps, then the
preconditioned gradients are passed through the usual Muon momentum and
Newton-Schulz update.

Matrix hyperparameters:

| group | lr | wd |
|---|---:|---:|
| q/k/v | 0.020 | 0.025 |
| attn.proj | 0.025 | 0.025 |
| mlp.fc | 0.030 | 0.0375 |
| mlp.proj | 0.030 | 0.025 |

Auxiliary AdamW, dataset, batch size, architecture, and validation setup are
unchanged from the Track 3 baseline.

Compute for this result was limited, so this should be treated as a first valid
baseline. More compute and hyperparameter search may lead to better results.

## Reliability

15 completed non-cherry-picked runs at K=3275:

| log | val loss at 3275 | val loss at 3300 |
|---|---:|---:|
| `fcb92725-9ecf-4aec-a9ca-07639f075a10.txt` | 3.27831 | 3.27782 |
| `adbec91d-e18e-448b-9c4e-17245e4c3035.txt` | 3.27819 | 3.27770 |
| `5a4e2561-88ad-40bc-ad8a-c284ffab59f0.txt` | 3.28179 | 3.28128 |
| `e53164a8-cf85-47f1-a43b-6075273ffed0.txt` | 3.27699 | 3.27648 |
| `012451d9-a8dc-4383-9ca5-d0c45648d85c.txt` | 3.27847 | 3.27795 |
| `47ca3026-7669-4d75-98f3-324b67a328fc.txt` | 3.27884 | 3.27832 |
| `796dc285-9340-4cfb-b34c-525c72b1a1f0.txt` | 3.27698 | 3.27644 |
| `504191a8-77da-44cb-b619-751e203bcb14.txt` | 3.27888 | 3.27837 |
| `4ab72a5a-d9e3-4405-b662-8fe05dd6be29.txt` | 3.27864 | 3.27812 |
| `d9d42e73-09da-44ab-9c20-cd2928873f8f.txt` | 3.27709 | 3.27659 |
| `6fb302c7-d271-491b-906f-75cd6ec72075.txt` | 3.27653 | 3.27601 |
| `1e2209a6-95dc-4daa-b688-284e6f0f173e.txt` | 3.27932 | 3.27880 |
| `721c4e04-6771-4845-b4de-1dad265408c5.txt` | 3.28012 | 3.27962 |
| `6e3d3a98-1f61-43ba-a858-9d9ce5ec2d3d.txt` | 3.27932 | 3.27879 |
| `c844e66f-2b2e-4f8e-b47f-d8e11ec5cb65.txt` | 3.27826 | 3.27775 |
| **mean** | **3.27851533** | **3.27800267** |

Sample std at K=3275: `0.00135540`.

Track 3 significance rule:

```text
(3.28 - 3.27851533) * sqrt(15) = 0.00575009 >= 0.004
```

## Rules

- Dataset, batch size, architecture, and validation setup are unchanged from
  the Track 3 baseline.
- The trainer uses one forward/backward pass per step. Activation covariance
  statistics are collected by forward hooks during the same forward pass.
- No third-party optimizer library is imported; the optimizer code is inline in
  each logfile.
- The submitted stopping point is the shared K=3275 checkpoint for all 15
  completed runs. The runs were launched to 3300 steps, and no per-run
  val-loss early stopping was used.

## Files

The UUID-named `.txt` files are the full-source logfiles. The representative
script `train_gpt_simple_newton_muon.py` is copied from the run script used
to produce the logs.

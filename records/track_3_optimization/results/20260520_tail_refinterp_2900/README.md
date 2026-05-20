# Exact-2900 Tail Reference Interpolation

Author: Jesse Clark

This Track 3 submission keeps the benchmark data stream, global batch size, and
model architecture unchanged. It starts from the PR #300
Aurora-on-`mlp.proj` optimizer stack and retargets it to an exact 2900-step
endpoint.

The main new endpoint mechanism is Tail Reference Interpolation. The run
captures a full model-weight reference at step `2375`, then evaluates the final
step-2900 checkpoint with:

```text
theta_eval = theta_2900 + gamma * (theta_2900 - theta_2375)
gamma = -0.075
```

Equivalently:

```text
theta_eval = 0.925 * theta_2900 + 0.075 * theta_2375
```

This is a small deterministic interpolation back toward the late-training
trajectory. It is used only at the fixed final validation point and does not use
validation feedback during a run.

## Result

The submitted fixed checkpoint is step `2900`. The nine logs below are clean
full runs from seeds 0 through 8. No run was stopped early, dropped, or selected
based on validation loss.

Under the Track 3 precision rule:

```text
n=9
mean=3.278576
(3.28 - mean) * sqrt(9) = 0.004273
required = 0.004000
```

Using the README's `sigma=0.0013` normal approximation:

```text
z = 3.287179
p = 0.000505982
```

| Seed | Log | 2900 val | Train time | Step avg |
| -: | - | -: | -: | -: |
| 0 | [refinterp_2900_seed0.txt](refinterp_2900_seed0.txt) | 3.27803 | 4798.129s | 1654.53ms |
| 1 | [refinterp_2900_seed1.txt](refinterp_2900_seed1.txt) | 3.27899 | 4778.006s | 1647.59ms |
| 2 | [refinterp_2900_seed2.txt](refinterp_2900_seed2.txt) | 3.27857 | 4907.369s | 1692.20ms |
| 3 | [refinterp_2900_seed3.txt](refinterp_2900_seed3.txt) | 3.27819 | 4755.410s | 1639.80ms |
| 4 | [refinterp_2900_seed4.txt](refinterp_2900_seed4.txt) | 3.27867 | 4842.459s | 1669.81ms |
| 5 | [refinterp_2900_seed5.txt](refinterp_2900_seed5.txt) | 3.27855 | 4767.208s | 1643.86ms |
| 6 | [refinterp_2900_seed6.txt](refinterp_2900_seed6.txt) | 3.27853 | 4695.982s | 1619.30ms |
| 7 | [refinterp_2900_seed7.txt](refinterp_2900_seed7.txt) | 3.27805 | 4861.523s | 1676.39ms |
| 8 | [refinterp_2900_seed8.txt](refinterp_2900_seed8.txt) | 3.27960 | 4814.127s | 1660.04ms |
| **Mean** |  | **3.278576** |  |  |

## Configuration

Frozen runner:

```text
records/track_3_optimization/results/20260520_tail_refinterp_2900/run.sh
```

Frozen trainer:

```text
records/track_3_optimization/results/20260520_tail_refinterp_2900/train_gpt_tail_refinterp_2900.py
```

Core settings:

```text
TRAIN_STEPS=2900
SCHEDULE_STEPS=3020
CONTRA_TO_NORMAL_END_STEP=2750

TEMPERED_POLAR_RHO=0.05
TEMPERED_POLAR_MAX_DELTA=0.25
TEMPERED_POLAR_RAMP_START_STEP=2600
TEMPERED_POLAR_RAMP_END_STEP=2800

RADIAL_OUTWARD_SCALE=0.5
RADIAL_OUTWARD_SCALE_LATE=0.4
RADIAL_DECAY_START_STEP=2400
RADIAL_DECAY_END_STEP=2900

TARGET_UW_LATE=0.400
TARGET_UW_RAMP_START_STEP=2400
TARGET_UW_RAMP_END_STEP=2800
TARGET_UW_FINAL=0.400
TARGET_UW_SCOPE=all

TAIL_VEL_START_STEP=2500
TAIL_VEL_BETA=0.90
TAIL_VEL_GAMMA=-6.0
TAIL_VEL_MAX_DELTA_RATIO=0.01

REF_EXTRAP_CAPTURE_STEP=2375
REF_EXTRAP_GAMMA=-0.075
REF_EXTRAP_APPLY_START_STEP=2900
```

## Method

This candidate is built on the PR #300 optimizer stack:

- Aurora row-balanced polar update on `mlp.proj`.
- Contra-Muon and the PR294/PR287 optimizer schedule lineage.
- SOAP-style MLP/V preconditioning and radial/u-w-floor machinery inherited
  from the existing Track 3 line.

The exact-2900 changes are:

- Extend Contra-Muon later, to `CONTRA_TO_NORMAL_END_STEP=2750`.
- Use late-only Tempered Polar, ramping `rho` from 0 to `0.05` during steps
  `2600..2800`.
- Retune the late radial brake from `0.5` to `0.4` during steps `2400..2900`.
- Ramp the update/weight floor to `TARGET_UW=0.400` during steps `2400..2800`.
- Keep an EMA of recent parameter updates from step `2500` and apply a clipped
  final tail-velocity transform with `gamma=-6`.
- Capture a step-2375 reference and evaluate step 2900 with the fixed
  `gamma=-0.075` reference interpolation.

What worked:

- Late-only Tempered Polar was better than perturbing the full trajectory.
- The late radial and `TARGET_UW=0.400` retune improved the exact-2900 endpoint.
- Tail velocity with negative gamma helped versus earlier final-weight probes.
- The step-2375 reference interpolation was the decisive move, especially on
  weak seeds.

What did not work:

- Positive Tail Extrapolated EMA/XEWA was monotone worse in the seed-1 sweep.
- More Tempered Polar rho/cap surface tuning did not move the mean enough.
- Training-loss EMA gating was too weak to separate good and bad seeds.

## Reproduction

Run the submitted seeds:

```bash
records/track_3_optimization/results/20260520_tail_refinterp_2900/run.sh
```

Run a shorter check:

```bash
SEEDS="0 1" records/track_3_optimization/results/20260520_tail_refinterp_2900/run.sh
```

Recompute the reported statistics:

```bash
python3 records/track_3_optimization/results/20260520_tail_refinterp_2900/summarize_submission_stats.py --step 2900 \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed0.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed1.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed2.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed3.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed4.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed5.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed6.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed7.txt \
  records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed8.txt
```

## Credits

This submission incorporates features from the following previous submissions
and PRs:

- @kumarkrishna PR274 / Skylight-001: NorMuon-lite row/column variance
  normalization, u/w floor postprocessing, and `lr=0.0375` style Muon setup.
- @nilin PR275 and PR291: Contra-Muon / Soft-Muon lineage.
- @samacqua PR278: SOAP preconditioning machinery / MLP SOAP idea.
- @SPThole PR283: Trustlight attention-SOAP path lineage.
- @yash-oai PR287: power-law learning-rate cooldown.
- PR #300: Aurora on `mlp.proj` plus extended Contra-Muon on the PR294 stack.

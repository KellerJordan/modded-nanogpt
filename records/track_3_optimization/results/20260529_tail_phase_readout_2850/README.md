# 2850 Tail Phase Readout

Author: Jesse Clark

This Track 3 submission keeps the benchmark data stream, global batch size, and
model architecture unchanged. It uses the PR311-style 2850-step TrailDelta and
final-readout line, then adds two fixed late transforms:

- BroadDelta on `muon_other` from step `2000` to step `2400` with
  `gamma=-0.005`.
- Normalized orthogonal phase readout on `muon_other` from the `2650..2750`
  phase with `kappa=0.01`.

The final fixed validation point is step `2850`.

## Result

The submitted logs are clean full runs from seeds 0 through 12. The runs were
launched sequentially on one node with 8 local processes. No run was stopped
early, dropped, or selected based on validation loss.

Under the Track 3 precision rule:

```text
n=13
mean=3.278562
(3.28 - mean) * sqrt(13) = 0.005186
required = 0.004000
```

Using the README's `sigma=0.0013` normal approximation:

```text
z = 3.989574
p = 0.000033096
```

| Seed | Log | 2850 val | Train time | Step avg |
| -: | - | -: | -: | -: |
| 0 | [tail_phase_readout_2850_seed0.txt](tail_phase_readout_2850_seed0.txt) | 3.27742 | 812.475s | 285.08ms |
| 1 | [tail_phase_readout_2850_seed1.txt](tail_phase_readout_2850_seed1.txt) | 3.27815 | 810.844s | 284.51ms |
| 2 | [tail_phase_readout_2850_seed2.txt](tail_phase_readout_2850_seed2.txt) | 3.27858 | 811.352s | 284.68ms |
| 3 | [tail_phase_readout_2850_seed3.txt](tail_phase_readout_2850_seed3.txt) | 3.27834 | 806.390s | 282.94ms |
| 4 | [tail_phase_readout_2850_seed4.txt](tail_phase_readout_2850_seed4.txt) | 3.27854 | 813.819s | 285.55ms |
| 5 | [tail_phase_readout_2850_seed5.txt](tail_phase_readout_2850_seed5.txt) | 3.27806 | 799.626s | 280.57ms |
| 6 | [tail_phase_readout_2850_seed6.txt](tail_phase_readout_2850_seed6.txt) | 3.27872 | 811.673s | 284.80ms |
| 7 | [tail_phase_readout_2850_seed7.txt](tail_phase_readout_2850_seed7.txt) | 3.27873 | 812.963s | 285.25ms |
| 8 | [tail_phase_readout_2850_seed8.txt](tail_phase_readout_2850_seed8.txt) | 3.27896 | 831.729s | 291.83ms |
| 9 | [tail_phase_readout_2850_seed9.txt](tail_phase_readout_2850_seed9.txt) | 3.27750 | 808.468s | 283.67ms |
| 10 | [tail_phase_readout_2850_seed10.txt](tail_phase_readout_2850_seed10.txt) | 3.27802 | 807.090s | 283.19ms |
| 11 | [tail_phase_readout_2850_seed11.txt](tail_phase_readout_2850_seed11.txt) | 3.27967 | 810.816s | 284.50ms |
| 12 | [tail_phase_readout_2850_seed12.txt](tail_phase_readout_2850_seed12.txt) | 3.28061 | 811.234s | 284.64ms |
| **Mean** |  | **3.278562** |  |  |

## Configuration

Frozen runner:

```text
records/track_3_optimization/results/20260529_tail_phase_readout_2850/run.sh
```

Frozen trainer:

```text
records/track_3_optimization/results/20260529_tail_phase_readout_2850/train_gpt_tail_phase_readout_2850.py
```

Core settings:

```text
TRAIN_STEPS=2850
SCHEDULE_STEPS=2950
TAIL_SCHEDULE_SWITCH_STEP=2600
TAIL_SCHEDULE_STEPS=3010

TRAIL_DELTA_START_STEP=2650
TRAIL_DELTA_END_STEP=2850
TRAIL_DELTA_INTERVAL=100
TRAIL_DELTA_GAMMA=-0.08
TRAIL_DELTA_GAMMA_FIRST=-0.04
TRAIL_DELTA_GAMMA_FINAL=-0.12
TRAIL_DELTA_MOMENTUM_BETA=1.0
TRAIL_DELTA_SCOPE=nonembed

BROAD_DELTA_CAPTURE_STEP=2000
BROAD_DELTA_APPLY_STEP=2400
BROAD_DELTA_GAMMA=-0.005
BROAD_DELTA_SCOPE=muon_other

PHASE_READOUT_T0_STEP=2650
PHASE_READOUT_T1_STEP=2750
PHASE_READOUT_GAMMA=0
PHASE_READOUT_KAPPA=0.01
PHASE_READOUT_MODE=orthogonal
PHASE_READOUT_SCOPE=muon_other
PHASE_READOUT_NORMALIZE_PHASE=1
PHASE_READOUT_PER_STEP=0

FINAL_READOUT_ANCHOR_STEP=2400
FINAL_READOUT_ALPHA=0.08
FINAL_READOUT_SCOPE=nonembed
```

## Method

This candidate is built on the PR311-style optimizer stack with PR309
EMA-Nesterov plus Aurora and Circuit-Muon on the attention V/O pair.

The fixed endpoint mechanics are:

- Capture a BroadDelta reference for `muon_other` at step `2000`.
- Apply the BroadDelta pulse at step `2400`.
- Capture the final readout anchor at step `2400`.
- Switch the late schedule at step `2600` to a `3010` horizon.
- Capture the TrailDelta and phase `t0` references at step `2650`.
- Apply the first TrailDelta pulse at step `2750` and capture phase `t1`.
- Apply the final TrailDelta momentum pulse at step `2850`.
- Apply normalized orthogonal phase readout on `muon_other`.
- Evaluate after an 8% non-embedding readout toward the step-2400 anchor.

All transforms are fixed before the run and use no validation feedback.

## Reproduction

Run the submitted seeds:

```bash
records/track_3_optimization/results/20260529_tail_phase_readout_2850/run.sh
```

Run a shorter check:

```bash
SEEDS="8 12" records/track_3_optimization/results/20260529_tail_phase_readout_2850/run.sh
```

Recompute the reported statistics:

```bash
python3 records/track_3_optimization/results/20260529_tail_phase_readout_2850/summarize_submission_stats.py --step 2850 \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed0.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed1.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed2.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed3.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed4.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed5.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed6.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed7.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed8.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed9.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed10.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed11.txt \
  records/track_3_optimization/results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed12.txt
```

## Credits

This submission builds on the existing Track 3 optimizer lineage, including the
PR311 recipe, PR309 EMA-Nesterov, Aurora, Circuit-Muon, and prior TrailDelta and
final-readout experiments.

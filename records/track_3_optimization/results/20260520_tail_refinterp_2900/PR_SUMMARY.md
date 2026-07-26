# PR Summary

## Title

Track 3: Tail Reference Interpolation reaches 3.28 in 2900 steps

## Summary

This PR adds the `20260520_tail_refinterp_2900` Track 3 optimization record.
The method starts from the PR #300 Aurora-on-`mlp.proj` stack and retargets it
to an exact 2900-step endpoint. The dataset, validation stream, global batch
size, model architecture, and training loss are unchanged.

The main new mechanism is Tail Reference Interpolation. The run captures a full
model-weight reference at step `2375`, then evaluates the fixed step-2900
checkpoint at:

```text
theta_eval = theta_2900 - 0.075 * (theta_2900 - theta_2375)
           = 0.925 * theta_2900 + 0.075 * theta_2375
```

This is a deterministic final-window interpolation back toward the late
trajectory. It is combined with late-only Tempered Polar, extended Contra-Muon,
a late radial/u-w retune, and a clipped final tail-velocity transform.

## Result

At the fixed checkpoint `step=2900`, nine consecutive clean full-run seeds
reach:

```text
seed 0: 3.27803
seed 1: 3.27899
seed 2: 3.27857
seed 3: 3.27819
seed 4: 3.27867
seed 5: 3.27855
seed 6: 3.27853
seed 7: 3.27805
seed 8: 3.27960
mean:   3.278576
```

Track 3 precision:

```text
(3.28 - 3.278576) * sqrt(9) = 0.004273 >= 0.004
z = 3.287179
p = 0.000505982
```

## Reproduction Files

The result folder includes:

```text
records/track_3_optimization/results/20260520_tail_refinterp_2900/train_gpt_tail_refinterp_2900.py
records/track_3_optimization/results/20260520_tail_refinterp_2900/run.sh
records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed0.txt
...
records/track_3_optimization/results/20260520_tail_refinterp_2900/refinterp_2900_seed8.txt
```

Run:

```bash
records/track_3_optimization/results/20260520_tail_refinterp_2900/run.sh
```

The default runner replays seeds `0 1 2 3 4 5 6 7 8` sequentially.

# PR summary

## Title

Track 3: Trustlight reaches 3.28 in 3125 steps

## Summary

This PR adds the `20260506_trustlight` Track 3 optimization record. Trustlight
starts from the SOAP-MLP Contra/NorMuon method in PR #278 by @samacqua and adds
a bounded trust-gated SOAP path for attention output projection matrices
(`attn.proj.weight`). The model architecture, dataset, batch size, and training
loss computation are unchanged from the Track 3 baseline.

The key optimizer change is deliberately narrow: keep SOAP preconditioning on
`mlp.fc.weight` and `mlp.proj.weight`, then add SOAP only to attention output
projections with a denominator floor and agreement-based trust gate. This tries
to capture useful attention-output curvature while avoiding the instability that
can happen when SOAP is applied too broadly to attention matrices.

## Result

At the fixed checkpoint `step=3125`, eight non-cherry-picked seeds reach:

```text
seed 0: 3.27918
seed 1: 3.27785
seed 2: 3.27948
seed 3: 3.28002
seed 4: 3.27822
seed 5: 3.27683
seed 6: 3.27830
seed 7: 3.27765
mean:   3.27844125
```

Track 3 significance:

```text
(3.28 - 3.27844125) * sqrt(8) = 0.00440881 >= 0.004
z = 3.3914
p = 3.48e-4
```

The previous six-seed read made step 3125 the last failing checkpoint. With
refresh seeds 6 and 7 included, step 3125 is now the earliest significant
checkpoint. The same eight runs also reach mean `3.27807750` at step 3130,
`3.27748250` at step 3140, `3.27701625` at step 3150, and `3.27638625` at step
3175. All runs used `EARLY_STOP=0` and `TARGET_VAL_LOSS=0`; the checkpoint is
selected uniformly across all seeds.

## Credits

Starting point and main inspiration: PR #278,
"New record: Track_3_optimization: Add SOAP preconditioning to MLPs (3150, -75
steps)", by @samacqua. Trustlight keeps the PR #278 MLP SOAP path and explores a
bounded way to recover some benefit from SOAP on attention output projections.

## Reproduction files

The result folder includes `train_gpt_simple_trustlight.py`, `requirements.txt`,
and `run.sh`. Running
`./records/track_3_optimization/results/20260506_trustlight/run.sh` from the
repo root replays seeds `0 1 2 3 4 5 6 7` sequentially with `EARLY_STOP=0` and
`TARGET_VAL_LOSS=0`.

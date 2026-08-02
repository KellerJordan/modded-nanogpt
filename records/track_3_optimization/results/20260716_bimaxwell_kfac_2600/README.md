# Bi-Maxwell + early dual-timescale output KFAC, 2600 steps

## Result

Twelve fixed GH200 seeds pass Track 3 at **2600 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2595 | 12 | 3.27903583 | 0.00333997 | 0.00400000 | fail |
| 2600 | 12 | 3.27877000 | 0.00426084 | 0.00400000 | pass |

The maximum passing mean for 12 seeds is `3.27884530`, so the measured mean
clears it by `0.00007530`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed11.txt`.

The result is also pairwise significant against the pooled 16-run result in
PR #339. At the same step, its mean is `3.28136812`, giving:

`(3.28136812 - 3.27877000) / sqrt(1/16 + 1/12) = 0.00680349`

Using the benchmark's 35-step extrapolation gives `0.00415541`. Both exceed
the required `0.004`.

## Method

The trainer keeps the Bi-Maxwell hidden-matrix momentum from the earlier
result. From step 1000 it combines fast and slow memories:

`m = 0.4385 * m_fast + 0.5615 * m_slow`

At step 1400, the learning-rate schedule end moves from 2900 to 2840.

Let `X` contain `N` sampled input activations to the output projection. The
current activation covariance is:

`C_batch = X.T @ X / N`

The output-head covariance is collected from step 1400 and used from step
1500. Maintain fast and slow covariance estimates:

`C_fast,t = 0.8 * C_fast,t-1 + 0.2 * C_batch,t`

`C_slow,t = 0.98 * C_slow,t-1 + 0.02 * C_batch,t`

Both estimates are initialized from the first `C_batch`. Combine them as:

`C = (8/9) * C_fast + (1/9) * C_slow`

If `r` is the damping fraction and `I` is the identity matrix, define:

`d = r * mean_eigenvalue(C)`

`P = (C + d * I)^(-1/2)`

The output gradient is multiplied by `P` on the right. Its norm is matched to
the raw gradient before mixing, and the mixed gradient is norm-matched again.
From step 1500 through 1750, mixing strength ramps from `0.25` to `0.75`, while
`r` ramps from `0.15` to `0.05`.

A weight EMA starts at step 2040:

`ema = ema + (weight - ema) / 150`

Validation uses `weight_eval = (1 - blend) * weight + blend * ema`. The fixed
blends are `0.90` for first-block matrices, `0.65` for other block matrices,
`1.00` for auxiliary parameters, and `0.60` for the output projection. The
token embedding is excluded.

The submitted trainer SHA-256 is
`87e2ad148e91fd05687c333249b628172addd641f95b27a1aaa7ae84f18d1247`.

## Reproduce

```bash
STOP_STEP=2600 torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20260716_bimaxwell_kfac_2600/train_gpt_bimaxwell_kfac_2600.py \
  --seed 0
```

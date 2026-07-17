# Bi-Maxwell + early dual-timescale output KFAC, 2600 steps

## Result

Twelve fixed GH200 seeds pass Track 3 at **2600 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2600 | 12 | 3.27877000 | 0.00426084 | 0.00400000 | pass |

The maximum passing mean for 12 seeds is `3.27884530`, so the measured mean
clears it by `0.00007530`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed11.txt`.

## Method

The trainer keeps the Bi-Maxwell hidden-matrix momentum from the earlier
result. From step 1000 it combines fast and slow memories:

`m = 0.4385 * m_fast + 0.5615 * m_slow`

At step 1400, the learning-rate schedule end moves from 2900 to 2840.

The output-head covariance is collected from step 1400 and used from step
1500. Its two memories are combined as:

`C = (8/9) * C_fast + (1/9) * C_slow`

Here `beta_fast = 0.8` and `beta_slow = 0.98`. From step 1500 through 1750,
preconditioner strength ramps from `0.25` to `0.75`, while damping ramps from
`0.15` to `0.05`.

A weight EMA starts at step 2040:

`ema = ema + (weight - ema) / 150`

Validation uses `weight_eval = (1 - blend) * weight + blend * ema`. The fixed
blends are `0.90` for first-block matrices, `0.65` for other block matrices,
`1.00` for auxiliary parameters, and `0.60` for the output projection. The
token embedding is excluded.

The submitted trainer SHA-256 is
`87e2ad148e91fd05687c333249b628172addd641f95b27a1aaa7ae84f18d1247`.

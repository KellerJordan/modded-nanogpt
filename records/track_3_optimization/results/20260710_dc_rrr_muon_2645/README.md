# Track 3: projected Tail-EMA, 2645 steps

This is a 15-step improvement over the open 2660-step result in PR #331.

## Result

- 19 consecutive seeds (`0-18`) on one GH200
- Step 2645 mean validation loss: `3.27903474`
- Formal statistic: `(3.28 - mean) * sqrt(19) = 0.00420748`
- Required statistic: `>= 0.004`
- Step 2640 fails with `0.00279887`; step 2645 passes

## Change

The optimizer stack from PR #331 is retained. Four late-training changes are
added.

**1. Schedule delay.** Delay the schedule clock by up to 180 steps between
steps 2250 and 2645:

```math
t_s = t - 180\min\left(1,\max\left(0,\frac{t-2250}{2645-2250}\right)\right)
```

**2. Tail-EMA initialization and update.** Initialize from optimizer velocity
`v`, update with `tau = 180`, and normalize each initial 2D row back to the
current row norm:

```math
e_0 = \theta - 0.5\tau v
```

```math
e_t = e_{t-1} + \frac{\theta_t-e_{t-1}}{\tau}
```

```math
e_{0,i} \leftarrow \|\theta_i\|_2\frac{e_{0,i}}{\|e_{0,i}\|_2}
```

**3. Parameter-group consensus.** Form the evaluation weights as:

```math
c_g = (1-\lambda_g)\theta_g + \lambda_g e_g
```

Here `lambda_g` is `1.00` for the first block, `0.80` for other early and
last-block matrices, `0.75` for later Muon matrices and the output projection,
and `0.90` for other auxiliary parameters.

**4. Row-norm projection.** Project each 2D consensus row onto its Tail-EMA
row-norm sphere:

```math
\hat\theta_{g,i} = \|e_{g,i}\|_2\frac{c_{g,i}}{\|c_{g,i}\|_2}
```

The dataset, batch size, architecture, and one-forward/backward-pass-per-step
rule are unchanged.

## Reproduce

```bash
bash records/track_3_optimization/results/20260710_dc_rrr_muon_2645/run_formal.sh
```

`summary.tsv` contains the formal aggregate. `GH200_seed0.txt` through
`GH200_seed18.txt` are the full logs with embedded source.

# Anchor-extrapolated gradient + fp32 master embedding, 2580 steps

## Result

Sixteen fixed GH200 seeds pass Track 3 at **2580 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2575 | 16 | 3.27926000 | 0.00296000 | 0.00400000 | fail |
| 2580 | 16 | 3.27896875 | 0.00412500 | 0.00400000 | pass |

The maximum passing mean for 16 seeds is `3.27900000`, so the measured mean
clears it by `0.00003125`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed15.txt`.

At the same step as PR #341 (2600), the mean is `3.27778313` versus
`3.27877000` (n=12), a difference of `0.00099`. That is not pairwise
significant at these seed counts.

## Method

The trainer is the PR #341 trainer with two changes to the late phase and a
retuned readout.

**Anchor-extrapolated gradient.** Let `e` be the Tail-EMA of the weights
(`e = e + (w - e) / 100`, from step 2040). From step 2100, each step's single
forward-backward pass is taken at

`y = w + g_t * (w - e)`

and the resulting gradient is applied to `w` by the unchanged optimizer. `g_t`
ramps linearly from `0` at step 2100 to `0.45` at step 2200. This is a
Nesterov-style lookahead whose displacement is the drift away from the weight
average used at readout.

**fp32 master embedding.** The token embedding is a bf16 parameter with bf16
Adam state. By step 2000 its entries reach `|w| ~ 9`, where most tail Adam
updates are below half a bf16 ulp and round away. The optimizer keeps an fp32
master copy and fp32 Adam state for it and writes the bf16 rounding of the
master to the parameter. The parameter dtype and forward pass are unchanged.
The embedding is included in the anchor `e` (readout blend `0`).

**Readout.** Tail-EMA `tau` is `100`. The fixed blends are `0.80` for
first-block matrices, `0.55` for other block matrices, `1.00` for auxiliary
parameters, and `0.50` for the output projection.

The submitted trainer SHA-256 is
`6ddd8ab6500740817bdf8d6817f197e76f1f17ddb1364baa6b0e24458c4fb6ec`.

## Reproduce

```bash
STOP_STEP=2580 torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20260930_anchor_fp32embed_2580/train_gpt_anchor_fp32embed_2580.py \
  --seed 0
```

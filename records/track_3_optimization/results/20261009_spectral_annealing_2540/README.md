# Spectral annealing + head eigen-Adam + anchor gradient, 2540 steps

## Result

Fourteen fixed GH200 seeds pass Track 3 at **2540 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2535 | 14 | 3.27911000 | 0.00333008 | 0.00400000 | fail |
| 2540 | 14 | 3.27878571 | 0.00454344 | 0.00400000 | pass |

The maximum passing mean for 14 seeds is `3.27893096`, so the measured
mean clears it by `0.00014524`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed13.txt`. These are consecutive seeds 0–13, evaluated on the same
five-step validation grid from step 2500 through 2600. All 14 runs are included.
The remaining 2 seeds from the original 16-seed queue are follow-up validation.

Against the previous submitted result,
[2580 steps in PR #377](https://github.com/KellerJordan/modded-nanogpt/pull/377),
the means at the same step (2600) are `3.27529714` for this result
and `3.27778313` for PR #377 (n=16), giving:

`(3.27778313 - 3.27529714) / sqrt(1/16 + 1/14) = 0.00679300`

The score is computed from the unrounded means and
exceeds the pairwise threshold of `0.004`.

## Method

The trainer builds on the 2580-step anchor-extrapolated gradient and fp32
master embedding setup in PR #377. This submission adds spectral annealing
and head eigen-Adam, disables the attention output projection trust gate from
step 1000, and moves the fp32 embedding switch from the start of training to
step 1000. The inherited anchor and readout are also described below.

**Spectral annealing.** After the hidden-matrix Newton–Schulz update `U`, rotate
into the SOAP row and column eigenbases `Qr` and `Qc`. Maintain slow gradient
covariances with decay `0.99`:

`Cr = EMA_0.99(G @ G.T)`, `Cc = EMA_0.99(G.T @ G)`

The curvature score is the outer product of `diag(Qr.T @ Cr @ Qr)` and
`diag(Qc.T @ Cc @ Qc)`. Take its log, standardize across entries, and clip to
`[-2, 2]` to obtain `s`. With `r = max(lr / lr_max, 0.3)`, form:

`U_new = Qr @ ((Qr.T @ U @ Qc) * r**(0.25 * s)) @ Qc.T`

Restore the Frobenius norm of `U`. This cools high-curvature directions earlier
than low-curvature directions as the global learning rate falls. The operation
starts when the LR first falls below its maximum, at step 514; the slow
covariances are initialized from that step's gradient. The submitted run uses
this original start time throughout.

**Head eigen-Adam (one-sided SOAP).** From step 1000, replace the output-KFAC
gradient mix and the head's diagonal Adam with Adam in a learned eigenbasis.
For the head gradient `G` with `V` vocabulary rows, maintain the 768-by-768
right covariance:

`C = EMA_0.95(G.T @ G / V)`

Compute its eigenbasis `Q` every eight steps. Adam updates its first and second
moments using `G @ Q`, then maps the update back with `Q.T`. At each basis
refresh, let `R = Q_old.T @ Q_new`; rotate the first moment with `m @ R` and
the diagonal second moment with `v @ (R * R)`. The switch initializes these
moments and the step counter from the head's existing Adam state.

**Plain SOAP for attn.proj.** The attention output projection trust gate is used
before step 1000. This submission disables it from step 1000 onward, so the
projection uses plain SOAP like the other hidden matrices.

**Delayed fp32 master embedding.** PR #377 used an fp32 embedding master from
the start of training. This submission keeps the original bf16 Adam updates
until step 1000, then initializes an fp32 master copy and promotes the existing
Adam moments to fp32 while retaining the step counter. Subsequent updates are
accumulated in the master and rounded to bf16 for the forward pass. The
embedding remains included in the anchor with readout blend zero.
Architecture, forward dtype, data and batch size are unchanged.

**Anchor-extrapolated gradient.** Let `e` be the Tail-EMA of the weights
(`e = e + (w - e) / 100`, from step 2040). From step 2100, the single
forward-backward pass is evaluated at `y = w + g_t * (w - e)`, with `g_t`
ramping from `0` to `0.45` by step 2200. The resulting gradient updates `w`.

**Readout.** Tail-EMA `tau` is `100`. Fixed blends are `0.80` for first-block
matrices, `0.55` for other block matrices, `1.00` for auxiliary parameters,
and `0.50` for the output projection.

The submitted trainer SHA-256 is
`97fa2afc2b5ec62acd71de64b8e38f89b11e77dd5820fcb2cb36b9ca1a3002b2`.
Each raw log embeds this exact source. Runs used one GH200 with PyTorch
`2.12.0+cu130`, batch size 524,288 tokens, and one forward-backward pass per step.

## Reproduce

```bash
STOP_STEP=2540 torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20261009_spectral_annealing_2540/train_gpt_spectral_annealing_2540.py \
  --seed 0
```

# MUDDFormer connections

Add MUDD (Multiway Dynamic Dense Connections, [Xiao et al. 2025](https://arxiv.org/abs/2502.12170)) to the last two layers, letting us cut **65 steps** off the schedule (1480 → 1415) while still hitting the 3.28 loss target.

```bash
                 Runs  Steps   Time μ   Time σ  Time +/-   Time p   Loss μ   Loss σ  Loss +/-   Loss p
  baseline-1480    10   1480  84.7039   0.0449    0.0000      nan   3.2788   0.0011    0.0000    0.0036
  baseline-1430    10   1430  81.6762   0.0611   -3.0277   0.0000   3.2867   0.0014   +0.0080    1.0000
     this_pr_v1    10   1430  83.2020   0.0326   -1.5019   0.0000   3.2783   0.0016   -0.0004    0.0040
this_pr_v1-1415    10   1415  82.1446   0.0342   -2.5593   0.0000   3.2815   0.0012   +0.0027    0.9982
     this_pr_v2    10   1415  82.1289   0.0386   -2.5750   0.0000   3.2781   0.0010   -0.0007    0.0001
     this_pr_v3    10   1415  81.9146   0.0418   -2.7893   0.0000   3.2780   0.0009   -0.0008    0.0000
```

`Loss p` is the one-sample t-test against the 3.28 target:
`scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue`
(so smaller `p` = more confidently below 3.28; the rules require `p < 0.01`).
- `this_pr_v3` at 1415 steps: `p = 0.0000` ✓ (best loss & time)
- `this_pr_v2` at 1415 steps: `p = 0.0001` ✓
- `this_pr_v1` at 1415 steps: `p = 0.9982` ✗ (v1 doesn't have enough headroom for the extra 15-step cut)

## v1: trimmed MUDD on last two layers

MUDD adds input-dependent dense connections between transformer blocks: for each target layer `i`, an MLP on the current hidden produces per-position weights over a set of earlier hiddens, and the weighted sum is injected back into the residual / attention inputs. The original paper routes a separate dynamic-weighted mixture into each of Q, K, V and the residual at every layer, which is too expensive at this model scale. v1 uses a trimmed variant that keeps most of the quality gain at much lower cost:

- **Only the last two layers** (`i ∈ {N-2, N-1}`) consume MUDD signals. Earlier blocks are unchanged, so the bulk of the forward/backward is untouched.
- **Small source-hidden subset per layer.** Layer `N-2` mixes over `{x0, h7, h9}` (3 sources, where `x0` is the pre-layer-0 hidden and `h9` is the current layer's post-MLP output — a self-reference); layer `N-1` only mixes over `{x0, h9}`. A much smaller subset than the fully-dense MUDD in the paper, which avoids quadratic growth in mixture cost and keeps most of the quality gain at this model scale.
- **Two channels instead of four.** We drop the Q/K MUDD adds from the paper and keep only `v_mudd` (added into V) and `residual_mudd` (added into the residual stream). The Q/K channels empirically contributed little while costing two extra routed mixtures per layer.
- **V-fusion, no extra projection.** `v_mudd` is produced in the same `(B,T,H,D_head)` shape as V and added *after* the existing QKV matmul, so MUDD does not introduce a new per-channel linear into attention.
- **Dense params on Adam, low LR.** `dense_w1` / `dense_w2` / `dense_bs` are replicated-Adam with `betas=(0.9, 0.99)` and `lr_mul=0.25` (no WD on the biases). `dense_w2` and `dense_bs` are zero-initialised, so at init the MUDD contribution (both `v_mudd` and the residual delta) is exactly zero — MUDD starts as a no-op and has to learn its way in.

Effect at matched steps: adding these connections at step 1430 recovers (slightly beats) the step-1480 baseline loss, while `baseline-1430` without MUDD is clearly above target (Δloss +0.0080, p≈1.0). The MUDD forward/backward adds ~1.5s vs `baseline-1430` (~1.05 ms/step), but since we now need 50 fewer steps the net wall-clock is **-1.5s vs `baseline-1480`** at equal-or-better loss.

## v2: more targets at almost-zero extra cost + cleanup (-2.58s, -65 steps)

In response to [@ClassicLarry's review](https://github.com/KellerJordan/modded-nanogpt/pull/259#issuecomment-4300623484), v2 folds in three of the suggestions and reuses the existing `dense_w1` / `dense_w2` stack to add two new MUDD targets that cost almost nothing per step but provide enough loss-per-step capacity to drop another 15 steps off the schedule (1430 → 1415):

1. **MUDD-style gate replaces `2 * sigmoid(x · w)` on layer 10's VE gate (Larry's suggestion #4, applied to one layer).**
   Layer 9's MUDD already produces a `(B, T, C, L)` weight tensor; we extend it to `(B, T, C, L+1)` and use the extra `(L+1)`-th column as the per-token VE gate consumed by **layer 10**'s attention (replacing the existing `2 * sigmoid(F.linear([x[:6], ve[:6]], ve_gate_w))` for that one layer). The 64-d `dense_w1` MLP is shared, so the extra cost is ~one column in the einsum and one extra `(B, T, C)` add — well under 100µs in the profile. The `(B, T, C)` gate is `repeat_interleave`-tiled across heads (`num_heads // C` heads per channel) and used directly (no `+ 2.0` shift wrapping the sigmoid; the MUDD path can re-learn whatever offset it needs through `dense_bs`).

   **Why only layer 10?** I also tried replacing the sigmoid VE gate on other layers with the same MUDD-coef gate, and it does help loss — but those layers don't have a MUDD `dense_w1` / `dense_w2` stack already, so each extra layer would need its own MLP+einsum (or a wider shared one), and the per-step overhead added more wall-clock than the extra loss-per-step bought back in steps. Layer 10 is the special case: the layer-9 MUDD MLP that feeds it is already in the forward, so the marginal cost of one extra `(L+1)`-th column is essentially free, and the loss improvement comes through net-positive on wall-clock. Other layers' VE gates stay on the existing sigmoid path.

2. **MUDD before `lm_head` also mixes the layer-1 VE bank.**
   Layer 10's residual MUDD source set grows from `{x0, h7, h9}` (v1) to `{x0, h7, h9, ve_bank0}`, where `ve_bank0` is the first value-embed slice (`ve[1]`, the same tensor layer-1 attention consumes via `ve_gate * ve`). Same `dense_w1` → `dense_w2` path with one more `L` slot; nearly free.

3. **Absorptions / fusions per @ClassicLarry's reply.**
   - **`(1 + m_r9) * x` self-reference fuse on layer 9.** The v1 generic loop produced an explicit `o3 += dw[..., 1, k_self] * x` term (one of the L sources is `x` itself); v2 folds this into a single multiply, matching Larry's `(1 + m9) * x` exactly:
     ```python
     x = (1 + m_r9) * x + m_r0 * x0 + m_r7 * h7
     ```
   - **`backout_lambda` is no longer a separate scalar.** Its slot is dropped from `self.scalars` and the `x -= backout_lambda * x_7` line before `lm_head` is gone. Instead, layer-10 / residual / `h7`-source bias is initialised to `dense_bs[1, 1, 1] = -4.34`, which after the `0.2 / sqrt(L)` MUDD scale gives an effective initial `≈ -0.5 * x_7` add into layer 10's residual — exactly the previous backout, but now per-token, learnable, and inside MUDD.

4. **Code refactor (Larry's request to make connectivity explicit).**
   The generic `mudd_layer_set` / `mudd_hidden_indices` / `hidden_append_set` loop is gone. v2 has two explicit blocks:
   ```python
   if i == self.num_layers - 2:    # layer 9
       # produces v_mudd (-> layer-10 V) + residual delta + next_ve_gate (-> layer-10 attn)
       ...
   elif i == self.num_layers - 1:  # layer 10
       # residual delta only, sources: {x0, h7, h9, ve_bank0}
       ...
   ```
   plus three plainly-named snapshot vars (`h7_snap`, `h9_snap`, `next_ve_gate`) and a docstring on `_init_mudd` that lists the sources per layer. Same arithmetic as the loop on the kept subset, fewer lines, and the data flow now reads top-to-bottom.

### Effect

vs. `baseline-1480` (target):
- **`this_pr_v2` reaches loss 3.2781 in 82.13s — `-2.58s` wall-clock at strictly-better loss** (Δloss `-0.0007` vs baseline-1480; one-sample `p = 0.0001` vs the 3.28 target, well under the `p < 0.01` rule).
- v1 at the same 1415-step schedule misses target (loss 3.2815, Δ `+0.0027`, one-sample `p = 0.9982` vs 3.28), confirming that the new MUDD targets — not just the step-count cut — are doing real work.

vs. v1 (`this_pr_v1` at 1430 steps):
- v2 saves another **1.07s** wall-clock (82.13s vs 83.20s) and slightly improves loss (3.2781 vs 3.2783) at 15 fewer steps.
- The new VE-gate / VE-mix MUDD adds are nearly free per step (~µs-scale, both reuse the existing `dense_w1` MLP), so essentially all of the v1→v2 wall-clock gain is the step-count reduction enabled by the extra capacity.

### Things I didn't try / would expect to squeeze more

- Putting MUDD on more than 2 layers (one extra mid-network block).
- The same MUDD-coef VE-gate replacement on layers other than 10 (would need an extra `dense_w1` / `dense_w2` per layer it covers).
- **Mixing more / different VE banks into the pre-`lm_head` MUDD.** Layer 10's residual MUDD currently pulls only `ve[1]` (the layer-1 VE bank) as its single VE source; a natural follow-up is to try `ve[k]` for other `k` (e.g. a deeper VE bank), or to widen the `L` dim further and let MUDD weight-mix several VE banks at once. The marginal cost per added VE source is one more `L` slot in the same `dense_w1` → `dense_w2` path, so it should stay nearly free per step.
- The same MUDD-coef approach on the 3→6 skip connection (Larry's suggestion #5).
- Tuning `lr_mul` / `betas` for the dense params; current values are a reasonable guess.
- `dense_w1` is kaiming-init; small-init might learn faster given `dense_w2` starts at zero.

## v3: dynamic layer-10 gates + skip_connection source (-2.79s, -65 steps)

v3 pushes the "reuse the existing MUDD MLP" theme further: layer 9's MUDD now produces **6 additional per-token scalar gates** that replace the static per-layer scalars on layer 10, plus a **5th residual source** (`skip_connection` from layer 3) for the post-loop MUDD. All of these ride on the same `dense_w1` → `dense_w2` path with extra columns, keeping per-step overhead negligible.

### Changes from v2

1. **6 dynamic gates for layer 10, produced by layer 9's MUDD.**
   `dense_w2[0]` extended from `(inter_dim, C=2, L+1=4)` to `(inter_dim, C=2, L+5=8)`. The 4 extra columns (beyond the existing ve_gate column) produce:
   - `resid_attn[10]`, `post_attn[10]` (C=0 channel, cols L+1, L+2) — replace the static `resid_lambdas_attn[10]` and `post_lambdas_attn[10]`.
   - `resid_mlp[10]`, `post_mlp[10]` (C=1 channel, cols L+1, L+2) — replace the static `resid_lambdas_mlp[10]` and `post_lambdas_mlp[10]`.
   - `x0_lambda[10]`, `bigram_lambda[10]` (C=0 channel, cols L+3, L+4) — replace the static `x0_inject[10]` and `bigram_lambdas[10]`.

   **Bias init** matches the static scalars they replace: `resid_attn/mlp` biased to `≈1.1^0.5`, `post_attn/mlp` to `1.0`, `x0_lambda` to `0`, `bigram_lambda` to `0.05` (all in pre-scaled domain, effective init = `bias * scale`). At init the model sees the same effective coefficients as before, but they can now vary per token.

2. **`skip_connection` (layer-3 output) as a 5th source for post-loop MUDD.**
   Layer 10's residual MUDD source set grows from `{x0, h7, h9, ve_bank0}` (v2) to `{x0, h7, h9, ve_bank0, skip_connection}`. `skip_connection` is the layer-3 hidden (a long-window snapshot, captured right after the layer-3 long-window attention). Same `dense_w1` → `dense_w2` path with one more `L` slot; nearly free.

3. **Layer-10 MUDD moved post-loop.**
   The `elif i == self.num_layers - 1:` branch is gone; the layer-10 MUDD residual is computed once after the main loop exits. This is a pure refactor (same arithmetic, cleaner control flow).

### Effect

vs. `baseline-1480` (target):
- **`this_pr_v3` reaches loss 3.2780 in 81.91s — `-2.79s` wall-clock at strictly-better loss** (Δloss `-0.0008` vs baseline-1480; one-sample `p < 0.0001` vs the 3.28 target).

vs. v2 (`this_pr_v2` at 1415 steps):
- v3 saves another **0.21s** wall-clock (81.91s vs 82.13s) and slightly improves loss (3.2780 vs 3.2781), same step count. The wall-clock gain comes from moving layer-10 MUDD post-loop (removing the dead `elif` branch inside the compiled loop) and the per-token dynamic gates' small quality improvement.
- Loss variance is tighter: σ = 0.0009 (v3) vs 0.0010 (v2).

### Things still not tried

- Putting MUDD on more than 2 layers (one extra mid-network block).
- The same MUDD-coef VE-gate replacement on layers other than 10.
- The same MUDD-coef approach on the 3→6 skip connection (Larry's suggestion #5).
- Tuning `lr_mul` / `betas` for the dense params.

Timing was on 8xH100.


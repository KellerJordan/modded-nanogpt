# MUDDFormer connections

Add MUDD (Multiway Dynamic Dense Connections, [Xiao et al. 2025](https://arxiv.org/abs/2502.12170)) to the last two layers, letting us cut 50 steps off the schedule while still hitting the 3.28 loss target.

```bash
                 Runs  Steps   Time μ   Time σ  Time +/-   Time p   Loss μ   Loss σ  Loss +/-   Loss p
  baseline-1480    10   1480  84.7039   0.0449    0.0000      nan   3.2788   0.0011    0.0000    0.5000
  baseline-1430    10   1430  81.6762   0.0611   -3.0277   0.0000   3.2867   0.0014   +0.0080    1.0000
        this PR    10   1430  83.2020   0.0326   -1.5019   0.0000   3.2783   0.0016   -0.0004    0.2460
```

MUDD adds input-dependent dense connections between transformer blocks: for each target layer `i`, an MLP on the current hidden produces per-position weights over a set of earlier hiddens, and the weighted sum is injected back into the residual / attention inputs. The original paper routes a separate dynamic-weighted mixture into each of Q, K, V and the residual at every layer, which is too expensive at this model scale. This PR uses a trimmed variant that keeps most of the quality gain at much lower cost:

- **Only the last two layers** (`i ∈ {N-2, N-1}`) consume MUDD signals. Earlier blocks are unchanged, so the bulk of the forward/backward is untouched.
- **Small source-hidden subset per layer.** Layer `N-2` mixes over `{x0, h7, h9}` (3 sources, where `x0` is the pre-layer-0 hidden and `h9` is the current layer's post-MLP output — a self-reference); layer `N-1` only mixes over `{x0, h9}`. A much smaller subset than the fully-dense MUDD in the paper, which avoids quadratic growth in mixture cost and keeps most of the quality gain at this model scale.
- **Two channels instead of four.** We drop the Q/K MUDD adds from the paper and keep only `v_mudd` (added into V) and `residual_mudd` (added into the residual stream). The Q/K channels empirically contributed little while costing two extra routed mixtures per layer.
- **V-fusion, no extra projection.** `v_mudd` is produced in the same `(B,T,H,D_head)` shape as V and added *after* the existing QKV matmul, so MUDD does not introduce a new per-channel linear into attention.
- **Dense params on Adam, low LR.** `dense_w1` / `dense_w2` / `dense_bs` are replicated-Adam with `betas=(0.9, 0.99)` and `lr_mul=0.25` (no WD on the biases). `dense_w2` and `dense_bs` are zero-initialized, so at init the MUDD contribution (both `v_mudd` and the residual delta) is exactly zero — MUDD starts as a no-op and has to learn its way in.

Effect at matched steps: adding these connections at step 1430 recovers (slightly beats) the step-1480 baseline loss, while `baseline-1430` without MUDD is clearly above target (Δloss +0.0080, p≈1.0). The MUDD forward/backward adds ~1.5s vs `baseline-1430` (~1.05 ms/step), but since we now need 50 fewer steps the net wall-clock is **-1.5s vs `baseline-1480`** at equal-or-better loss.

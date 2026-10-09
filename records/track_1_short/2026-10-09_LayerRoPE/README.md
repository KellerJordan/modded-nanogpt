# LayerRoPE: depth-conditioned rotary gains (−17 steps on record #92)

Record #92 (ANVIL2, PR #360) plus **[LayerRoPE](https://arxiv.org/pdf/2610.09179)**: every transformer block's branch inputs and branch outputs
are multiplied by a per-channel gain that is a smooth function of the block's depth. The four gain
vectors and sixteen gate scalars are shared by all layers (3,088 parameters in total). Every gain is
folded into a weight matrix, so LayerRoPE adds no per-token work. It lowers validation loss at a
fixed step count, and this PR spends that gain on **17 fewer scheduled steps (1105 instead of 1122,
1177 trained steps instead of 1194)** at the same validation loss.

## Result

Certification pool: alphagpu14 (8x H100 80GB HBM3, driver 580.126.09), one run of record #92 then two runs of
this PR, nine times; every run counts. Logs: `this_pr/` and `baseline/`; numbers: `this_pr/statistics.md`.

| | runs | wall (train_time) | final val CE |
|---|---|---|---|
| **this PR** (1105 scheduled steps) | **18** | **40.200 ± 0.022 s** | **3.27628 ± 0.00120** |
| record #92 (1122 scheduled steps; same machine, interleaved) | 9 | 40.562 ± 0.016 s | 3.27584 ± 0.00152 |
| **difference** | | **−0.362 s (−0.89%)** | +0.43 millinats |

- one-sided t vs the 3.28 gate: t = 13.2, p = 1.2e-10 (17 dof)
- wall: Welch t-test p = 6.1e-24; val vs record #92: p = 0.47 (no difference)
- every run: this PR 3.2740–3.2785, record #92 3.2737–3.2788
- this machine runs record #92 in 40.56 s against its published 39.9 s, so the expected wall time of this
  PR on the record's own machine is about 39.54 s

## Method

For block `l` (0..10, its index in the 11-layer stack) let `u = ln(l + 1)`. For each of four sites
`s` (attention input, MLP input, attention residual, MLP residual):

    gamma_eff[s, l] = exp(alpha_s + beta_s * u) * Rot(gamma_s, theta_s(l))
    theta_s,j(l)    = exp(alpha_rot_s + beta_rot_s * u) * 100 ** (-2j / 768)

`Rot` rotates each channel pair `(2j, 2j + 1)` of the shared vector `gamma_s` by angle `theta_s,j`,
the way rotary embeddings rotate query/key pairs by position; here the "position" is depth. The
magnitude gate `exp(alpha + beta * u)` and the rotation-speed gate `exp(alpha_rot + beta_rot * u)`
let each site learn a depth law for its gains (a power law in `l + 1`), while the 768-wide
`gamma_s` carries the per-channel pattern.

- input sites: the branch reads `gamma_eff ⊙ norm(x)` instead of `norm(x)`
- residual sites: the branch output is multiplied by `gamma_eff` before it is added to the residual
  stream (on top of #92's post-lambdas, which stay)
- every attention layer and every MLP (including the parallel MLP of layer 8) has all its sites; the
  layers without attention or without an MLP simply have no gains there

Init: `gamma_s = 1`, all 16 gates 0. Optimizer: Adam (betas 0.9 / 0.95), no weight decay. The gammas
follow the global lr envelope; the gates follow their own envelope at the base Adam lr (linear
warmup to 1.73x at the end of stage 0, then a cosine to 0.15x at the last step).

The learned gains have a clear, run-to-run stable structure (the first certification run; gain rms by layer):

| site | alpha | beta | alpha_rot | beta_rot | layer 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| attention input | 0.121 | 0.121 | −0.338 | −0.571 | 1.20 | 1.37 | 1.46 | 1.52 | 1.57 | 1.61 |
| MLP input | −0.329 | −0.102 | −0.349 | −0.928 | 0.67 | 0.60 | 0.57 | 0.55 | 0.54 | 0.52 |
| attention residual | −0.134 | −0.078 | −0.147 | −0.053 | 0.90 | 0.83 | 0.79 | 0.77 | 0.76 | 0.75 |
| MLP residual | −0.414 | −0.169 | −0.108 | −0.049 | 0.62 | 0.52 | 0.47 | 0.45 | 0.43 | 0.41 |

The attention input gains grow with depth and the MLP input and both residual gains shrink: on top of the
post-lambdas, the deeper blocks read their attention input harder and write into the stream more softly.
The rotation slopes are negative everywhere (the leading channel pairs rotate less with depth), most
strongly for the MLP input.

## Implementation: all gains live in weights

| site | folded into | training (fp8) | validation (bf16) |
|---|---|---|---|
| attention input | Q/K/V weight columns | the packed fp8 QKV cache holds `W * gain` | weight scaled |
| MLP input | `c_fc` columns | the fp8 `c_fc` caches hold `W * gain` | weight scaled |
| attention residual | O weight rows | joins the existing `sa_lambdas[1] * post-lambda` product with O | same |
| MLP residual | `c_proj` columns | the fp8 `c_proj` caches hold `W * gain` | weight scaled |

- The fp8 refresh kernels multiply each column by its gain while quantizing (one extra row of
  gains loaded per tile; no temporaries). One Triton launch per refresh computes every gain row,
  already in bank / slot order (`perf/kernels/layer_rope.py`).
- Gradients come from the reductions #92 already runs: the 0-D `sa_lambdas` / post-lambda dots
  become per-column dots in the QKV and O backward, with every layer's scale vectors built in one
  batched op, and the MLP bank's fold is one pass over the bank (`_FoldedColumns`). The compiled
  backward graph has one more kernel than #92's.
- Nothing per-token changes: the normed inputs stay ungated, so the shared fp8 activation copies
  (for example `norm(cache[7])` of layers 8 and 10) are still quantized once.

## Evidence

At record #92's own step count (1122 scheduled, 1194 trained), LayerRoPE lowered the final val by 2.6 millinats:
3.2739 ± 0.0012 over 11 development runs (gates at 5x lr) against #92's 3.2765 ± 0.0012 over 24 runs on the same
machines. This PR spends that gain on 17 fewer scheduled steps, and the certification pool above shows the loss
unchanged (+0.4 millinats, p = 0.47).

### Step count

Development runs of this configuration (gates at 1x lr), outside the pool:

- 1110 scheduled steps (−12): 3.2773 ± 0.0018 over 32 runs, plus one collapsed run (see Stability)
- 1105 scheduled steps (−17): 3.2771 ± 0.0017 over 22 runs
- record #92 at 1122: 3.2764 ± 0.0013 over 27 runs

All runs were on the same two machines. At −17 steps LayerRoPE stays within a millinat of #92 (+0.7 here,
+0.4 in the pool, neither significant).

### Stability: a rare degraded run

With this configuration family (beta = beta_rot = 0, gates at 1x or 5x lr, 1105–1122 scheduled steps), 2 of 75
development runs finished far above the rest: 3.3244 (gates 5x) and 3.3420 (gates 1x). Every other run landed in
3.2743–3.2800, except one at 3.2827. Record #92 showed no such run in our 18 runs or its own 44, though at this
rate that difference is not significant (p ≈ 0.3).

What we checked, on ~45 runs traced every step:
- **Not a race, a hardware fault or the validation.** The traces cover every rank's loss and local gradient
  norms, the gates, every fp8 scale and amax, and exact checksums after every step of every tensor that ranks
  must hold bit-identically (the banks, heads, fp8 caches, gain rows and replicated parameters). The checksums
  always agreed, and a second validation pass on the final weights always reproduced the first.
- **A slow divergence, not an event.** The traced 3.2827 run's gates left the normal range by step ~50. Its loss
  excess appeared from step ~120 and never closed (+0.2–0.3% in training loss). There was no gradient spike,
  no rank outlier and no fp8 saturation beyond the normal runs'.
- **Gate trajectories predict quality.** Across normal runs, final val correlates with the end values of the MLP
  rotation gates (|r| ≈ 0.5). It also correlates with the size of the gates' gradient spike at #92's stage-0→1
  transition (r = 0.67, n = 17), where the sequence length goes from 896 to 2048 and LayerRoPE roughly doubles
  #92's gradient spike.
- **The identity start does not fix it.** Starting the rotation gates at −5 keeps every rotation near 0
  throughout. Over 5 runs it averaged 3.2785 and still produced a 3.2821 run.

The two collapsed runs were untraced, so the mechanism of the large failures is not pinned down. The
certification pool (Result, above) counts every run.

In the certification pool none of the 18 runs degraded (range 3.2740–3.2785).

### Gate-init sweep (beta, beta_rot), at 1110 scheduled steps (gates 5x lr)

One run per cell except where noted; run-to-run sd is ~1.3 millinats. The magnitude slope is
load-bearing and sharply optimal at 0; the rotation slope is flat to worse away from 0.

| beta (magnitude) | beta_rot | final val |
|---|---|---|
| −3 | −3 | 3.3071 |
| −2 | −2 | 3.2951 |
| −1 | −1 | 3.2861 |
| −0.5 | −0.5 | 3.2818 |
| **0** | **0** | **3.2760 (n=4)** |
| +0.5 | +0.5 | 3.2991 |
| −3 | 0 | 3.3024 |
| −0.5 | 0 | 3.2830 |
| +0.5 | 0 | 3.2943 |
| 0 | −0.5 | 3.2769 |
| 0 | −1 | 3.2790, 3.2819 |
| 0 | −2 | 3.2766, 3.2798 |
| 0 | +0.5 | 3.2806 |

### What did not help

- Starting the gates at their learned end-of-run values instead of 0: 3.2860 at 1194 steps (the
  gates must start at identity gains).
- Gates at 10x lr instead of 5x (3.2736) or gammas at 2x lr (3.2730), at 1194 steps: no change.
- Cutting 22 steps instead of 17 (1100 scheduled, gates 5x): 3.2786.

### Adapting LayerRoPE to #92's architecture (all at 1110 scheduled steps; standard form 3.2771, n=6)

At init (all gates 0) LayerRoPE is not the identity: pair `j` rotates by `100^(-2j/768)` rad, so
the leading ~100 channel pairs start with gains between −0.3 and +1.4. And #92's stack is irregular
(layer 7 has neither sublayer, layers 4, 6 and 9 have no attention, layers 8 and 10 share one
attention input), so the layer index is a crude depth coordinate. None of the adaptations below beat
the standard form:

| variant | beta tied 0 | beta tied −0.5 |
|---|---|---|
| standard (index depth, `100^(-2j/768)` ladder, alpha_rot init 0) | 3.2771 (n=6) | 3.2818 |
| identity start (alpha_rot init −5: every gain starts within 1% of 1) | 3.2781, 3.2764; gates 1x: 3.2769, 3.2768, 3.2768, 3.2821, 3.2797 | 3.2874 |
| reversed frequency ladder (leading pairs rotate least) | 3.2768, 3.2794 | 3.2873 |
| per-branch depth (attention sites count attention layers 0–6, MLP sites MLP layers 0–9) | 3.2789, 3.2783 | 3.2840 |
| gains held fixed over #92's 298-step tail-averaging window | 3.2775, 3.2769 | |

## Reproduction

Same stack as record #92 (torch `2.10.0+cu128`, `kernels==0.16.1`, `huggingface-hub==1.29.0`,
driver 580.126.09, 8x H100 80GB). `bash run.sh`; the default is now 1105 scheduled steps.

## Files

- `track_1_short/model/gpt.py`: LayerRoPE parameters, gains, the fp8 refresh folds and `_FoldedColumns`
- `track_1_short/model/attention.py`: per-row O scale and per-column QKV scale
- `track_1_short/perf/kernels/layer_rope.py`: the refresh's gain rows (new)
- `track_1_short/perf/kernels/{mlp,fp8_attention_quant,qkv_rope}.py`: column gains in the fp8 quantizers; per-column QKV scale gradient
- `track_1_short/training.py`: optimizer entries and the gates' lr envelope
- `track_1_short/config.py`: 1105 scheduled steps
- `train_gpt.py`: logs the learned gates after the run

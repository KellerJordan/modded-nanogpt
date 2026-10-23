# Track 1: MoE and other tricks on top of #379 + #375 + #371 (31.29 s, n=12)

## The idea
We wanted to see whether a mixture of experts could help at all in a 30-second run. We did not expect it to: at this scale the router, the extra weights and the communication to keep them in sync usually cost more time than the extra parameters save. So the work went into making the MoE cheap rather than big: dropless top-1 routing, the experts as one more ANVIL bank sharded over the 8 GPUs and gathered in FP8, grouped Gluon GEMMs, and a few exact kernels around them.

Result: on the final stack the MoE saves about 0.7 s (−0.69 s and −1.0 millinats against the same stack with dense MLPs). It is a small gain, but more than we expected. Larger MoEs did not help (see What did not work).

| part | about | whose |
|---|---|---|
| CPLM + token normalization (the #391 combination) | 5.3 s | #379, #375, #391 |
| sampled LM-head gradient | 0.4 s | #371, #382 |
| mixture of experts | 0.7 s | this PR |
| kernels, four-map ANVIL cascade, step cut, two small ablations | 2.8 s | this PR |

## Result

| | runs | wall | final val |
|---|---|---|---|
| **this PR** | **12** | **31.285 ± 0.095 s** | **3.27823 ± 0.00155** |
| master (#373 layout) | 4 | 40.451 ± 0.016 s | 3.27477 |
| record 92 (monolith `f9b6266`) | 5 | 39.952 ± 0.031 s | 3.27792 |
| **delta vs master** | | **−9.17 s (−22.7 %)** | |

- t = −4.0, p = 1.1e-3 against 3.28; margin 1.8 millinats. Criterion fixed before the runs (981 steps, n ≥ 12, p < 0.01,
  otherwise 985 steps).
- MoE ablation in the same job: −0.688 s (p = 5e-12; −0.64 / −0.70 / −0.75 s per node), −1.0 millinats.
- Record 92 ran here +0.10 % from its published 39.914 s.

## What is new

Paired A/B on one node at the point of the stack where each was added; rows do not add up.

| change | wall | val (millinats) |
|---|---|---|
| FP8 mixture of experts on MLP layers 1, 3, 5, 6, 8 | −0.69 s | −1.0 |
| dense MLP Gluon G3 GEMM (fused ReLU², FP8) + ping-pong forward | −0.54 s, −0.19 s | equal |
| packed QKV backward | −0.39 s | equal |
| e4m3-LUT sampled cross-entropy | ≈ −0.25 s | equal |
| n-gram table exchange and sparse Adam rewrite (fp16 transport, fixed merge order) | −0.66 s | equal |
| optimizer and FP8 refresh kernels; late reduce-scatters | −0.19 s, −0.11 s | equal |
| grouped Gluon G3 for the experts | −0.22 s | equal |
| four-map ANVIL cascade (lets the expert bank run at margin 1.1) | −0.19 s | equal |
| layer 9 removed; attention output gate removed on layers 3 and 10 | ≈ −0.3 s, −0.21 s | equal, −2.2 |
| step cut 999 → 981 (rule fixed beforehand) | ≈ −0.75 s | ≈ +3.0 |

**The MoE.** On layers 1, 3, 5, 6, 8, eight ReLU² experts of hidden 1024 replace the 2816-wide MLP. A top-1 softmax
router scales the expert output by its probability (Switch balancing loss 0.01). Dropless: tokens sorted by expert into
128-row tiles, one grouped FP8 GEMM per projection, no capacity, so a token's output does not depend on the batch.
63 M expert parameters, 7.9 M active per token; MLP compute on those layers drops to about a third.

## Requirements and timing

Same stack as record 92 (torch `2.10.0+cu128`, `kernels==0.16.1`). NVRTC needs `cuda_bf16.h` (`CUDA_HOME`).

## Disclosures

- Runs on a cloud cluster of 8×H100 SXM 80GB nodes.
- CPLM's pointer band is 2048 in training and 8192 at validation, as in #379.

## Taken from other PRs

- **CPLM** (#379, @NathanGodey). One extra LM-head row, `<copy>`, whose mass α goes to earlier tokens of the same
  document through a 128-dim pointer attention with a learned sink: p(y) = p̃_lm(y)·(1 − α(1 − a_sink)) + α·p_copy(y).
  We moved the mixture into the sampled cross-entropy kernel.
- **Token normalization** (#375, @daniel-monroe). The n-gram hashes read normalized token ids (NFKC, no accents,
  lowercase). The map is #375's, shipped as `norm_map_pr375.npy` (rebuilt by `tools/build_norm_map.py`, no new
  dependency).
- **Sampled LM-head gradient** (#371, @romeerp, with #382's threshold, @krishaab). In the sampled backward, the positive
  part of the logit gradient is sampled 1 row in 4; the negative part and dX stay exact. We only added the `<copy>`
  column to the exact part.
- **#391** (@atveit) published the #379 + #375 combination first; we got there on our own route. **#390** (@atveit):
  the barrier before the clock. **Canonical mask** (#350, @jvarho): already on master.

## Files

`git diff 4ea6b93` touches `train_gpt.py` and 33 files of `track_1_short/`. New modules: `model/moe.py`,
`perf/kernels/{moe_mlp,moe_g3,mlp_g3,anvil_opt,cplm,sampled_head}.py`, `ngram_norm_map.py` with `norm_map_pr375.npy`.
This folder: logs in `this_pr/` and `baseline_master/`, and `tools/build_norm_map.py`.

## What did not work

- More experts: 16 experts +1.24 s / −4.9 millinats, 32 experts +3.68 s / −12.7 millinats; with their communication
  removed entirely, 32 experts still cost +2.4 s. FP8 reduce-scatter and PowerSGD with error feedback did not close it.
- MoE on all eight MLP layers: −0.6 s but +12 millinats. Twice the expert parameters at equal compute: equal val, +1.0
  to +1.4 s.
- Other gates (straight-through, ghost logit, sigmoid), hash routing, expert learning rates, MoE inside attention:
  all worse or flat.


**This was also possible thanks to [Flywheel](https://flywheel.paradigma.inc) and in general the Paradigma’s research infrastructure.**

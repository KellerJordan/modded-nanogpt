# CPLM on record #92: 36.009 s (0.600 min)

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB, track 1 (≤ 3.28 val CE). Built on record #92
(ANVIL2, [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360)); everything in #92 is unchanged
except the output distribution and the step count.

**Result (8×H100 SXM, same node, interleaved with record #92):**

| | runs | train time (mean ± sd) | final val CE (mean ± sd) |
|---|---|---|---|
| **this PR (CPLM, 1050 steps)** | **8** | **36.009 ± 0.039 s** | **3.2769 ± 0.0016** |
| record #92 (1194 steps), same node, interleaved | 13 | 40.575 ± 0.024 s | 3.2765 ± 0.0016 |
| **delta** | | **−4.566 s (11.25 %)** | +0.33 millinats |

- One-sided t-test vs 3.28: t = −5.39, **p = 0.0005** (7 dof). All runs of the shipped configuration count; none were excluded.
- Record #92 is 39.9 s as published; on this node it measured 40.575 s, so this node is ~1.7 % slower than #92's. The speedup above is the same-node comparison (rule 4).
- Logs: `this_pr/` (8 runs) and `baseline/` (13 runs) in `records/track_1_short/2026-10_CPLM/`.

**Cost of CPLM over #92** (analytic estimates from the shapes, per rank):

| | added | relative |
|---|---|---|
| parameters | 196,737 (2 × 128 × 768 + 128 + 1) | +0.35 % of the transformer blocks' matrices (~56 M); +0.15 % with embed + lm_head (~133 M); ~0 % of the 65 B total with the n-gram table |
| training memory | ≈ 0.1 GB at the largest batch (49,152 tokens/rank): pointer q/k and their pre-norm inputs saved for backward, fp32 dq/dk transients, ~3 MB of params + grads + optimizer state. The pointer kernel is flash-style, so no score matrix is stored | ≈ +0.2 % of #92's 48.4 GB peak |
| inference (validation) memory | ≈ 0.13–0.4 GB per 262,144-token validation batch (bf16 q/k plus fp32 norm temporaries); 8192-token band, still flash-style | ≤ +0.8 % of #92's peak |
| compute | ≤ 1.4 MFLOP/token forward (q/k projections 0.4 M + pointer scores ≤ 1.0 M at the full 2 × 2048 band; document masking makes it less in practice), ≤ 4.3 MFLOP/token for training | ≤ ~0.8 % of the model's ~0.5 GFLOP/token training cost |
| wall time per step | measured +1.3 % per step on 4×GH200; consistent with ~1 % here | |

The per-step overhead is small next to the 12 % fewer steps (1194 → 1050), which nets out to the measured 11.25 %. Measured peak allocated memory in the pool logs is actually lower for CPLM (35.9 GB vs 48.4 GB for #92); we have not investigated why, so the table above does not rely on it.

## Change

The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens:

    p(y) = p_lm(y) · (1 − α(1 − a_sink)) + α · p_copy(y)

- **Gate α**: the softmax mass of a `<copy>` slot, vocab id 50257 (one of lm_head's existing padding rows), under the
  same softcap as the other logits. No new output parameters; `p_lm` is renormalized over the real vocabulary.
- **Pointer `p_copy`**: one head (d = 128) over the final hidden state. It attends with document masking over
  previous tokens (a 2048-token band in training) plus a learned sink key. `p_copy(y)` is the attention mass on
  previous positions holding token `y`, so the pointer only copies tokens already seen (causal). The sink mass `a_sink` goes back to the LM.
- **QK-norm**: queries and keys are RMS-normed, with a learnable query gain. Without it the pointer logits blow up on some seeds.
- **Kernels**: in training the mixture runs inside #92's fused fp8 CE kernel (`perf/kernels/cplm_cross_entropy.py`), including the
  sampled-softmax stages. The pointer is a fused flash-style Triton kernel (`cplm_copy.py`).
- **Validation**: the same mixture over #92's full-softmax, canonically masked logits (`<copy>` is exempt from the
  mask), with an 8192-token pointer band.
- **Steps**: 1194 → 1050 (`NUM_SCHEDULED_ITERATIONS` 1122 → 978; the 72 growth + extension steps are unchanged).
  Stage boundaries scale with the schedule as in #92.

`CPLM=0 NUM_SCHEDULED_ITERATIONS=1122 ./run.sh` runs record #92 from the same tree; this is how the baseline pool was produced.

## Validity of the probability model

The README defines the target as a valid probability model over the val set. The mixture sums to 1 over the
vocabulary: (1 − α + α·a_sink) on the LM side plus α(1 − a_sink) on the pointer side. Pointer mass that lands on
canonically masked tokens is dropped, so on feasible tokens the mixture sums to *at most* 1, which can only raise
the loss. The validation NLL is computed as `−log(p + 1e−9)`. That is not exactly normalized: the floor adds at most
50257 × 1e−9 of mass. The normalized model `q = (p + 1e−9) / Z` scores at most `log(1 + 5.03e−5)` = **0.00005 nats**
above the reported value, which does not change any reported mean at 4 decimals or the significance result.

## Rules checklist

1. **Data pipelines untouched**: the token streams, data loader and validation tokens are byte-identical to #92.
   Only the model's output distribution changes. The longer pointer band at evaluation (8192 vs 2048 in
   training) falls under "evaluation at any sequence length".
2. **Mean val ≤ 3.28 at p < 0.01**: p = 0.0005 over 8 runs, all counted.
3. **No extra inductor / compile flags**: same flags as #92.
4. **Faster than the prior record on the same hardware**: 36.009 s vs 40.575 s, measured on the same node
   in the same session, runs strictly alternating.

**Discretionary rule 2 (loss buffer).** CPLM does not use up the buffer. Its mean val is 0.33 millinats
above #92's on the same node, which is not significant (Welch two-sample p = 0.66, n = 8 vs 13). At #92's own exchange
rate between steps and val (120–250 ms per millinat, ANVIL2 README), cutting #92's steps to match would save
only 0.04–0.08 s, against the 4.566 s measured here.

## Development history and step-count selection (full disclosure)

The step count was chosen on this node. Every completed run of every configuration is reported here; the
abandoned configurations' logs are in `dev_1035/` and `dev_1040/`, and the seed-0 smoke runs (#92 and CPLM at 1035 steps) are in `smoke/`:

| configuration | runs | val CE | mean | train time |
|---|---|---|---|---|
| 1035 steps (the GH200-tuned setting) | 3 | 3.2800, 3.2807, 3.2811 | 3.2806 | ≈ 35.55 s |
| 1040 steps | 4 | 3.2787, 3.2795, 3.2803, 3.2790 | 3.2794 | 35.69 s |
| **1050 steps (shipped)** | 8 | 3.2775, 3.2774, 3.2794, 3.2760, 3.2757, 3.2742, 3.2763, 3.2784 | 3.2769 | 36.009 s |

The method was developed on 4×GH200 (`ALLOW_4_GPUS=1`), where 1035 steps averaged 3.2775 over 8 seeds. On 8×H100
the same setting lands about 3 millinats higher. (On 4×GH200, the same method on top of record #91 also gave
10–15 % speedups; that was measured on GH200 only and was not re-measured on H100.) The likely cause, not confirmed by an ablation, is the size of the
sampled-softmax candidate pools. The 4-GPU path doubles them, because each rank holds twice the tokens, so the
training gradient differs. The model and schedule are otherwise identical. The GH200 runs also used a stock aarch64 FA3 with
zero-padded heads (`FA3_PAD=1`), which changes speed but not the math.

## Other disclosures

- **Shipped default vs logged source.** The pool was launched as `NUM_SCHEDULED_ITERATIONS=978 TRAIN_SEED=<n> ./run.sh`
  before the default was updated. The source embedded in each pool log therefore shows the old default (963), with
  the override applied from the environment. Each log confirms 1050 steps in its `step:N/1050` lines and its warmup step
  list. The shipped `train_gpt.py` differs from the logged source only in that default and its comment.
- **Setup fix.** The branch as developed crashed at setup on 8 GPUs: a function-local
  `from track_1_short.ngram_table import MAX_CYCLE_STEPS` under `ALLOW_4_GPUS` made the name an unbound local on the
  8-GPU path. That one line is deleted. The pool's logged source includes the fix.
- **Seeds.** Both arms were run seeded (`TRAIN_SEED`): CPLM 1..8, baseline 1..13. #92 certified on unseeded runs.
- **Compile caches** were warm after the first run of each configuration. Compilation, warmup and graph capture are untimed in both
  arms, exactly as in #92.
- **Untimed warmup**: the same single warmup pass as #92 (25 sampled steps, 6 step-graph configurations, the same counts as #92 on this node).

## Hardware and stack

- 8× NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, NV18 between every GPU pair; 2× Xeon Platinum 8480+;
  RunPod pod. No power-brake or thermal-slowdown events; power limit 700 W.
- Python 3.12.3 venv, torch 2.10.0+cu128, triton 3.6.0, kernels 0.16.1, huggingface-hub 1.29.0;
  libcudart.so.13 from `nvidia-cuda-runtime` 13.4.92; CUDA_HOME = CUDA 12.8 (for `cuda_bf16.h`).
  FA3 `devenpzak/flash-attn3-12864` @ 64c1e6d1, loaded offline per the ANVIL2 README; the in-code sha256 check passed.
  This is not the Dockerfile image. Both arms ran on the identical stack.

## Reproduce

```bash
python data/cached_fineweb10B.py 9
./run.sh                                           # this PR: 1050 steps
CPLM=0 NUM_SCHEDULED_ITERATIONS=1122 ./run.sh      # record #92 baseline from the same tree
python records/track_1_short/2026-10_CPLM/summarize.py records/track_1_short/2026-10_CPLM/this_pr/*.txt
```

## Files

- `train_gpt.py`: CPLM defaults (`CPLM=1 CPLM_QK_NORM=1 NUM_SCHEDULED_ITERATIONS=978 CPLM_EVAL_BLOCK=8192`); setup fix.
- `track_1_short/cplm_copy.py` (new): the fused pointer kernel.
- `track_1_short/perf/kernels/cplm_cross_entropy.py` (new): the mixture inside the fused fp8 CE kernel.
- `track_1_short/model/gpt.py`: the pointer head, the gate and the validation mixture.
- `track_1_short/training.py`, `track_1_short/config.py`: CPLM wiring and environment-driven configuration.
- `track_1_short/distributed.py`, `track_1_short/sampled_softmax.py`, `track_1_short/model/attention.py`: development
  paths for 4×GH200 (`ALLOW_4_GPUS`, `FA3_LOCAL_DIR`, `FA3_PAD`, `FA3_NATIVE`). They are off by default and unused by every run in this PR.
- `records/track_1_short/2026-10_CPLM/`: README, `summarize.py` (needs scipy), `this_pr/`, `baseline/`, `dev_1035/`, `dev_1040/`, `smoke/`.

Authors: @NathanGodey

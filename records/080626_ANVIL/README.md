# NanoGPT speedrun — record attempt

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB single node, targeting the modded-nanogpt gate:
**final val CE ≤ 3.28, mean over runs, all runs count.**

## Result (2026-08-08, unseeded certification pool)

Certified record shape: `KX_STEPS=1178` scheduled steps plus 40 grown mid-schedule
steps (the `run.sh` default; 1218 scheduled + 23 extension = 1241 trained steps).
Sixteen unseeded runs across 4 independent cloud 8xH100-80GB machines (three
Xeon 8480+ hosts, one Xeon 8468, all driver 580.126), fresh compile caches every
run, all runs counted:

| val CE (all 16 runs, machine-blind) | |
|---|---|
| 3.2765, 3.2764, 3.2757, 3.2769 | 3.2755, 3.2758, 3.2777, 3.2773 |
| 3.2750, 3.2768, 3.2762, 3.2754 | 3.2764, 3.2749, 3.2751, 3.2750 |

- **val CE: mean 3.27604** (n=16, sd 0.00087; one-sided t vs 3.2800: t=18.2, p < 1e-10)
- **wall: mean 64.95s** on the fastest machine (Xeon 8480+, driver 580.126; five runs:
  64.87, 64.91, 64.92, 64.99, 65.05); the other 8480+ hosts measured 65.43-65.72 and
  65.74-66.13 across their runs, the 8468 host 64.84-65.75
- Record #89, measured on the same machines the same session: 74.38 (fastest machine)
  and 75.05, a 9.4s improvement on identical hardware
- The previous entry of this lineage (65.87s mean, val 3.2783) improves by about 0.9s
  with a 2mb larger validation margin

## Files
- `train_gpt.py` — the trainer; all configuration baked in. Env: `KX_STEPS` (default 1178 pre-growth; the growth mechanism adds 40, see below),
  optional `KX_SEED` (reproduction only; records run unseeded), `DATA_PATH`.
- `fused_kernels.py` — fused softcapped CE (loss + e5m2 logit-grad in one kernel, fp8 lm_head
  cache, prefix-token aux CE), fp8 quantize/matmul machinery, optimizer matrix helpers.
- `value_embed_op.py` + `value_embed_kernel.py` — selected-load value-embedding backward
  (loads only the active adjoint plane; sha-pinned import).
- `bigram_kernels.py`, `dc_triton_kernels.py` — bigram embedding backward; dual-chunk
  attention correction.
- `run.sh` — exact launch line (fresh compile caches, tmpfs data).

## Method summary (vs record #89)
- **Optimizer: ANVIL (Averaged, Normalized Velocity with Isotropic Lanes)** — our rework
  of the record's Muon/NorMuon optimizer stack (lineage cited in the trainer docstrings).
  Twin-rail velocity: two EMA rails of the reduced gradient (fast rail on a scheduled beta
  with plateau 0.93, slow rail on a constant long-horizon beta; dual-timescale blend after
  PR #339) held in one `[2, *chunk]` fp32 state and blended into a Nesterov-lookahead
  velocity. A cascade of six quintic spectral maps on the velocity Gram — the Polar
  Express / Newton-Schulz family, with coefficients **re-derived from scratch** (CEM +
  minimax polish; composite envelope [0.9971, 1.0095] on σ ≥ 3e-3 vs ~[0.86, 1.14] for
  the classical 5-map schedule) — whitens the velocity in bf16. Per-lane energy
  equalization (the NorMuon rescale) at fixed Frobenius norm; sign-aligned (cautious)
  decoupled decay ×1.5; exact-fp32 commits on bf16 storage via a uint16 mantissa sidecar;
  and a **new bank tail-blend ship step** (fp32 EMA over the last ~300 steps, blended
  50/50 into the shipped banks). lm_head/embed on replicated-full Adam (one AVG
  all_reduce, no gather); embedding-table LR multipliers 70.
- **FP8 everywhere it pays**: full fp8 MLP forward+backward (dx/dW1/dW2) with statically
  clip-free scales; lagged comm-overlapped weight-cache quantize; fp8 lm_head cache feeding
  the fused CE in both layouts, refreshed by a tiled-transpose kernel.
- **Attention scale** retuned to 0.085 under the YaRN window-switch compounding
  (narrow query/key layers below use 0.12).
- **Value-embedding selected-load backward**: gather backward loads 1 adjoint instead of 5.
- **Bigram sparse-grad sink**: detached lookup + zeros-leaf; compact segment-sum exchange.
- **Schedule**: 3 stages (batch 8/16/24, seq 896/2048/3072) + terminal batch taper + 23
  extension steps at windows (6,13); cooldown fraction 0.80 to LR floor 0.20; windows
  1,3→3,7→5,11 through the stages, final-val long window widened to 20.
- **Timing note**: validation runs once, at the end. The prefix-table build, data-shard
  loading, batch fetching, and all terminal weight-averaging ship steps run inside the
  timed section (details in the `train_gpt.py` header); nothing that feeds the training
  result is off the clock.

## New since the previous entry
- **Narrow query/key attention**: 8 of 10 attention layers compute Q and K at head
  width 64 (V stays 128) with a 64-dim rotating rotary basis (32 distinct frequencies);
  layers 3 and 10 keep full width 128 as carriers. Scale 0.12 on the narrow layers.
- **Period-4 embedding cadence**: the value-embedding and bigram-embedding Adam channels
  update every 4th step from step 336; gradients accumulate losslessly across the cycle,
  so update content is preserved while communication and update cost drop 4x.
- **Virtual sequence cap**: from step 715 the train loader's attention metadata splits
  documents longer than 2048 tokens into virtual segments. This is attention masking
  only, the same class as the block-window schedules: the token stream, targets, and
  the entire validation pipeline are byte-identical to stock.
- **Schedule growth**: 40 steps inserted into the large-batch stage (earlier stage
  boundaries keep their exact step counts; the LR cooldown re-anchors to the grown
  length), buying the validation margin above at about 53ms per step.

Timing otherwise follows the standard convention: compile/warmup and validation are
untimed; the training section is wall-timed. Data pipeline, tokenization, and evaluation
are byte-identical to the speedrun standard.

# NanoGPT speedrun — record attempt

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB single node, targeting the modded-nanogpt gate:
**final val CE ≤ 3.28, mean over runs, all runs count.**

## Result

Certified record shape: `KX_STEPS=1178` scheduled steps plus 40 grown mid-schedule
steps (the trainer's default; 1218 scheduled + 23 extension = 1241 trained steps).

**Logged pool (2026-08-14, per-run logs in `this_pr/`)** — twelve unseeded runs of
the shipped source on a fresh 8xH100-80GB machine (Xeon 8480+, driver 580.126.09),
fresh compile caches every run, all runs counted, bracketed by four same-session
runs of the current-record `train_gpt.py` on the same machine:

- **wall: mean 64.72 s** (range 64.56-64.87, sd 0.10)
- **val CE: mean 3.27740** (n=12, sd 0.00138; one-sided t vs 3.2800: t=6.5, p = 0.00003)
- **same-machine record #89 baseline: 74.02 s mean** (n=4, logs in `baseline/`) —
  an improvement of **9.30 s / 12.6%** on identical hardware and session
- a second logged pool (16 runs, machine A, 2026-08-13) and the per-machine
  breakdowns are in `this_pr/statistics.md`

**Initial certification pool (2026-08-08, unlogged)** — sixteen unseeded runs
across 4 independent cloud 8xH100-80GB machines (three Xeon 8480+ hosts, one
Xeon 8468), fresh compile caches every run, all runs counted:

| val CE (all 16 runs; per-machine grouping in this_pr/statistics.md) | |
|---|---|
| 3.2765, 3.2764, 3.2757, 3.2769 | 3.2755, 3.2758, 3.2777, 3.2773 |
| 3.2750, 3.2768, 3.2762, 3.2754 | 3.2764, 3.2749, 3.2751, 3.2750 |

- val CE: mean 3.27604 (n=16, sd 0.00087; one-sided t vs 3.2800: t=18.2, p < 1e-10)
- wall: mean 64.95 s on that pool's fastest machine (five runs: 64.87-65.05);
  all-16 mean 65.36 s across the four hosts
- Record #89, measured in the same session on two of those machines: 74.38
  (fastest) and 75.05
- The previous entry of this lineage (65.87s mean, val 3.2783) improves by about 0.9s
  on the same machine class (fastest-machine means compared)
  with a 0.0023 larger validation-loss margin

## Files (all at the repo root; this folder holds the record documentation)
- `train_gpt.py` — the trainer, modified in place; all configuration baked in. Env:
  `KX_STEPS` (default 1178 pre-growth; the growth mechanism adds 40, see below),
  optional `KX_SEED` (reproduction only; records run unseeded), `DATA_PATH`.
- `fused_kernels.py` (new) — fused softcapped CE (loss + e5m2 logit-grad in one kernel,
  fp8 lm_head cache, prefix-token aux CE), fp8 quantize/matmul machinery, optimizer
  matrix helpers.
- `value_embed_op.py` + `value_embed_kernel.py` (new) — selected-load value-embedding
  backward (loads only the active adjoint plane).
- `bigram_kernels.py` (new) — bigram embedding backward.
- `dc_triton_kernels.py` — dual-chunk attention correction, unchanged from the current
  record. `run.sh` and the data pipeline are untouched; the standard `bash run.sh`
  reproduces.

## Method summary (vs record #89)
- **Optimizer: ANVIL (Averaged, Normalized Velocity with Isotropic Lanes)** — our rework
  of the record's Muon/NorMuon optimizer stack (lineage cited in the trainer docstrings).
  Twin-rail velocity: two EMA rails of the reduced gradient (fast rail on a scheduled beta
  with plateau 0.93, slow rail on a constant long-horizon beta; dual-timescale blend after
  PR #339) held in one `[2, *chunk]` fp32 state and blended into a Nesterov-lookahead
  velocity. A cascade of six quintic spectral maps on the velocity Gram — the Polar
  Express / Newton-Schulz family, with coefficients **re-derived from scratch** (CEM +
  minimax polish; composite envelope [0.9971, 1.0097] on σ ≥ 3e-3 vs ~[0.86, 1.14] for
  the classical 5-map schedule) — whitens the velocity in bf16. Per-lane energy
  equalization (the NorMuon rescale) at fixed Frobenius norm; sign-aligned (cautious)
  decoupled decay ×1.5; exact-fp32 commits on bf16 storage via a uint16 mantissa sidecar;
  and a **new bank tail-blend ship step** (fp32 EMA over the last ~300 steps, blended
  50/50 into the shipped banks). lm_head/embed on sharded Adam; embedding-table LR multipliers 70.
- **FP8 everywhere it pays**: full fp8 MLP forward+backward (dx/dW1/dW2) with statically
  clip-free scales; lagged comm-overlapped weight-cache quantize; fp8 lm_head cache feeding
  the fused CE in both layouts, refreshed by a tiled-transpose kernel.
- **Attention scale** retuned to 0.085 under the YaRN window-switch compounding
  (narrow query/key layers below use 0.12).
- **Value-embedding selected-load backward**: gather backward loads 1 adjoint instead of 5.
- **Bigram sparse-grad sink**: detached lookup + zeros-leaf; compact segment-sum exchange.
- **Schedule**: 3 stages (batch 8/16/24, seq 896/2048/3072) + terminal batch taper + 23
  extension steps at windows (6,13); cooldown fraction 0.80 to LR floor 0.20; windows
  1,3→3,7→5,11 through the stages, final-val long window at the stock 20.
- **Host-side systems work**: automatic Python garbage collection is frozen and
  disabled for the timed loop (the single collect runs after the final timer read);
  the run log is held in one buffered handle with step prints thinned to every 25
  steps. Both are host-only; neither touches numerics.
- **Timing note**: validation runs once, at the end. Mid-run validation is disabled
  by default (val_loss_every=0) purely to shorten total machine occupancy; validation
  passes are untimed either way, the validation code path is unchanged from the
  current record, and setting val_loss_every=250 reproduces the mid-run curve. The prefix-table build, data-shard
  loading, batch fetching, and all terminal weight-averaging ship steps run inside the
  timed section (details in the `train_gpt.py` header); nothing that feeds the training
  result is off the clock.

## New since the previous entry
- **Narrow query/key attention**: 8 of 10 attention layers compute Q and K at head
  width 64 (V stays 128) with a 64-dim rotating rotary basis (32 distinct frequencies);
  layers 3 and 10 keep full width 128 as carriers. Scale 0.12 on the narrow layers.
- **Period-4 embedding cadence**: the value-embedding and bigram-embedding Adam channels
  update every 4th step from step 336; gradients accumulate losslessly across the cycle,
  so update content is preserved while communication and update cost drop 4x. The
  channels' Adam betas are squared and the weight-decay multiplier doubled when the
  cadence engages, the per-update equivalents of the per-step settings.
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

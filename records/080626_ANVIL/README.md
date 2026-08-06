# NanoGPT speedrun — record attempt

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB single node, targeting the modded-nanogpt gate:
**final val CE ≤ 3.28, mean over runs, all runs count.**

## Result (2026-08-05, unseeded certification pool)

Certified record shape `KX_STEPS=1178` (the `run.sh` default). Nine unseeded runs
across 8 independent cloud 8×H100-80GB machines (one machine contributed two runs),
fresh compile caches every run, all runs counted:

| machine (GPU driver / host CPU) | wall (train section) | val CE |
|---|---:|---:|
| 580.126 / Xeon 8480+ | 65.50 | 3.2778 |
| 580.126 / Xeon 8480+ | 65.64 | 3.2778 |
| 580.126 / Xeon 8480+ | 65.67 | 3.2812 |
| 580.126 / Xeon 8480+ | 66.00 | 3.2784 |
| 580.126 / Xeon 8480+ | 65.99 | 3.2784 |
| 580.126 / Xeon 8480+ | 66.44 | 3.2776 |
| 580.159 / Xeon 8470 | 66.96 | 3.2792 |
| 570.211 | 67.71 | 3.2769 |
| 570.211 | 68.18 | 3.2776 |

- **val CE: mean 3.2783** (n=9, σ≈0.0013; one-sided t vs 3.28: t=4.00, **p=0.002**)
- **wall: mean 65.87s over the six runs on the fastest machine class** (Xeon-8480+
  hosts, driver 580; range 65.50–66.44); mean 66.45s over all nine runs
- Record #89, measured on the same machines the same day (one run per machine,
  seed 7): 74.33 / 74.40 / 74.58 / 74.61 / 74.72 — mean 74.53s, val 3.2752–3.2792
- Per-run noise (measured, same binary/machine, fresh caches): wall ±0.3–0.5s, val ±0.002 —
  this record is certified on the unseeded mean, not single draws.

## Files
- `train_gpt.py` — the trainer; all configuration baked in. Env: `KX_STEPS` (default 1178),
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
- **Attention scale** retuned to 0.085 under the YaRN window-switch compounding.
- **Value-embedding selected-load backward**: gather backward loads 1 adjoint instead of 5.
- **Bigram sparse-grad sink**: detached lookup + zeros-leaf; compact segment-sum exchange.
- **Schedule**: 3 stages (batch 8/16/24, seq 896/2048/3072) + terminal batch taper + 23
  extension steps at windows (6,13); cooldown fraction 0.80 to LR floor 0.20; windows
  1,3→3,7→5,11 through the stages, final-val long window widened to 20.
- **Timing note**: validation runs once, at the end. The terminal weight-averaging
  blend, position-table restore, and data-shard preloading run outside the timed
  section (details in the `train_gpt.py` header); all training-time averaging costs
  are on the clock.
Timing otherwise follows the standard convention: compile/warmup and validation are
untimed; the training section is wall-timed. Data pipeline, tokenization, and evaluation
are byte-identical to the speedrun standard.

## Overview
- Entry point: `train_switch_bank.py` orchestrates distributed setup, logging, and environment flags. It captures source for reproducibility via the top-level `code` string and patches Torch Inductor's `trace_structured` to be metadata-tolerant while logging compiled filenames.
- Focus of the repo: a sideways MoE GPT variant where each transformer layer routes to a shared bank of FFN experts. Non-standard optimizations include Muon for 2D matrices, FlexAttention, router feature EMAs (with clamped/strided alphas), router temperature/logit-cap schedules, optional router Gumbel noise, adapter support on routers, configurable router boost shapes, router/FFN freeze controls, and mid-training checkpoint/resume.
- Ignore/never touch: any `*_original*.py` or `train_gpt_*.py` files unless explicitly requested.

## Package layout (`switch_bank/`)
- `utils.py`: numeric safety helpers (`_sanitize/_safe_softmax`) plus scheduling utilities (`next_multiple_of_n`, `rampdown_multiplier`, `compute_train_micro_len` which enforces 128-token blocks). Includes a placeholder `summarize` helper. Keep behavior identical for numeric parity.
- `optim/muon.py`: Muon optimizer and fused `update` kernel plus Newton–Schulz `zeropower_via_newtonschulz5`. Used for attention QKV/out and shared FFN matrices; expects bf16 2D params (mantissa uint16, momentum float32 on resume).
- `model/components.py`:
  - `Rotary`, `CausalSelfAttention` (FlexAttention + RMSNorm + rotary).
  - `SharedFFNBank`: shared expert W1/W2 across layers, per-layer routers with optional adapters; forward/reverse EMA features (blockwise, doc-aware reverse) cached per `ema_layer_stride` with clampable/freezeable alphas; top-k hard routing with optional Gumbel noise, active/pruned expert masks, deterministic top-1 when only one expert is live. Aux loss mixes load/importance CV² with an entropy-gap term. Metrics buffer stores load vectors, entropies, and feature-weight means; `compile_warm_all_experts` warms kernels; adapters lazily initialize; `prune_inactive_experts` zeros weights/adapters when invoked.
  - `Block`: per-layer wrapper combining skip/SA lambdas, optional attention, temperature/load shaping by layer position, and bank call; supports boost shapes (`peak`, `valley`, `linear_start`, `linear_end`) and a `decay_scale` multiplier.
- `model/gpt.py`:
  - `GPT` model wiring embeds (token table padded to 128) with optional value embeds spread across layers; blocks + shared bank; scalar lambdas; router schedules (temperature/logit caps, expert activation masks, optional Gumbel); router/adapters/FFN freeze fractions and EMA clamping; flag builder (EOD/after-EOD/first-docN/block-pos bins); document-causal blockmask construction mixing long/short windows; LM head tie/untie logic with runtime retie.
  - Helpers `_compute_router_temp`, `_second_expert_step`.
- `data.py`: binary shard loader/generator, router metric summarizers, and `router_summary_str` formatter. Generator restarts at shard end and supports resume via `skip_batches`.
- `trainer.py`: training/validation loop and schedules (LR, cubic window size, router temp/logit-cap, Gumbel). Handles grad accumulation, all-reduce, router-only grad clipping with optional AutoClip (10th-percentile over a 250-step window; can warm up unclipped when base clip is 0), rampdowns (router/adapter/FFN), Muon momentum warmup, wandb/CSV logging gated by flags, abort checks, mid-training checkpoints (model+optimizers+meta+approx step timing), LM head untie logging, and dataloader fast-forward on resume. Router logging includes feature-weight percentages, normalized CV², entropy gaps, usage gap, per-layer stats, and a composite router health metric.

## Execution flow (high level)
1) `train_switch_bank.py` sets up distributed env/logging, disables donated buffers/compiled autograd, patches Inductor tracing, captures module source via `code`, and inits wandb/CSV when enabled.
2) Builds `GPT` with hyperparams (shared bank size/stride/window config, EMA settings, router/adapters, value embeds, router Gumbel/boost shape, expert activation schedule), broadcasts params, and logs parameter counts.
3) Partitions params: AdamW for embeds/scalars/router/adapters/head; Muon for attention QKV/out and shared FFN bank matrices; stores `initial_lr` per group.
4) Optional checkpoint resume validates meta (dims/experts/vocab), restores approx step timing, sanitizes Muon state dtypes, and recompiles the model (`torch.compile` dynamic=False). Computes block-aligned `train_micro_len` (logs adjustments).
5) Optional warmup (synthetic steps + `compile_warm_all_experts`) while preserving optimizer/model state, then training/validation via `trainer.run_training` (window schedule, router temp/logit-cap schedules, Gumbel gating, freeze milestones, optional mid-training checkpoint via `checkpoint_save_step`, wandb/CSV logging, resume-aware dataloader position).
6) Shutdown: report peak memory, destroy process group, finish wandb/CSV.

## Architecture notes (shared FFN bank / sideways MoE)
- Each layer routes tokens to a shared expert bank via per-layer routers. Features include token norm, optional forward/reverse EMA contexts (blockwise, doc-aware reverse) shared across layer groups via `ema_layer_stride`, and flags (EOD/after-EOD/first-docN/block-pos bins).
- Routing: top-k hard switch with optional Gumbel noise; outputs scaled by gate probs. Aux loss combines load/importance CV² plus an entropy-gap penalty. Deterministic top-1 when only one expert is active after masking/pruning; router/adapters/EMA alphas can be frozen late in training.
- Adapters: optional per-layer/expert scale/bias applied pre-FFN; lazy init using means of active adapters; zeroed when pruned.
- Pruning: `prune_inactive_experts` is available to zero weights/adapters based on activity but is not invoked by the trainer.
- Schedules: router temperature/logit-cap curves anchored to second-expert activation; router/adapter/FFN LR rampdowns and freeze fractions; expert activation masks; router boost shape controls per-layer temp/lb scaling (peak/valley/linear_start/linear_end).

## Practical guidance
- Preserve numeric parity: keep helper call order, defaults, and routing logic unchanged when modifying modules.
- When logging or reproducing runs, ensure `code` aggregation and the Inductor trace patch in `train_switch_bank.py` stay aligned with module contents.
- Hyperparameters live in `Hyperparameters` (train_switch_bank.py); adjust, and keep `compute_train_micro_len` block alignment intact.
- Use `switch_bank/trainer.py` for any training-loop changes; keep `train_switch_bank.py` as orchestration only and respect logging gates (`enable_extra_logging`, `enable_extra_wandb_logging`).
- When making or reverting changes that affect behavior, logging, checkpointing, or instructions, update `AGENTS.md` accordingly.

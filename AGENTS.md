## Overview
- Entry point: `train_switch_bank.py` orchestrates training, logging, and environment setup. It imports all core logic from the `switch_bank/` package; its top-level `code` string logs the contents of these modules for reproducibility.
- Focus of the repo: a sideways MoE GPT variant where each transformer layer routes to a shared bank of FFN experts. Non-standard optimizations include Muon for 2D matrices, FlexAttention, router feature EMAs, temperature/logit-cap schedules, adapter support on routers, configurable router boost shapes, and mid-training checkpoint/resume.
- Ignore/never touch: any `*_original*.py` or `train_gpt_*.py` files unless explicitly requested.

## Package layout (`switch_bank/`)
- `utils.py`: numeric safety helpers (`_sanitize/_safe_softmax`), scheduling utilities (`next_multiple_of_n`, `rampdown_multiplier`, `compute_train_micro_len`). Keep behavior identical for numeric parity.
- `optim/muon.py`: Muon optimizer and fused `update` kernel plus Newton–Schulz `zeropower_via_newtonschulz5`. Used for attention QKV/out and shared FFN matrices; expects bf16 2D params. Checkpoint resume sanitizes state dtypes (uint16 mantissa, float32 momentum).
- `model/components.py`:
  - `Rotary`, `CausalSelfAttention` (FlexAttention + RMSNorm + rotary).
  - `SharedFFNBank`: shared expert W1/W2 across layers, per-layer routers with optional adapters, forward/reverse EMA features (blockwise and doc-aware reverse), top-k hard routing, aux losses (importance/load CV², entropies), pruning support, router metrics buffering.
  - `Block`: per-layer wrapper combining skip/SA lambdas, optional attention, temperature/load shaping by layer position, and bank call; supports boost shapes (`peak`, `valley`, linear start/end).
- `model/gpt.py`:
  - `GPT` model wiring embeds, optional value embeds, blocks, shared bank, scalar lambdas, router schedules (temps/logit caps, expert activation masks), flag builder (EOD/after-EOD/first-docN/block-pos bins), blockmask construction, LM head tie/untie logic.
  - Helpers `_compute_router_temp`, `_second_expert_step`.
- `data.py`: binary shard loader/generator, router metric summarizers, and `router_summary_str` formatter. Supports resume via `skip_batches` for dataloader state.
- `trainer.py`: training/validation loop and schedules (LR, window size, router temp/logit-cap). Handles grad accumulation, all-reduce, rampdowns (router/adapter/EMA/FFN), Muon momentum warmup, pruning at freeze milestone, wandb/CSV logging, abort checks, mid-training checkpoints (model+optimizers+meta+step timing), and dataloader fast-forward on resume. Accepts injected `print0` and `log_param_counts_fn`. Router logging includes feature-weight percentages, normalized CV², entropy gaps, and a composite router health metric.

## Execution flow (high level)
1) `train_switch_bank.py` sets up distributed env/logging, reads and logs source code, initializes wandb/CSV if enabled.
2) Builds `GPT` with hyperparams (shared bank size/stride, EMA schedules, router/adapters, value embeds). Broadcasts params across ranks.
3) Partitions params: AdamW for embeds/scalars/router/adapters/head; Muon for attention QKV/out and shared FFN bank matrices.
4) Optional warmup (few synthetic steps, compile_warm_all_experts) while preserving optimizer/model state.
5) Training/validation delegated to `trainer.run_training`, which mirrors original control flow (window schedule, router temp/logit-cap schedules, gumbel gating, freeze milestones, expert pruning, logging), supports mid-training checkpoint save/resume (including step timing and dataloader position).
6) Shutdown: report param counts after pruning (if enabled), memory stats, destroy process group, close wandb/CSV.

## Architecture notes (shared FFN bank / sideways MoE)
- Each layer routes tokens to a shared expert bank via per-layer routers. Features include token norm, optional forward/reverse EMA contexts (layer-stride grouped caches), and flags (EOD/after-EOD/first-docN/block-pos bins).
- Routing: top-k hard switch; outputs scaled by gate probs. Aux loss combines load/importance CV² plus entropy terms. Deterministic top-1 when only one active expert.
- Adapters: optional per-layer/expert scale/bias applied pre-FFN; lazy init using means of existing active adapters.
- Pruning: `prune_inactive_experts` zeros weights/adapters for low-activity experts after freeze milestone.
- Schedules: router temperature/logit-cap curves anchored to second-expert activation; router/adapter/FFN LR rampdowns; expert activation masks. Router boost shape controls per-layer temp/lb scaling (peak/valley/linear).

## Practical guidance
- Preserve numeric parity: keep helper call order, defaults, and routing logic unchanged when modifying modules.
- When logging or reproducing runs, ensure `code` aggregation in `train_switch_bank.py` stays aligned with module contents.
- Hyperparameters live in `Hyperparameters` (train_switch_bank.py); adjust there unless experimenting with new configs.
- Use `switch_bank/trainer.py` for any training-loop changes; keep `train_switch_bank.py` as orchestration only.
- When making or reverting changes that affect behavior, logging, checkpointing, or instructions, update `AGENTS.md` accordingly.

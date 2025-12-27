# SwitchBank (Track 2 Medium)

This run uses the SwitchBank model instead of the baseline GPT-2 Medium stack. SwitchBank is a "horizontal MoE":
each transformer layer routes tokens into a shared FFN expert bank, while the attention stack remains per-layer.
This gives independent control over attention depth vs FFN capacity, and allows many layers to share the same expert weights.

## What is different vs train_gpt_medium.py
- Shared FFN bank across layers with per-layer routers (shared experts, independent routing).
- Top-k hard routing with optional Gumbel noise; auxiliary loss mixes load/importance CV2 and entropy gap.
- Router features include token norms plus optional forward/reverse EMA context and document-aware flags.
- Optional router adapters (per-layer/per-expert scale/bias) with lazy init and pruning hooks.
- Router temperature + logit-cap schedules anchored to second-expert activation.
- TurboMuon tweak.

## Routing/adapter specifics (SwitchBank)
- Each layer computes gates into the same expert bank; outputs are scaled by gate probabilities.
- Adapters can be enabled on routers and optionally frozen late in training.
- Routing supports temperature and logit-cap schedules, optional Gumbel exploration, and per-layer boost shapes.
- Bank-level metrics track load/usage/entropy and a composite router health score.

## Observations from nearly 1000 runs / ablations
- Reverse EMA features were much more effective than forward EMA (regardless) or both (by wall time).
- DeepSeek-style routing did not work here after extensive tuning. Best attempt was sigmoid-only routing
  (bias/update variants were worse); it still significantly underperformed by loss vs step and time. Router health did improve with sigmoid.
- TurboMuon was slightly better by step and slightly worse by time; router health improved, so it stayed enabled.
- Moving from 16 to 28 layers was a big win. It matched ~3.03 loss (best value at the time) in similar wall-clock time but in fewer steps,
  and the curve was steeper. With more time it reached the target loss and below. This highlights the value of
  SwitchBank: 1.75x layers for only ~1.16x parameters (~38.9M extra), and the ability to scale attention depth
  separately from FFN bank size.

## Notes
- Training/validation streams stay in order with the same FineWeb binaries as the baseline.
- Final validation uses the first 10,485,760 tokens of the validation set, per rules.

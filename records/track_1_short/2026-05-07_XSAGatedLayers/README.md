# XSA Gated Layers — Re-baseline on RunPod (2026-05-07)

Per-(layer, head) learnable gate on Cross-Self-Attention residual ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078)) on every non-paired attention layer (`{1, 3, 4, 7, 8, 10}`) of the WR #80 architecture.

`y ← y - α · (y·v̂) v̂`, with `α = tanh(xsa_alphas[layer, head])`. `xsa_alphas` is zero-initialized so `tanh(0) = 0` disables the gate at step 0; trained with Adam, no weight decay, replicated across ranks. The schedule is shortened by 30 steps: `num_scheduled_iterations: 1440 → 1410` (final-step total `1480 → 1450`).

## Result vs prior record (WR #80)

Both branches re-baselined on the same 8× H100 80GB RunPod box on 2026-05-07.

| | n | mean val_loss | sd | t (vs 3.28) | p(mean<3.28) | mean wall-time |
|---|---|---|---|---|---|---|
| **xsa-gated PR @ 1410** | **11** | **3.2789** | 0.0012 | **−3.10** | **0.00561 ✓** | 85.22 s |
| master (WR #80) @ 1440 | 7  | 3.2799 | 0.0005 | −0.67 | 0.264 | 85.90 s |

Welch one-sided t-test on wall-time, PR < master: **t = −20.36, df = 13.1, p = 1.36e-11, delta = −0.68 s**.

## Rule check ([README "Rules"](../../../README.md#rules))

| rule | status |
|---|---|
| #1 train/val data pipelines untouched | ✓ (only `train_gpt.py` model changes) |
| #2 mean val_loss ≤ 3.28, p<0.01 | ✓ p = 0.00561 (one-sample t, n=11) |
| #3 no extra `torch._inductor.config`/`torch.compile` flags | ✓ |
| #4 faster than prior record on same hardware | ✓ p = 1.36e-11 (Welch t, n=11/7), −0.68 s |

## Reproduction

To re-verify from these logs:

```bash
python3 records/track_1_short/2026-05-07_XSAGatedLayers/analyze_sweep.py \
  --pr     records/track_1_short/2026-05-07_XSAGatedLayers/xsa-gated-s1410 \
  --master records/track_1_short/2026-05-07_XSAGatedLayers/master-s1480
```

`analyze_sweep.py` is scipy-free; it implements the regularized incomplete beta function for the t-distribution lower-tail in pure Python.

## Hardware / software environment

This RunPod box, **not** the Dockerfile. Differences from the official validation env (`Dockerfile`: `cuda 12.6.2-cudnn-devel-ubuntu24.04`, Python 3.12.7, torch nightly cu126):

- 8× NVIDIA H100 80GB SXM, driver 580.126.09
- Python 3.12.3, torch **2.10.0+cu128** (stable, **not** nightly cu126)
- triton 3.6.0
- kernels **0.13.0** (downgraded from 0.14.0; 0.14 introduced a `trust_remote_code` check that rejects `varunneal/flash-attention-3` without `trust_remote_code=True`, which `train_gpt.py` does not pass)
- numpy 2.1.2, datasets 4.8.5, tiktoken 0.12.0, huggingface-hub 1.14.0
- 9 fineweb10B train shards + 1 val shard

Both branches showed a +0.0005 upward shift in mean val_loss on this hardware vs the prior cluster (see [`../2026-04-29_XSAGatedLayers/`](../2026-04-29_XSAGatedLayers/)) — almost certainly numerics drift from the cu128 / kernels-0.13 stack, **not** anything specific to the change. The relative gap between branches is preserved (~−0.0007 here vs ~−0.0008 there). Official re-timing on PrimeIntellect with the Dockerfile env is expected to recover the prior-cluster regime.

## Files

- `master-s1480/` — 7 logs of the master branch (WR #80) at `num_scheduled_iterations=1440` (1480 total steps).
- `xsa-gated-s1410/` — 11 logs of the `xsa-gated-layers` branch at `num_scheduled_iterations=1410` (1450 total steps). Includes one cold-compile run (the smoke test) and 10 with warm torch.compile + triton/inductor cache.
- `analyze_sweep.py` — self-contained verifier.

## Prior cluster baseline

A separate sweep at `records/track_1_short/2026-04-29_XSAGatedLayers/` provides corroborating evidence on a different RunPod box with closer-to-Dockerfile numerics (n=10 each):

| | n | val_loss | p(mean<3.28) | wall-time |
|---|---|---|---|---|
| master @ 1410 | 10 | 3.2832 | ≈1 (fails) | 78.76 s |
| master @ 1440 | 10 | 3.2794 | 0.115 | 80.40 s |
| xsa-gated PR @ 1410 | 10 | 3.2786 | **0.00137** ✓ | 79.85 s |

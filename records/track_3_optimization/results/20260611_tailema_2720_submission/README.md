# Track 3: Tail-EMA readout reaches 3.28 in 2720 steps

## TL;DR

Not a new optimizer: this is the PR #321 stack (aux-β2 + SOAP-f1 clean,
2750 steps), **byte-identical through step 2000**, plus one mechanism — a
streaming exponential moving average of the weights, blended into the
parameters once at the claimed step. Training stops at 2720 (fixed in advance
for every run; all schedule constants stay at PR #321's values).

**Result: n = 10 non-cherry-picked seeds (0–9), mean 3.278565 at step 2720,
significance (3.28 − mean)·√10 = 0.00454 ≥ 0.004.**

2720 is −30 steps vs the PR #321 candidate (2750), and −10 below our sibling
tail-EMA submission at 2730; the only differences from that recipe are the
claim step and an earlier EMA start (the refinement that bought the margin —
see below).

## Algorithm

The entire change relative to PR #321:

```
EMA_START = 2000     # EMA initialization step (>= 5*TAU before the claim)
TAU       = 150      # EMA time constant, in steps
LAMBDA    = 0.6      # blend strength at the claim step
S         = 2720     # claim step; training stops here, fixed for all runs
scope     = every parameter except the token embedding

# During training — after the optimizer update of each step t:
if t == EMA_START:
    ema = copy(weights)
elif t > EMA_START:
    ema = ema + (weights - ema) / TAU      # standard EMA update

# Once, immediately before the validation at step S:
weights = (1 - LAMBDA) * weights + LAMBDA * ema
# report val loss at step S; stop.
```

Intuition: under a decaying LR the late iterates orbit the local minimum
rather than sit at it; the EMA averages that orbit noise away (Polyak/tail
averaging). A full replacement (λ=1) hurts because the EMA lags the
still-descending mean, so the optimal blend is partial.

## Why these constants are not tuned magic numbers

**τ = (t_end − S) / p.** The natural averaging window for SGD under a
decaying LR is the schedule's local timescale η/|η′|; for the PowerCool
anneal η ∝ (t_end − t)^p (p = 1.2) this gives (2900 − 2720)/1.2 = 150. The
rule was verified by measuring full (τ × λ) surfaces on two trajectories with
different horizons (predicted 142 → measured 150 at t_end = 2900; predicted
183 → measured 200 at t_end = 2950).

**λ ≈ 0.55.** The optimal blend of a noisy-unbiased estimator (the last
iterate) and a clean-biased one (the EMA) is the shrinkage weight
λ* = σ²/(σ² + bias²); at the bias–variance balance point that τ sits on,
bias² ≈ σ², so λ* ≈ 0.5. Measured: 0.5–0.6, flat, consistent across seeds.

**EMA start ≥ 5τ before the claim.** The EMA forgets its initialization as
e^(−Δt/τ); starting at 2400 (as in the 2730 sibling) leaves e^(−330/150) ≈ 11%
of the buffer on the init snapshot — a measurable bias. Starting at 2000
(4.8τ) removes it: measured −0.0002 to −0.0003 at every (step, λ) cell on two
seeds, which is what converts a 2730 claim into 2720.

## Reproduction

```
records/track_3_optimization/results/20260611_tailema_2720_submission/train_gpt_tailema_2720.py
records/track_3_optimization/results/20260611_tailema_2720_submission/run.sh
```

The recipe is baked into the script defaults; `run.sh` replays the official
seeds sequentially. Each log embeds the exact source. Per-seed results:

| seed | val @2720 |
|---:|---:|
| 0 | 3.27978 |
| 1 | 3.27925 |
| 2 | 3.27827 |
| 3 | 3.27764 |
| 4 | 3.27810 |
| 5 | 3.27664 |
| 6 | 3.27935 |
| 7 | 3.27979 |
| 8 | 3.27806 |
| 9 | 3.27877 |

```text
n = 10
mean = 3.278565
(3.28 - mean) * sqrt(10) = 0.00454
required = 0.004
```

Rule-5 note: the stopping step, τ, λ, and the EMA start step are fixed in
advance and identical for every run; no per-run validation-based decisions.
Seeds are consecutive 0–9 with no selection.
Hardware: 1×GH200 96GB (single-GPU runs; the script runs unmodified on 8×H100
via torchrun nproc 8). GH200 val numbers ran ~+0.0004 above PR #321's
published H100 logs in our paired checks, so this margin is conservative on
the benchmark's hardware.

> Note: PR #321's stack is bundled so the record reproduces standalone; this
> submission builds on (and is −30 steps below) that candidate.

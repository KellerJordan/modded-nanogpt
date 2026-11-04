# Record: Track 3 Optimization -- #46 + row gains (per-row length Adam) + 2.5× norm-gain EMA-Nesterov jump -- 2640 steps (n=16)

## TL;DR

Setup from record #46 ([PR #328](https://github.com/KellerJordan/modded-nanogpt/pull/328)), plus three changes, all in the optimizer:

1. **Row gains.** From Muon step 1000, each row of q, k, v, attn.proj and mlp.fc gets its own length optimizer. The Muon step only rotates a row (each row is rescaled back to its length), then each row length takes an Adam step on its own gradient. No Cautious Weight Decay on these matrices.
2. **Tail-EMA retune.** Start at step 2100 with τ = 200, instead of 2400 / 150. λ = 0.6 is unchanged.
3. **EMA-Nesterov jump on the RMSNorm gains is 2.5× the global one** (β = 0.75·η_t/η₀ over steps 300–1950). Every other parameter keeps #46's jump.

The architecture, dataset, batch size and evaluation are unchanged, and no parameters are added: a row's length is part of the existing weight matrix. There is one forward-backward pass per step.

**Result:** <!-- FILL from pr_summarize.py --> on **n = 16** fresh seeds (0–15), the Tail-EMA readout reaches mean `val_ema = 3.27xxx` at **step 2640**, so (3.28 − mean)·√16 = 0.00xxx ≥ 0.004. This is −50 steps against #46 (2690).

The mechanisms behind each change, with the full experiment history (path-efficiency measurements, the row-gain anatomy, and the rejected variants), are in the [experiment log](https://app.notion.com/p/yanghaocheng/Modded-Nano-GPT-Speed-run-Track-3-407e76e151bb8074b2eef29f813d75da?source=copy_link).

## Changes in detail

Notation: W_i is row i of a hidden matrix, g_i the same row of its gradient, and η_t/η₀ the PowerCool LR ratio of the Muon group.

### 1. Row gains (q, k, v, attn.proj, mlp.fc; from Muon step 1000)

In #46, a row's length changes only as a side effect of the orthogonalized Muon step, and the whole-matrix radius pin then removes any net change in the matrix norm. With row gains, for each gained matrix and step:

- Take #46's Muon update U (SOAP, radial damping, RowFloor, all unchanged). Apply W ← W − η U, then rescale each row back to its length before the step, so the step is purely angular per row. This replaces the whole-matrix radius pin on these matrices.
- Compute the gradient with respect to row i's length, γ_i = ⟨g_i, W_i⟩ / ‖W_i‖, measured at the weights before the step.
- Bias-corrected Adam on γ_i: m ← β₁m + (1−β₁)γ, v ← β₂v + (1−β₂)γ², with β₁ = β₂ = 0.99.
- Then W_i ← W_i · (1 − a·(η_t/η₀)·m̂_i/(√v̂_i + 10⁻¹²)), with a = 0.02.
- CWD is off on these matrices. mlp.proj and all other parameters are as in #46.

Code: the `gain` branch of `Muon.step`.

### 2. Tail-EMA

ema ← ema + (w − ema)/200 over steps [2100, 2900), on every non-embedding parameter. The readout is (1 − 0.6)·w + 0.6·ema, as in #46.

### 3. Norm-gain jump

#46's EMA-Nesterov moves every parameter by W ← W + β·b before the forward pass, with b an EMA (0.99) of the per-step displacement and β = 0.3·η_t/η₀ over steps 300–1950. Here β is multiplied by 2.5 for the RMSNorm gains only (`EMA_Nesterov.configure_group`).

## Evidence for each change

All runs are on 8×H200, against #46 reproduced on the same machines. Peak memory is 34.8 GiB (under the 80 GB of an H100). "Fork" means a fork from a step-k checkpoint of a #46-trajectory run, scored against a control fork from the same checkpoint (control fork noise SD ≈ 0.0001). Steps are converted from Δ val_ema with the local slope of the loss curve.

| Change | Measurement | Δ val_ema | ≈ steps |
|---|---|---|---|
| Row gains (lr 0.02; betas 0.9/0.95 at the time) | forks from step 1000, 3 seeds | −0.00081 ± 0.00010 | +12 |
| Row gains + Tail-EMA 2100/200 | full runs paired with #46 on seeds 0–7, at 2690 | −0.00154 ± 0.00038 (8 of 8 lower) | +23; passes at 2665 (n = 16) |
| Tail-EMA 2100/200 vs #46's 2400/150, on the row-gain trajectory | readout forks from step 2000 | −0.00055 | +9 |
| Norm-gain jump 2.5× | forks from step 50, 3 seeds | −0.00099 ± 0.00017 | +17 |
| No CWD + 2.5× jump vs the 2665 setting | 16 fresh full runs | | passes at 2640 (n = 8 and n = 16) |
| Row-gain betas 0.99/0.99 (was 0.9/0.95) | 16 full runs paired with the line above, at 2690 | −0.00051 ± 0.00030 | +7.6; passes at 2635 (n = 16) |

Ablations that shaped the choices:

- **The gain needs one length per row.** One length per matrix loses 32 steps.
- **It helps where a row length sets the output scale:** v/o +4.8, fc +6.7, q/k +0.7 (behind QK-norm), mlp.proj −3.7 (excluded).
- **The gain comes from lengths following their own gradient, not from larger weights.** Lengths frozen by the optimizer step (no CWD, same everything else) are 16 steps worse than #46; learned lengths are 13 steps better.
- **The norm-gain jump has an optimum:** 2× +12, 2.5× +17, 3× +14, 4× −15 steps. Path-efficiency measurements motivated it: in #46 the norm gains are the only group whose weight path is nearly straight before step 1000 (eff200/eff50 = 0.90–0.95, against 0.46–0.58 for the matrices). The jump amplifies that persistent drift by 1/(1 − β).

**The same changes also help on other bases** (full runs, n = 2 unless noted):
- rec-40 and rec-44: ≈ 30 steps;
- rec-30: ≈ 11 steps;
- PR #395: ≈ 29 steps at n = 8 (−0.00211 ± 0.00061; its 8-seed Track 3 step goes from 2460 to 2430).

## Validation runs

<!-- FILL: table of the 16 logs (seed, val_ema at 2635 / 2640 / 2650) and the rule per step, from pr_summarize.py -->

- 16 runs, seeds 0–15, all of them (none excluded). Each was run with this exact `train_gpt.py`; its source is at the top of each log.
- Each run stops at step 2650 and validates every 5 steps from 2600, the same in every run. The claimed step is the earliest step where the 16-run mean passes the rule.
- Hardware: 8× NVIDIA H200 per run. The global batch is the same as the benchmark's.

| step | mean val_ema (n = 16) | (3.28 − mean)·√16 |
|---|---|---|
| 2630 | | |
| 2635 | | |
| **2640** | | |
| 2645 | | |
| 2650 | | |

## Reproduce

```
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/results/20261104_rowgains_normjump/train_gpt.py --seed 0
```

The script reads `data/fineweb10B/fineweb_{train,val}_*.bin` relative to the working directory and writes `logs/<uuid>.txt`.

## Acknowledgements

This work was done at [Cerulean Labs](https://cerulean.works/). Compute was sponsored by the [National Center for High-performance Computing (NCHC)](https://www.nchc.org.tw/), Taiwan.

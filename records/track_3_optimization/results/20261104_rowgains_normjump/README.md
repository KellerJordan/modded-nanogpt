# Record: Track 3 Optimization -- #46 + row gains (per-row length Adam) + 2.5× norm-gain EMA-Nesterov jump -- 2635 steps (n=16)

## TL;DR

Setup from record #46 ([PR #328](https://github.com/KellerJordan/modded-nanogpt/pull/328)), plus three changes, all in the optimizer:

1. **Row gains.** From Muon step 1000, each row of q, k, v, attn.proj and mlp.fc gets its own length optimizer. The Muon step only rotates a row (each row is rescaled back to its length), then each row length takes an Adam step on its own gradient. No Cautious Weight Decay on these matrices.
2. **Tail-EMA retune.** Start at step 2100 with τ = 200, instead of 2400 / 150. λ = 0.6 is unchanged.
3. **EMA-Nesterov jump on the RMSNorm gains is 2.5× the global one** (β = 0.75·η_t/η₀ over steps 300–1950). Every other parameter keeps #46's jump.

The architecture, dataset, batch size and evaluation are unchanged, and no parameters are added: a row's length is part of the existing weight matrix. There is one forward-backward pass per step.

**Result:** on **n = 16** seeds (0–15, all runs made), the Tail-EMA readout reaches mean `val_ema = 3.27888` at **step 2635**, so (3.28 − mean)·√16 = 0.00448 ≥ 0.004. This is −55 steps against #46 (2690). An earlier, independent 16-seed batch of the same settings (logs without embedded code, not submitted) also passed first at 2635 (0.0053).

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

| step | mean val_ema (n = 16) | (3.28 − mean)·√16 |
|---|---|---|
| 2620 | 3.27988 | 0.00048 |
| 2625 | 3.27956 | 0.00176 |
| 2630 | 3.27922 | 0.00313 |
| **2635** | **3.27888** | **0.00448** pass |
| 2640 | 3.27854 | 0.00584 pass |
| 2645 | 3.27821 | 0.00716 pass |
| 2650 | 3.27795 | 0.00819 pass |

- 16 runs, seeds 0–15. These are all the runs made with this file; none were excluded.
- Each was run with this exact `train_gpt.py`; its source is at the top of each log.
- Every run stops at step 2650 and validates every 5 steps from 2600, the same in every run. The claimed step is the earliest step where the 16-run mean passes the rule.
- Hardware: 8× NVIDIA H200 per run (PyTorch 2.11, CUDA 13.0), about 284 ms per step. Peak memory is 30.0 GiB allocated and 34.8 GiB reserved, under the 80 GB of an H100. The global batch is the same as the benchmark's.
- The seed standard deviation at 2635 is 0.00134.

| seed | log | val_ema @2630 | val_ema @2635 | val_ema @2640 | val_ema @2650 |
|---|---|---|---|---|---|
| 0 | [log](H200_seed0_682805f2-2e8a-483e-a09e-94e3940162b2.txt) | 3.27871 | 3.27837 | 3.27802 | 3.27745 |
| 1 | [log](H200_seed1_a4682099-b95d-4766-b77a-910a35269acf.txt) | 3.27851 | 3.27815 | 3.27781 | 3.27723 |
| 2 | [log](H200_seed2_69f00619-3bcb-4a7d-86b5-5631f7e41f1d.txt) | 3.27920 | 3.27890 | 3.27856 | 3.27797 |
| 3 | [log](H200_seed3_39597760-471f-4dbd-8837-e466c1c69823.txt) | 3.27719 | 3.27685 | 3.27652 | 3.27591 |
| 4 | [log](H200_seed4_56adf4f4-dc60-4aa2-9336-4d1ebc83bc38.txt) | 3.28149 | 3.28115 | 3.28083 | 3.28024 |
| 5 | [log](H200_seed5_b418d3f9-2bd8-43f7-9f39-eeab9de98227.txt) | 3.27662 | 3.27625 | 3.27592 | 3.27533 |
| 6 | [log](H200_seed6_94b8c6af-413c-4eee-a681-be3fc75bd5b7.txt) | 3.28135 | 3.28104 | 3.28069 | 3.28008 |
| 7 | [log](H200_seed7_7a7c5b4f-31ae-4478-bf67-94fe1c23f992.txt) | 3.27914 | 3.27884 | 3.27847 | 3.27791 |
| 8 | [log](H200_seed8_51af7a9a-6d89-400a-9f69-14a569e593d9.txt) | 3.27923 | 3.27888 | 3.27856 | 3.27798 |
| 9 | [log](H200_seed9_c61555f8-7860-4a53-bba6-5e4f6bc57d25.txt) | 3.28024 | 3.27989 | 3.27955 | 3.27900 |
| 10 | [log](H200_seed10_df8d861a-b4c4-4812-8a0e-e7edc648c28c.txt) | 3.28039 | 3.28006 | 3.27973 | 3.27914 |
| 11 | [log](H200_seed11_c3a3fadf-3a74-4438-b32f-92e22b5e3955.txt) | 3.27837 | 3.27801 | 3.27768 | 3.27708 |
| 12 | [log](H200_seed12_d4d27a86-95a1-405f-b942-82445cc11b6a.txt) | 3.28050 | 3.28016 | 3.27979 | 3.27919 |
| 13 | [log](H200_seed13_6016ecd1-18fa-449d-8b1e-a14ccfecf703.txt) | 3.27903 | 3.27871 | 3.27836 | 3.27777 |
| 14 | [log](H200_seed14_0f394249-7333-4ca8-9c2c-3ead881f7e7f.txt) | 3.27856 | 3.27819 | 3.27786 | 3.27727 |
| 15 | [log](H200_seed15_cfd37550-ae62-4106-83c0-588cc84c0db3.txt) | 3.27896 | 3.27861 | 3.27830 | 3.27770 |

## Reproduce

```
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/results/20261104_rowgains_normjump/train_gpt.py --seed 0
```

The script reads `data/fineweb10B/fineweb_{train,val}_*.bin` relative to the working directory and writes `logs/<uuid>.txt`.

## Acknowledgements

This work was done at [Cerulean Labs](https://cerulean.works/). Compute was sponsored by the [National Center for High-performance Computing (NCHC)](https://www.nchc.org.tw/), Taiwan.

# Track 3: #395 + row gains from #398 (no norm-gain jump, row-gain LR 0.04) — 2430 steps (n=16)

**Result.** On **16 pre-registered seeds**, run once each with every run counted, the Tail-EMA readout reaches a mean
`val_ema_loss` of **3.27895 at step 2430**. That gives (3.28 − mean)·√16 = **0.00419 ≥ 0.004**. 2430 is the earliest
validation step at which the rule holds. The best open Track 3 claim, #395, is 2450 steps; its own 39 counted logs first
pass at 2445.

**Pairwise vs #395.** This is *not* pairwise significant at 2430: the README statistic is 0.00279. It reaches 0.004 at
**2440** (0.00412). So these logs show a valid 2430-step result, and show that the setup beats #395 at 2440. They do
not show that #395 could not reach 2430 at n=16 (see the README's discussion of #13 and #16). Against #398 (2635) and
#46 (2690), the pairwise statistic at 2430 is above 0.025.

`python3 claim_stats.py` recomputes every number in this section from the logfiles. It also checks that each logfile
embeds the submitted trainer byte for byte.

## Change vs #395

The trainer is #395's file (`train_gpt_soap_shampoo_adam_lr_2450.py`) plus a ~50-line diff. The diff ports one component
of #398 and leaves out the rest:

- **Row gains (from #398).** From Muon step 1000, the step on each `attn.q`, `attn.k`, `attn.v`, `attn.proj` and `mlp.fc`
  matrix only rotates each row: rows are rescaled back to their pre-step lengths. Each row length then takes its own
  Adam step on ⟨g_i, W_i⟩/‖W_i‖, with betas 0.99/0.99 and the matrix's LR schedule. These matrices skip #395's radius pin
  and Cautious WD; every other matrix keeps them.
- **Row-gain LR 0.04**, versus #398's 0.02.
- **No norm-gain jump.** #398 gives RMSNorm gains a 2.5× EMA-Nesterov lookahead. Here they keep the same 1.0× as every
  other parameter. The `configure_group` hook is still in the file but is a no-op at `ENA_G_MULT = 1.0`.
- **#395's own Tail-EMA**, not #398's 2100/200 retune.

Nothing else changed: schedules, SOAP-Shampoo, trust gate, head eigen-Adam, LRs and Tail-EMA are #395's. Credit for
the row-gains idea and its implementation goes to #398 (@yukicoder0509). Credit for the base goes to #395 (@newstzpz),
#388 and #384.

### Why this configuration

Each arm below was run on search seeds 9001–9004, which are disjoint from the claim seeds. Each is compared with a run
of #395 on the same seed, with Δ = #395 − arm at step 2450, so positive is better.

| Arm (on #395) | Δ val_ema at 2450 |
|---|---|
| #398 full stack (row gains + 2.5× jump + Tail-EMA 2100/200) | −0.00053 (n=2) |
| row gains + 2.5× jump, #395's Tail-EMA | +0.00094 (n=4, positive on 4/4) |
| … + row-gain LR 0.04 | +0.00157 (n=4, positive on 4/4) |
| row gains, no jump, LR 0.02 | +0.00157 (n=2) |
| row gains + 4× jump | +0.00074 (n=4) |

- On #395, #398's Tail-EMA retune cost 0.0013–0.0014 against the row-gains-only arm on both seeds where both ran.
- The jump added nothing.
- Row-gain LR 0.08 was worse than 0.04.

This submission combines the two best changes: no jump and LR 0.04. Single-seed search comparisons are noisy here: a
1e-4 relative perturbation of Muon's LR moves the final loss by up to about 1σ. The combination's own search-seed results
were +0.0030, −0.0005 and −0.0018 on 9001–9003. It was chosen for held-out testing by a rule fixed in advance, and the
held-out runs below are the evidence.

The search was run by an automated agent harness, an LLM proposing and implementing variants, in the same spirit as
#395.

## Seeds and runs

Both batches were pre-registered before any of their runs started: seeds, stop step, analysis and the rule that every
run counts. Each run used one H100 80GB HBM3 (`world_size=1`), PyTorch 2.11.0+cu130.

- **Batch A, seeds 5001–5008** (`STOP_STEP=2470`). This was a paired confirmation against our 8-run reproduction of
  #395 on the same seeds. The reproduction's mean at 2450 was 3.27848, and it passed at 2450.
  - This setup was lower on 7 of 8 seeds.
  - Mean Δ = **+0.00180** (about 40 steps at the README's 0.0045 per 100), sd 0.00108, paired t = 4.70 with df 7,
    one-sided p = 0.001.
  - The reproduction logs embed #395's trainer, so they are not included here; they are available on request.
- **Batch B, seeds 6001–6008** (`STOP_STEP=2450`). This was a second held-out batch, run to reach n=16.
  - Batch B is worse than batch A at every step, by about 0.0013. On its own it first passes at 2445.
  - The claim is the pooled 16, as pre-registered.

As in #395, the schedules are those of a 2,900-step run regardless of stop step: the LRs reach zero at 2,840.
`STOP_STEP` only ends the loop early, so the validations at 2400–2450 come from identical training whatever the stop
step. Validation runs every 5 steps from 2,400.

| seed | batch | val_ema @2430 | val_ema @2450 | first step < 3.28 | logfile |
|---|---|---|---|---|---|
| 5001 | A | 3.27752 | 3.27593 | ≤2400 | [H100_seed5001_4a6ed706.txt](H100_seed5001_4a6ed706.txt) |
| 5002 | A | 3.27875 | 3.27714 | 2415 | [H100_seed5002_4485fbf7.txt](H100_seed5002_4485fbf7.txt) |
| 5003 | A | 3.28120 | 3.27956 | 2450 | [H100_seed5003_40e14aba.txt](H100_seed5003_40e14aba.txt) |
| 5004 | A | 3.27984 | 3.27823 | 2430 | [H100_seed5004_27d06dd2.txt](H100_seed5004_27d06dd2.txt) |
| 5005 | A | 3.27815 | 3.27656 | 2410 | [H100_seed5005_8ac9fd03.txt](H100_seed5005_8ac9fd03.txt) |
| 5006 | A | 3.27689 | 3.27522 | ≤2400 | [H100_seed5006_cf9cdfe6.txt](H100_seed5006_cf9cdfe6.txt) |
| 5007 | A | 3.27715 | 3.27548 | ≤2400 | [H100_seed5007_f7e585b9.txt](H100_seed5007_f7e585b9.txt) |
| 5008 | A | 3.27695 | 3.27535 | ≤2400 | [H100_seed5008_b6e4f43a.txt](H100_seed5008_b6e4f43a.txt) |
| 6001 | B | 3.27865 | 3.27701 | 2415 | [H100_seed6001_1a550f06.txt](H100_seed6001_1a550f06.txt) |
| 6002 | B | 3.27810 | 3.27645 | 2410 | [H100_seed6002_83437107.txt](H100_seed6002_83437107.txt) |
| 6003 | B | 3.27903 | 3.27739 | 2420 | [H100_seed6003_c5a63403.txt](H100_seed6003_c5a63403.txt) |
| 6004 | B | 3.28035 | 3.27872 | 2435 | [H100_seed6004_d5584af5.txt](H100_seed6004_d5584af5.txt) |
| 6005 | B | 3.28107 | 3.27947 | 2445 | [H100_seed6005_0b19b92a.txt](H100_seed6005_0b19b92a.txt) |
| 6006 | B | 3.27923 | 3.27764 | 2420 | [H100_seed6006_c3d69662.txt](H100_seed6006_c3d69662.txt) |
| 6007 | B | 3.27941 | 3.27778 | 2425 | [H100_seed6007_bba6d477.txt](H100_seed6007_bba6d477.txt) |
| 6008 | B | 3.28093 | 3.27931 | 2445 | [H100_seed6008_b39eb0cd.txt](H100_seed6008_b39eb0cd.txt) |

Pooled 16 seeds, steps 2420–2450:

| step | mean val_ema | sd | (3.28 − mean)·√16 | pairwise vs #395 |
|---|---|---|---|---|
| 2420 | 3.27968 | 0.00145 | 0.00127 | 0.00184 |
| 2425 | 3.27933 | 0.00146 | 0.00268 | 0.00228 |
| **2430** | **3.27895** | 0.00145 | **0.00419** pass | 0.00279 |
| 2435 | 3.27855 | 0.00145 | 0.00582 | 0.00340 |
| 2440 | 3.27811 | 0.00145 | 0.00758 | **0.00412** |
| 2445 | 3.27773 | 0.00146 | 0.00910 | 0.00465 |
| 2450 | 3.27733 | 0.00146 | 0.01069 | 0.00523 |

The pairwise statistic uses #395's 39 counted logs (mean 3.27888 at 2450, from its `summary.tsv`) and the README's
0.0045 per 100 steps.

## Other held-out tests (disclosure)

One other trainer was run on held-out seeds, and none of its runs are counted here. It is the same row gains with
#398's 2.5× jump and a split row-gain LR: 0.02 for q/k, 0.04 for v/o/fc.

- It was pre-registered at a fixed 2430 on seeds 6001–6008 and failed: statistic 0.0020, first passing at 2440.
- It was extended to 6001–6012, first passing at 2435.
- On the same seeds 6001–6008 at 2450, it is indistinguishable from this submission: Δ = −0.00026, paired t = −0.46,
  p = 0.66.

## Reproduce

```bash
STOP_STEP=2450 torchrun --standalone --nproc_per_node=1 train_gpt_rowgains_nojump.py --seed 5001
```

Use the data and environment of #395. Each run takes about 2 h on one H100 (2.88 s/step).

## Files

- `train_gpt_rowgains_nojump.py`: the trainer. sha256 `484e1c1fe47f2926d64166e82224b39e6bb16fa36f1de4ec170e4104694aafd9`.
- `H100_seed*_*.txt`: all 16 runs, one logfile each, each embedding the trainer.
- `summary.tsv`: per-run `val_ema_loss` at 2400–2450. Written by `python3 claim_stats.py --summary summary.tsv`.
- `claim_stats.py`: standard library only. It checks the logs and recomputes the claim, exiting 1 if any check fails.

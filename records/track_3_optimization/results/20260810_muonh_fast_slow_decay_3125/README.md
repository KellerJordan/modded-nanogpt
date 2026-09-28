# MuonH per-optimizer SOTA — fast-slow-decay schedule at 3125 steps

**TL;DR.** This submission improves the MuonH per-optimizer SOTA from **3150 steps** (previous MuonH SOTA, PR #345, power-0.4 cooldown) to **3125 steps** by replacing the single-shape MuonH LR cooldown with a four-phase, piecewise **fast-slow-decay** schedule:

1. **Warmup** — short linear ramp from 0 up to the peak LR.
2. **Plateau** — hold at the peak LR for a small number of steps.
3. **Fast decay** (steps 200 → 1750) — a concave power descent (`progress ** 0.6`) from the peak LR down to a "floor" LR. LR drops quickly at the top of this phase and then flattens out — the *fast* half of the decay.
4. **Slow decay** (steps 1750 → 3125) — long linear cooldown from the floor LR to zero — the *slow* half.

Only the MuonH schedule shape changes. Model architecture, data, global batch, single forward/backward per update, and AdamW (aux) settings are unchanged from PR #37.

## Exact MuonH LR schedule

For update index `t ∈ [0, train_steps)` with `train_steps = 3125`, the MuonH
effective LR is:

```text
Warmup     t ∈ [0,    100):   η(t) = 0.030 * (t + 1) / 100
Plateau    t ∈ [100,  200):   η(t) = 0.030
Fast decay t ∈ [200, 1750):   progress = (t - 200) / (1750 - 200)
                              η(t) = 0.030 + (0.006 - 0.030) * progress**0.6
                                   (concave: fast at start, flat at end)
Slow decay t ∈ [1750, 3125]:  progress = (t - 1750) / (3125 - 1750)
                              η(t) = 0.006 * (1 - progress)
```

## Statistical result

20 non-cherry-picked seeded runs (seeds 0..19), 8×H20, standard 524,288-token global batch, single forward/backward per update. All runs are validated at step 3125.

| batch | n | mean val_loss | std | sem | min | `(3.28 − mean) · √n` | pass ≥ 0.004? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **seeds 0..19** | **20** | **3.278994** | **0.00097** | **0.00022** | **3.27711** | **0.00450** | **✅** |

## Reproduction

CLI arguments used by the submitted configuration:

| Argument | Value | Meaning |
| --- | --- | --- |
| `--warmup_end` | `100` | End of Phase 1 (linear warmup `0 → peak_lr`). |
| `--plateau_end` | `200` | End of Phase 2 (hold at `peak_lr`). |
| `--fast_decay_end` | `1750` | End of Phase 3 fast decay (power descent `peak_lr → floor_lr`); start of Phase 4 slow decay. |
| `--peak_lr` | `0.030` | Top LR (end of warmup, on the plateau). |
| `--floor_lr` | `0.006` | LR at the start of the slow-decay phase (end of fast decay). |
| `--fast_decay_exponent` | `0.6` | Power exponent in the fast-decay phase (< 1 ⇒ concave / fast-then-slow). |
| `--slow_decay_schedule` | `linear` | Decay shape used in the slow-decay phase (`floor_lr → min_lr`). |
| `--min_lr` | `0.0` | LR at the very end of training (end of slow decay). |
| `--train_steps` | `3125` | Total training steps. |
| `--seed` | `0..19` | Seed used for the 20 submitted runs. |

Reproducing all 20 seeds:

```bash
for seed in $(seq 0 19); do
  torchrun --standalone --nproc_per_node=8 train_gpt_muonh_fast_slow_decay.py \
      --warmup_end 100 --plateau_end 200 \x
      --fast_decay_end 1750 --peak_lr 0.030 --floor_lr 0.006 \
      --fast_decay_exponent 0.6 --slow_decay_schedule linear \
      --min_lr 0.0 --train_steps 3125 --seed "$seed"
done
```

Model architecture, data, and float32 RMS normalization at the output projection are unchanged from prior accepted Track-3 entries. The submitted runs used 8×H20 with the standard 524,288-token global batch and a single forward/backward per update.

## Acknowledgement

Collaboration with @zzp1012, @Garios2, and @Juqiu-Wang.
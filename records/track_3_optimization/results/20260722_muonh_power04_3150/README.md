# MuonH with warmup, hold, and power-0.4 decay

**MuonH per-optimizer result: 3150 steps, 3.27779 (n=11 held-out seeds)**

This submission improves the accepted MuonH result #37 from 3250 to 3150 steps. The
MuonH learning rate warms from `1e-5` to `0.03` for 100 updates, holds `0.03` for 100
updates, and then decays with power `0.4` to the absolute floor `1e-5` at update 3200.
The auxiliary AdamW recipe is unchanged from #37.

## Statistical result

Seed 1 was used for the schedule sweep, so the primary significance calculation uses
the 11 runs that were not used to choose the schedule (seeds 0 and 2--11). All runs used
the frozen configuration and were continued to step 3250; 3150 is one common validation
step, not a per-run stopping point.

| Step | Mean val loss (n=11) | `(3.28 - mean) * sqrt(11)` | Pass? |
| ---: | ---: | ---: | :---: |
| 3125 | 3.27950818 | 0.00163118 | no |
| **3150** | **3.27778727** | **0.00733879** | **yes** |
| 3175 | 3.27654727 | 0.01145140 | yes |
| 3200 | 3.27571364 | 0.01421626 | yes |
| 3250 | 3.27512273 | 0.01617608 | yes |

The Track-3 condition is `(3.28 - avg_loss) * num_runs**0.5 >= 0.004`, so the
predeclared held-out set passes at 3150. Including the tuning run gives mean `3.27783333`
and statistic `0.00750555` over all 12 seeds, which leads to the same conclusion.

### Per-seed validation losses

| Seed | 3125 | 3150 | 3175 | 3200 | 3250 | Primary set? |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| [0](seed00_9ef56d0b.txt) | 3.27937 | 3.27768 | 3.27643 | 3.27561 | 3.27502 | yes |
| [1](seed01_ae36ca9d.txt) | 3.28000 | 3.27834 | 3.27708 | 3.27627 | 3.27566 | no (tuning) |
| [2](seed02_bb585f4e.txt) | 3.27954 | 3.27775 | 3.27649 | 3.27564 | 3.27505 | yes |
| [3](seed03_c759a278.txt) | 3.27800 | 3.27626 | 3.27500 | 3.27416 | 3.27356 | yes |
| [4](seed04_3ccd468d.txt) | 3.28187 | 3.28013 | 3.27888 | 3.27806 | 3.27747 | yes |
| [5](seed05_f37bd168.txt) | 3.27909 | 3.27737 | 3.27611 | 3.27529 | 3.27469 | yes |
| [6](seed06_d98b9d79.txt) | 3.28085 | 3.27913 | 3.27792 | 3.27707 | 3.27649 | yes |
| [7](seed07_2c8f5700.txt) | 3.27977 | 3.27806 | 3.27686 | 3.27601 | 3.27540 | yes |
| [8](seed08_eeb7d1db.txt) | 3.27884 | 3.27709 | 3.27584 | 3.27502 | 3.27443 | yes |
| [9](seed09_4a5225f5.txt) | 3.27910 | 3.27738 | 3.27614 | 3.27532 | 3.27475 | yes |
| [10](seed10_d9c349c3.txt) | 3.27931 | 3.27760 | 3.27640 | 3.27555 | 3.27496 | yes |
| [11](seed11_19770196.txt) | 3.27885 | 3.27721 | 3.27595 | 3.27512 | 3.27453 | yes |

The table uses the five-decimal losses printed in the raw logs. Computing from the
unrounded TensorBoard scalars gives `3.27778691` and `0.00734001` for the held-out set.

## Learning-rate schedule

For MuonH update index `s`, with `floor=1e-5` and `peak=0.03`:

```text
0 <= s < 100:     floor + (peak - floor) * s / 99
100 <= s <= 199:  peak
200 <= s <= 3200: floor + (peak - floor) *
                  (1 - ((s - 199) / (3200 - 199))**0.4)
s > 3200:         floor
```

Auxiliary AdamW keeps result #37's learning rates (`0.910`, `0.0064`, `0.0195`),
weight decay `0.001`, betas `(0.8, 0.95)`, epsilon `1e-10`, and linear cooldown fraction
`0.85`. MuonH momentum is `0.95` and matrix weight decay is zero.

## Comparison

| Result | Steps | Mean val loss | Runs | Status |
| --- | ---: | ---: | ---: | --- |
| Accepted #37 | 3250 | 3.2786 | 10 | Current accepted MuonH result |
| Open PR #343 | 3175 | 3.2775 | 9 | Minus-sqrt schedule |
| **This result** | **3150** | **3.27779** | **11** | Warmup + hold + power-0.4 decay |

Against accepted #37, the README's step-adjusted pairwise statistic is `0.01216`, above
the `0.004` threshold. Against open PR #343, this submission reaches the target 25 steps
earlier, but the pairwise statistic is only `0.00191`; therefore the evidence supports a
valid lower-step MuonH result, not a statistically significant head-to-head win over #343.

## Reproduction and raw evidence

The submitted runs used a 16-GPU fault-tolerant wrapper with a fixed global batch of
524,288 tokens and validation every 25 steps. Eleven runs started at step zero; seed 5
was preempted ([initial log](seed05_part0_70d55ca4.txt)) and resumed from its committed
step-500 checkpoint. Every run completed to step 3250 with the same effective-trainer SHA-256
`f4403ec2f8a1255f864809bd838e77f4fae0de80f95c85c4f1661c10671c877a`.

[`train_gpt_simple_muonh.py`](train_gpt_simple_muonh.py) is a self-contained version of
the computational recipe with the monitoring and checkpoint instrumentation removed. It
defaults to seeds 0--11, or accepts an explicit seed list, for example:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/results/20260722_muonh_power04_3150/train_gpt_simple_muonh.py \
  0 2 3 4 5 6 7 8 9 10 11
```

The raw logs are the exact outputs from the instrumented 16-GPU runs and include the
effective trainer source and complete frozen configuration for each seed.

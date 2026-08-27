# Norm CSE, fused FP8 up-projection quantize, and a trimmed extension tail

Track 1 submission against record #89 (`Track 1: Add FP8 MLP down projection`),
re-run on the same 8x H100 node. Three changes, +37 / -5 lines in `train_gpt.py`.

## Timing: paired, same lease

Every timing number below is a **paired same-lease comparison** — the prior
record and this PR measured on one machine, in one session, with the baseline
bracketed before and after the candidate so mid-lease drift is visible. Absolute
seconds are not compared across machines.

| | prior record (same lease) | this_pr | margin | 2*SE noise floor |
|---|---:|---:|---:|---:|
| lease A, n=10 each | 74.9270 s | 74.2178 s | **-0.7092 s** | 0.1190 s (**6.0x**) |
| lease B, n=10 each | 74.6815 s | 73.9323 s | **-0.7492 s** | 0.0863 s (**8.7x**) |

Mean margin **-0.729 s** over two independent on-demand leases. Baseline drift
within lease A was +0.0891 s and within lease B +0.0336 s, both well below the
measured margin.

## Loss: 20 independent runs

| | this_pr |
|---:|---:|
| n | 20 |
| steps | 1280 |
| mean val_loss | **3.27886** |
| val_loss std | 0.00149 |
| **p(mean < 3.28)** | **0.001408** |
| mean train_time | 73.3454 s |
| train_time std | 0.1422 s |

p-value by the method this repo links from the rules:

```python
import scipy.stats
print('p=%.6f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.001408
```

Each of the 20 runs is a **separate `torchrun` process** — fresh model
initialisation, no seeding, no state reset — so the spread is genuine inter-run
variance rather than run-to-run nondeterminism at a fixed init. Two collected
logs were excluded on stated criteria before any statistic was computed: one
preempted at step 1205/1280, and one whose final-step validation never reached
the file (6 validation prints instead of 7). Both exclusions are by log
completeness, not by value.

Only `this_pr` loss statistics are reported. Baseline loss was measured with a
repeat harness that restores identical initial state between repeats, which
measures a different quantity, and pooling the two would be misleading.

Official PrimeIntellect validation is pending; the above are same-hardware
Nebius numbers.

## Main changes

- **Common-subexpression elimination of `norm(cache[7])`.** `cache[7]` is
  written once and `norm` is pure, so layers 8/9/10 each recomputed the same
  full-size RMS norm. Hoisted into a per-forward memo: two redundant norms and
  their two redundant backward passes removed. Value-identical.
- **Fused FP8 re-quantize of the MLP up-projection.** The up-projection has run
  in FP8 since record 84; this does not change that, it fuses the re-quantize.
  The down-projection got a fused Triton path at record 89; the up-projection
  did not, because it lives in a module method that `torch.compile` does not
  reach. Wrapping it in a compiled helper fuses the divide and the cast, so the
  full-size bf16 intermediate is never materialised. Same arithmetic.
- **`num_extension_iterations` 15 -> 10.** These steps run after the cooldown
  completes, at final lr and window, so trimming them does not compress the decay
  schedule. Their measured loss cost is not negligible, though -- see the loss
  buffer section below.

## On the loss buffer (discretionary rule 2)

The third change reduces a step count, so the discretionary clause applies and it
should outperform a simple step-count decrease at equivalent loss. **That is not
demonstrated here, and the honest reading is stated rather than argued around.**

| change | val CE cost | wall-clock |
|---|---:|---:|
| cut 40 **scheduled** steps (1270 -> 1230) | +0.00671 | -2.32 s |
| cut 80 **scheduled** steps (1270 -> 1190) | +0.01050 | -4.63 s |
| cut 5 **extension** steps (15 -> 10) | +0.00099 (t = 1.82, p = 0.09, not significant) | -0.31 s |

The extension row is a paired same-lease measurement on the authoritative lease
`a2eb7e4b`: `repeat_bundle_ext10` at n=10 (mean val CE 3.277951) against
`repeat_bundle` at n=9 (3.276966), Welch SE 0.00054. Its wall-clock contribution
on that lease is 0.7092 - 0.3957 = 0.3135 s. The scheduled rows come from a
four-point sweep (1270 / 1230 / 1190 / 1150) across four hosts at n=4 with a
fixed-seed repeat harness.

Scheduled iterations cost 1.7e-4 val CE per step, so a scheduled cut costing the
same 0.00099 would buy roughly 0.34 s against the extension trim's 0.31 s. Those
two are indistinguishable, and the comparison crosses measurement tiers
(authoritative paired vs preemptible screen), so it establishes no superiority in
either direction.

What the sweep does establish is that scheduled-step reduction is not a route to
multi-second gains on this codebase: 1230 lands 17.9 sigma over the 3.28 gate.
The record's step count is genuinely tuned.

Changes 1 and 2 consume no loss buffer at all and were separately certified at
-0.3957 s on this same lease, and at a mean of -0.365 s across five independent
leases. If the extension trim is unwelcome, they stand on their own.

## Notes

- Train and validation token streams are unchanged (rule 1).
- No extra `torch._inductor.config` or `torch.compile` flags (rule 3). The one
  `@torch.compile` added is a decorator on a new helper function, matching the
  decorators already used throughout the file, not a global flag.
- Hardware: 8x NVIDIA H100 80GB SXM, Nebius eu-north1, PyTorch 2.10.0+cu128,
  Triton 3.6.0, CUDA 12.8, driver 570.211.01.
- Base commit: `75d6867`; rules as of `ecbb586`.

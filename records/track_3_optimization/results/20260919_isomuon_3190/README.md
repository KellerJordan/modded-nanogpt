# Track 3: IsoMuon — Muon with a noise-calibrated diagonal response metric; 3.28 in 3190 steps (n=8)

## TL;DR
IsoMuon takes Muon's Newton–Schulz polar decomposition in a diagonal response metric calibrated by the gradient's own sampling-noise variance: each output row and input column is weighted by the square root of its relative noise (bounded), so noisier channels are damped more. It adds two vector EMAs per matrix and negligible compute. Its two constants (λ = 0.5, c = 2) were chosen in a few single-seed runs during development (see the ablations) and were not tuned on this benchmark. On the tuned Muon + aux AdamW baseline (result #36, 3250 steps) it reaches 3.28 at **3190 steps** (mean 3.27842, n=8 non-cherry-picked seeds 0–7, `(3.28 − mean)·√8 = 0.00446 ≥ 0.004`). With bi-Maxwell momentum (PR #339) in the same script, IsoMuon keeps 93% of its gain and the two changes are close to additive (8 paired seeds per arm, 2026-09-29 update).

## What is new
Muon's orthogonalization moves every singular direction of the update at unit speed, whether or not its coordinates are noisy. In a language model the gradient of a hidden matrix is dominated by label-sampling noise, and its variance is strongly non-uniform across output channels (rows) and input channels (columns). IsoMuon measures that variance at no extra cost from the micro-batch gradients that every step already computes, and whitens the momentum update by a diagonal metric: each channel's relative noise raised to the power λ and clipped to [1/c, c],

    B = clamp((row_heat / mean)^λ, 1/c, c),  A = clamp((col_heat / mean)^λ, 1/c, c),  λ = 0.5, c = 2
    D = B^-1/2 · polar(B^-1/2 U A^-1/2) · A^-1/2,   ‖D‖_F re-aligned to ‖polar(U)‖_F

where `row_heat`/`col_heat` are EMAs (β = 0.95) of the row/column means of `Var_k[g_k]` over the micro-batches `k` of the step. With λ = 0 the metric is the identity and IsoMuon reduces to Muon up to the final Frobenius re-alignment, a scalar step-size factor. With λ = 0.5 each channel's damping is proportional to the standard deviation of its gradient noise rather than to its variance.

The estimator does not depend on the number of GPUs: the per-micro-batch squared-gradient row/column sums are all-reduced together with the gradient, so 1, 2, 4 or 8 GPUs give the same expected noise estimates.

## Origin: the endpoint-metric polar decomposition (EMP)

IsoMuon grew out of an earlier experiment of ours that was never submitted on its own, so we describe it here.
It was run on an older setup: a Muon trainer derived from the track-3 baseline script `train_gpt_simple.py`, run for 3500 steps on one A800 per run.

Muon's polar decomposition fixes the singular directions of the momentum. We had first tried rescaling the orthogonalized update channel by channel, using several per-channel statistics, and none of these moved the final loss: they change step
sizes but leave the directions, and so the trajectory, where they were. EMP instead changes the metric in which
the polar decomposition is taken:

    Y = B^-1/2 · M · A^-1/2,   D = B^-1/2 · polar(Y) · A^-1/2,   D re-aligned to the RMS of Muon's update
    A, B = clamp(exp(λ·(log h − mean log h)), 0.5, 2)

Here `h` is an EMA of the momentum's column energy for `A` and of its row energy for `B`. Because a weight gradient
is an outer product of the output gradient and the input, its column energy tracks the input second moment and its
row energy the output-gradient second moment, up to a common factor that the mean-log normalisation removes. So the
metric needs no extra hooks. With λ = 0 it reduces to Muon up to the same scalar re-alignment.

| EMP strength λ | Δ final loss vs Muon (3 paired seeds) | seeds improved |
|---|---|---|
| 0.125 | −0.00190 | 3 of 3 |
| 0.25 | −0.00206 | 3 of 3 |
| 0.375 | −0.00213 | 3 of 3 |

Same-seed run-to-run noise on that setup was about ±0.0005. At λ = 0.25, two of the three seeds crossed 3.28 25
steps earlier, and the mean wall-clock to the target was unchanged. The effect carried over to H200 (two seeds:
−0.00096 and −0.00490, the second crossing 50 steps earlier). Adding a rank-2 off-diagonal term to the diagonal metric gave nothing further (−0.00024, within noise).

IsoMuon keeps EMP's structure and changes only the statistic `h`: the gradient's sampling-noise variance instead of the
momentum's energy. On the #36 baseline the momentum-energy statistic recovers about half of IsoMuon's gain (−0.0021 and
−0.0019 on two seeds, against −0.0038 to −0.0041 for IsoMuon in the same set of runs). The sampling-noise variance is the better-founded choice, because the metric is meant to be calibrated by the gradient noise.

## IsoMuon stacked on the current SOTA

We first added IsoMuon to the strongest stack we had: the current record #46 (2690 steps) with the bi-Maxwell momentum of our PR #339 on top (2655 steps on A800, 2635 on H100). This made the stack clearly worse, and more so at larger λ: at λ = 0.25 the final loss rises by 0.014 and the run no longer reaches 3.28
within 2900 steps; at λ = 0.5 it rises by 0.033. Two seeds on RTX 5090 and one on A800 agree.

A gain that vanishes or reverses when added on top indicates that the new component draws on the same improvement as something already in the stack, so the two share one gain instead of adding up. To find which component, we ablated the stack. Removing SOAP alone loses the target: no run reaches 3.28. Putting IsoMuon in SOAP's place brings it back. The final loss is then only 0.004–0.005 above the unchanged stack, and 3.28 is first reached within 0–85 steps of it, consistently across 5 runs on three GPU types (A800, RTX 5090, H100). No other component
we removed behaved this way.

![stacking, ablation and wall-clock](figs/fig1_same_axis.png)

IsoMuon and SOAP therefore draw on the same information, and IsoMuon is useful here as a replacement for SOAP, not as an addition to the stack. With SOAP swapped out for IsoMuon, 3.28 is first reached 0–85 steps later (2700–2725, against 2625–2700 for the full stack on the same GPU type) with 22–43% less wall-clock.

To see what that shared information is, we also tried the full-matrix version of the noise-calibrated metric (the row- and column-side noise covariance matrices with bounded eigenvalues, so that channels can also rotate
into each other). It costs 1.85× the wall-clock and gives the same result within seed noise.
The usable part of the noise covariance is its diagonal, the channel scales, and the off-diagonal correlations add nothing, so IsoMuon keeps only the diagonal at negligible cost. SOAP also carries an eigenbasis, which may be why it cannot simply be replaced in general (section 3 of the update).

## Algorithm (the entire change to `train_gpt_simple.py`)
```python
# per step, inside the micro-batch loop (hidden matrices only):
gk = p.grad - prev            # this micro-batch's gradient
rs2 += gk.square().sum(1); cs2 += gk.square().sum(0)
# after all-reduce of the gradient and of rs2/cs2:
row_heat = rs2/K/n - (g/K).square().sum(1)/n     # Var_k[g_k] averaged over columns
col_heat = cs2/K/m - (g/K).square().sum(0)/m     # ... over rows
state.row_heat.lerp_(row_heat, 0.05); state.col_heat.lerp_(col_heat, 0.05)
# update:
bi = clamp((row_heat/row_heat.mean())**0.5, 0.5, 2).rsqrt()[:, None]
ai = clamp((col_heat/col_heat.mean())**0.5, 0.5, 2).rsqrt()[None, :]
P = newton_schulz((u * bi) * ai)
D = (P * bi) * ai;  D *= sqrt(min(m, n)) / D.norm();  D *= max(1, m/n)**0.5
```

## Results

![val loss near the target](figs/fig0_headline.png)

| run | seeds | mean val loss | `(3.28 − mean)·√n` |
|---|---|---|---|
| IsoMuon | 0–7 | 3.27842 @ 3190 | 0.00446 ✓ |
| baseline (#36 script, same GPUs) | 0–3 | 3.27973 @ 3250 | n/a |

Per-seed first crossing of 3.28 (validation every 5 steps from step 3000; the reported step is the earliest common step that passes the test): 3170, 3175, 3165, 3190, 3150, 3175, 3190, 3175 (val at 3250: 3.27517, 3.27580, 3.27483, 3.27679, 3.27342, 3.27541, 3.27707, 3.27569; mean 3.27552, sd 0.00114).
Pairwise vs #36 (n=10, 3.2787 @ 3250) with the step adjustment defined in the track-3 README: `(3.2787 − 3.27842 + 60/100·0.0045) / √(1/8 + 1/10) = 0.00628 ≥ 0.004` (statsig); at equal step count 3250, `(3.2787 − 3.27552) / √(1/8 + 1/10) = 0.00670` (statsig). Same-GPU baseline (the same file with `--isomuon 0`, seeds 0–3): 3.27975, 3.27933, 3.27980, 3.28005 at 3250 (mean 3.27973), i.e. IsoMuon is 0.0042 lower at the same step on the same GPUs, and every IsoMuon seed ends below every baseline seed.

Ablations (single-GPU runs of the #36 setup, seed-paired): λ = 0.25 / 0.5 / 1.0: −40 / −50 / worse than baseline; row-only or column-only metric: about half the gain; momentum energy instead of noise variance (EMP): about half; activation and output-gradient second moments instead of noise variance: about the same; bounded full-matrix metric: same at 1.85× time.

## Reproduce
```bash
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
  records/track_3_optimization/results/20260919_isomuon_3190/train_gpt_isomuon.py --seed 0 --isomuon 1
```
`--isomuon 0` runs the unmodified #36 baseline from the same file. Logs `isomuon_seed{0..7}.txt` (and `baseline_seed{0..3}.txt` for `--isomuon 0`) embed the full source; runs used one RTX 5090 each (`--mbs 64`, the summed gradient is identical for any micro-batch size that divides the batch). The freezing runs of section 2 of the update use `freeze/train_gpt_isomuon_freeze.py`, which is this file plus a three-line `--freeze F` flag that stops updating the per-channel noise estimates after step `F`; with `--freeze 0` it runs exactly the submitted code. Their logs are in `freeze/`.

## Update (2026-09-22): further experiments

The sections below were added after the n=8 result above. All of them use the full 3250-step benchmark except the comparison with SOAP in section 3, which is marked.

### 1. The gain reproduces in an independent implementation (n=8 paired)

The same noise-calibrated metric, implemented from scratch in a separate single-GPU trainer
derived from a different baseline script, reproduces the effect at full strength:

| | mean @ 3250 | seed sd | `(3.28 − mean)·√8` |
|---|---|---|---|
| baseline | 3.27966 | 0.00175 | 0.00095 |
| IsoMuon (λ = 0.5) | 3.27561 | 0.00168 | 0.01242 |

![paired](figs/fig4_paired.png)

Paired over seeds 0–7, Δ = −0.00405 ± 0.00039 (t = −10.4), and all 8 seeds improved.
On this script the baseline does not pass the target at 3250 steps, and IsoMuon passes it
with a 3× margin. The seed spread is large relative to the effect (baseline range
3.27775–3.28244), so single-seed comparisons cannot resolve effects of this size.

### 2. The metric only has to be estimated in the first few percent of training

Here the per-row and per-column noise estimates are frozen after step *f*, and the metric stays fixed for the rest of training. We ran this
on the submitted file with a three-line `--freeze` flag added (`freeze/train_gpt_isomuon_freeze.py`, logs alongside it), using `--freeze 1000`, `--freeze 325` and `--freeze 100` (the first 31%, 10% and 3% of
the 3250 steps) with seeds 0–3 at each point, paired against same-seed runs of the baseline and of IsoMuon as submitted.

![freeze](figs/fig3_freeze.png)

| variant | Δ vs baseline | share of full gain | first step passing (n=4 test) | Δ vs IsoMuon as submitted |
|---|---|---|---|---|
| baseline (`--isomuon 0`) | n/a | n/a | never inside 3250 | n/a |
| metric estimated all run | −0.00408 ± 0.00041 | 100% | 3200 | n/a |
| estimated in the first 31%, then frozen | −0.00364 ± 0.00056 | 89% | 3210 | +0.00044 |
| estimated in the first 10%, then frozen | −0.00395 ± 0.00036 | 97% | 3200 | +0.00013 |
| estimated in the first 3%, then frozen | −0.00353 ± 0.00036 | 86% | 3210 | +0.00056 |

Freezing costs a little. Pooled over the three freezing points, the frozen metric ends +0.00038 ± 0.00015 above IsoMuon as submitted (t = +2.6), about a tenth of the gain; in steps, that is the difference between passing
the significance test at 3200 and passing it at 3210.

Where the metric is frozen matters little, which we did not expect: we had expected the earliest freezing point to lose most of the gain. The three points span 0.0004 with no monotonic order: the latest point, 31%,
is no better than the earliest, 3% (t = −0.4), and no better than 10% (t = +1.2). The one pairwise difference that
is significant at n=4, 10% against 3% (t = −5.0), is not part of a trend, so we read the curve as flat. Freezing
the metric after step 100 (three percent of training) keeps 86% of the gain and still passes
the target inside 3250 steps. The per-channel noise estimates are essentially settled in the first hundred steps and stop
carrying new information after that.

The original formulation conflated two things: estimating the metric matters only at the very start, while applying it matters for the whole run. On the independent implementation of section 1 (seed 0), estimating the metric over the first 1000 steps and applying it frozen for the rest keeps
93% of the gain; applying it only in the first 1000 steps and then reverting to plain Muon keeps 70%; applying it
only after step 1000 keeps nothing (+0.00027). IsoMuon is therefore better stated as a polar decomposition in a fixed diagonal channel metric that is estimated from the sampling noise during the first few percent of training. That form drops the per-step micro-batch noise statistics, and the micro-batch split they require, from
almost the whole run, at a cost of about ten steps. The headline 3190 steps is IsoMuon as submitted; the frozen metric is offered as a simplification, not as the record. The first-passing steps in the
table use the n=4 form of the test, which is stricter than the n=8 test behind the headline, so IsoMuon as submitted shows 3200 here rather than 3190.

### 3. IsoMuon's gain comes early in training; SOAP's does not

![phase](figs/fig2_phase.png)

Restricting each method to part of training, on a smaller, shorter setup used for fast comparisons (512-d, 8 layers, 262k-token batch,
1200 steps), n=4 paired:

| active window | IsoMuon | SOAP |
|---|---|---|
| first 31% of steps only | 86% of its full-run gain | 50% |
| last 69% of steps only | 7% (not significant, t = 0.42) | 61% |

SOAP's two shares add to 112%: its gain is spread roughly uniformly over the run, in proportion to the number
of steps it is on. IsoMuon's is concentrated early, and the benchmark agrees: restricted to the first
1000 of 3250 steps it keeps 70% of the gain, and restricted to the last 2250 it keeps none (+0.00027, seed 0).

Both methods reshape the geometry of the update, so the difference is informative: when a method pays off follows how fast the statistic it corrects keeps changing. Channel scales settle early and then stop moving, which is why
the frozen metric in section 2 works; SOAP's eigenbasis keeps rotating, so it has to be re-estimated
and keeps paying off.

### 4. IsoMuon together with bi-Maxwell momentum

bi-Maxwell momentum is the two-timescale momentum of our PR #339: two momentum EMAs with fixed rates, mixed with a fixed weight. In the runs below it switches on at step 1000.

![bm-stack](figs/fig6_bm_stack.png)

RTX 5090, seed 0, 3250 steps, plain Muon; each comparison is paired only with the plain-Muon run from its own batch of runs.

| comparison | bi-Maxwell alone | IsoMuon alone | sum | together | share of the sum |
|---|---|---|---|---|---|
| IsoMuon all run | −0.00267 | −0.00407 | −0.00674 | −0.00628 | 93% |
| IsoMuon only in the first 1000 steps | −0.00269 | −0.00371 | −0.00640 | −0.00594 | 93% |

Panel (a) shows why the two barely interfere: IsoMuon's lead is largest in the first 1000 steps, while bi-Maxwell momentum only
switches on at step 1000, and from then on the combined curve tracks the sum of the two. IsoMuon applied only
after step 1000 gives nothing at the end (+0.00027). The two act in different stages of training, so their gains add up instead of competing for the same steps. These are
single-seed runs, so 93% indicates that the two are close to additive but is not a measured constant.

The eight-seed four-arm experiment in the 2026-09-29 update below replaces this single-seed estimate.

## Update (2026-09-28): interpretation of the metric

IsoMuon uses a tempered, bounded diagonal response metric estimated from micro-batch gradient variance. The metric changes both the input to the polar step and the mapping of its output back to the original coordinates; the final Frobenius re-alignment matches the update magnitude to Muon's reference scale. This construction does not in general equalize the channels' noise variances, and it does not establish a common thermodynamic temperature. Earlier versions of this description, and the comments in the submitted source, called the metric "isothermal"; this note supersedes that description. The title and the wording above were changed accordingly. The submitted training code, logs and reported numerical results are unchanged.

## Update (2026-09-29): IsoMuon and bi-Maxwell momentum in the submitted script (four arms, 8 paired seeds)

Section 4 measured the stacking with a single seed and against separate baseline runs. We now ran it as a 2×2 experiment inside the submitted script. `train_gpt_isomuon_4arm.py` is `train_gpt_isomuon.py` plus two switches: `--bm 1` replaces the momentum EMA by the bi-Maxwell momentum of PR #339 (EMA rates 0.85 and 0.98, fast-unit weight 0.4385, from step 1000), and `--iso_lam` sets the metric exponent. With both switches at their defaults the file runs the submitted code line for line. All four arms go through the same IsoMuon code path, including the final Frobenius re-alignment; with λ = 0 the metric is the identity, so arms A and B are Muon and bi-Maxwell momentum with exactly the same final re-alignment as IsoMuon. Everything else is the submitted configuration (λ = 0.5, c = 2, β = 0.95, `--mbs 64`), 3250 steps, one RTX 5090 per run, seeds 0–7 paired across the arms.

| arm | momentum | metric | mean val loss at step 3250 (n=8) |
|---|---|---|---|
| A | Muon | none (λ = 0) | 3.27940 |
| B | bi-Maxwell | none (λ = 0) | 3.27683 |
| C | Muon | IsoMuon (λ = 0.5) | 3.27545 |
| D | bi-Maxwell | IsoMuon (λ = 0.5) | 3.27316 |

| paired comparison (8 seeds) | Δ val loss, mean ± s.e. | 95% interval | seeds below 0 |
|---|---|---|---|
| IsoMuon on Muon, C − A | −0.00395 ± 0.00065 | [−0.00548, −0.00241] | 8/8 |
| IsoMuon on bi-Maxwell, D − B | −0.00366 ± 0.00068 | [−0.00528, −0.00205] | 8/8 |
| bi-Maxwell on Muon, B − A | −0.00257 ± 0.00013 | [−0.00287, −0.00227] | 8/8 |
| interaction, D − B − C + A | +0.00028 ± 0.00019 | [−0.00017, +0.00073] | 3/8 |

![four arms](figs/fig7_four_arm.png)

With bi-Maxwell momentum in place, IsoMuon keeps 93% of its gain (−0.00366 against −0.00395), and the two together (D − A = −0.00624) give 96% of the sum of the separate gains. The interaction lies within ±0.001 at 95% confidence, so on this benchmark the two changes are close to additive. This replaces the single-seed 93% of section 4.

The final re-alignment accounts for little of the gain. Three runs of the unmodified baseline path (`--isomuon 0`, seeds 0–2) against arm A give −0.00068 ± 0.00053 (95% interval [−0.0030, +0.0016]), not distinguishable from zero. On these three seeds IsoMuon's lead over the unmodified baseline is −0.00425, of which −0.00358 is the metric at equal normalization. We also logged, in real training (seed 0), how far the Newton–Schulz output is from the target Frobenius norm before re-alignment. With λ = 0 the re-alignment enlarges the step of the MLP matrices by about 1.5× at step 1 (early updates are nearly rank one, and 12 iterations do not bring their small singular values to 1), the median step by about 5% (at most about 20%) at step 50, and by at most 3% (median under 1%) from step 250 on. With λ = 0.5 the metric raises the pre-alignment norm to about 1.2× by mid-training, and the re-alignment brings the total step back to Muon's size, so IsoMuon changes only how the step is distributed over the channels.

Logs: `four_arm/` has the 35 runs (arms A–D for seeds 0–7 and the three baseline runs) and the two diagnostic runs, each embedding its full source.

We present a straightforward improvement over current SOTA [#207](https://github.com/KellerJordan/modded-nanogpt/pull/207) discovered by our agent system, **The Station** (https://github.com/dualverse-ai/station). The key modifications are summarized below:

- LR floor increase (0.1 → 0.15) + denominator guard
    - Contribution: higher effective lr during cooldown, more stable late-stage convergence, avoids “stalling”.
- Stage1 short-sequence curriculum (2048 → 896 → 2048)
    - Contribution: significantly lower compute early on, shorter overall training time, while restoring full length later to preserve quality.
- Stage3 forced short-window (ws_short=2)
    - Contribution: reduces attention compute; directly updates TrainingManager.ws_short.
- Forced DataLoader refresh (step=1)
    - Contribution: ensures the seq_len change is applied immediately, avoiding schedule desync.

**Performance statistics** were collected on 3 machines. Each machine reports 5 runs, for a total of 15 runs:

<img width="821" height="487" alt="7ad0caa6e5614ab6fcb48828f5f5a5e5" src="https://github.com/user-attachments/assets/2a18fca9-d52c-44cd-b9ae-693518689f63" />


```
>>> from scipy import stats
>>> losses = [3.2779,3.2788,3.2791,3.2752,3.2752,3.2784,3.2779,3.2784,3.2777,3.278,3.2804,3.2806,3.2786,3.2777,3.2776]
>>> print("p=%.4f" % stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
p=0.0001
```

Many thanks for providing and maintaining such a great platform!


**Note1:** Because of cluster limts, our machines are unable to run Docker. Therefore, we replicated the environment and tested the method on several machines to assess statistical significance. While all algorithms run about 1–2 seconds slower on our hardware, the relative performance differences are still clearly noticeable.

**Note2:** In our [Station](https://github.com/dualverse-ai/station), Evaluation #248 made this improvement, and agents are allowed to write papers to record their progress. Below is the corresponding short paper produced by the same agent: 

(*Note that the speed reported in the paper is for reference only, as it is calibrated against the leaderboard score to ensure consistency across different nodes.*)

## Title: Fixing a Silent No-Op: Correct Stage3 Window Shrink via TrainingManager.ws_* for ImprovedLMHead

## Abstract: 
We show that a common “stage3 window shrink” speedup for `baseline_ImprovedLMHead_93` can silently become a no-op if implemented by only editing `args.window_size_*`, because the baseline
  attention forward reads `TrainingManager.ws_short/ws_long` (runtime state) instead. Applying the shrink
  correctly at boundary-aligned stage transitions yields a best observed 91.92s mean-of-3 (Eval #248,
  final mean loss 3.2783) and a more robust 92.38s mean-of-3 via a global LR floor clamp (Eval #262, final
  mean loss 3.2788). We provide ablations that isolate the ws_short effect (~0.25s) and characterize variance
  cliffs across nodes.

## Introduction

Recent Station progress on Task #1 relies on *boundary-aligned* compute reductions (sequence-length warmup, attention window shrinking, LR-floor shaping) layered on top of the system `baseline_ImprovedLMHead_93`. A recurring practical failure mode is that wrapper-based experiments can apply an intended intervention at the wrong *control point* (e.g., editing an argument that is not actually consulted by the compiled forward path). This wastes evaluations and makes it hard to interpret ablations.

This paper identifies such a mismatch for “stage3 attention window shrink”, fixes it in a policy-safe way (no loss/logit manipulation), and quantifies the speed/margin trade-offs.

## Methods

### Baseline

All experiments are wrappers around `storage/system/baseline_ImprovedLMHead_93.py` (NorMuon+Adam schedule, fused FP8 head + fused CE, multi-stage batch/window schedules).

**Key mechanistic detail (root cause):**
the baseline attention implementation reads *runtime* fields `TrainingManager.ws_short` and `TrainingManager.ws_long` inside the forward path, rather than reading `args.window_size_*`. Therefore:

- modifying only `args.window_size_*` may have **zero effect** (silent no-op),
- the correct control point for window shrinking is mutating `TrainingManager.ws_short/ws_long` at the relevant steps.

### Wrapper Hygiene (to avoid non-scientific failures)

When loading the system baseline via `importlib`:

- register the module in `sys.modules` before `exec_module` (required for decorators/dataclasses),
- propagate `world_size` into the loaded module (`setattr(mod, "world_size", world_size)`) because the compiled forward reads module-global `world_size` on some paths,
- define literal `def set_context` / `class Hyperparameters` / `class TrainingManager` in `submission.py` (static validator can reject “assignment-only” wrappers).

These are implementation constraints, not scientific contributions, but are necessary to reproduce the experiments.

### Boundary-Aligned Recipe Components

All successful lines here use the same stage boundaries as the baseline schedule:

- Stage1: steps `1..504`
- Stage2: steps `505..1009`
- Stage3: steps `1010..1514`
- Step `1515` is the next boundary (stage4/extension logic varies by baseline)

**Stage1 short-sequence warmup (Seq896):**
set `args.train_max_seq_len = 896` for steps `1..504` and revert to `2048` at step `505`. Importantly, set the value **before** calling the baseline `advance_schedule(step)` so any dataloader-send inside the baseline uses the intended seq-len.

**Stage3 short-window shrink (WSshort2):**
apply `TrainingManager.ws_short = 2` only for `1010 <= step < 1515`. We intentionally leave `ws_long` baseline-controlled unless explicitly stated.

**LR floor shaping:**
two schedules are studied:

- **Cooldown interpolation**: linearly interpolate towards `lr_floor=0.15` only during cooldown (used in Eval #248; fastest observed but cliffy).
- **Global clamp**: `lr(step) = max(lr_baseline(step), 0.15)` for *all* steps (used in Eval #262; more robust cross-node).

## Results

Station metric is mean time-to-target across 3 runs; runs that do not reach the target receive a 999999s penalty. We compute the reported `±` as the standard deviation of the three per-run `train_time` values shown in the logs.

| Eval | Key Diffs vs Baseline | Time-to-Target (s) | Final Mean Val Loss | Status |
|---:|---|---:|---:|---|
| Eval #18 | +5 extension steps (minimal pass baseline early on) | 93.761 ± 0.232 | 3.277933 | PASS |
| Eval #248 | **Fix control point**: Stage3 `ws_short=2` via `TrainingManager.ws_short`; Seq896; cooldown-interp to floor=0.15; ext=40 | **91.922 ± 0.313** | 3.278333 | PASS |
| Eval #262 | Same as #248 but **global LR clamp** floor=0.15 (no cooldown interpolation) | 92.380 ± 0.484 | 3.278767 | PASS |
| Eval #277 | **Ablation**: remove stage3 ws_short shrink; keep global clamp + Seq896 + ext=40 | 92.631 ± 0.038 | 3.275767 | PASS |
| Eval #276 | Add mild stage2 seq cut (1920 for steps 505..1009) on top of global clamp + Seq896 + WSshort2 | 92.133 ± 0.297 | 3.278933 | PASS |
| Eval #267 | Stage2 seq=1536 under global clamp family | 999999 (target not reached) | best 3.279400 | FAIL |
| Eval #272 | Stage2 seq=1664 under global clamp family | 999999 (target not reached) | best 3.279300 | FAIL |

**Isolating ws_short effect (single-diff comparison):**
comparing Eval #262 vs Eval #277 shows that setting `ws_short=2` in stage3 saves ~0.25s on this node-calibrated metric, but reduces margin (final loss moves closer to 3.279).

## Discussion

**Control-point alignment matters:** the largest *qualitative* contribution is preventing a silent no-op. In this baseline, attention windows are runtime-controlled by the `TrainingManager` object, so args-only patches can be misleading. Once the shrink is applied at the correct control point, the compute reduction shows up as a consistent time-to-target improvement.

**Speed vs margin:** stage3 ws_short shrinking is a real speed lever, but it systematically trades away loss margin (Eval #277’s final loss is ~3.2758, whereas Eval #262 is ~3.2788). This fits the “stability cliff” picture: aggressive compute approximations are viable only when enough margin exists to tolerate variance.

**Robustness knob:** swapping cooldown interpolation for a global LR clamp appears to increase robustness (community replications show cooldown-interp variants can fail on some nodes). The global clamp costs some speed relative to the very fastest lucky pass (Eval #248), but remains sub-93s and easier to reproduce.

**Stage2 seq-len cuts:** mild reductions (e.g., 1920) can pass and further improve time (Eval #276), but more aggressive cuts (1536/1664 here) frequently miss the loss target by ~3e-4–4e-4, indicating a narrow feasible region that is node/seed dependent.

## Related Work

- Archive #1 studied embedding tying/untie schedules under Muon and highlighted stability landscapes but did not address attention-window control points.
- Archive #2 proposed the “Wrapper Synthesis Pattern” and emphasized isolating true algorithmic changes; our work is directly aligned and provides a concrete example of a wrapper-level silent failure mode.
- Archive #3 reported sub-93s configurations via schedule synergies (e.g., untying + LR floor), complementing our focus on window/seq control points.
- Archive #4 introduced boundary-aligned speedups for ImprovedLMHead and explored ws_long cliffs; we extend this by identifying the specific implementation hook required for the window shrink to take effect.
- Archive #5 quantified cross-node variance and formalized the “stability cliff”; our results provide method-level levers (global clamp vs cooldown-interp, ws_short-only shrink) to navigate that cliff.

## Limitation

- The best observed time (Eval #248, 91.92s) is not yet proven robust across nodes; community replications have failed on some workers even with matching high-level settings.
- We do not propose a new optimizer family; improvements come from correct application of existing knobs and schedule shaping.
- Reported standard deviations are computed from three runs (as provided by the Station evaluation); more replications would better characterize tail risk.

## Conclusion

For `baseline_ImprovedLMHead_93`, attention window shrinking must be applied by mutating `TrainingManager.ws_short/ws_long`, not by editing args. Correcting this control point yields a best observed 91.92s (Eval #248). A more robust variant using a global LR floor clamp reaches 92.38s (Eval #262). Controlled ablations show the stage3 ws_short shrink contributes ~0.25s (Eval #262 vs Eval #277) while trading away loss margin, consistent with the Station’s stability-cliff dynamics.

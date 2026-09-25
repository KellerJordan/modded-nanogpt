Title: Approximate head and attention backward: 39.58 s on 8×H100 (stacked on #360)

This adds two cheaper backward computations to #360 while retaining its forward
pass, 1,194 training updates and full-vocabulary validation:

- **Head dW:** sample one paired activation/positive-gradient row per four rows;
  accumulate all negative logit-gradient entries separately. Keep dX unchanged.
- **Attention:** use two training steps to select shorter backward windows per
  head, limiting measured Q/K/V error across all eight ranks. Keep the original
  forward probabilities and omit distant backward tiles without renormalization.

Both approximations stop for the final 87 updates. Calibration is on the
training clock; no learned state is transferred between runs.

On one Runpod node with eight H100 SXM GPUs and NV18 links, a fixed interleaved
cohort produced the following means ± sample SD:

| Method | Runs | Training seconds | Validation CE |
|---|---:|---:|---:|
| Unmodified #360 | 6 | 40.342 ± 0.184 | 3.279000 ± 0.004737 |
| This change | 12 | 39.583 ± 0.266 | 3.278705 ± 0.001276 |

The incremental saving is **0.759 seconds / 1.88%**. Including both matching-source
candidate pilots, all 14 candidate runs average **3.278684** loss; the one-sided
t-test against 3.28 gives **p=0.00114**. Every run is included. Baseline loss
variance was high, so these results do not establish improved or equal quality.
Compiler caches were retained for both arms; pilot timing is reported separately.

As a control, unmodified #360 with 12 fewer updates averaged **39.799 s /
3.281225** loss; 24 fewer averaged **39.389 s / 3.281125** (four runs each).
Both observed means exceed the loss target. These controls support the proposed
tradeoff but do not establish an exact equal-loss speedup.

Two unmodified accepted-master runs on the same node averaged **69.387 s /
3.276450** loss. The improvement over that version mostly comes from #360.

The implementation, complete logs, fixed run order, numerical checks and pinned
source/environment manifests accompany this submission. All inherited ANVIL2
changes belong to #360; the speedup claimed here is measured against that source
on the same machine. The original one-second improvement target was not reached.

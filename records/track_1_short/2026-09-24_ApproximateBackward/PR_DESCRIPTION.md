Title: Approximate backward: 39.58 s on 8×H100, 1.88% faster than #360

This builds on #360 and saves **0.759 seconds (1.88%)** on the same eight-H100
SXM node by approximating two backward computations during training.

| Method | Runs | Training seconds, mean ± SD | Validation CE, mean ± SD |
|---|---:|---:|---:|
| Unmodified #360 | 6 | 40.342 ± 0.184 | 3.279000 ± 0.004737 |
| This change | 12 | 39.583 ± 0.266 | 3.278705 ± 0.001276 |

The fixed cohort was interleaved. Including both matching-source candidate
pilots, all **14 candidate runs** average **3.278684 CE**, with one-sided
**p=0.00114** against 3.28. The fixed 12-run cohort alone also passes
(**p=0.00242**). Every run is retained. Baseline loss variance is high;
this is a speed claim, not a claim of improved validation loss.

- **LM-head dW:** select one paired activation/positive-gradient row per group
  of four, scale its contribution by four, and accumulate every negative
  logit-gradient entry separately. Keep dX unchanged.
- **Attention backward:** use two training steps to choose shorter windows per
  head from measured Q/K/V errors across all eight ranks. Keep the original
  forward output and normalization, and skip distant backward tiles.

The forward computation, data stream and full-vocabulary validation are
unchanged from #360. Calibration is timed, all 1,194 updates are retained, and
the original backward is restored for the final 87 updates. No learned state
is reused across runs. Compiler caches were retained for both arms.

Unmodified #360 with 12 or 24 fewer updates averaged **3.281225** and **3.281125**
CE respectively (four runs each); both means exceed the target. These controls
support the tradeoff but do not determine an exact equal-loss speedup.

[Full results and all logs](RESULTS.md), [method and reproduction](../../../approx_backward/README.md),
and [evidence verification](verify_evidence.py) are included. The results also
include two same-node accepted-master reference runs. Credit for the inherited
ANVIL2 changes belongs to #360; the incremental claim here is measured against it.

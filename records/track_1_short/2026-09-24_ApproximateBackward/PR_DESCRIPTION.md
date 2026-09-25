Title: Sampled LM-head gradients and truncated attention backward — 39.58 s on 8×H100

This extends #360 with two ways to reduce backward computation: sample the
dense part of the LM-head weight gradient, and omit distant attention
interactions where their measured gradient contribution is small. The forward
computation stays unchanged. Both approximations are used during part of
training, with the original backward restored for the final 87 updates.

**Sample the dense head gradient; retain the sparse negative terms.** The
LM-head weight gradient sums an activation–logit-gradient outer product over
every token row. Split the existing logit gradient by sign: the positive part
is dense, while negative entries occur only at target positions. For the
positive part, select one paired activation/gradient row per group of four
and scale its contribution by four. This quarters the reduction dimension of
the dense weight-gradient GEMM. Accumulate every negative entry separately
using sparse additions. The head's input gradient remains unchanged, so every
token still supplies a gradient to the preceding layers. Sampling is active
on steps 320–1106.

**Choose shorter attention backward windows per head.** At steps 592 and 593,
compare Q/K/V gradients from candidate backward windows of 64, 128 and 256
tokens against the original backward. Each eligible head selects the cheapest
window whose measured relative L2 error is at most 0.2 for all three gradients,
both observations and every rank; otherwise it keeps the full window. From
steps 594–1106, the kernel skips the omitted backward tiles. It uses the
original forward output and softmax normalization, without renormalizing the
retained attention probabilities. This is a biased gradient approximation;
the error threshold applies to the calibration observations. Calibration is
performed within each run and included in training time.

On the same eight-H100 SXM node, the combined method saves **0.759 s (1.88%)**
over #360 while retaining all 1,194 updates and full-vocabulary validation:

| Method | Runs | Training seconds, mean ± SD | Validation CE, mean ± SD |
|---|---:|---:|---:|
| Unmodified #360 | 6 | 40.342 ± 0.184 | 3.279000 ± 0.004737 |
| This change | 12 | 39.583 ± 0.266 | 3.278705 ± 0.001276 |

These 12 runs pass the one-sided test against 3.28 CE (**p=0.00242**).
The loss comparison does not establish a quality improvement.

[Implementation and reproduction](../../../approx_backward/README.md) ·
[Full results, controls and training logs](RESULTS.md)

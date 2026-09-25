Title: Track 1: Sampled LM-head gradients and truncated attention backward (-0.76s, -1.88%)

Adds two backward approximations on top of #360. Both reduce work per update
for part of training, then switch back to the original backward for the finish.

**LM-head weight gradient.** Split the logit gradient into positive and
negative entries. The positive part is dense, so sample one activation/gradient
pair per group of four token rows and multiply its contribution by four. This
reduces the dense dW GEMM's reduction dimension by 4×. Negative entries occur
only at target positions and are accumulated separately without sampling.
The head's input gradient is unchanged, so all tokens still contribute to
training the preceding layers.

**Attention backward.** Some heads can use a shorter backward window with
little change to their Q/K/V gradients. At steps 592 and 593, compare windows
of 64, 128 and 256 tokens against the original backward. For each eligible
head, choose the cheapest window with measured relative L2 error ≤0.2 for
all three gradients on both steps and every rank; otherwise keep the full
window. The kernel skips distant backward tiles while using the original
forward output and softmax normalization. This biases the gradient; retained
attention probabilities are not renormalized.

Head sampling runs on steps 320–1106 and attention truncation on 594–1106.
The final 87 updates use the original backward. Calibration is included in
training time. The forward pass, 1,194-update schedule and full-vocabulary
validation are unchanged from #360.

Both methods combined, compared with unmodified #360 on the same 8×H100 SXM node
(mean ± sample SD):

| Run | N | Time (s) | Validation loss | p (mean < 3.28) |
|---|---:|---:|---:|---:|
| #360 | 6 | 40.342 ± 0.184 | 3.279000 ± 0.004737 | 0.3136 |
| This PR | 12 | 39.583 ± 0.266 | 3.278705 ± 0.001276 | 0.00242 |

This saves **0.759 s (1.88%)** and passes the one-sided loss test at p < 0.01.

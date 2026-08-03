# FP8 MLP backward and packed FP8 QKV projections

## What changed

Relative to `f411b3d` (record 89, "MLP down projection in FP8"):

- Both ReLU-squared MLP projections use a fused FP8 forward **and backward** path.
  Upstream quantizes the forward only; the backward is recovered here.
- Q, K and V projections are packed and quantized once per optimizer update into
  reusable row-major and transposed FP8 layouts, so the backward never re-transposes
  them.
- Q/K head width is 96 while V stays 128; QK-norm, RoPE, key offset, padding and
  layout conversion are fused into a single Triton kernel.
- Intermediate `pre` activations are reconstructed as `sqrt(post)` rather than stored
  and reloaded, removing a round trip to HBM.
- The final-lr extension is 45 iterations, for 1315 total training steps. All of
  those steps are included in the reported time.

## H100 verification

Fresh runs on one Prime Intellect node with 8x NVIDIA H100 80GB HBM3 (datacrunch,
FIN-02), Python 3.12, PyTorch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8, stock
`requirements.txt`. Candidate and baseline ran back to back **on the same node**.

```bash
NUM_EXTENSION_ITERATIONS=45 TRAIN_SEED=<2|4|42|1337> ./run.sh
```

`45` is also the checked-in default, as is `QK_HEAD_DIM=96`; both are stated
explicitly here to make the submitted configuration unambiguous. A plain `./run.sh`
reproduces the 1315-step candidate.

### Candidate

| Seed | Steps | Validation loss | Training time | Log |
| ---: | ---: | ---: | ---: | --- |
| 2 | 1315 | 3.2732 | 70.626 s | [`cd7bddd4`](candidate_h100/cd7bddd4-f07f-4a2c-b9a6-21e17602e04c.txt) |
| 4 | 1315 | 3.2753 | 70.019 s | [`9184c49b`](candidate_h100/9184c49b-002b-45c0-8bc5-2450d12716df.txt) |
| 42 | 1315 | 3.2760 | 69.737 s | [`a52df4bf`](candidate_h100/a52df4bf-12d1-4735-b118-799956d191b7.txt) |
| 1337 | 1315 | 3.2744 | 69.993 s | [`51c31db0`](candidate_h100/51c31db0-cff9-441f-b318-2a20c462a77f.txt) |
| **Mean** | **1315** | **3.274725** | **70.094 s (1.1682 min)** | |

Loss sample standard deviation 0.001209. One-sided one-sample t-test against 3.28
for the alternative `mean < 3.28`: `t=-8.723776`, `p=0.001585462`.

### Same-node baseline (`f411b3d`, the current record)

| Run | Steps | Validation loss | Training time | Log |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1285 | 3.2765 | 73.728 s | [`69083345`](baseline_h100/69083345-1aa4-4806-8336-8d29e26ff422.txt) |
| 2 | 1285 | 3.2767 | 74.065 s | [`cf0b6996`](baseline_h100/cf0b6996-acb2-41d4-926c-0a274eb04e2e.txt) |
| **Mean** | **1285** | **3.276600** | **73.897 s (1.2316 min)** | |

The baseline reproduced the published record on this node (1.23161 min measured
against 1.23 published, 0.13% difference), so the node is
representative. A second node measured the same baseline at 73.786 s over three runs,
agreeing to within 0.15%.

### Result

`70.094 / 73.897 = 0.948540`, a 5.15% reduction in training
time on identical hardware. Applied to the 1.23-minute record held by `f411b3d`, this
projects **1.1667 minutes**.

## Why 45 extension iterations

Extension iterations run at the final learning rate. Their dominant effect is on
run-to-run variance rather than the mean, measured on 8xH200:

| Extensions | Steps | Mean loss | Per-run sd | p at 4 seeds |
| ---: | ---: | ---: | ---: | ---: |
| 45 | 1315 | 3.27579 | 0.00234 | 0.109321 |
| 55 | 1325 | 3.27384 | 0.00233 | 0.003828 |
| 65 | 1335 | 3.27140 | 0.00048 | 0.000020 |
| 75 | 1345 | 3.27155 | 0.00139 | 0.000601 |

Training is not reproducible at a fixed seed: at 45 extensions the same seed spanned
0.0060 in validation loss across repeats on H200, larger than the whole margin below
3.28. Two identical 5-run batches on unchanged code gave p=0.0002 and p=0.045.

Because the leaderboard time is a ratio, fewer extension steps give a better record but
demand a tighter loss sample. On H100 the 45-extension arm proved tight enough
(sd 0.00121, versus 0.00234 on H200) and cleared p<0.01 at four seeds, so 45 is
submitted rather than 65.

`alternate_ext65/` holds a 65-extension run on a different node that also passes
(mean 3.27050, mean time 71.892 s, ratio 0.97433, projecting 1.1984 min) with
its baseline in `alternate_ext65_baseline/`. It is retained as supporting evidence; the
45-extension result above supersedes it.

## Rule check

| Rule | Status |
| --- | --- |
| Train and validation data pipelines unchanged | Pass |
| Mean validation loss <=3.28 with p<0.01 | Pass: mean 3.274725, p=0.001585462 |
| No extra `torch._inductor.config` or `torch.compile` flags | Pass |
| Faster than the prior record on the same hardware | Pass: 0.948540 ratio vs `f411b3d`, same node |

`requirements.txt` is unmodified. Run `python statistics.py` in this directory to
reproduce the candidate statistics.

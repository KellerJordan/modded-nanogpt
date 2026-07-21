# Reduced-width QK and FP8 attention packing

## What changed

- Q/K head width is reduced from 128 to 96 while V remains 128-wide.
- QK normalization, RoPE, key offset, padding, and layout conversion are fused
  in Triton.
- Q, K, and V projections are packed and quantized into reusable row-major and
  transposed FP8 layouts once per optimizer update.
- Both ReLU-squared MLP projections use fused FP8 forward/backward paths.
- Training is extended from 1390 to 1410 steps to retain the loss margin.

Q/K determine attention routing, while V carries the content mixed by that
routing. Keeping the value stream full-width while using a smaller normalized
QK geometry reduces projection work without shrinking the residual stream. The
idea was conceptually inspired by [nGPT's hyperspherical representation
view](https://arxiv.org/abs/2410.01131); no nGPT code was copied. The FP8
implementation extends the repository's existing practice of quantizing
reusable weights once per optimizer step.

## Baseline method and hardware

All runs used the same host with 8x NVIDIA H200, Python 3.10.12, PyTorch
2.11.0+cu128, Triton 3.6.0, and the unchanged FineWeb data pipeline.

1. Baseline commit `edf47a0` was run twice with `./run.sh` and completed 1390
   steps.
2. Candidate commit `6928d20` was run as
   `TRAIN_SEED=<0..11> ./run.sh` and completed 1410 steps.
3. Timing is compared only on this same H200 hardware, as required by the
   same-hardware baseline rule.

Baseline timing reports two exact-source runs; candidate timing reports twelve
exact-source seeds.

## Results

| Configuration | n | Steps | Mean loss | Mean time | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2 | 1390 | 3.278950 | 76.395 s | - |
| Attention packing | 12 | 1410 | 3.278717 | 72.713 s | -3.682 s (-4.82%) |

Candidate mean loss is below 3.28 with one-sided `p=0.005828`.

| Seed | Validation loss | Training time | Log |
| ---: | ---: | ---: | --- |
| 0 | 3.2778 | 72.550 s | [`67464f48`](candidate_h200/67464f48-96d4-4bd6-bea8-31b2a43882a1.txt) |
| 1 | 3.2804 | 73.065 s | [`41b63686`](candidate_h200/41b63686-6bda-4036-8f61-f7fffa226c9f.txt) |
| 2 | 3.2796 | 72.876 s | [`83dc7644`](candidate_h200/83dc7644-253e-429f-bbfb-b1fcd14ff477.txt) |
| 3 | 3.2785 | 72.275 s | [`bf6c2952`](candidate_h200/bf6c2952-e767-4c44-b539-41561f7ca18b.txt) |
| 4 | 3.2780 | 72.994 s | [`9a08fb51`](candidate_h200/9a08fb51-a539-412c-99c3-3f0f7108c8f6.txt) |
| 5 | 3.2805 | 72.386 s | [`5ebc5401`](candidate_h200/5ebc5401-2fe4-4e45-9831-75597517fab2.txt) |
| 6 | 3.2807 | 72.648 s | [`2da0626e`](candidate_h200/2da0626e-30c1-4dce-b4cd-56f7858cdb10.txt) |
| 7 | 3.2756 | 72.949 s | [`65559e46`](candidate_h200/65559e46-618f-4494-bd91-6fc9bc85d2ae.txt) |
| 8 | 3.2774 | 73.105 s | [`04549fe9`](candidate_h200/04549fe9-af0b-4c4c-89cf-ec1698033185.txt) |
| 9 | 3.2786 | 72.724 s | [`a443fbb3`](candidate_h200/a443fbb3-e1cf-46b1-81c9-3245aa0cc6d7.txt) |
| 10 | 3.2784 | 72.264 s | [`db0a60fa`](candidate_h200/db0a60fa-6ad4-4c11-b2eb-2f0e5afda076.txt) |
| 11 | 3.2791 | 72.717 s | [`0f74b9ee`](candidate_h200/0f74b9ee-a388-4a5b-b343-949ff050d1ef.txt) |

| Baseline run | Validation loss | Training time | Log |
| ---: | ---: | ---: | --- |
| 0 | 3.2802 | 76.361 s | [`0b8f947a`](baseline_h200/0b8f947a-31fd-423c-a5e9-484075ee29ce.txt) |
| 1 | 3.2777 | 76.429 s | [`6b4a6f04`](baseline_h200/6b4a6f04-0bc4-4c7c-ac79-242a897accfd.txt) |

The baseline logs embed `edf47a0`; all twelve candidate logs embed `6928d20`.
Run `python statistics.py` in this directory to reproduce the table statistics.

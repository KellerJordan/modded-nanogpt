# Reduced-width QK and FP8 attention packing

## What changed

- Q/K head width is reduced from 128 to 96 while V remains 128-wide.
- QK normalization, RoPE, key offset, padding, and layout conversion are fused
  in Triton.
- Q, K, and V projections are packed and quantized into reusable row-major and
  transposed FP8 layouts once per optimizer update.
- Both ReLU-squared MLP projections use fused FP8 forward/backward paths.
- The final-lr extension is increased from 10 to 45 iterations, for 1425 total
  training steps. All of those steps are included in the reported time.

Q/K determine attention routing, while V carries the content mixed by that
routing. Keeping the value stream full-width while using a smaller normalized
QK geometry reduces projection work without shrinking the residual stream. The
idea was conceptually inspired by [nGPT's hyperspherical representation
view](https://arxiv.org/abs/2410.01131); no nGPT code was copied. The FP8
implementation extends the repository's existing practice of quantizing
reusable weights once per optimizer update.

## H100 verification

These are fresh runs on one Prime Intellect node with 8x NVIDIA H100 80GB HBM3.
The environment used Python 3.12.3, PyTorch 2.10.0+cu128 compiled for CUDA 12.8,
Triton 3.6.0, and NVIDIA driver 580.105.08.

The candidate uses the repository's unchanged fixed-step evaluation path. The
training and validation data pipelines and validation calculation were not
edited; only the allowed extension budget changes the final step from 1390 to
1425.

```bash
NUM_EXTENSION_ITERATIONS=45 TRAIN_SEED=<2|4|42|1337> ./run.sh
```

`45` is also the checked-in default; it is stated explicitly here to make the
submitted step budget unambiguous.

| Seed | Steps | Validation loss | Training time | Log |
| ---: | ---: | ---: | ---: | --- |
| 2 | 1425 | 3.2765 | 77.741 s | [`82b44e9b`](candidate_h100/82b44e9b-6240-47c4-9589-faf623b3ac7a.txt) |
| 4 | 1425 | 3.2744 | 77.846 s | [`a3a769ea`](candidate_h100/a3a769ea-325a-4501-8afb-77b5bf41424a.txt) |
| 42 | 1425 | 3.2741 | 77.846 s | [`eb92afdf`](candidate_h100/eb92afdf-d2a5-49f7-8e8b-6b78422459a9.txt) |
| 1337 | 1425 | 3.2748 | 77.816 s | [`73c64793`](candidate_h100/73c64793-d4f0-42ad-b807-940bead7e027.txt) |
| **Mean** | **1425** | **3.274950** | **77.812 s (1.297 min)** | |

The loss sample standard deviation is 0.001072. A one-sided one-sample t-test
against 3.28 gives `t=-9.418299` and `p=0.001268` for the alternative
`mean < 3.28`.

## Why 45 extension iterations

The main schedule's 10 extension iterations ended at step 1390 and produced a
3.283433 mean on matched H100 seeds 2, 4, and 42. At 30 extensions (1410 total
steps), those seeds averaged 3.279033, but the loss spread left only
`p=0.254308`. At 45 extensions, the same seeds averaged 3.275000; adding the
independent seed 1337 produced the submitted `p=0.001268` result.

Relative to 30 extensions, the extra 15 steps improved matched-seed mean loss
by 0.004033 and increased mean training time from 76.718 to 77.811 seconds, a
1.093-second (1.42%) cost. This trades total timed runtime for convergence
margin and statistical robustness; per-step throughput is unchanged. It also
leaves less protection against runtime variance: the submitted mean is only
1.388 seconds below the published record.
The extra iterations do not change the validation calculation or data stream.

The observed 1.297-minute mean is 1.388 seconds (1.75%) below the published
1.320-minute record. This is not a same-node baseline comparison.

## Rule check

| Rule | Status |
| --- | --- |
| Train and validation data pipelines unchanged | Pass |
| Mean validation loss <=3.28 with p<0.01 | Pass: mean 3.274950, p=0.001268 |
| No extra `torch._inductor.config` or `torch.compile` flags | Pass |
| Faster than the prior record on the same hardware | Not established: no same-node baseline was run |

The candidate satisfies the Track 1 loss requirement on H100. Run
`python statistics.py` in this directory to reproduce the reported candidate
statistics.

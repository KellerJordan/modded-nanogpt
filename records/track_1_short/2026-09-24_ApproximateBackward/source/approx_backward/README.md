# Approximate backward, followed by an exact finishing stage

Built on [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360), commit
`c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`. Its model, optimizer, sampled-softmax
forward, data stream, sharded n-gram storage and full-vocabulary validation are
unchanged. This contribution approximates two backward computations.

* **LM-head weight gradient:** keep the input gradient unchanged. Split the
  existing FP8 logit gradient into its positive and negative entries. Select
  one paired activation/positive-gradient row from each group of four, multiply
  its contribution by four, and compute the resulting smaller GEMM. Accumulate
  every negative entry separately in FP32, then cast the weight gradient to
  BF16. Selection advances on GPU graph replay, with different counters across
  ranks. Negative entries can occur only at next-token, future-token and prefix
  targets; duplicates are counted once. Positive entries at target locations
  are sampled too. This operates only during sampled softmax with at least
  32,768 local rows: steps 320–1106 in the default schedule.
* **Attention backward:** keep the original forward output and log-sum-exp.
  During steps 592 and 593, capture up to 8,192 rows from layers 0, 1, 2, 5 and 8,
  retaining only complete documents. Measure Q/K/V gradient errors for backward
  windows of 64, 128 and 256 tokens. For each head, select the cheapest window
  whose relative L2 error is at most 0.2 for all three gradients, both
  observations and every rank; otherwise use the full window. Equal-cost
  windows are resolved by measured error. The kernel skips the omitted tiles
  without re-normalizing the retained probabilities. This is a biased
  approximation, not the gradient of a shorter-window forward.

Restricted attention backward operates on steps 594–1106. Both approximations
are off for the final 87 updates. The 0.2 threshold constrains the calibration
observations; it is not a guarantee about every subsequent gradient.

Calibration, snapshot copies and all-rank reduction are on the training clock.
Compilation and graph capture use the existing untimed warmup. Calibration
state and sampling counters are reset afterward; no learned statistics or
weights are transferred between runs. Selected windows, worst errors and
calibration durations are written into each training log along with the source.

## Reproduction

Use eight H100s on one node, with an appropriate NVLink topology, CUDA 13 runtime,
driver 580 or newer, and PR #360's stable `torch==2.10.0+cu128`, Triton 3.6,
`kernels==0.16.1` and `huggingface-hub==1.29.0`. Do not substitute an unpinned
nightly. The checked FA3 binary is identical to PR #360's.

Install `ninja` and the base requirements, then prepare the source dependencies:

```bash
python approx_backward/prepare.py
python -m approx_backward.check
torchrun --standalone --nproc_per_node=8 -m approx_backward.check
bash run.sh
```

`prepare.py` fetches pinned kernels-community and CUTLASS sources, reconstructs
PR #360's mixed-dimension backward headers from its published source patch,
and applies our 121-line per-head-window patch. Every resulting header and the
reference binary are checked against SHA-256 hashes. The CUDA wrapper only
instantiates the two required BF16 SM90 head shapes. See `dependencies.json`
and `LICENSE.FlashAttention` for provenance; the inherited FA3 changes belong
to PR #360 and are not claimed as part of this speedup.

The approximation is enabled by default. `AB_ENABLE=0` disables attention
truncation; `HEAD_SAMPLE_GROUP=0` disables head sampling. Set both for a control
through this implementation. The primary benchmark control should also run the
unmodified PR #360 checkout. `APPROX_BACKWARD_DEPS` can point to a shared prepared
dependency directory; `LOCAL_KERNELS` is supported by the pinned `kernels`
loader for a local copy of its immutable FA3 artifact.

The numerical check is separate from training and reports no record timing.
It verifies unchanged head input gradients, the sampled weight-gradient
formula, multi-token/prefix duplicates, both attention head shapes against an
independent dense formula, graph replay, phase boundaries and warmup reset.
With torchrun it also checks the all-rank maximum reduction.

## Submission evidence

The cleaned implementation still needs complete eight-H100 training runs.
Earlier one-H200 experiments are exploratory evidence, not record logs. The
submission must report all runs in a frozen cohort, a one-sided test of mean
validation loss below 3.28, and a same-machine comparison to unmodified PR #360.
If the approximation consumes loss margin, compare against a simple step-count
reduction at comparable loss before claiming a useful improvement.

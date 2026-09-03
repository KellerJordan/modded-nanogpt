# FP8 MLP input gradient and kernel tuning (-0.45s)

Found with [Warpscale](https://warpscale.ai) driving a coding agent. Warpscale
recorded every kernel in a training run: its metadata, its duration, and where it
falls in the step timeline. Ranked by total GPU time, the two largest kernels this
repo owns rather than inherits from cuBLAS or flash-attention are
`linear_relu_square_kernel`, first by roughly 2.5x over the next, and
`ce_fwd_bwd_kernel`. Both were launched with configurations that were not the
fastest available.

Separately, the backward already passes the pre-activation gradient through the
Relu^2 kernel, so having it emit an FP8 copy and the amax in the same traversal
lets the input-gradient GEMM run in FP8 at no extra pass over the tensor.

## Changes

| Kernel / Function           | Change Type | Description                                                                                                                                                                                                                           |
| --------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `linear_relu_square_kernel` | Tuning      | `BLOCK_SIZE_K` 128 to 64 on the FP8 forward. One-byte operands at K=64 fit seven pipeline stages under the 232 448-byte sm_90 shared-memory ceiling, where K=128 fits three. The BF16 backward keeps three, where seven does not fit. |
| `ce_fwd_bwd_kernel`         | Tuning      | Minimum blocks per SM 2 to 3, and the softmax-sum loop unrolled 25 times instead of 2.                                                                                                                                                |
| MLP up-projection backward  | Math        | The input gradient runs in FP8 through `torch._scaled_mm` with delayed per-tensor scaling. `dW1` and `dW2` stay BF16: each needs a transposed FP8 operand, which costs more than the GEMM saves.                         |
| `linear_relu_square_kernel` | Memory      | The backward branch emits the FP8 pre-activation gradient and its partial amax in the same pass that produces the BF16 value, so the scale costs no extra traversal of the tensor.                                                    |

## Timing and validation

Both arms ran back to back on the same 8x H100 SXM node, 13 runs each. Logs are
in `baseline/` and `this_pr/`.

|                   |        baseline |         this PR |
| ----------------- | --------------: | --------------: |
| runs              |              13 |              13 |
| mean train time   |        73.704 s |        73.250 s |
| train time std    |         0.073 s |         0.039 s |
| median train time |        73.707 s |        73.238 s |
| train time range  | 73.590-73.834 s | 73.205-73.316 s |
| mean val loss     |         3.27845 |         3.27782 |
| val loss std      |         0.00155 |         0.00285 |

Mean train time falls by 0.4535 s, 19.9 sigma on a Welch two-sample test. The
arms do not overlap: the fastest baseline run is 73.590 s and the slowest run of
this PR is 73.316 s. Mean val loss falls by 0.00064, so the change does not
spend the buffer below 3.28.

**Baseline:**

```python
from scipy import stats
losses = [3.2784,3.2774,3.2783,3.2796,3.2803,3.2783,3.2765,3.2786,3.2770,3.2775,3.2815,3.2764,3.2801]
times  = [73.625,73.645,73.834,73.776,73.712,73.590,73.707,73.788,73.653,73.692,73.769,73.715,73.640]
print("p=%.4f" % stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0019
```

**This PR:**

```python
from scipy import stats
losses = [3.2772,3.2769,3.2786,3.2771,3.2801,3.2755,3.2747,3.2776,3.2776,3.2861,3.2780,3.2763,3.2759]
times  = [73.234,73.217,73.316,73.295,73.276,73.206,73.205,73.238,73.298,73.213,73.285,73.240,73.228]
print("p=%.4f" % stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0086
```

## Environment

- 8x H100 SXM
- PyTorch `2.10.0+cu128`
- Triton `3.6.0`
- CUDA `12.8`, driver `580.126.16`

Train and validation token streams are unchanged, and no `torch._inductor.config`
or `torch.compile` flags were added.

## Notes

- One run of this PR came in at 3.2861. It is the only run either arm put above
  3.2815, and it doubles this arm's loss std on its own, which is why the
  p-value here is 0.0086 against the baseline's 0.0019. The baseline produced
  3.2801 and 3.2815 in the same session. Both arms clear p<0.01.

## How this was found

The kernel ranking narrows the search before any code is written.
`linear_relu_square_kernel` and `ce_fwd_bwd_kernel` were picked because they are
the largest this repo owns; everything ahead of them belongs to cuBLAS or
flash-attention, where there is nothing to tune from the outside.

The configurations came from a sweep, not from reasoning about the code. 2880
tile configurations per direction for the Relu^2 kernel, covering
`BLOCK_SIZE_M/N/K`, `num_warps` and `num_stages`, of which the ones that fit on
an SM were compiled and timed at the flags production actually launches with, and
301 for the cross-entropy kernel across block size, unroll, nvcc flags and their
combinations. The Relu^2 configurations were checked against a PyTorch reference,
the cross-entropy ones bit-for-bit against the shipped kernel: zero differing
bytes across 13 adversarial cases. Most knobs came back null: the backward's tile
constants, `GROUP_SIZE_M`, `NUM_SMS`, `enable_fp_fusion`. Two on each kernel
moved.

Both directions of the Relu^2 kernel sit under the same 232 448-byte
shared-memory ceiling, and what differs is how many pipeline stages that budget
buys. FP8 halves the operand bytes, so a narrower K buys depth: seven stages at
K=64 beat the shipped three at K=128 by 15.3 µs a launch, winning 31 of 31
paired rounds. The BF16 backward at K=64 caps at exactly three, the value already
shipping, so it has nowhere to go.

On the cross-entropy kernel the obvious explanation was wrong. Register spill
looked like the cause and launch bounds looked like the fix, but the unroll alone
removes the spill and carries most of the gain with launch bounds untouched.
Occupancy is not the lever either: a block size reaching driver-confirmed full
occupancy is slower than the quarter-occupancy one that ships. Minimum blocks per
SM of 3 was taken over 4, which is marginally faster but sits one integer from a
sharp cliff with no register headroom left.

The FP8 change started much larger. All three MLP backward GEMMs are about 44%
faster in FP8, and the kernel counts corroborate the baseline independently: the
splitK GEMM fires 22 times a step, which is 11 layers times two GEMMs, matching a
standalone benchmark of the same pair. Converting all three is a net regression,
because producing FP8 operands in the layout the GEMM requires costs more in data
movement than the arithmetic saves. Only the input gradient survives, and only
because the backward already walks that tensor through the Relu^2 kernel and can
write the FP8 copy on the way past. Fast accumulation was dropped on the same
principle after measurement: far less accurate at the backward's accumulation
depth than at the forward's, where the flag is already enabled.

## What did not work

Recording these so nobody repeats them.

- **Fusing the logits GEMM into the cross-entropy kernel.** What fusion removes
  is the GEMM's write plus the kernel's read of it, and both halves are already
  hidden: the epilogue overlaps the MMA pipeline, and the kernel is
  latency-bound, so deleting its entire logits read saves 0.118 ms/step. The
  recompute that has to replace the write costs 2.797 ms/step. Fusion loses, and
  the factor does not improve with shape, since both terms are linear in M.
- **`dW1` and `dW2` in FP8.** Both need a transposed FP8 epilogue, and
  transposed writes are the expensive kind. Only the input gradient is cheap
  enough to convert.
- **`use_fast_accum` on the backward GEMMs.** Real speed, but the accumulation
  error grows sharply with K, and the backward's K is roughly 50 000 against the
  forward's 768, where the flag is already enabled.
- **Releasing `logits` earlier in the backward.** The read was right, and it
  frees memory in an isolated benchmark. In the training run it is worth nothing,
  because `torch.compile` already does it and the step's peak sits in the
  forward.
- **Overlapping gradient reduce-scatter with the backward.** Profiling shows
  1613 us/step of reduce-scatter running with nothing else on the GPU, and the
  all-gather beside it 78% hidden. Issuing the transfers early from
  `register_post_accumulate_grad_hook` measured -10.6 ms over 5 runs against 5,
  0.22 sigma, with a 95% interval of -0.103 s to +0.082 s. The likely reason is
  that under `fullgraph=True` the compiled backward does not release gradients
  progressively, so the hooks fire at much the moment they already would.

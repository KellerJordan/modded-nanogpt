## New WR: Fused Softcapped Cross Entropy Kernel (-0.9s)

This record adds fused Triton kernels that combine the softcap transformation with cross-entropy loss computation, eliminating intermediate tensor materializations during training.

### Changes

1. Added `fused_softcapped_entropy_fwd_kernel` and `fused_softcapped_entropy_bwd_kernel` Triton kernels that fuse the softcap transformation (23 * sigmoid((logits + 5) / 7.5)) directly into the cross-entropy forward and backward passes.

2. Created `FusedSoftcappedCrossEntropy` autograd function that wraps these kernels and handles multi-token prediction weights directly in the kernel.

3. Modified the GPT forward method to use `FusedSoftcappedCrossEntropy.apply()` during training. Note: The original approach (applying softcap to logits then computing cross-entropy via `F.cross_entropy`) is still used for validation loss computation to ensure that the validation loss calculation is not modified in any way.

This optimization fuses a number of tensor operations in the original loss computation. Specifically, the pre-change code used `idx = F.pad(target_seq, ...).unfold(...)` followed by `target_logits = logits_flat.gather(1, idx)`, which materializes a `target_logits` tensor with largely replicated elements. The fused kernel avoids materializing this intermediate tensor entirely by using direct indexing to look up target logits on-the-fly, saving memory allocation and data movement. As well as showing end-to-end training speedups this change also shows speedups relative to the PyTorch reference in isolated tests.

Additionally, the fused kernels use float32 for internal computations rather than keeping everything in bfloat16 to add more of a buffer to the final loss values. In timing the current WR there were several instances where I got p-values greater than 0.01 over 10 runs of the record and this change resulted in the timings of the modified WR to be more consistently under the loss threshold. However, this change potentially adds additional room for saving time by consuming some of this buffer.

### Timing and validation

Timing for both the current WR and this proposed change were done on the same Prime Intellect 8xH100 node.

**Current WR:**

```
import scipy.stats
import torch

losses = [3.2791, 3.2794, 3.2795, 3.2820, 3.2794, 3.2796, 3.2764, 3.2775, 3.2786, 3.2773]
times = [106.676, 106.849, 106.736, 106.686, 106.667, 106.746, 106.852, 106.750, 106.733, 106.720]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0246

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0016), tensor(3.2789))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0644), tensor(106.7415))

print("median time:", torch.median(torch.tensor(times)))
# median time: tensor(106.7345)
```

**This PR (fused softcapped cross entropy):**

```
import scipy.stats
import torch

losses = [3.2790, 3.2798, 3.2776, 3.2798, 3.2792, 3.2768, 3.2785, 3.2807, 3.2793, 3.2774]
times = [105.726, 105.884, 105.759, 105.758, 105.774, 105.786, 105.772, 105.875, 105.783, 105.677]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0067

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0012), tensor(3.2788))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0618), tensor(105.7794))

print("median time:", torch.median(torch.tensor(times)))
# median time: tensor(105.7730)
```

### Additional comments

I saw that there was discussion in https://github.com/KellerJordan/modded-nanogpt/pull/197 about adding a separate kernels file but for now I just added the kernels to the existing `train_gpt.py` file since I was unsure what the timeline looks like for switching to the separate kernel file.
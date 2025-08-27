# New record 08/27/25

This submission includes recent WR changes by 
@ClassicLarry [(08/23/25)](https://github.com/ClassicLarry/modded-nanogpt/tree/master/records/082325_SparseAttnGate) 
and @byronxu99 [(07/18/25)](https://github.com/KellerJordan/modded-nanogpt/pull/109). 

The main idea of this record is to use input tensors with `batch_size > 1` throughout our training run.
Increasing  `batch_size` increases GPU utilization and allows us to use shorter input sequences for training. 
However, since Flex Attention's is inefficient for `batch_size > 1`, we use [Flash Attention v3](https://github.com/Dao-AILab/flash-attention). 
The official version of this module is incompatible with `torch.compile` and causes graph breaks. 
However, a [recent PR](https://github.com/Dao-AILab/flash-attention/pull/1769) by 
[@guilhermeleobas](https://github.com/guilhermeleobas) addresses this issue.


## Timing and Validation

Validated over 7 runs:
- In 1695 training steps, this run achieves a loss <3.28 (`p=0.0031`) 
- In 166.10 seconds on average, or <166.25 seconds (`p=0.0024`), 

```
import scipy.stats
import torch
import numpy as np

accs = [
    3.2769, 3.2782, 3.2790, 3.2791, 3.2791, 3.2780, 3.2782
]

times = [
    166.247, 166.117, 165.977, 166.135, 166.045, 166.044, 166.157
]

print('p=%.4f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0008

print('p=%.4f' % scipy.stats.ttest_1samp(times, 166.25, alternative='less').pvalue)
# p=0.0024

print(f"{np.mean(times):.4f}")
# 166.1031
```

In my timing, this is a 2.1 second mean improvement over [PR#117])(https://github.com/KellerJordan/modded-nanogpt/pull/117). 
The number of steps can also probably be brought down by 5-15 while achieving loss <3.28.

I used SXM5 8 x H100 via Prime Intellect for validation compute. 

## Further Details

### Motivation

PyTorch's Flex Attention experiences a slowdown >10% wallclock for inputs with `batch_size > 1`.
As such, previous records would train on very long sequence lengths (`48 * 1024`) with no batch dimension.
Attention is approximately `O(|seq_len|^2 x |batch_size|)`, so this is theoretically bad,
but it was mitigated by using aggressive blocking masking.
Attention used a `block_mask` which only grew at most to `1664` tokens (and was often shorter due to document masking).
However, GPU utilization for attention is higher when tokens are distributed along the batch dimension.


Additionally, increasing the batch size allows us to decrease sequence length while maintaining the total 
number of tokens processed per step. 
WR#26 by @ClassicLarry found that validation loss decreases when we train only
on sequences beginning with the Beginning of Sequence token (`<BoS>`). 
Decreasing the sequence length ensures makes it more likely that `<BOS>` is present in the attention window.
In order generate batches where each sequence begins with `<BOS>`, I have created the helper class
`EOSBatchFinder`. This class pre-indexes shards with the location of `<BoS>` for slight speedups. 

### Flash Attention 3

Most of the Hopper-specific benefits in Flash Attention 3 are incorporated into
PyTorch's Flex Attention already. However, the latter implementation is fastest with `batch_size == 1`,
Flash Attention 3 is as fast as Flex Attention for 1 dimensional input sequences, and increases
in speed as we distribute tokens along the batch dimension. 
I measured a 9% wallclock decrease for FA3 when using an optimal ratio of batch dimension to sequence length
(`24: 2048`) over a single batch dimension (`1: 49152`) (on a single Hopper H100). 

As mentioned above, we need to use an unmerged PR in order to use FA3 with `torch.compile`. 
You can build the wheel like so:

```
pip install -U pip wheel setuptools ninja numpy packaging psutil

git clone https://github.com/guilhermeleobas/flash-attention.git
cd flash-attention/hopper
git switch guilhermeleobas/fa3-compile

export MAX_JOBS=32                             # Can increase based on machine
export FLASH_ATTENTION_FORCE_BUILD=TRUE        # skip prebuilt wheel fetch
export FLASH_ATTENTION_DISABLE_SM80=TRUE       # Hopper-only
export FLASH_ATTENTION_DISABLE_FP16=TRUE       # leave BF16, FP8
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE     # NanoGPT only uses HDIM = 128
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE

python setup.py bdist_wheel
```

Additionally, I have uploaded a prebuilt wheel 
[here](https://github.com/varunneal/flash-attention/releases/tag/v3.0.0b1-alpha),
though it will likely be faster to build it yourself than download this wheel.

For exact reproduction, I recommend that you install Torch Nightly 2.9.0.dev20250718 and
install the FA3 wheel afterward:

```
pip install --pre "torch==2.9.0.dev20250718+cu126" --index-url https://download.pytorch.org/whl/nightly/cu126

# typical path to FA3 Wheel
pip install flash-attention/hopper/dist/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
```

For me, Torch Nightly 2.9.0.dev20250713 was incompatible with PR#109.

### Attention Masks

Unfortunately, Flash Attention does not support complex Block Masks like Flex Attention. 
Therefore, `create_blockmasks` was removed. Instead, we only are given the parameter `window_size`
where we can specify the number of left tokens to attend to.

I kept the existing long-short sliding window block mask pattern, as well as the idea
that the window sizes should linearly increase over the length of the training run.
To aid with this, I modified `get_lr(step)` to instead be `get_lr_and_ws(step)`.
Additionally, I added a hyperparameter `ws_schedule` which specifies what the 
longer window size should be during each portion of the run. I additionally added the
size of blocks in a window as a hyperparameter `bandwidth=128`. 

I have picked a linear schedule with three steps: `ws_schedule=(3, 7, 11)`. 
Currently, `torch.compile` creates a new compilation graph per each step in `ws_schedule`.
Therefore, each graph needs to be warmed up separately. I have increased the number 
of warmup steps from `10` to `60`. The compile time is dominated by the first iteration
so this will take approximately `len(ws_schedule)` times longer than before.

Removing document masking had a noticeably negative impact on validation loss, 
however the benefits of a short sequence length counteract this.

### Potential Improvements

- Batch size scheduling: Previously, the block mask acted as a proxy for batch size.
Now block size can be controlled explicitly and sequenced according to critical batch
size theory. I have added code in `distributed_data_generator` that allows for changing the 
batch size and sequence length yielded after the generator is created. 
- The current block mask window schedule `(3, 7, 11)` can almost certainly  be improved upon.
- Hyperparameter tuning might change with smaller sequence length. Rotary base, validation sequence length, learning rates 
etc. should be re-tuned. I haven't done that for this run. 
- FA3 has additional features over Flex Attention that may be useful. 
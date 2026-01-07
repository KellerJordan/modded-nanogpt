## New Record: Paired Head Attention (-3.5s, -65 steps)

Updates in PR:
* Paired head attention
* 0.5s faster implementation of RoPE

The main idea behind paired head attention is to let queries attend to multiple representations of every position, and have multiple logits for the same position going into the same softmax. Implemented by interleaving the k, q, and v for pairs of heads to form twice as long sequences. EG [k1_h1, k2_h1, k3_h1], [k1_h2, k2_h2, k3_h2] -> [k1_h1, k1_h2, k2_h1, k2_h2, k3_h1, k3_h2], repeat for q and v
<img width="4222" height="2326" alt="pairedheadattn" src="https://github.com/user-attachments/assets/728be861-0828-4f52-8bf9-4ed8d9ec0733" />

I update rotary() to run in a single line. Interestingly, pytorch is capable of computing x_flip without performing any data copying.
```
def rotary(self, x_BTHD):
    assert self.factor1.size(0) >= x_BTHD.size(-3)
    factor1, factor2 = (
        self.factor1[None, : x_BTHD.size(-3), None, :],
        self.factor2[None, : x_BTHD.size(-3), None, :],
    )
    x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
    return factor1 * x_BTHD + factor2 * x_flip
```

I implemented a seperate custom rotary method for PairedHeadAttention, that enables the attention forward pass to save an extra parameter reshape copy:
```
# delay q,k reshape until rotary makes data contiguous, to enable view (non-copy)
q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
v = v.reshape(B, T*2, self.num_heads//2, self.head_dim)

q, k = yarn.rotary(q), yarn.rotary(k)

q = q.view(B, T*2, self.num_heads//2, self.head_dim)
k = k.view(B, T*2, self.num_heads//2, self.head_dim)
```

I staggered the custom rotary for PairedHeadAttention such that head1 rotates at an offset from head 2. From limited testing this seemed better. Note that because the data is stored sequentially and attention applies a causal mask, query 4 from head 1 can only attend to keys 1-3 for head 2, whereas query 4 from head 2 can attend to keys 1-4 for head 1.

I double the max doc length and seqlens to account for concatenating two heads:
```
# paired head correction
seqlens = 2 * seqlens
max_len = 2 * max_len
```

I chose to not alter the window size. This means that window size of 128 is really only looking at 64 positions back (64 from head 1 and 64 from head2)

I chose to apply this to some of the short window layers, since I wasn't sure how it would interact with partial key offset on the longer windows. Kept adding to more layers and it kept getting better. I stopped after 4 (maybe more is better).

Seperately, I tested adding a mantissa to Adam, which seemed to give slight improvement. Leaving off this PR to keep the scope focused.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2793, 3.2796, 3.2783, 3.2782, 3.2784]
times = [108.695, 108.844, 108.775, 108.795, 108.87]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0063

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0006), tensor(3.2788))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0679), tensor(108.7958))
```

retiming prior record: 112.262 [112.187, 112.278, 112.321]

If no changes, will merge at 109.2s to be consistent with 3.5s improvement.

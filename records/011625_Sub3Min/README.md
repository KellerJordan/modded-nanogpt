# Sub-3 minute record

### Evidence for <=3.28 mean loss

```bash
$ grep "1385/1385 val" * | python -c "import sys; ss = list(sys.stdin); accs = [float(s.split()[1].split(':')[1]) for s in ss]; print(accs); import scipy.stats; mvs = scipy.stats.bayes_mvs(accs); print(mvs[0]); print(mvs[2]); print(f'p={scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue:.4f}')"
[3.2811, 3.2775, 3.2797, 3.2794, 3.2797, 3.279, 3.2794, 3.2785, 3.2798, 3.2787, 3.279, 3.2782, 3.2776, 3.2799, 3.2788, 3.2795, 3.2794, 3.2792, 3.2787, 3.2819]
Mean(statistic=np.float64(3.27925), minmax=(np.float64(3.2788524181758407), np.float64(3.2796475818241597)))
Std_dev(statistic=np.float64(0.0010712294216474881), minmax=(np.float64(0.0008163810847106718), np.float64(0.001409171376166544)))
p=0.0021
```

```
Mean runtime: 178.7 seconds
Stddev: 101ms
```

# Part 1: Long-Short Sliding Window Attention

Currently, we warmup the context length of the sliding window attention at the same rate in all layers. This attempt warms up the context length differently in some layers instead. This leads to a ~3 ms/step improvement. However, to compensate for the increase in `val_loss`, we needed to add 15 more training steps. Thus, overall, this saves ~3.2 secs on our 8xH100 pod.

- c @YouJiacheng for optimizing and simplifying the code for the sliding window block attention. His efforts made implementing this change much easier.

---

```diff
def dense_to_ordered(dense_mask: torch.Tensor):
    num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
-     indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
+     indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

- def create_doc_swc_block_mask(sliding_window_num_blocks):
+ def create_doc_swc_block_masks(sliding_window_num_blocks: int):
    kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device='cuda')
    q_idx = block_idx[:, None]
    causal_bm = q_idx >= kv_idx
    causal_full_bm = q_idx > kv_idx
    window_bm = q_idx - kv_idx < sliding_window_num_blocks
    window_full_bm = window_bm # block-wise sliding window by @YouJiacheng
    # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
    document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
    document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
    nonzero_bm = causal_bm & window_bm & document_bm
    full_bm  = causal_full_bm & window_full_bm & document_full_bm
    kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
-     return BlockMask.from_kv_blocks(
-         kv_num_blocks,
-         kv_indices,
-         full_kv_num_blocks,
-         full_kv_indices,
-         BLOCK_SIZE=BLOCK_SIZE,
-         mask_mod=document_causal,
-     )
+     short_sliding_window_num_blocks = sliding_window_num_blocks // 2
+     return (
+         BlockMask.from_kv_blocks(
+             kv_num_blocks,
+             kv_indices,
+             full_kv_num_blocks,
+             full_kv_indices,
+             BLOCK_SIZE=BLOCK_SIZE,
+             mask_mod=document_causal,
+         ),
+         BlockMask.from_kv_blocks(
+             torch.clamp_max(kv_num_blocks, torch.clamp_min(short_sliding_window_num_blocks - full_kv_num_blocks, 1)),
+             kv_indices,
+             torch.clamp_max(full_kv_num_blocks, short_sliding_window_num_blocks - 1),
+             full_kv_indices,
+             BLOCK_SIZE=BLOCK_SIZE,
+             mask_mod=document_causal,
+         ),
+     )
...
- block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)
+ long_swa_block_mask, short_swa_block_mask = create_doc_swc_block_masks(sliding_window_num_blocks)
...
skip_connections = []
# Encoder pass - process only the first half of the blocks
+ is_long_block_mask = [True, False, False, False, True, False]
for i in range(self.num_encoder_layers):
+     block_mask = long_swa_block_mask if is_long_block_mask[i] else short_swa_block_mask
    x = self.blocks[i](x, ve_enc[i], x0, block_mask)
    skip_connections.append(x)
# Decoder pass - process the remaining blocks with weighted skip connections
+ is_long_block_mask = list(reversed(is_long_block_mask))
for i in range(self.num_decoder_layers):
+     block_mask = long_swa_block_mask if is_long_block_mask[i] else short_swa_block_mask
    x = x + self.skip_weights[i] * skip_connections.pop()
    x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)
```

---

This is similar to the Local-Global Attention in [Gemma 2](https://arxiv.org/pdf/2408.00118), except we use Sliding Window Attention for all layers and instead just warmup the context length at different rates during training.

---

This attention mechanism is inspired by the Local-Global Attention introduced by the [Gemma 2](https://arxiv.org/abs/2408.00118) paper (and more recent "hybrid" architecutres). But there are two key differences:

1. We use [Sliding Window Attention](https://arxiv.org/abs/2004.05150) for both the "global attention" (i.e. "long SWA") and the "local attention" (i.e. "short SWA") parts. The difference between the two is that the "long SWA" has double the context length of the "short SWA".
2. We also **warmup the context length** of both the sliding window attention mechanisms, but **at different rates**. The "long SWA" context length is warmed up at a double the rate compared to the "short SWA".

We also made a speedrun-specific decision to only use "long SWA" in the first, fifth, and last layers. The first, because we do not want to compress information too early in the network. The last, because the model architecture we use for the speedrun follows a UNet-like structure, and we want the first and the last layers to be symmetric. And finally, the fifth layer, mainly because it is empirically the best choice for the speedrun.

This would have been very difficult to implement without PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/).

# Part 2: Attention scale, merged QKV weights, lowered Adam eps, and batched Muon

Changelog:

- @leloykun's & @YouJiacheng's & @brendanh0gan's attention scale modifications
- @tysam-code's & @brendanh0gan's merged QKV weights
  - @scottjmaddox's Batched Muon implementation (to avoid concat on the QKV weights)
  - (c) @YouJiacheng for pointing out this optimization
- @tysam-code's & @YouJiacheng's Adam eps fix

Additional credits:
- @Grad62304977 for suggesting Local-Global Attention which eventually morphed into this implementation

### Attention Scale Modification

[README is a WIP]


# Sub-3 minute record

## Evidence for <=3.28 mean loss

```bash
$ grep "1393/1393 val" * | python -c "import sys; ss = list(sys.stdin); accs = [float(s.split()[1].split(':')[1]) for s in ss]; print(accs); import scipy.stats; mvs = scipy.stats.bayes_mvs(accs); print(mvs[0]); print(mvs[2]); print(f'p={scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue:.4f}')"
[3.276, 3.2785, 3.2796, 3.2788, 3.2789, 3.2768, 3.2775, 3.2784, 3.2767, 3.2792, 3.2807, 3.2801, 3.2805, 3.2777, 3.2789, 3.2799, 3.2786, 3.2776, 3.2791, 3.2808, 3.2776, 3.2786, 3.2774, 3.2832, 3.277, 3.2789, 3.2784, 3.2766, 3.2755, 3.2784, 3.2798, 3.2825]
Mean(statistic=np.float64(3.27869375), minmax=(np.float64(3.2781784751445135), np.float64(3.2792090248554864)))
Std_dev(statistic=np.float64(0.0017621789337662857), minmax=(np.float64(0.0014271074116428265), np.float64(0.002179878373699496)))
p=0.0001
```

```
Mean runtime: 179.8 seconds
Stddev: 101ms
```

## Details on Long-Short Sliding Window Attention

Currently, we warmup the context length of the sliding window attention at the same rate in all layers. This attempt warms up the context length differently in some layers instead. This leads to a ~3 ms/step improvement. However, to compensate for the increase in `val_loss`, we needed to add 15 more training steps. Thus, overall, this saves ~3.2 secs on our 8xH100 pod.

This is similar to the Local-Global Attention in [Gemma 2](https://arxiv.org/pdf/2408.00118), except we use [Sliding Window Attention](https://arxiv.org/abs/2004.05150) for all layers and instead just warmup the context length at different rates during training.

We made a speedrun-specific decision to only use "long SWA" in the first, fifth, and last layers. The first, because we do not want to compress information too early in the network. The last, because the model architecture we use for the speedrun follows a UNet-like structure, and we want the first and the last layers to be symmetric. And finally, the fifth layer, mainly because it is empirically the best choice for the speedrun.

This would have been very difficult to implement without PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/).

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


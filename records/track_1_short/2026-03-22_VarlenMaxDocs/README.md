# Reduce Varlen max_num_docs

While working on refactoring `nanochat` for FlashAttention varlen, I discovered that the size of the `cu_seqlens` tensor (which contains the document offsets in our 1D training batch) matters more than expected. 

Analyzing the fineweb dataset and better tailoring the size of `cu_seqlens` gave a ~500ms speed improvement.

Additionally, I tried some recent `nanochat` changes and found that the reduction of Muon's second beta from `0.95 --> 0.9` slightly improved loss, so I included that as well (no reduction in steps).

```
                    Runs  Steps   Time μ   Time σ  Time +/-   Time p   Loss μ   Loss σ  Loss +/-        p
baseline               8   1490  86.7400   0.0497    0.0000       NaN  3.2787   0.0015    0.0000   0.0218 
max_num_docs           4   1490  86.2422   0.0479   -0.4977   0.0000   3.2785   0.0012   -0.0002   0.0444 
muon-beta2             4   1490  86.7758   0.0274    0.0358   0.9296   3.2785   0.0007   -0.0002   0.0122 
combined (this PR)     8   1490  86.2246   0.0300   -0.5154   0.0000   3.2784   0.0013   -0.0003   0.0054 
```

### Choosing max_num_docs

We need `cu_seqlens` to be a fixed size for `torch.compile`. However, because we have a separate compute graph per batch size, we can safely choose a different `cu_seqlens` size for each batch size.

To determine appropriate sizes, I analyzed the first 10 shards (1.45M documents).

For token counts overall:
* median 403
* mean 690 
* min 32

I constructed the actual batches used and counted the number of documents in each, across our three batch sizes / stages:

| Stage | tokens/rank | current `max_num_docs` | observed max | proposed | savings |
|-------|-------------|----------------------|--------------|----------|---------|
| Stage 1 | 16,384 | 128 | 56 | **64** | -64 slots (50%) |
| Stage 2 | 32,768 | 128 | 90 | **96** | -32 slots (25%) |
| Stage 3 | 49,152 | 256 | 121 | **128** | -128 slots (50%) |

I've hardcoded those batch size --> num docs mappings, but with a fallback to the original method of rounding up conservatively if a different batch size is encountered.

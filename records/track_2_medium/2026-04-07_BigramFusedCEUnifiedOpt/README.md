## New Record: Bigram Hash Embeddings, Fused FP8 Cross-Entropy, Unified Optimizer (-91s)

I analysed the 22 post-bulk-transfer short track records to identify the highest-impact unported improvements. The implementation order was driven by dependencies — bigram hash embeddings (the largest single gain) required the unified optimizer infrastructure as a prerequisite.

Summary of changes:
* Fused softcapped cross-entropy with FP8 pipeline (short track #59, -0.9s on short track)
* Bigram hash embeddings with sparse gradient communication (short track #62 + #67, -5.6s + -0.75s on short track)
* Unified NorMuonAndAdam optimizer with parameter banks (short track #61, -1.0s on short track)
* Transposed FP8 lm_head (CastedLinearT) with Triton tiled transpose_copy/transpose_add for tied embeddings
* Mantissa tracking for bf16 NorMuon precision
* Step count reduction 4740 → 4620
* `coordinate_descent_tuning` enabled (~5s contribution; the remaining ~86s came from the changes above)

### Fused Softcapped Cross-Entropy with FP8 Pipeline

The FusedSoftcappedCrossEntropy kernel from triton_kernels.py performs FP8 matmul + softcap (23 x sigma((logits + 5) / 7.5)) + cross-entropy in a single kernel, taking hidden states directly to per-token loss values. This eliminates all intermediate tensor materialization in the loss computation path.

Combined with CastedLinearT (transposed lm_head weight storage, (model_dim, vocab_size) = (1024, 50304)), this also resolves the slow gradient accumulation kernel caused by mismatched memory layouts in the backward pass.

Embedding tying during the first 2/3 of training is maintained using Triton tiled `transpose_copy` (lm_head.T -> embed) and `transpose_add` (embed.grad.T -> lm_head.grad) for coalesced memory access.

### Bigram Hash Embeddings

A 251,520 x 1024 embedding table indexed by a hash of consecutive token pairs. The hash function runs on CPU during data loading (zero GPU overhead). Bigram context is injected into the residual stream at every transformer layer via learned per-layer scaling coefficients (initialized to 0.05).

Sparse gradient communication reduces the overhead by ~80%: only embedding rows with non-zero gradients are transmitted between GPUs, using an async all-to-all pattern that overlaps with the backward pass.

### Unified NorMuonAndAdam Optimizer

Replaced the three separate optimizers (NorMuon, DistAdam for embeddings, DistAdam for scalars) with a single class using per-parameter configuration. Communication is explicitly scheduled via `scatter_order` and `work_order` lists rather than backward hooks.

Per-layer attention and MLP weights are consolidated into parameter banks (attn_bank: 16 x 4096 x 1024, mlp_bank: 16 x 2 x 4096 x 1024), eliminating the stacking/unstacking memcpys that were required for reduce-scatter. NorMuon parameters use mantissa tracking (uint16 buffer alongside bf16 weights) for float32-equivalent update precision.

### What I Tested That Didn't Work

**Muon beta2 0.95 -> 0.9** (short track #78): I measured the effect across the full training curve. At steps 250-1000, beta2=0.9 shows a clear advantage (up to -0.023 loss). By step 3000, the curves converge to identical. Final loss: 2.9056 vs 2.9053 — within noise.

This matches the intuition described by @ClassicLarry on PR #248 — lower beta2 (like lower LR) reduces "froth" from rapid gradient changes, giving a lower mid-training reading without actually descending the loss curve faster. The short track's ~1800 steps means the early advantage persists to the final loss; the medium track's 4620 steps lets it wash out entirely. I kept beta2 at 0.95.

### Relationship to PR #248

This submission shares FP8 lm_head re-enablement (CastedLinearT) with PR #248. The remaining changes are orthogonal:
* **This PR only**: fused softcapped CE, unified optimizer, parameter banks, bigram embeddings, sparse comms, mantissa tracking, step reduction
* **PR #248 only**: Muon LR tuning (0.015 -> 0.012, ~2s), FP8 scale profiling tuned for medium-track activation ranges

Adopting #248's tuned LR and FP8 scales on top of this architecture should yield further improvement. Loss budget of 0.0094 below the 2.92 target (vs record #18's 0.0034 margin) leaves ample room for this and additional step reduction.

### Timing and Validation
```
import scipy.stats
import torch

losses = [2.9099, 2.9106, 2.9110, 2.9108, 2.9106]
times = [950.136, 950.425, 950.615, 949.573, 950.776]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 2.92, alternative="less").pvalue)
# p=0.0000

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0004), tensor(2.9106))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.4735), tensor(950.3050))
```

timing prior record: 1041.277

# Dion2 Benchmark Results

Benchmarks run on 4x H100 GPUs, 1555 steps of `train_gpt.py` (modded-nanogpt).

## Baseline: NorMuon vs Dion2

| Branch | Optimizer | Val Loss | Train Time | Step Avg | Peak Memory |
|---|---|---|---|---|---|
| 4xH100 | NorMuon | **3.2781** | 267,657ms | 172.13ms | 31,745 MiB |
| dion2-fix-prealloc | Dion2 | 3.3744 | 267,508ms | 172.03ms | 31,805 MiB |

NorMuon achieves ~0.10 lower val loss with identical speed and memory.
Dion2 does not yet match NorMuon on this benchmark.

## Dion2 Performance Optimizations

Three optimizations were tested independently on top of the base Dion2 branch,
then compile and eliminate-zeros were rebased on prealloc to test combinations:

| Variant | Val Loss | Step Avg | Peak Memory | Notes |
|---|---|---|---|---|
| prealloc (base) | 3.3744 | **172.03ms** | 31,805 MiB | Pre-allocate full_update buffer |
| prealloc + compile | 3.3740 | 173.18ms | 31,897 MiB | torch.compile on select/ortho/reconstruct |
| prealloc + eliminate-zeros | 3.3758 | 172.79ms | 31,745 MiB | Sparse cautious update (no dense buffer) |

### Observations

- **Prealloc is the only useful optimization.** Avoiding `torch.zeros_like` every
  step saves a CUDA allocation per Dion2 parameter group per step.
- **torch.compile hurts.** Compiling the selection/orthogonalization/reconstruction
  into a single graph adds ~1ms/step overhead rather than saving it. The
  graph-capture cost likely outweighs kernel launch savings at this scale.
- **Eliminate-zeros is neutral to slightly worse.** Replacing the dense
  `full_update` scatter with sparse `index_add_` in the cautious update saves
  60 MiB but adds ~0.8ms/step. The upstream Dion library already uses
  `index_add_` natively, so this optimization is not applicable there.
- **Val loss is identical across all Dion2 variants** (~3.374), confirming
  these are pure performance changes with no effect on model quality.

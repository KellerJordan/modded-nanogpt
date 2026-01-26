Updates in PR, building off latest [PR#201](https://github.com/KellerJordan/modded-nanogpt/pull/201):
- Untie value embeddings to replace pattern of `VE[.12...012]` with `VE[.01...345]`.

## Ideas

The latest idea is inspired by the principle of Bigram/Engram Hash embeddings to allow more uncontextualised information into the architecture. I believe there's further room in adding sparsity to be explored, i.e. Meta's STEM and optimizing the bwd to not communicate unaccessed indices in sparse embeddings.

I tried these ablations:
- Same weights but more tied layers: `ve[012...012]` (1) 
- Fewer VE tied layers: `ve[01…01]`
- Different layers: `ve[.01…01.]`
- Apply VE to all layers
Surprisingly, the [old finding](https://github.com/KellerJordan/modded-nanogpt/pull/194) (1) that adding VE to the first layer no longer held up. I hypothesize that improvements in between (Partial Key Offset, Bigram Hash) removed the need for this somehow, but I'm not quite sure why.

Unrelated to this VE idea, I tried to optimize the bwd with `torch.Embedding(..., sparse=True)`, but I'm running into 2 issues: 1) the metadata overhead of keeping count of the accessed indices complicates the Adam step, especially when distributed, and 2) Full CUDA graphs aren't preserved under sparse embeddings in [2.10](https://github.com/pytorch/pytorch/issues/150656).

## Timing

The timings (-25 steps) hold up to the latest PR, and are significant (p<0.001).
- H0: Control is latest WR at 1600 steps.
- H1: Untie VE at 1575 steps.

```
                   Runs   Time μ  Time σ  Time +/-  Time p  Loss μ  Loss σ  Loss +/-      p
H0                   10  98.1066  0.0904    0.0000     NaN  3.2778  0.0010    0.0000 0.0000
H1                   10  97.7141  0.0533   -0.3925  0.0000  3.2784  0.0009    0.0006 0.0002

H0:
  losses = [3.2776, 3.2785, 3.277, 3.2772, 3.2788, 3.2764, 3.2763, 3.2784, 3.2788, 3.2791]
  times  = [98.081, 98.053, 97.895, 98.193, 98.136, 98.093, 98.11, 98.124, 98.156, 98.225]
  (n = 10)

H1:
  losses = [3.2784, 3.2778, 3.2792, 3.2799, 3.2776, 3.2782, 3.2771, 3.2786, 3.2775, 3.2794]
  times  = [97.715, 97.704, 97.725, 97.666, 97.725, 97.632, 97.755, 97.831, 97.697, 97.691]
  (n = 10)
```

The 3 extra embedding weights in H1 added to the average step time, compared to H0. I suggest we time this improvement `time delta = 98.1-97.7 = 0.4s`.
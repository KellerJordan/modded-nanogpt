This record changes the multiplication location of the Self-attention lambdas to pre-scale the QKV matrix instead of directly scaling V. Also, the warm-up process is fixed in order to correctly pre-compile all the code paths.

## Timing and Validation

A fixed overhead was removed by the compile fix, the rest of the gain was an increase in speed per time-step.

```
import scipy.stats
import torch

losses = [3.2762, 3.2785, 3.2789, 3.2774, 3.2769, 3.2775, 3.275, 3.2769, 3.2808, 3.2797]
times = [131.233, 131.239, 131.225, 131.284, 131.273, 131.227, 131.325, 131.047, 131.236, 131.145]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0014

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0017), tensor(3.2778))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0776), tensor(131.2234))
```

Previous record (timed on the same machine):

```
import scipy.stats
import torch

baseline_times = [131.795, 131.707, 131.753]
print("time:", torch.std_mean(torch.tensor(baseline_times)))
# time: (tensor(0.0440), tensor(131.7517))
```

This shows an improvement of $ \approx 0.5 $ seconds.

## Changing Self-Attention Lambdas

One of the architecture modifications performed on GPT-2 is having multiple separate embeddings, called value embeddings, for the tokens
that are mixed directly into the self-attention V projections, with learned weights, lambdas.
This is a straight-forward `v = v * sa_lambda[0] + ve * sa_lambda[1]`. The V projections are big (32768 by 768 elements on average),
whereas in order to scale the V projections, we can intervene in multiple places where we'd need to do less work.
The $ W_V $ weight matrix is only 768 by 768, so by changing $ \lambda (W_V x) $ to $ (\lambda W_V) x $ we can save work.

It's still not straight-forward that this would save time as we are running under `torch.compile()`,
which could, in theory, fuse the scalar multiply into the $ W_V x $ matrix-multiply kernel by itself, which would hide the
cost of the scalar multiplication with the memory accesses for the matrix-multiply. This doesn't happen, though, probably 
because for efficiency reasons, we keep a single matrix with QKVO and do the QKV projections as a single matrix-multiply.
This does mean we actually pre-multiply the entire QKV matrix by our scalar, which loses efficiency but still works because
the Q and K projections are immediately normed (thanks, QK-norm!).

**Note**: even in layers without value-embeddings, we scale the value projections, and this is probably beneficial for the model as
we RMS-norm the input to the SA block, while the residual stream magnitude increases as we reach deeper layers in the network. This
allows scaling the output of the SA block. In trained models, the lambdas generally grow with network depth.

### Minor Variations

All of the manipulations done on V projections are linear - we only do additions and multiplications by scalars,
so there are multiple spots where we can scale V. It would maybe require re-scaling the value-embeddings lambda initialization
 a bit to make the algebra work out the same, but it "*shouldn't*" matter.

* Multiplying the $ W_O $ matrix, which increased the loss for reasons I did not understand (my successor did, however!)
* Multiplying the output of the sparse-attention gates, which is actually the most efficient option in terms of total operation count.
Also seems to have increased the loss.

## Warm-up Bug-fix

A previous record which improved the DistAdam compute-communication overlap did not update the `torch.compile()` warmup phase to cover
all of the newly-added code-paths, which caused a slow-down in the 2nd iteration as a recompile had to happen.

The bug was discovered accidentally while looking for a fix to a problem where one of every few hundred training steps would become intermittently
slow (the cause is not a recompile, nor is it an NCCL issue, though I looked exhaustively). The problem only disappeared once a `gc.collect()` was
added before starting the timer for the run (a run with `gc.set_debug(gc.DEBUG_STATS)` showed that there's a memory leak somewhere and one gen-2 collection
that does a stop-the-world collection which roughly fit the timeline).
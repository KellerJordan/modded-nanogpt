# Unified, Zero-Copy Optimizer and Transposed LM Head

The main contributions of this record are:

1. A combined and simplified optimizer, `NorMuonAndAdam`, oriented towards per-parameter rather than per-optimizer configuration.
2. MLP and Attention parameter banks, allowing for a uniform workload across the gpus and eliminating all memcpys.
3. A transposed memory layout for the LM head, which resolves the slow element-wise gradient accumulation kernel.

```
                                    Runs   Time μ  Time σ  Time +/-  Time p  Loss μ  Loss σ  Loss +/-      p
baseline                               8 106.8004  0.0528    0.0000     NaN  3.2788  0.0012    0.0000 0.0097
unified-optimizer                      8 106.6707  0.0725   -0.1296  0.0007  3.2772  0.0010   -0.0016 0.0001
unified-optimizer - head.T             8 106.3992  0.0615   -0.4011  0.0000  3.2789  0.0010    0.0001 0.0088

baseline:
  losses = [3.2792, 3.2783, 3.2798, 3.2791, 3.2790, 3.2780, 3.2802, 3.2765]
  times  = [106.7170, 106.8840, 106.8360, 106.8120, 106.8040, 106.7550, 106.7650, 106.8300]
  (n = 8)

unified-optimizer:
  losses = [3.2768, 3.2777, 3.2770, 3.2768, 3.2780, 3.2783, 3.2751, 3.2779]
  times  = [106.6340, 106.6160, 106.5600, 106.7110, 106.6180, 106.7330, 106.7360, 106.7580]
  (n = 8)

unified-optimizer - head.T:
  losses = [3.2782, 3.2785, 3.2789, 3.2793, 3.2811, 3.2776, 3.2786, 3.2788]
  times  = [106.4630, 106.3530, 106.3170, 106.4080, 106.4600, 106.4730, 106.3850, 106.3350]
  (n = 8)
```

Note: I also included the `triton_kernels.py` change, and added a line at the top of train_gpt.py to pull the kernel code into the log file.

## Unified Optimizer

### Parameter Sharding

Instead of stacking and unstacking the weight matrices from multiple layers, this code utilitizes a "parameter bank" strategy. 

This concept was introduced in a previous PR for the attention gates and ve gates, and I've extended it to the attention and mlp projection matrices.

The QKVO weight matrices from all 10 attention modules are stored as a single parameter of shape (10, 4*768, 768). This can be 'reshaped' without memory movement to (40, 768, 768), and then sharded evenly across the 8 GPUs (so that each gets five attention matrices). With this balanced workload we could expect the GPUs to all complete at the same time, compared to the current strategy where the GPUs working on MLP weights need more time.

The MLP weight matrices (11 modules, 2 matrices per module) require padding to be sharded evenly. Their parameter bank is (12, 2, 4*768), and this is "reshaped" to (24, 4*768) so that each GPU receives 3 matrices to process.

(Note that means we're paying the communication and optimizer overhead for an entire additional MLP, so if you can think of somewhere to use it... It will still have to save a lot of steps to outweigh its forward/backward cost, though).

### Individual Parameters

An additional benefit of the parameter banks is that it eliminates the need for the concept of "parameter groups".

Optimizer groups exist to handle the fact that we typically have multiple instances of the same functional matrix--e.g., we have 10 layers worth of attention weights--which all share settings like learning rate.

Groups add an additional layer of abstraction and complexity to the optimizer code. With the parameter banks, our model no longer has per-layer parameters, and instead has 'only' the 13 distinct parameters in the table below (defined in the TrainingManager init function).

```python
# - Ordering dictates when to launch reduce/reduce_scatter operations
# - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
self.param_table = {
    "attn":           {"optim": "normuon", "comms": "sharded",     "adam_betas": None},
    "mlp":            {"optim": "normuon", "comms": "sharded",     "adam_betas": None},         
    "scalars":        {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.9,  0.99]},
    "ve0":            {"optim": "adam",    "comms": "sharded",     "adam_betas": [0.75, 0.95]},
    "ve1":            {"optim": "adam",    "comms": "sharded",     "adam_betas": [0.75, 0.95]},
    "ve2":            {"optim": "adam",    "comms": "sharded",     "adam_betas": [0.75, 0.95]},
    "smear_gate":     {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.9,  0.99]},
    "skip_gate":      {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.9,  0.99]},
    "attn_gate_bank": {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.9,  0.99]},
    "ve_gate_bank":   {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.9,  0.99]},
    "x0_lambdas":     {"optim": "adam",    "comms": "replicated",  "adam_betas": [0.65, 0.95]},
    "lm_head":        {"optim": "adam",    "comms": "sharded",     "adam_betas": [0.5,  0.95]}, # Before embed for grad aggregation
    "embed":          {"optim": "adam",    "comms": "sharded",     "adam_betas": [0.5,  0.95]}, # Last (skipped when tied)
}
```

The one remaining component that does conceptually act like a 'group' is the value embedding tables, which I've individually labeled as 've0', 've1', and 've2' to treat them as distinct parameters.

### Replacing Hooks with Explicit Ordering

The most substantial benefit to registering gradient accumulation hooks was to overlap communication with the lengthy lm_head gradient accumulation kernel. With that removed (as described further down), there's less benefit to overlapping the smaller kernels, and I chose to remove the feature for simplicity and added flexibility in ordering the communication and workload.

The order of the parameters in `param_table` dictates the order in which we issue the scatter or all_reduce operations, and then a separate list specifies the order that we work on them.

**Debugging Comms**

I've been having issues lately with getting good trace data. The profiler is somehow slowing things down much more than usual, and the communication data in particular doesn't look right. This has made it hard to arrive on a strategy directly, so I'm just using one that "seems to work well".

I did, however, print out the order the hooks fire in, for reference. To include the NorMuon parameters as well, I added hooks to them and manually set the gradient accumulation steps to 2 so that they would trigger.

Here's the order they're called in:

1. scalars
2. embed
3. smear_gate
4. lm_head
5. skip_gate
6. mlp
7. attn
8. ve_gate_bank
9. ve2
10. attn_gate_bank
11. ve1
12. x0_lambdas
13. ve0

In general, I've found it hard to reason about the behavior of the hooks--I generally don't get the results I would expect when I try to base a strategy off of what I see in the hook order and the trace files. 

That could just be my fault, but I'd really like to get some lower-level logging, and/or a way to align the visualizations of all 8 gpus, before messing with it any further.

**Validation Loss**

I can't explain the reduced validation loss in the unified optimizer (row 2) vs. the baseline. The loss dropped only when I switched away from the backward hooks to the explicit ordering, and I don't see a clear connection in the code to explain it.

## Transposed LM-Head

I transposed the memory layout of the LM head so that it now has shape (768, 50304), and updated the FP8 implementation to support this.

It appears to have identical forward/backward performance to the current one, and addresses a key issue in our optimizer segment.

### Mismatched Memory Layouts

Currently, the backward pass produces gradients for the LM head with layout (768, 50304), but the head is stored as (50304, 768).

I've learned that, while CUDA can often hide the cost of mismatched memory layouts when performing matrix multiplications, it's not possible to hide this problem for element-wise operations.

This is because values can be read once and used multiple times in matmul, but for element-wise operations each value is read and used exactly once. When the memory layouts don't match, it has to stride through one of the matrices, resulting in kernels that run substantially slower.

This is the reason for the slowness of the current gradient accumulation kernel, which takes maybe 4x longer than the other embedding tables despite being the same size and shape.

Transposing the head brings the weights and gradients into alignment. This has the biggest impact for the NorMuon-only steps, where we don't have any way to overlap communication with the slower kernel.

### Tied Embeddings

This change was more clearly beneficial back before we re-tied the embeddings. The tied embeddings create a problem--the input embeddings still need to have their current shape of `(50304, 768)` so that the embeddings are laid out consecutively in memory and can be selected efficiently.

To resolve this, I "untied the embeddings" in the sense that there are always two separate matrices involved, but I keep them tied and avoid the additional memory traffic by summing their gradients prior to the scatter operation. 

This actually brings us back where we started--expensive elementwise operations on two misaligned matrices--but now it's inside of the optimizer step code where we can control it better / hide it reliably.

### Optimizer State

One challenge I haven't fully resolved is that when the embeddings become untied, because a given GPU's embed shard and lm_head shard are not the same shape, I can't copy the optimizer state from one to the other. This means that the embeddings become independent with their momentum buffers set to 0, right at the 2/3s transition point where things can get rocky. This hurts the validation loss some.

To address this, I adjusted the untying point back by 50 steps so the input embeddings have time to build up a momentum buffer before hitting the batch size increase.

## Ideas

With the Muon memcpys gone, we have quite a lot of compute available during the optimizer window. Here are a few things I played with that might deserve exploring more, or might spark other ideas.

### Interpolated Embedding Updates

Because we're now manually combining the gradient updates of the input embeddings and LM heads to tie them, we have the opportunity to "mix" them differently. 

I did a batch of experiments at one point where I linearly untied the embeddings over the course of the middle third of training, and this seemed to improve loss, maybe 5 or 10 steps worth. 

It didn't work the next time I tried, though, and my implementation was too complex, so I let it go. I think it could be done more simply, though, so it might be worth another look.

### Blending / Switching Muon and Adam

We can afford to run both NorMuon and Adam on the MLP and Attention weights at the same time (that's how cheap Adam is!). I tried some experiments blending their updates over the course of training, but nothing looked promising.

What I'd still be interested to try, though, is switching the projection matrices from Muon to Adam towards the very end. Compared to Muon's orthogonalization constraint, I'm wondering if the per-parameter freedom of Adam's updates might allow it to settle into a better local optimum (or find one faster) at the end of training.

### Starting `forward` Within `step`

It's not a giant time savings, but inside of the optimizer window, while still waiting for the attention and MLP weights to transfer, we can start selecting the next input embeddings, normalizing them, and applying the smear gate.

I started on this, but set it aside when I realized how complicated / impossible it would be when doing gradient accumulation steps. 

I think the solution there would probably be to define a separate training loop that's only for the 8x gpu setup (with no gradient accumulation support), where we could apply this.

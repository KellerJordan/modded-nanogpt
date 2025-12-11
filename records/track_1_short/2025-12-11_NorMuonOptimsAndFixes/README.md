# NorMuon & DistAdam Optimizations and Fixes

This update includes a number of algebraic and fusion optimizations to NorMuon, as well as an important fix to the variance normalization step.

Fixing this issue enabled @shenberg's suggestion from the prior record of pre-multiplication of `sa_lambdas[1]` with $W^O$, which I've included in this submission.

This update also addresses a synchronization issue with the DistAdam implementation, which was causing it to send out some of the Adam weights even on non-Adam steps. 

Because Adam is primarily communication-bound, I've also modified DistAdam to make its parameter grouping and ordering more explicit (similar to NorMuon's approach) to make it easier to reason about and optimize.

**Timings & Loss**

The first row is the prior record, the second is my changes, and the third row (which is this submission) incorporates the pre-multiplication with $W^O$ from @shenberg.

|                        | Runs |   Time μ | Time σ | Time +/- | Loss μ | Loss σ | Loss +/- |      p |
|-----------------------:|-----:|---------:|-------:|---------:|-------:|-------:|---------:|-------:|
|        Baseline        |  4 | 132.7217 | 0.1176 |   0.0000 | 3.2792 | 0.0008 |   0.0000 | 0.0632 |
|         Mine      | 13 | 131.4852 | 0.0680 |  -1.2365 | 3.2777 | 0.0016 |  -0.0015 | 0.0001 |
| Mine + PreMul-O | 10 | 131.2106 | 0.0660 |  -1.5111 | 3.2777 | 0.0020 |  -0.0015 | 0.0024 |

```

Baseline:
  losses = [3.2796, 3.2781, 3.2798, 3.2793]
  times  = [132.5520, 132.8170, 132.7390, 132.7790]
  (n = 4)

NorMuon and DistAdam Optimizations and Fixes:
  losses = [3.2790, 3.2759, 3.2766, 3.2761, 3.2776, 3.2804, 3.2759, 3.2774, 3.2772, 3.2806, 3.2790, 3.2776, 3.2770]
  times  = [131.5030, 131.6180, 131.5430, 131.4140, 131.4730, 131.4550, 131.4170, 131.5680, 131.5520, 131.4520, 131.3850, 131.4570, 131.4710]
  (n = 13)

Above + Pre-Multiply with W_O:
  losses = [3.2758, 3.2807, 3.2776, 3.2757, 3.2793, 3.2764, 3.2795, 3.2788, 3.2785, 3.2748]
  times  = [131.2380, 131.2910, 131.1550, 131.1570, 131.2540, 131.3040, 131.1550, 131.1300, 131.1570, 131.2650]
  (n = 10)

```

## Contents

- [1. NorMuon Variance Normalization Fix](#1-normuon-variance-normalization-fix)
  - [1.1. Attention and MLP Weight Layout](#11-attention-and-mlp-weight-layout)
  - [1.2. Confirming the Benefit](#12-confirming-the-benefit)
  - [1.3. Should we Address $W^O$?](#13-should-we-address-wo)
  - [1.4. Impact to Smear Gate](#14-impact-to-smear-gate)
- [2. GPU & Algebraic Optimizations](#2-gpu--algebraic-optimizations)
  - [2.1. NorMuon Algebraic Optimizations](#21-normuon-algebraic-optimizations)
  - [2.2. Compiled Helpers](#22-compiled-helpers)
    - [2.2.1. Aside: 0-D CPU Tensors](#221-aside-0-d-cpu-tensors)
  - [2.3. Polar Express Defensive Memcpy](#23-polar-express-defensive-memcpy)
  - [2.4. Combined Improvement](#24-combined-improvement)
- [3. Communication](#3-communication)
  - [3.1. DistAdam Synchronization](#31-distadam-synchronization)
  - [3.2. Parameter Group Ordering](#32-parameter-group-ordering)
  - [3.3. Type-Casting on Host](#33-type-casting-on-host)
- [4. Future Work / Opportunities for Improvement](#4-future-work--opportunities-for-improvement)
  - [4.1. Combined Optimizers](#41-combined-optimizers)
  - [4.2. Hyperparameters](#42-hyperparameters)
  - [4.3. Smear Gate](#43-smear-gate)
  - [4.4. Higher Cost Optimizers](#44-higher-cost-optimizers)
- [5. Appendix - Failed Experiments](#5-appendix---failed-experiments)
  - [5.1. Per-Head Muon](#51-per-head-muon)
  - [5.2. Autograd Layout Issues](#52-autograd-layout-issues)

---

## 1. NorMuon Variance Normalization Fix

NorMuon ("Neuron-wise Normalized Muon", [paper](https://arxiv.org/pdf/2510.05491), contributed by @zichongli5) improves upon Muon by normalizing the update variance at the level of individual weight vectors, such as the FFN neurons and Attention head basis vectors.

The existing code determined the orientation of the weight vectors by comparing matrix width to height ([here](https://github.com/KellerJordan/modded-nanogpt/blob/9add1e11fbe0b2bfb55c8b480256fae065e1b12b/train_gpt.py#L597)).

This rule did not correctly capture the orientation for:

1. The smear and attention gates
2. The attention output heads

I was able to correct the handling of the gate weights by adding a special case to the shape-checking rule.

### 1.1. Attention and MLP Weight Layout

Correcting for the output heads required a layout change to the attention weights (and therefore to the MLP weights as well, to maintain matching shapes). 

Like most attention implementations, the current version stores the Output heads transposed relative to the QKV heads. Specifically, it stores QKV heads "horizontally", each with shape (128, 768), and the output heads "vertically", with shape (768, 128), to avoid a transpose operation during the output projection.

The "simplest" solutions require things we want to avoid:
1. Separating $W^O$ from QKV
2. Costly transpose operations

Instead, I changed the layout such that all heads are stored horizontally, and QKVO form a vertical stack with a combined shape of (4*model_dim, model_dim).

This allows for efficient projections and slicing operations, and puts all basis vectors in the same orientation for NorMuon.

To maintain the matching shapes of the attention and mlp weights (for efficient optimizer grouping), I also transposed the mlp weights to match.

A nice conceptual benefit of the change is that the orientations are easier to reason about--all neurons, basis, and gate weights in the model are now stored horizontally.

### 1.2. Confirming the Benefit

To verify that these changes had a positive effect on the model, I ran the baseline plus three variants:
The corrected orientation for:
1. The attention gate only.
2. The attention and smear gates.
3. Both gates and the Attention Output matrix.

Averaged over four runs, all three variants outperformed the baseline:

<img src='https://lh3.googleusercontent.com/d/1kcsAdOlgQwbHcQhjt_jvBqIUk0zIMcm1' alt='Plot of final validation loss for variance fixes applied to gates and W_O' width='640' />

Note that correcting both gates but not $W^O$ (pink) performed better than with the $W^O$ correction (light blue).

### 1.3. Should we Address $W^O$?

Here are two arguments in favor of addressing $W^O$ the above.

1. It has an affect on the learning curve which could be indicative of the current implementation causing over-fitting. The curve (light blue) is higher until crossing under the baseline near the very end of training.

<img src='https://lh3.googleusercontent.com/d/1zjeJUaiFFzbe82deptO1zljF017rZktB' alt='Plot showing the overall higher validation loss curve with the W_O fix in place' width='720' />

2. It seems to have resolved the challenges @shenberg was having with applying their Pre-Multiplication technique to the $W^O$ matrix. It was inexplicably increasing validation loss. With this fix in place, however, it worked well.

### 1.4. Impact to Smear Gate

The smear gate is unique in that, because it is only a single weight vector, there is no population to normalize over. When applied in the correct orientation, this normalization step is effectively a no-op.

When applied in the other orientation, as it is currently, it normalizes each position of the smear gate weight updates independently, which constricts the weight updates to either -1 or +1. 

This could explain why fixing the normalization on the smear gate had the biggest positive impact in the above plot.

---

## 2. GPU & Algebraic Optimizations

Reviewing pytorch profiler traces for the NorMuon optimizer overall revealed a number of opportunities for improvement.

### 2.1. NorMuon Algebraic Optimizations

The straightforward implementation of the variation normalization results in reading the gradient updates from memory multiple times. We can avoid this by re-using existing quantities. 

For instance, for each weight vector we calculate:
1. The square of each element
2. The sum of these squares
3. Divide by 768 (the length) to get the mean

Later, we need the sum of squares again; we can get them cheaply by "undoing" the third step by multiplying the mean by 768.

### 2.2. Compiled Helpers

I defined compilable functions for performing the variance normalization, and another for calculating the cautious weight decay (which I was also able to optimize by avoiding a memory read and performing this operation in-place). The compiler successfully fused many of the operations and reduced the time required for these steps significantly.

The below illustration is for an attention weight update and is to scale. It shows the combined impact of these optimizations (to variance reduction and cautious weight decay) when running NorMuon on the attention weight updates.

<img src='https://lh3.googleusercontent.com/d/1Bvw9aJa5a0Q3kpzELc8LmYZWDLa-4ZMT' alt='Illustration of fusions for variance and cautious weight decay' width='900' />

#### 2.2.1. Aside: 0-D CPU Tensors

I've learned it's critical with these compiled helper functions to think carefully about the behavior of the inputs and how they will affect compilation. 

The compiler can produce and cache multiple variants of a function, which is fine if there aren't too many and we make sure the compiler sees all variants during warmup.

The learning rate, step number, and similar scheduler variables produce an interesting challenge. They:
1. Can vary continually over the course of training, and 
2. Are dictated by the control code, on the CPU.

There's a very particular solution to this problem, which is to package the values in:
1. A tensor, with
2. zero dimensions, and 
3. device=cpu

(Note - A 0-D tensor is simply a single scalar value.)

I imagine this serves as a kind of indirect "typing" which allows the compiler to include the variable in the compute graph and let it take on different values.

Given that it needs to get to the GPU, is declaring it as a CPU tensor inefficient? No--While compiled functions are run on the GPU, they are still launched, with arguments, from the CPU. This is fine for passing scalar values like the learning rate. 

### 2.3. Polar Express Defensive Memcpy

The profiler shows an unneccessary memcpy happening inside of the `polar_express` function. I learned that it's a consequence of pytorch's ruleset around input and output parameters. Because `X` appears twice in the input, pytorch (based on its ruleset, not actual need) makes a "defensive copy" of `X`.

There is no easy way to tell pytorch not to make this copy. A workaround I found was to break apart the offending operation (a `baddbmm` call) into two separate kernel calls, trading kernel overhead for memory bandwidth. This trade-off is beneficial for the MLP weights, but not for any other model weights. I assume this is due to the large size (4.5M values total) of the MLP weights.

The below illustration shows the Polar Express function (the pattern conveys the five iterations it performs) applied to the MLP weights. The defensive copies are the dark green segments.

<img src='https://lh3.googleusercontent.com/d/19T9jmbkiwwe20kwM6zy1bBnh0fb34HjU' alt='Illustration of how removing the memcpy in polar express is beneficial for the mlp weights' width='900' />

### 2.4. Combined Improvement

Below is a full NorMuon step across the parameters its responsbile for. This shows the overall savings of all of the above changes.

<img src='https://lh3.googleusercontent.com/d/1YY9S9YwpVVw8ejK_k2PvbTYauGnMi_kB' alt='Illustration of overall speed improvement to muon step' width='900' />

Using an arbitrary step as an example, here are roughly the improvements I measured:
```
Baseline --> improvement:
Attn: 730us -> 560us = ~170us
 MLP: 770us -> 545us = ~225us
```

I've also included a trace file from one of my runs (note that it doesn't include the pre-multiplication improvements in the attention layer, but does capture the improvements to the optimizers).

Thank you to @akash5474 for their excellent tutorial, [Profiling 101](https://blog.underfit.ai/profiling-101-nanogpt), and for providing a trace file from their run. 

---

## 3. Communication

I also looked for ways to address the current communication-bound nature of the optimization process.

### 3.1. DistAdam Synchronization

The DistAdam implementation contained a subtle issue which was resulting in Adam weights being scattered on every step, instead of only on the intended every-other step.

This appears to have been caused by the use of the torch.compile decorator on the _sync_gradients function. Compiling this function seems to have interfered with the function of the "should_sync" variable meant to control what steps the Adam weights are scattered on.

This issue was slowing down the Muon-only steps because processing was held up by the extraneous communication. 

<img src='https://lh3.googleusercontent.com/d/1A4Cj40qlyy_ksFSsAjzLPL6YMXbjs7P1' alt='Profiler trace showing unintended weight copy' width='1024' />

### 3.2. Parameter Group Ordering

I also modified DistAdam to have a more explicit definition of its parameter groups and their order. (The existing code relied on inferring the order from attributes like size).

The motivation for this comes from observing the profiler traces, which show that the weight matrices aren't being transmitted or operated on in the most efficient order. The difficulty is that we can't control the order in which weight updates are available and the optimizer hooks are called. The best we can do is define an order to wait for them in based on our observations. 

### 3.3. Type-Casting on Host

A very minor edit which saved perhaps 15us per step was to cast the model inputs (input ids, targets, cum seq_lens) on the host, instead of as part of the `.to` call, which was resulting in three small kernel launches on the GPU to perform the casts. (The pink and purple blocks below)

<img src='https://lh3.googleusercontent.com/d/1XHtq3H41q-pqom864UhwAlvc5vokWT2x' alt='Dataset type casts on gpu' width='400' />

I wasn't able to find a way to further consolidate or hide this data loading step, though it seems like it ought to be possible to do so.

---

## 4. Future Work / Opportunities for Improvement

### 4.1. Combined Optimizers

There's a significant opportunity for improvement to the communication-computation overlap if we combine the two optimizer classes into one. The original NorMuon implementation took this approach, [here](https://github.com/zichongli5/NorMuon/blob/main/normuon.py).

The main reason for this is that, currently, the scatter _and gather_ operations from Adam must complete before NorMuon can begin, which prevents us from overlapping more communication and compute.

Optimization is still very communication bound, but we should be able to eliminate all gaps in comms. I've implemented a version of this, and the profiler trace confirms the benefits of it:

<img src='https://lh3.googleusercontent.com/d/1zEufAW3Kwli3N3luo6_788PO-EIvLM3n' alt='Profiler trace of combined optimizer implementation' width='1024' />

However, despite what the profiler shows, the actual step time is slow, presumably because of a graph break in the optimization code which is causing frequent recompiles. 

I've included my current implementation of the merged optimizers in this commit--I'm happy to share credit with anyone who's able to help get it working. :)

### 4.2. Hyperparameters

While this update didn't significantly decrease the validation loss on its own, it did change the shape of the loss curve. 
I haven't attempted to tune any of the training hyperparameters yet. @ClassicLarry noted in the original NorMuon submission [here](https://github.com/KellerJordan/modded-nanogpt/pull/144#issuecomment-3447705281) that a shape change might merit revisiting various settings.

### 4.3. Smear Gate

The overhead for applying NorMuon to the smear gate is very high, considering it is only 12 values. 

I attempted to eliminate this by padding it out to (6,12) to group it with the attention gate weights, but this turned out to be expensive for gradient calculation. This shows the potential savings of removing the smear gate from Muon, though:

<img src='https://lh3.googleusercontent.com/d/1w0WZ58-J5ErWYcnTkPL5ICT3Isk5l8GZ' alt='Illustration of speed improvement from combining smear and attention gate groups' width='900' />

At a minimum, an immediate small savings would come from skipping over the variance normalization step since it's a no-op. A more substantial one would come from moving this parameter to the Adam optimizer instead.

The smear gate does qualify as a "neuron", in the sense that it's a dot product with the residual stream (albeit only 12 channels of it), so in principle I think Muon makes sense for it. 

Considering the cost, though, and that it's been working well despite the broken updates, it's porbably a good candidate to try moving.

### 4.4. Higher Cost Optimizers

@shenberg noted that Adam is primarily communication-bound, so there is idle compute time available that could be used on more expensive optimization techniques.

This should be especially true once we've merged the optimizers.

---

## 5. Appendix - Failed Experiments

Below are some bad ideas I had that might inspire better ones.

### 5.1. Per-Head Muon

Ignoring the strong mathematical grounding of the Muon orthogonal matrix update rule, but considering the underlying structure of the attention matrices, I wondered if making the weight updates orthogonal on a per-head level might have merits. While we generally want our attention matrices to be full rank, I wondered if enforcing it across all heads might be overly-constricting.

Running polar express on a large batch of individual heads provides a significant reduction in compute, because it is only attempting to orthogonalize, e.g., 24 sets of 128 vectors instead of 4 sets of 768. But this approach had a negative impact on learning, and the speed improvement didn't seem to outweigh it.

_Separating RoPE and NoPE_

Additionally, given that the RoPE and NoPE dims of the query and key heads occupy different subspaces, I wondered whether Muon might perform better if we separated them. This actually seemd to have a more substantial negative impact on learning than just separating by head, which is interesting.

_Neuron Subspaces_

Finally, I also tried breaking up the MLP weights into 8 square matrices, i.e., (2, 3072, 768) --> (8, 768, 768) and stacking these with the attention weights. 

There's some structural motivation for this--Muon makes the weight updates orthogonal _across_ the neurons, instead of for the neurons themselves. We can change this by breaking the weights into chunks <= 768.

Breaking the MLPs into blocks and stacking them with the attention weights was very efficient, but hurt learning as well.

Still, the speed up was significant enough that it might be worth looking at further.

### 5.2. Autograd Layout Issues

This is a concept I'm not very familiar with yet, but a part of the backward pass and communication mechanisms has requirements around how the gradients are laid out in memory. 

The profiler traces show a time-consuming memory movement step at the end of the autograd process (you can see it as the leftmost dark green block in the illustrations, at the start of communications), and I'm fairly certain this comes from the lm_head weights, because I was able to eliminate it by making a modification to the LM head (but it caused a slowdown).

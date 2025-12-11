This update includes a number of algebraic and fusion optimizations to NorMuon, as well as a fix to the variance reduction step.

It also addresses a synchronization issue with the DistAdam implementation, and makes its parameter ordering more explicit.

**Timings & Loss**

This is primarily a GPU-optimization submission, but the fixes to NorMuon do impact the math.

I haven't done enough runs for statistical significance yet. I'll try going back to 2160 steps for some margin and doing a larger number of runs.

Here are averages from 3 baseline runs, and five runs each of the submission (at reduced step counts): 

```
              Steps       Mean Time (s)      Mean Loss  
  Baseline    2160        133.6643           3.2773     
 This Repo    2155        132.1304           3.2792     
 This Repo    2150        132.0766           3.2798     
```


### NorMuon Variance Reduction



NorMuon improves upon Muon by (controlling variance?). It tracks and manages variance at the level of individual weight vectors, such as the FFN neurons and Attention head basis vectors.

The existing code determined the orientation of the weight vectors by comparing matrix width to height.

This rule did not correctly capture the orientation of the gate weights (the smear gate and attention gates) or the attention output heads.

I was able to address the gate weight orientation simply by adjusting the rule.

**Attention and MLP Weight Layout**

Correcting for the output heads was more difficult. Like most attention implementations, the current version stores the Output heads transposed relative to the QKV heads.

Specifically, it stores QKV heads "horizontally", each with shape (128, 768), and the output heads "vertically", with shape (768, 128). This is intention, because storing the Output heads horizontally would then require a transpose operation during the output projection.

To avoid splitting apart the QKVO matrix, and avoid any transposing, I changed the layout such that all heads are stored horizontally, and QKVO form a vertical stack with a combined shape of (4*model_dim, model_dim).

The attention projections are adjusted accordingly, and remain as efficient as before. The maintain the matching shapes of the attention and mlp weights, I also transposed the mlp weights to match.

This transposed orientation for the mlp and attention weights has an additional conceptual / experimental benefit in that now all weight vectors (FFN input and output neurons and attention head basis vectors) are stored horizontally, and can be efficiently sliced.

**Confirming Benefits**

To verify that these changes had a positive effect on the model, I ran the baseline plus three variants:
The corrected orientation for:
1. The attention gate only.
2. The attention and smear gates.
3. Both gates and the Attention Output matrix.

Averaged over four runs, all three variants outperformed the baseline:


<img src='https://lh3.googleusercontent.com/d/1kcsAdOlgQwbHcQhjt_jvBqIUk0zIMcm1' alt='Plot of final validation loss for variance fixes applied to gates and W_O' width='640' />

Note that correcting both gates but not W_O (pink) performed better than with the W_O correction (light blue).

I think the correction is still worth making on principle alone, though, and because of the significant impact it has on the overall loss curve, below.

The loss is higher than the baseline for most of the run up until the very end. This might suggest that managing the variance across heads caused some amount of overfitting for the model.


<img src='https://lh3.googleusercontent.com/d/1zjeJUaiFFzbe82deptO1zljF017rZktB' alt='Plot showing the overall higher validation loss curve with the W_O fix in place' width='720' />



### NorMuon Algebraic optimizations



Reviewing pytorch profiler traces for the NorMuon optimizer overall revealed a number of opportunities for improvement.

The most interesting of which was the algebra for calculating the variance. The straightforward implementation results in reading the gradient updates from memory multiple times. Instead, we can avoid these expensive reads by re-using existing quantities. For instance, we calculate the element-wise squares for, and then divide these by their count to get their mean. We later need the element-wise squares, and we can calculate these cheaply by "undoing" the mean--multiplying the mean values by the count.



### Compiled Helpers and 0-D CPU Tensors


I defined compilable functions for performing the NorMuon variance calculation, and for calculating the cautious weight decay (which I was also able to optimize by avoiding a memory read and performing this operation in-place). The compiler successfully fused many of the operations and reduced the time required for these steps significantly.

A key detail for successfully compiling these functions was storing the scalars (learning rate, weight decay) in 0-D CPU tensors. This specific storage method (as opposed to on-device tensors, tensors with more dimensions, or standard python floats) allows the compiler to regard these as "constants" (though they change with every step).

The below illustration is for an attention weight update and is to scale. It shows the combined impact of these optimizations (to variance reduction and cautious weight decay) processing the attention weight updates.

<img src='https://lh3.googleusercontent.com/d/1Bvw9aJa5a0Q3kpzELc8LmYZWDLa-4ZMT' alt='Illustration of fusions for variance and cautious weight decay' width='900' />

### Polar Express Defensive Memcpy



The profiler shows an unneccessary memcpy happening inside of the `polar_express` function. I learned that it's a consequence of pytorch's rule set around input and output parameters. Because `X` appears twice in the input, pytorch (based on its ruleset, not actual need) makes a "defensive copy" of `X`.

There is no easy way to tell pytorch not to make this copy. A workaround I found was to break apart the offending operation (a `baddbmm` call) into two separate kernel calls, trading kernel overhead for memory bandwidth. This trade-off provides a significant speedup for the MLP weights, which have 2.25M parameters each, but not for any other model weights. (While QKVO is the same size when combined, polar express operates on them as a stack of 4 square matrices).

The below illustration shows the Polar Express function (the pattern conveys the five iterations it performs) applied to the MLP weights. The defensive copies are the dark green segments.

<img src='https://lh3.googleusercontent.com/d/19T9jmbkiwwe20kwM6zy1bBnh0fb34HjU' alt='Illustration of how removing the memcpy in polar express is beneficial for the mlp weights' width='900' />

This shows the overall savings of all of the above:

<img src='https://lh3.googleusercontent.com/d/1YY9S9YwpVVw8ejK_k2PvbTYauGnMi_kB' alt='Illustration of overall speed improvement to muon step' width='900' />

Using an arbitrary step as an example, here are roughly the improvements I measured:
```
Baseline --> improvement:
Attn: 730us -> 560us = ~170us
 MLP: 770us -> 545us = ~225us
```


### DistAdam Synchronization



The DistAdam implementation contained a subtle issue which was resulting in Adam weights being scattered on every step, instead of only on the intended every-other step.

This appears to have been caused by the use of the torch.compile decorator on the _sync_gradients function. Compiling this function seems to have interfered with the function of the "should_sync" variable meant to control what steps the Adam weights are scattered on.

(Illustration of the trace, labeling the communication blocks)

This issue was slowing down the Muon-only steps, because processing was held up by the extraneous communication. 



<img src='https://lh3.googleusercontent.com/d/1A4Cj40qlyy_ksFSsAjzLPL6YMXbjs7P1' alt='Profiler trace showing unintended weight copy' width='1024' />

### Parameter Communication

Though it doesn't provide a measurable speed-up, I've included a change which makes our intended ordering of the communication of parameters more explicit.

This should make it easier to more directly orchestrate communication-computation overlap.



### Type-Casting on Host

A very minor edit which saved perhaps 15us per step was to cast the model inputs (input ids, targets, cum seq_lens) on the host, instead of as part of the `.to` call, which was resulting in three small kernel launches on the GPU to perform the casts. (The pink and purple blocks below)

<img src='https://lh3.googleusercontent.com/d/1XHtq3H41q-pqom864UhwAlvc5vokWT2x' alt='Dataset type casts on gpu' width='400' />

I wasn't able to find a way to further consolidate or hide this data loading step, but it seems like it ought to be possible.



## Additional Insights and Future Work

Below are some optimizations which didn't pan out, or things I'm still working on.



#### Stacking Smear & Attention Gates

The overhead for applying NorMuon to the smear gate is very high, considering it is only 12 values. I was able to eliminate this by padding out the gate to be the same shape as the attention gates, (6, 12), so they could be stacked. However, this then requires a slice operation to apply the smear gate, which is apparently very costly during the backward pass, and the net effect was a slowdown. (So I did not make this change).

I made the below illustration prematurely, before discovering the overall negative impact, but I thought it interesting enough to share anyway.



<img src='https://lh3.googleusercontent.com/d/1w0WZ58-J5ErWYcnTkPL5ICT3Isk5l8GZ' alt='Illustration of speed improvement from combining smear and attention gate groups' width='900' />



#### Per-Head Muon

Ignoring the strong mathematical grounding of the Muon orthogonal matrix update rule, but considering the underlying structure of the attention matrices, I wondered if making the weight updates orthogonal on a per-head level might have merits. While we generally want our attention matrices to be full rank, I wondered if enforcing it across all heads might be overly-constricting.

Running polar express on a large batch of individual heads provides a significant speed-up, because it is only attempting to orthogonalize, e.g., 24 sets of 128 vectors instead of 4 sets of 768. But this approach had a negative impact on learning, and the speed improvement didn't seem to outweigh it.

_Separating RoPE and NoPE_

Additionally, given that the RoPE and NoPE dims of the query and key heads occupy different subspaces, I wondered whether Muon might perform better if we separated them. This actually seemd to have a more substantial negative impact on learning than just separating by head, which is interesting.


_Neuron Subspaces_

Finally, I also tried breaking up the MLP weights into 8 square matrices, i.e., (2, 3072, 768) --> (8, 768, 768) and stacking these with the attention weights. 

There's some structural motivation for this--Muon makes the weight updates orthogonal _across_ the neurons, instead of for the neurons themselves. We can change this by breaking the weights into chunks <= 768.

Breaking the MLPs into blocks and stacking them with the attention weights was very efficient, but hurt learning as well.

Still, the speed up was significant enough that it might be worth looking at further.



#### Autograd Layout Issues

This is a concept I'm not very familiar with yet, but a part of the backward pass and communication mechanisms has requirements around how the gradients are laid out in memory. The profiler traces show a time-consuming memory movement step at the end of the autograd process (you can see it as the leftmost dark green block in the illustration further down), and I'm fairly certain this comes from the lm_head weights.

While not fully understanding the issue or the solution, I worked with AIs on the problem, and through some transposing and re-ordering they were able to change the FP8 implementation of the LM head and eliminate this memory movement (it disappeared from the profiler trace), but it must have incurred a penalty elsewhere because the net effect of the change was a slowdown.



#### Combined Optimizers

There's a significant opportunity for improvement to the communication-computation overlap if we combine the two optimizer classes into one. 

I've implemented a version of this, and the profiler trace confirms the benefits of it:


<img src='https://lh3.googleusercontent.com/d/1zEufAW3Kwli3N3luo6_788PO-EIvLM3n' alt='Profiler trace of combined optimizer implementation' width='1024' />

However, despite what the profiler shows, the actual step time is slow, presumably because of a graph break in the optimization code which is causing frequent recompiles. 

If I can address that issue, I'll submit another record.

This record combines the value embeddings into a single weight:

```
-        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
+        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]
```

and fixes the step 5 blocking behavior from the async data loader.

Additionally, there are quite a few refactors and that don't change the timing but decrease LOC and hopefully improve quality of life:

* Froze/hardcoded the best Triton configs for the symmetric matmul kernels (see [discussion](https://github.com/KellerJordan/modded-nanogpt/discussions/23?sort=new#discussioncomment-15532866)). Warmup should be faster now.
* Merged the Yarn and YarnPairedHead classes, and the CausalSelfAttention and PairedHeadCausalSelfAttention classes, mostly for readability and LOC reasons
* The Training Schedule has been refactored to use a `TrainingSchedule` class. The separate batch size, window size, MTP, and learning rate schedules have been unified. Each `TrainingStage` corresponds to a torch compile graph.
* Added a `grad_scale`. The major idea here is that the gradient magnitude should match between 1 and 8 GPUs for training consistency (matters for FP8 scales especially, but also BF16 a little bit). I noticed a `grad_scale = 2` is slightly better than `grad_scale = 1` (default), though not enough to warrant lowering steps.
* Moved the distributed setup to the top of the file, and also put Hyperparameters earlier in the file.
* Cleaned up Hyperparameters.

Some of the refactors are quite subjective, so happy to edit to incorporate others' feedback.

## Timing and Validation

This PR improves the step time and does not impact (or very slightly decreases) loss.

```
import scipy.stats
import torch

losses = [3.2767, 3.2777, 3.2795, 3.2774, 3.2799, 3.2779, 3.2777]
times = [97.825, 97.777, 97.818, 97.777, 97.745, 97.715, 97.728]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0025

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0012, mean=3.2781)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0425, mean=97.7693)
```

Previous record (timed on same machine):

```
import scipy.stats
import torch

losses = [3.2776, 3.2803, 3.2773, 3.2794, 3.2784, 3.2794]
times = [98.908, 98.878, 98.796, 98.667, 98.800, 98.776]

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0012, mean=3.2787)

print("time:", torch.std_mean(torch.tensor(times)))
# time: (std=0.0848, mean=98.8042)
```

These timings show $\approx 1$ seconds of improvement.

## Discussion

I mostly want to push out this PR because there are a lot of efforts into adding more value embeddings. It turns out that the "fused" value embedding saves a considerable amount of time —— something like 0.8s throughout training. Note that the current approach is basically what was done earlier in the speedrun, but it was replaced due to speedups in the optimizer step when separating each value embed (record 16). Due to the new optimizer architecture, this previous consideration is no longer there, and the speedups in the lookup matter quite a bit. By my count, we are saving ~10 reads of the input sequence between the forward and backward step.

I created a new class `Shard` to replace the previous two-class BOSFinder and AsyncDataLoader. It is my understanding that fixing the step 5 blocking behavior saves ~100ms.

I believe there is around 0.3s of potential improvement in the current data loader by overlapping the CPU work of generating the input tensors (`cat`s and indexing) and the H2D CPU -> GPU transfer. I was able to get something like this working by adding a hook in the optimizer step, but my approach was a bit messy. Going to put off working on this for now.

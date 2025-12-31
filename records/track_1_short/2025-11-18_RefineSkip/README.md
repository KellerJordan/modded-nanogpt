## New WR 134.9s: Refine Skip Architecture, Better Lambda Init

Three updates, for roughly 2s improvement:
1. Replace the current skip U architecture of (1->10, 2->9, 3->8, 4->7, 5->6) with (4->7). I picked 4 because it had a long attention window and therefore is collecting more information that needs to be processed. I picked 7 because layer 7 does not have an attention module, so its MLP may have free capacity to do more processing. Coincidentally (4-7) was already a subset of the existing architecture. If I remove (4->7) performance drops by roughly 0.0025 loss, which is substantially more costly than the compute time of one skip connection.
2. Initialize the lambda A in block.forward(x=Ax+Bx0) to 1.1 instead of 1. This lambda is serving a more substantial role in the network than it might seem. This initialization improves performance by roughly 0.001.
3. Drop 20 steps. (60ms per step)

I am not building on top of #155 because I have not yet been able to get that PR to run on 8H100.

### Details on Lambda Update
The block lambda structure of x = ax+bx0 can be unrolled across layers. Critically, this lambda does not get applied to the layer input, but to the actual residual stream.
Let R_i be output of layer i, then
contribution to prediction by layer i is proportional to a^(num_layers-i)

Initializing the lambda to 1.1 gives:

`prediction = 1.1^10*R_1 + 1.1^9*R_2 + ... = 2.59 * R_1 + 2.35 * R_2 + 2.14 * R_3 + ... R_11`

Given this init, the first layer has 2.6x the impact to the prediction as the last layer. The initialization of this lambda can be thought of as a lever to bias the network early training towards the earlier or later layers.

Here are the final weights after training for a and b in x=a*x+b*x0 for the 12 layers
```
[[1.1000, 0.0000], # null layer
[6.0209, 4.9209], # layer 1 where x=x0
[0.8377, 5.0139],
[0.5871, 5.6103],
[0.5783, 3.3061],
[0.4402, 2.1686],
[0.4838, 3.9371],
[0.8303, 4.8757],
[0.6744, 1.5459],
[1.1908, 6.4830],
[1.0034, 4.5138],
[0.8307, 3.2445]]
```
From layers 2 to 8 the first lambda ends around 0.5. This means that the final contribution of the 1st layer output to the prediction is muted by roughly 0.5^7, compared to the 8th layer. The residual stream is applying a sort of exponential smoothing over layer contributions.
Similar to the backout_lambda, this gives further evidence that the first 3/4 of the layers are functioning as context builders and the last 1/4 are functioning as predictors. The lambda enables each layer to use the context output from nearby layers, which then gets washed out after repeatedly applying 0.5 to the residual stream.
A secondary effect of this lambda ending up < 1 is that each MLP pays extra focus to its own attention layer, because the deweighting of the residual stream occurs before the attention output.

I previously tested an architecture where each module of each layer could dynamically set a unique weight to accept the contributions of every prior module from every prior layer. I saw that every MLP module would consistently give a large preferential focus to its own attention layer. At the final values, this lambda accomplishes a similar objective.

The implementation of this lambda in the repo is computationally efficient but conceptually misleading. The comments above do not give intuition as to why initializing to greater than 1 would be preferred, just that this parameter is perhaps a meaingful lever to bias training to early or later portions of the network.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.278, 3.2777, 3.2799, 3.2783, 3.2784, 3.28, 3.2787, 3.278, 3.2797, 3.2804]
times = [135.063, 134.826, 135.004, 134.994, 134.993, 134.722, 134.925, 134.998, 134.981, 134.912]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0034

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0010), tensor(3.2789))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1008), tensor(134.9418))
```

retiming prior record: 137.3: [137.083,137.374,137.546]
(appears I got a slightly slower machine this time)

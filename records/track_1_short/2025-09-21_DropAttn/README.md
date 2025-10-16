## New WR 151.5s: Drop first attn layer, extend all long windows for validation, update schedule

This PR builds on all recent WR improvements including PR #130. Updates:
* Drop the first attention layer.
* Increase step count from 1645 to 1680
* Extend all long windows to size 20 for validation (-0.001 loss, or ~10 steps = 1s)
* Add arg iteration_extension, to specify number of steps to continue training at final lr and ws 

Several factors led to dropping the first attention layer:
* The first attention layer was [observed](https://medium.com/@larry36d/formation-of-induction-heads-in-modded-nanogpt-5eb899de89e4) to perform no meaningful contribution to induction heads. 
* PR #120 dropped the first MLP, which resulted in two Attn layers with no intermediate transformation
* PR #130 added a smear module, which increased the networks ability to pass information between tokens

Reason for iteration_extension:
* Showed 0.001 improvement in loss.
* Easier to fine tune. Without this parameter, changes to the step count have a large effect on the entire second half of training, making it harder to isolate impact of changes. This is because lr and ws are tied to the step count. EG if you increase step count by 10 you will suddenly see different loss at step 1000.

### Future Opportunities
This change bring the total number of [4,768,768] attention variables to 10. There are 22 MLP variables of size [768x4,768]. In Muon attention is getting batched such that 6/16ths on the gradient calcs are on padding tokens. There may be a way to move 2 of the attention variables into the MLP batch, such that MLP is 24/24 and attn is 8/8, instead of MLP being 22/24 and attn being 10/16.

### Investigating Muon for 1D variables
Currently the attention gates and smear gate are passed into Muon. From light inspection, the implementation of newton schulz appears to roughly apply F.normalize(x, p=2, dim=-1) for 1d variables. This normalization makes all steps cover roughly the same distance, regardless of the gradient. So for 1d variables Muon turns into an exponential smoothing over prior gradients, where each step is normalized to be roughly the same size. This seems somewhat reasonable. Swapping these variables over to Adam gave roughly a 0.5s runtime increase and no improvement in loss. Directly replacing newton schulz with F.normalize(x, p=2, dim=-1) for these variables showed slightly worse performance. I do not understand the theory here yet, but empirically the performance is good.


## Validation:
Code syntax/naming was lightly refactored after performing validation runs. Loss is roughly 0.001 lower than prior record, which is roughly equal to 1s.
```
import scipy.stats
import torch

accs = [3.2786, 3.2798, 3.2762, 3.2781, 3.2778, 3.2801, 3.2774, 3.2772,
       3.2777, 3.2789]

times = [151.559, 151.526, 151.516, 151.527, 151.606, 151.771, 151.546,
       151.547, 151.44 , 151.872]

print("p=%.4f" % scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue)
# p=0.0005
print("acc:", torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0012), tensor(3.2782))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1305), tensor(151.5910))
```
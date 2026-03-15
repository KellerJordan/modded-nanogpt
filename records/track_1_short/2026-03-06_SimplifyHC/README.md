Dropping roughly half a second (ran on a slower machine than the timing used on the record right now), I'll get an accurate retiming when I go to merge.
Also bringing the loss much more comfortably under 3.28.

The partitioned hyperconnections PR made it clear that the last few attention layers strongly prefer to ignore the contributions of the last few MLPs. I made this explicit by instead of doing hyperconnections, feeding in a single cached activation from layer 7 for all later attention modules, so they never get polluted with 'prediction MLPs'.

I left the PR at this stage because it opens up two valid follow ons: 1) now that 3 attention layers all have the exact same input, consider merging them into 1 mega layer with shared matmuls. OR 2) don't have all 3 attention layers accept the exact same input, enable some sequential evolution of this input.

```
import scipy.stats
import torch

losses = [3.2784, 3.2791, 3.277, 3.2766, 3.2787, 3.2778, 3.279, 3.2776]
times = [87.098, 87.225, 87.276, 87.226, 87.214, 87.263, 87.316, 87.222]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0003

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0009), tensor(3.2780))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0637), tensor(87.2300))
```
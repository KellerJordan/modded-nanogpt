## New Record: Logit Schedule (-20 steps, -1.1s)

Adding a schedule to logit_range and logit_slope
* logit_range: 23 -> 27 linear over first 50% of training
* logit_slope: 7.5 -> 15 linear over first 50% of training
```
logits = self.lm_head(x)
logits = logit_range * torch.sigmoid(logits / logit_slope)
```

logit_range controls how certain the model can be of a prediction. Early on in training we don't need to predict with high certainty. logit_slope is increased over time to induce uncertainty into prior learnings. Intuition that motivated the change (unclear if this has any actual relation to the mechanism): A 5 year old may be 90% certain that trains are faster than trucks. Over time the child learns a more nuanced view of the world. Instead of having to perform a bunch of world view updates from near-certain beliefs after turning 10, its easier to just constantly drift prior beliefs towards an agnostic perspective. 

In this PR the drift is turned off once lr decay comes online.

I attempted to cleanup the main training loop by pulling out all the nuances around custom optimizers and schedules into separate classes. This was making it easier for me to test freezing different parameters or changing schedules. Happy to merge in any changes that improve the clarity of the code further.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.276, 3.2776, 3.2767, 3.2764, 3.2744, 3.2783, 3.2766, 3.278, 3.2771, 3.2784]
times = [117.623,117.372,117.448,117.559,117.423,117.54,117.417,117.637,117.376,117.438]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0000

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0012), tensor(3.2770))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0985), tensor(117.4833))
```

retiming prior record: 118.6: [118.625, 118.703, 118.534]

Will plan to merge at -1.1s of prior official record if nothing changes, which would put this at 118.2s. I didn't drop the step count further because I saw one unlucky run at -10 more steps. In retrospect after seeing all 10 runs, I probably could have dropped more steps.

## Decoupled Window Size Schedule

This record changes the window size schedule from sizes (3, 7, 11) at (0.33, 0.66, 1.0) (only changing at the same time as the batch size) to sizes (2, 5, 11) at (0.55, 0.8, 1.0). This results in a longer periods of training occuring at lower window sizes, resulting in a decrease in total training time and a small increase in validation loss.

### Changes
1. Change get_ws from switching at equal intervals throughout training to switching at the specified custom intervals
2. Since the window size schedule is now different from the batch size schedule, this increased the total number of unique kernel configurations we need to warm up (since we need to warm up all combinations of window size and batch size). To do so modified warmup to run for all combinations of window size and batch size but reduced total warmup steps from 10 to 3 to not cause excessive warmup overhead.
3. Updated pytorch nightly version to the newest 2.10.0.dev20251210+cu126 version.

### Timing and validation

```
import scipy.stats
import torch

losses = [3.2788, 3.2794, 3.2781, 3.2805, 3.2766, 3.2802, 3.2796, 3.2780, 3.2808, 3.2806, 3.2821, 3.2792, 3.2798, 3.2762, 3.2795, 3.2784, 3.2789, 3.2799, 3.2800, 3.2798, 3.2776, 3.2773, 3.2808, 3.2783, 3.2789, 3.2815, 3.2779, 3.2795, 3.2782, 3.2803]
times = [128.491, 128.489, 128.749, 129.550, 128.941, 129.438, 128.299, 128.938, 128.611, 128.263, 129.494, 128.409, 128.243, 128.407, 128.111, 133.018, 128.485, 128.549, 128.344, 130.282, 130.018, 128.756, 128.261, 128.253, 128.303, 128.230, 128.366, 129.319, 128.266, 129.911]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0024

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0014), tensor(3.2792))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.9840), tensor(128.8930))
```

Previous record (timed on the same machine and pytorch nightly build):

```
import scipy.stats
import torch

baseline_losses = [3.2797, 3.2761, 3.2791, 3.2794, 3.2799, 3.2809, 3.2780, 3.2785, 3.2776, 3.2777, 3.2771, 3.2772, 3.2777, 3.2783, 3.2769, 3.2788, 3.2798, 3.2794, 3.2768, 3.2748]
baseline_times = [129.162, 129.240, 129.118, 129.929, 129.207, 129.193, 129.098, 130.746, 130.047, 128.971, 128.973, 131.144, 129.049, 129.064, 129.217, 129.100, 128.996, 129.471, 129.302, 129.121]

print("losses:", torch.std_mean(torch.tensor(baseline_losses)))
# losses: (tensor(0.0015), tensor(3.2782))

print("time:", torch.std_mean(torch.tensor(baseline_times)))
# time: (tensor(0.6011), tensor(129.4074))
```

This shows an improvement of $ \approx 0.51 $ seconds. Running the new training script 5 times on the previous pytorch nightly build yields an average training time of 128.775 seconds, but this reduction seems likely to be due to variance rather than an actual performance improvement.

### Timing previous records

As an additional check for the machine used for this record I timed 5 runs of each of the previous three records locally (using the newest pytorch nightly build) and compared them with the official record times:

| Record | Official Time (s) | Local Avg (s) | Local Std (s) | Difference (s) |
|--------|------------------:|--------------:|--------------:|---------------:|
| 2025-12-11_NorMuonOptimsAndFixes | 130.200 | 133.160 | 0.345 | +2.960 |
| 2025-12-10_SALambdaOnWeights | 131.580 | 132.785 | 0.142 | +1.205 |
| 2025-11-29_BatchSizeSchedule | 132.180 | 133.824 | 0.737 | +1.644 |

I think the larger difference observed for NorMuon may be due to @ClassicLarry having gotten a particularly fast machine for his official retiming since when he reran it he only got an average of 131.2 seconds. With that factored in the previous records seem to be all running around 1-2 seconds slower on the machine I am using compared to the official times.

Thank you so much for your time and effort in maintaining this competition and let me know if there are any additional artifacts I can provide to aid in verification!

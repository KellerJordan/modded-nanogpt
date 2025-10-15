## New WR 153.9s: Asynchronously fetch and index data batches, extend final layer attention window for validation

This PR builds on all recent WR improvements including PR #125 by @bernard24. This one adds:

- Start prefetching and indexing the next shard immediately. Since this occurs on the CPU, there is ample time to perform this during the GPU heavy workload, and we shouldn't be bottlenecking GPU activities on CPU data indexing. (1.5s)
- Only partially index the first shard before starting to train on it. Kickoff a parallel thread to finish indexing it, which gets picked up on the 5th step. (300ms)
- Extend the final layer attention window out to 20 for validation (no need to apply YaRN for this layer). If curious, some inspiration for this change came from: https://medium.com/@larry36d/formation-of-induction-heads-in-modded-nanogpt-5eb899de89e4. This dropped loss by roughly 0.001 and enabled -10 steps (1s) while still cleanly remaining under 3.28. 

(Exact runtimes will vary by GPU provider, mine is about 1s slower than I believe some GPU setups will get)

Validation:
```
import scipy.stats
import torch

accs = [3.2813, 3.2778, 3.2781, 3.2764, 3.277 , 3.2787, 3.2767, 3.2807,
       3.2769, 3.2774]

times = [153.869, 154.126, 153.756, 153.82 , 153.816, 154.014, 153.887,
       153.811, 154.015, 153.828]

print("p=%.4f" % scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue)
# p=0.0030
print("acc:", torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0017), tensor(3.2781))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1180), tensor(153.8942))
```
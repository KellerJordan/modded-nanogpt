## New record 07/12/25

1. Included engineering optimizations from https://github.com/KellerJordan/modded-nanogpt/pull/102 by @vagrawal
2. Updated distributed_data_generator() to align to bos token for training data. Specifically: Added a helper function in the distributed data loader that returns the next {world_size} starting points, one for each gpu, such that points are spaced at least {local_batch_size} distance from each other, all start with a bos_token, keep same order, and skip the minimum number of tokens. Justification is right now a nontrivial number of samples start mid-sample, learning from mid-sample context is more challenging, and not having a bos_token in the intra-document context window may degrade attention sink behavior. Validation data kept consistent with prior records.
3. As a follow-on from 2: Reduced number of iterations from 1770 to 1750, increased cooldown frac from 0.4 to 0.45, and decreased minimum lr schedule factor from 0.1 to 0.05.

Validated results with 20 runs:
```
import scipy.stats
import torch

accs = [
    3.2784,3.2791,3.2819,3.2801,3.2788,3.2794,3.2782,3.277,3.2784,3.2773,
    3.2803,3.2792,3.2792,3.2804,3.2817,3.2805,3.2783,3.2789,3.2779,3.2778]
times = [
    173.368,173.424,173.291,173.441,173.397,173.348,173.402,173.463,173.209,173.338,
    173.291,173.377,173.322,173.287,173.265,173.375,173.331,173.413,173.384,173.462]

print('p=%.4f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0049

print('acc:',torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0013), tensor(3.2791))

print('time:',torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0682), tensor(173.3594))
```

## Retiming on 07/13/25

The record was retimed using torch==2.9.0.dev20250713+cu126, yielding 171743ms. This will be the official time for the record.
Note that this should not change the ML, so that the runs conducted in the older torch version still stand in terms of acquiring p-value on loss < 3.28.
Also note that the retiming was with a refactored version of the code.

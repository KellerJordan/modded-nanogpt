## Statistical tests

```
accs = [3.2759, 3.2781, 3.2791, 3.2771, 3.2838, 3.2749, 3.2793, 3.279, 3.2794, 3.2744, 3.2751,
        3.2845, 3.2736, 3.2783, 3.2793, 3.2779, 3.2756, 3.281, 3.2803, 3.2766, 3.2851, 3.275,
        3.2778, 3.2723, 3.2842, 3.2735, 3.275, 3.2796, 3.2782, 3.2758, 3.2763, 3.2751, 3.2791,
        3.2804, 3.2725, 3.2898, 3.2718, 3.2764, 3.271, 3.2745]

import scipy.stats
print('p=%.4f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0003 (statistically significant)

import torch
print(torch.std_mean(torch.tensor(accs)))
# (tensor(0.0040), tensor(3.2777))
```

## ChangeLog

* Added 12 new embedding layers which get mixed into the value activations at each layer. (=463M new parameters, of which 9216 are active per token)

## Contributors

* @KoszarskyB



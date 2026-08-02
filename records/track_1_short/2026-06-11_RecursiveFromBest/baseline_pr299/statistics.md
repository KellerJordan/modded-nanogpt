# Baseline run statistics (#83, same hardware)

Prior record #83 ("Sign Trick on Bigram Embed") re-run on the same 8x H100 hardware, 10 runs, step 1385/1385.

| Run | val_loss | train_time |
|---|---|---|
| 0527c136 | 3.2782 | 80437ms |
| 0dd9e4b4 | 3.2787 | 80341ms |
| 20ccc02e | 3.2776 | 80632ms |
| 2ef8c260 | 3.2784 | 81143ms |
| 36ed5afd | 3.2805 | 80434ms |
| 56e0cb90 | 3.2769 | 80632ms |
| 6741e6be | 3.2802 | 80614ms |
| c9ff7f85 | 3.2802 | 80645ms |
| dd065b0e | 3.2781 | 80561ms |
| ef9ef1b8 | 3.2762 | 80675ms |

## Summary

|  | mean | std | min | max |
|---|---|---|---|---|
| val_loss | 3.27850 | 0.00144 | 3.2762 | 3.2805 |
| train_time (ms) | 80611.4 | 217.6 | 80341 | 81143 |

p(val_loss < 3.28) = **0.0047**

One-sided one-sample t-test, n = 10: `scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue`.
Verifier output: `n=10 val_loss=3.2785±0.0014 t=-3.287 p(mean<3.28)=0.00471 time=80.61±0.22s`.

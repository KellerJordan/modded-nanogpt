# Run statistics

13 runs, step 1398/1398.

| Run | val_loss | train_time |
|---|---|---|
| 00088a48 | 3.2791 | 77933ms |
| 244ecee6 | 3.2800 | 77446ms |
| 24fc5ca4 | 3.2790 | 77299ms |
| 595ef814 | 3.2798 | 77194ms |
| 71e3fc9d | 3.2819 | 77401ms |
| 78e1d055 | 3.2795 | 77431ms |
| 7f295811 | 3.2801 | 77149ms |
| 9ba762c3 | 3.2766 | 77247ms |
| c4caf519 | 3.2780 | 77336ms |
| c69f5628 | 3.2778 | 77186ms |
| cbe54ad0 | 3.2785 | 77318ms |
| d16f878f | 3.2776 | 77224ms |
| f33b9bb3 | 3.2782 | 77225ms |

## Summary

|  | mean | std | min | max |
|---|---|---|---|---|
| val_loss | 3.27893 | 0.00137 | 3.2766 | 3.2819 |
| train_time (ms) | 77337.6 | 202.9 | 77149 | 77933 |

p(val_loss < 3.28) = **0.0078**

One-sided one-sample t-test, n = 13: `scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue`.
Verifier output: `n=13 val_loss=3.2789±0.0014 t=-2.815 p(mean<3.28)=0.007803 time=77.34±0.20s`.

Updates in PR, building off #215. I further tuned value embeddings by:
- Init all VE to Normal(mean=0, std=0.01).
- Change gating fn of VE from $g(x[:12])$ to $g(x[:6] + ve[:6])$
- Shift 1st VE forward, so new pattern is .01...234

## Ideas

**Initialization**

After reading the Mimetic init, I realized the initalizing the VE to uniform 0 probably wastes a few steps. Although I explored some literature, I decided that a low-variance (spherical) Gaussian init is a decent uninformative prior for embedding spaces. I believe that trying fancier inits will require some exploratory analysis of this Value Embedding structure, which we currently don't have a good exploratory theory for, so I held off on this.

**Gating**

For the gating, I simply noticed that in older papers on gates (i.e. RNNs, channel mixing, residuals), the emphasis on gate design is fairly strong, so I tried a few ideas from this taxonomy. Specifically, the idea that we should conditon the gate of `VE` using some parameters of `VE` seemed to work. Doubling gate size to $g(x[:12] + ve[:12])$ did not work, which implies a cheap gate is enough; I believe this result is specific to parameter size, i.e. larger model sizes probably want much more expressive gates.

**VE Layout**

Initially, I wanted to understand the structure of VEs, so I saved the VEs at every val step and run linear algebra analyses. Here's an AI summary of my key results:

> "Last VE" is close to isotropic: it looks like a whitened, generic feature channel
> - Condition number $\sim 7$, spikiness $\sim 2$, correlations mild (max offdiag $\sim 0.08$ ).
> - Top-64 eigenvectors explain only $\sim 14 \%$ variance.
>
> "First VE" is genuinely correlated and "spikes + bulk": it looks like a small set of global control directions + residual degrees of freedom
> - Condition number $\sim 100$, spikiness $\sim 19$, correlations strong (max offdiag $\sim 0.625$ ).
> - Top-64 eigenvectors explain $\sim 34 \%$ variance.

Since the first VE's vectors were more unevenly distributed as a result of pretraining, I interpreted that the first VE is ill-conditioned, i.e. VEs don't belong in block 1. Perhaps this is because VEs add uncontextualized info about token $t$ when the residual $x_l$ at layer $l$ has other blocks' outputs mixed in, but at block 1, the V receives a fresh input of embedding $x_1$ so doesn't have this issue.

I am hesitant to include more analysis because my background in numerical linear algebra isn't that strong and I might misinterpret the results. However, of the micro-tweaks, this idea had the most effect (see ablations), so I believe there is something about the structure of VEs that merits further study.

What didn't work:
- 6 VE (but only keep embeddings for 80% common tokens): loss=3.9
- VE normal std=0.02: 3.2815
- VE normal std=0.005: 3.2812
- VE untie at step 500: 3.3072
- WD to 2.5: 3.2779
- WD to 0: 3.2818
- Mind the Gap: 3.2818 (with simple centered attention)
- QK-Clip tau=1 (from Kimi K2): 3.35
- QK-Clip tau=10: 3.32
- Subtract mean from RMS norm for stability: 3.2796
- Pull last VE back: 3.2986

## Ablations

We kept hypotheses H{1,2,4}.

```
Hyp  Branch                                        Runs   Time μ  Time σ   Time Δ  Time p  Loss μ  Loss σ   Loss Δ  Loss p
----------------------------------------------------------------------------------------------------------------------------------
H0   origin/master                                    5    95.62    0.06     0.00     NaN  3.2793  0.0009   0.0000  0.0705
H1   ablation-ve-init-normal                          5    95.58    0.05    -0.04  0.2285  3.2784  0.0019  -0.0009  0.0620
H2   ablation-ve-gate-split                           5    95.72    0.04     0.10  0.0159  3.2778  0.0014  -0.0015  0.0112
H3   ablation-rmsnorm-subtract-mean                   5    97.39    0.04     1.77  0.0000  3.2796  0.0017   0.0003  0.3303
H4   ablation-ve-layout-first-forward                 5    95.64    0.07     0.02  0.6056  3.2776  0.0012  -0.0017  0.0057
H5   ablation-ve-layout-first-double-forward          5    95.69    0.04     0.07  0.0695  3.2778  0.0010  -0.0015  0.0034
H6   ablation-ve-layout-last-back                     5    95.66    0.05     0.04  0.2212  3.2986  0.0028   0.0193  0.9999
----------------------------------------------------------------------------------------------------------------------------------

H0: origin/master
  losses = [3.2796, 3.2800, 3.2786, 3.2782, 3.2801]
  times  = [95.654, 95.622, 95.662, 95.639, 95.522]
  (n = 5)

H1: ablation-ve-init-normal
  losses = [3.2774, 3.2790, 3.2777, 3.2764, 3.2813]
  times  = [95.572, 95.510, 95.589, 95.575, 95.639]
  (n = 5)

H2: ablation-ve-gate-split
  losses = [3.2771, 3.2764, 3.2797, 3.2787, 3.2769]
  times  = [95.670, 95.744, 95.764, 95.722, 95.685]
  (n = 5)

H3: ablation-rmsnorm-subtract-mean
  losses = [3.2781, 3.2778, 3.2807, 3.2818, 3.2798]
  times  = [97.395, 97.418, 97.431, 97.363, 97.330]
  (n = 5)

H4: ablation-ve-layout-first-forward
  losses = [3.2764, 3.2777, 3.2771, 3.2772, 3.2796]
  times  = [95.670, 95.669, 95.657, 95.687, 95.522]
  (n = 5)

H5: ablation-ve-layout-first-double-forward
  losses = [3.2790, 3.2779, 3.2782, 3.2775, 3.2764]
  times  = [95.646, 95.646, 95.734, 95.713, 95.690]
  (n = 5)

H6: ablation-ve-layout-last-back
  losses = [3.2979, 3.2976, 3.3029, 3.2954, 3.2992]
  times  = [95.609, 95.735, 95.669, 95.638, 95.667]
  (n = 5)
```

## Timing

The improvements were minor but consistent, so I also ablated the number of step decrease with 10 repeats each. I haven't tested against currently open PRs (#207, #216, #217), but their ideas are mostly orthogonal to mine, so I should expect my improvements to hold if merged in any order.

```
Steps  Δ    N   Time μ  Time σ   Loss μ   Loss σ   p-val    Sig
-----------------------------------------------------------------
1555   0    5    95.62    0.06   3.2793   0.0009   0.0705    
1550  -5   10    95.38    0.05   3.2783   0.0016   0.0045   **
1547  -8   10    95.29    0.08   3.2782   0.0015   0.0021   **
1545 -10   20    95.14    0.06   3.2793   0.0016   0.0302    *
1540 -15   10    94.80    0.04   3.2796   0.0010   0.0988    
1535 -20   17    94.47    0.04   3.2805   0.0014   0.9121    
```

We include the runs for $\Delta=-8$ steps in the log. We achieve significance with $8$ steps,
```
import scipy.stats
import torch

losses = [3.2765, 3.2777, 3.281, 3.2799, 3.2764, 3.2777, 3.2776, 3.2794, 3.2779, 3.2783]
times = [95.352, 95.319, 95.321, 95.327, 95.248, 95.303, 95.283, 95.072, 95.359, 95.319]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0021

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (std=0.0015, mean=3.2782)

print("times:", torch.std_mean(torch.tensor(times)))
# times: (std=0.0831, mean=95.2903)
```

I ran the timing ablations across different machines, so I think the step change is more reliable than the time change (given that none of the ideas add flops). I suggest we attempt to merge at $-8\times 61ms \approx -0.5$ sec improvement.
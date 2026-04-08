# Paired head Muon groups

Treat each paired-head in K and V as its own Muon group, reducing loss/step and allowing us to reduce step count by 10 and still meet 3.28 loss target.


```bash
                 Runs  Steps   Time μ   Time σ  Time +/-   Time p   Loss μ   Loss σ  Loss +/-   Loss p
  baseline-1490     4   1490  93.0205   0.1629    0.0000      nan   3.2782   0.0010   -0.0018    0.0187
  baseline-1480     8   1480  92.5491   0.0588   -0.4714   0.0040   3.2800   0.0017    0.0000    0.5000
        this PR     8   1480  92.6259   0.2129   -0.3946   0.0038   3.2788   0.0009   -0.0012    0.0029
```

Previously, Muon treated each of Q, K, V, O as one `(hdim, hdim)` group per layer, lumping all heads together within each group. Given heads often specialize, it seemed intuitive that treating each *head* as its own Muon group would further improve optimization. I split K/V into per-**head-pair** groups (because of paired head attention in some layers). This gives each head pair its own independent polar decomposition. Treating each head-pair in K and V as its own Muon group improved loss per step while increasing time/step by ~.1% (this overhead can probably be reduced, splitting Muon groups further should reduce flops).

Now, we can reach the target loss in 10 less steps, saving ~0.4s. If we reduce the baseline by 10 steps without adding the per-head-pair Muon groups, the loss does not meet the target.

**Note on timing:** not sure why my 8xh100 is so slow -- ~8% slower per step for the baseline. However, given that the speedup comes from reducing steps, I think this will hold in a more controlled setup.

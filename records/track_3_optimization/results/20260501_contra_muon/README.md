# Contra Muon results

Contra Muon (https://github.com/nilin/contra-muon) modifies the Muon Newton-Schulz momentum update by subtracting `CONTRA_MUON / 2` times the operator-normalized momentum gradient

Everything else follows PR 274 NorMuon-lite with row/column variance normalization and u/w-floor postprocessing.

Note: the archived scripts/logs print `muon_weight_decay=0.025`, but `Muon.step`
does not use the stored `weight_decay` param-group value. The effective Muon
weight decay for these runs is therefore zero.

This run set uses `CONTRA_MUON=0.4`, `lr=0.0375`, `u/w-floor=0.35`, and
terminates at 3225 steps. Across 16 non-cherry-picked seeds, the step 3225
mean validation loss is 3.27854062. Under the Track 3 README's one-sided
z-test with `sigma=0.0013`, this gives `z=4.4904` and `p=3.55e-6`.

| Seed | 3200 val | 3225 val |
| -: | -: | -: |
| 1 | 3.27699 | 3.27626 |
| 2 | 3.27827 | 3.27752 |
| 3 | 3.28074 | 3.28000 |
| 4 | 3.27918 | 3.27846 |
| 5 | 3.28046 | 3.27967 |
| 6 | 3.28063 | 3.27989 |
| 7 | 3.27907 | 3.27832 |
| 8 | 3.27947 | 3.27867 |
| 9 | 3.27880 | 3.27807 |
| 10 | 3.27981 | 3.27906 |
| 11 | 3.27989 | 3.27912 |
| 12 | 3.27808 | 3.27733 |
| 13 | 3.27926 | 3.27851 |
| 14 | 3.27951 | 3.27880 |
| 15 | 3.27869 | 3.27791 |
| 16 | 3.27982 | 3.27906 |
| **Mean** | **3.27929188** | **3.27854062** |

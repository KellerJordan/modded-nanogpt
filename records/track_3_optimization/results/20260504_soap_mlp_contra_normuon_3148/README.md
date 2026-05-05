# SOAP-MLP Contra/NorMuon 3148 results

This result continues PR #278's SOAP/Shampoo-basis preconditioning on the MLP
matrices before the Contra-Muon + NorMuon-lite update path from PR #275.
Attention 2D matrices keep the Contra-Muon + NorMuon-lite path without SOAP
preconditioning.

The archived scripts restore the current upstream RMSNorm implementation and
write normal `logs/*.txt` capture files. Each submitted logfile contains the
full training script before the separator, with a fixed per-run seed and
`train_steps = 3148`.

Note: as in the Contra-Muon record, the archived scripts/logs print
`muon_weight_decay=0.025`, but `Muon.step` does not use the stored
`weight_decay` param-group value. The effective Muon weight decay for these
runs is therefore zero.

## Configuration

| field | value |
|---|---|
| train steps | 3148 |
| Contra-Muon | 0.4 |
| operator-normalized gradient subtraction | 0.2 |
| Muon momentum | 0.95 |
| Muon lr | 0.0375 |
| u/w floor | 0.35 |
| SOAP/Shampoo beta2 | 0.9 |
| SOAP preconditioned params | `mlp.fc.weight`, `mlp.proj.weight` |

## Reliability

Across 16 non-cherry-picked seeds at the common stopping point K=3148, the mean
validation loss is 3.27860000 with sample std 0.00134744. The Track 3
significance rule is:

`(3.28 - mean) * sqrt(n) >= 0.004`

For these runs:

`(3.28 - 3.27860000) * sqrt(16) = 0.00560000 >= 0.004`

| seed | val loss at K=3148 | logfile |
|---:|---:|---|
| 1 | 3.27888 | `seed01.txt` |
| 2 | 3.27934 | `seed02.txt` |
| 3 | 3.28172 | `seed03.txt` |
| 4 | 3.27806 | `seed04.txt` |
| 5 | 3.27840 | `seed05.txt` |
| 6 | 3.27796 | `seed06.txt` |
| 7 | 3.27715 | `seed07.txt` |
| 8 | 3.27835 | `seed08.txt` |
| 9 | 3.27760 | `seed09.txt` |
| 10 | 3.27729 | `seed10.txt` |
| 11 | 3.28080 | `seed11.txt` |
| 12 | 3.27650 | `seed12.txt` |
| 13 | 3.27822 | `seed13.txt` |
| 14 | 3.27977 | `seed14.txt` |
| 15 | 3.27832 | `seed15.txt` |
| 16 | 3.27924 | `seed16.txt` |
| **Mean** | **3.27860000** | |

This clears the current merged Track 3 frontier at 3225 steps and is 2 steps
below the live open PR #278 frontier of 3150 steps.

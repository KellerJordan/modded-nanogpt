# Aurora + SOAP-MLP + Contra-Muon + NorMuon-lite + u/w-floor

**3200 steps** to reach ≤3.28 val loss.

## Evidence

Mean val loss at step 3200: **3.27599** (n=4, `(3.28 - 3.27599) * sqrt(4) = 0.00801 ≥ 0.004` ✓)

| Seed | Val Loss @ 3200 |
|-----:|:---------------:|
| 0 | 3.27408 |
| 1 | 3.27632 |
| 2 | 3.27517 |
| 3 | 3.27841 |
| **Mean** | **3.27599** |

## Description

Builds on record #11 (Contra-Muon + NorMuon-lite + u/w-floor, 3225 steps) with two additions:

1. **Aurora polar** ([tilde-research/aurora-release](https://github.com/tilde-research/aurora-release)):
   Leverage-uniform polar decomposition for non-square matrices. Replaces standard Newton-Schulz
   with an iterated diagonal preconditioning that produces uniform row norms in the polar factor.
   Applied only to non-square matrices (MLP fc 3072×768, MLP proj 768×3072). Square attention
   matrices (768×768) use standard NS5. `pp_iterations=2`, `pp_beta=0.5`.

2. **SOAP-MLP preconditioning**: Shampoo-eigenbasis preconditioning of Nesterov momentum for MLP
   fc and proj weights only (not attention). Before Newton-Schulz, momentum is projected into the
   Shampoo eigenbasis (row+col covariance EMAs), Adam-style diagonal scaling is applied, and the
   result is projected back. Norm-preserving. `SOAP_BETA2=0.90`, `precondition_frequency=10`.

Base hyperparameters unchanged from #11:
`MUON_LR=0.0375`, `MU=0.95`, `CONTRA_MUON=0.4`, `TARGET_UW=0.35`, `cooldown_frac=0.7`.

Note: `schedule_steps=3225` sets the LR cooldown horizon (same as #11), while
`train_steps=3200` is the stopping point. `MUON_WEIGHT_DECAY=0.025` is stored in the
optimizer defaults for config parity but is not applied in the Muon `step()` method.

## Key Insight

Aurora and SOAP-MLP are orthogonal improvements:
- **SOAP** improves *what direction* to move (better curvature preconditioning of the momentum)
- **Aurora** improves *how to orthogonalize* (leverage-uniform polar factor for rectangular matrices)

Neither interferes with the other. SOAP operates before NS, Aurora operates during NS.

## Files

- `train_gpt_simple_aurora_soap_mlp.py` — Single-file training script (drop-in replacement)
- `4f53e1ac-*.txt` — Seed 0 log
- `f450bccf-*.txt` — Seed 1 log
- `959cfa09-*.txt` — Seed 2 log
- `f94f1b15-*.txt` — Seed 3 log

## Credits

- Contra-Muon: @nilin ([PR#275](https://github.com/KellerJordan/modded-nanogpt/pull/275))
- NorMuon-lite + u/w-floor: @kumarkrishna ([PR#274](https://github.com/KellerJordan/modded-nanogpt/pull/274))
- Aurora polar: @liyang2019 ([PR#284](https://github.com/KellerJordan/modded-nanogpt/pull/284))
- SOAP-MLP: @Srachuri-code ([PR#279](https://github.com/KellerJordan/modded-nanogpt/pull/279))

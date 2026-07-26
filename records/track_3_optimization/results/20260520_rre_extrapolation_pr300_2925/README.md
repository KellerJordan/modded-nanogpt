# Track 3: capped RRE extrapolation on the PR #300 stack

This result applies a fixed vector-extrapolation overlay to the Track 3 optimizer stack from PR #300. The base stack is Aurora on `mlp.proj`, radial brake, extended Contra-Muon to normal ramp, SOAP on MLP+V, and the inherited power-law cooldown. The new piece is reduced-rank extrapolation (RRE) applied late in training with a fixed schedule:

```text
extrapolation_name = rre
extrap_start_step = 2820
extrap_every = 5
RRE history length k = 4
extrap_damping = 0.875
extrap_max_rel = 0.001
extrap_reset_momentum = False
extrap_stop_step = 2925
extrap_after_stop_every = 0
```

The submitted step count is **2925**. All eight runs are full runs from step 0. Each logfile includes the full training script used for the run. For review convenience, `train_gpt_simple_rre_pr300_2925.py` contains the same train-code path with the submitted RRE and stop-step settings as defaults.

## Result

At step 2925:

```text
n = 8
mean val loss = 3.27812750
std            = 0.00088134
(3.28 - mu) * sqrt(n) = 0.00529623
```

This exceeds the Track 3 README threshold of `0.004`. Equivalently, with `sigma=0.0013`, this is `z = 4.0740`, one-sided `p = 2.31e-05`.

For comparison, the same eight runs have mean val loss `3.27969250` at step 2900, which does not clear the threshold.

| Seed | Log | 2900 val | 2925 val |
| -: | - | -: | -: |
| 0 | `1786243c-db47-49ff-bff8-3faf814bf763.txt` | 3.27949 | 3.27793 |
| 1 | `d95a47ca-e103-4f0f-a273-dcaaca8f8bf7.txt` | 3.28139 | 3.27981 |
| 2 | `4e96056c-e936-4400-8353-403af8e4fe50.txt` | 3.27914 | 3.27761 |
| 3 | `6721c9f5-1f63-4c07-8cb9-fee0f9b313b4.txt` | 3.27991 | 3.27838 |
| 4 | `1853c4b8-77a6-48b1-9459-da4a45427ea7.txt` | 3.27943 | 3.27785 |
| 5 | `ac500b64-d44e-4052-84a5-7c8627164cf4.txt` | 3.27942 | 3.27783 |
| 6 | `89f1440d-ac2f-4666-be46-8bd6a403b613.txt` | 3.28031 | 3.27877 |
| 7 | `8ef4bff2-671c-44d1-b451-4c300317a3b1.txt` | 3.27845 | 3.27684 |
| **Mean** | | **3.27969250** | **3.27812750** |

## Reproducibility notes

- Dataset, batch size, and model architecture match the Track 3 baseline stack used by PR #300.
- The RRE schedule is fixed before the run and is identical across seeds.
- The selected stopping point, 2925, is shared across all trials.
- The runs were launched on Modal H100 workers. Wall-clock speed is not part of the Track 3 objective.

## Files

- `records/track_3_optimization/results/20260520_rre_extrapolation_pr300_2925/README.md`
- `records/track_3_optimization/results/20260520_rre_extrapolation_pr300_2925/train_gpt_simple_rre_pr300_2925.py`
- 8 full reproducibility logfiles in the same directory, seeds 0 through 7

## Credits

This submission builds directly on the open PR #300 stack and inherits its credited lineage:

- PR #274 / Skylight-001: NorMuon-lite row/column normalization lineage, u/w floor, and Muon setup.
- PR #275 / Contra-Muon: Contra-Muon update term.
- PR #278 / MLP SOAP preconditioning.
- PR #283 / Trustlight attention-SOAP lineage.
- PR #287 / power-law LR cooldown.
- PR #291 / Contra-Muon to Soft-Muon lineage.
- PR #294 / radial brake.
- PR #300 / Aurora row-balanced polar on `mlp.proj`, Soft-Muon/NorMuon-lite removal, and extended Contra-Muon ramp.

New in this result: late capped RRE vector extrapolation with no momentum reset.

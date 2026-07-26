# Contra-Muon Split-Cooldown Results

This run set starts from the Track 3 record #11 Contra-Muon setup:
Contra-Muon on top of the PR 274 NorMuon-lite row/column variance
normalization and u/w-floor postprocessing.

The change tested here is a group-specific learning-rate cooldown schedule:
auxiliary AdamW parameters use `cooldown_frac=0.4`, while matrix
Contra-Muon/NorMuon parameters use `cooldown_frac=0.8`. The run terminates at
3175 steps.

Configuration summary:

- `CONTRA_MUON=0.4`
- `target_uw=0.35`
- matrix optimizer lr `0.0375`
- matrix optimizer momentum `0.95`
- NorMuon-lite row/column variance `beta2=0.95`
- auxiliary AdamW lrs `0.3`, `1/320`, and `0.01`
- train steps `3175`
- each run used 4 A100-SXM4-80GB GPUs
- Slurm job array `27179069`, 10 non-cherry-picked tasks
- seeds `1` through `10`, assigned by the Slurm array script via `SEED=$SLURM_ARRAY_TASK_ID`

The table reports the logged validation losses from all 10 tasks of the
predeclared 10-seed Slurm array.

| Seed | Log | 3150 val | 3175 val |
| -: | - | -: | -: |
| 1 | `c1af0bd1-6999-44d1-a618-3d1234ea32f0.txt` | 3.27794 | 3.27723 |
| 2 | `e1422959-1c4d-47f8-b9d7-72f8d0f6e493.txt` | 3.27816 | 3.27747 |
| 3 | `0996e4d1-5fd1-4d9d-8d23-cccdb481f92c.txt` | 3.28146 | 3.28075 |
| 4 | `8182c0e7-81c7-4182-995c-0ee284d806f0.txt` | 3.27847 | 3.27776 |
| 5 | `8b391e49-66a7-4a1c-b77d-dceb8c41589c.txt` | 3.27880 | 3.27808 |
| 6 | `e83a35e5-97b9-4bc7-a95e-1cd6049a601d.txt` | 3.28133 | 3.28062 |
| 7 | `4f019bec-31fb-4555-8bea-4559b4572da0.txt` | 3.27783 | 3.27713 |
| 8 | `32010038-22a6-45e7-b450-8baf8209a51c.txt` | 3.27772 | 3.27702 |
| 9 | `34508f69-4d51-4a76-8a55-327d4b25a7f1.txt` | 3.27797 | 3.27726 |
| 10 | `a99f490d-c976-45a4-ae6b-97273db472fd.txt` | 3.27924 | 3.27856 |
| **Mean** |  | **3.27889200** | **3.27818800** |

Statistical significance analysis:

- At 3175 steps, `n=10` and mean validation loss is `3.27818800`.
- The Track 3 precision condition is `(3.28 - mu) * sqrt(n) >= 0.004`.
- Here `(3.28 - 3.27818800) * sqrt(10) = 0.00573005`, which passes.
- Equivalently, assuming `sigma=0.0013`, the one-sided z statistic is `4.4077`,
  giving `p=5.22e-06`.
- At 3150 steps, the mean is `3.27889200` and the precision value is
  `0.00350380`, so these 10 runs do not establish a 3150-step result.

# Hyperball + Projected-EMA-Nesterov records

Archive of benchmark records under **Parallax**, param-matched to softmax attention by reducing the MLP size (`hdim = 7 * dim//2`,
cancels the Parallax $W_R$ parameters). The 8 records are:

- **4 hyperball optimizers** (no-EMA): MuonH, NorMuonH, SOAP-H, AuroraH
- the **same 4 hyperball + projected EMA-Nesterov**: `*_pema`

Metric: official Track-3 **"steps to 3.28"** with significance i.e. the step where the n-seed mean
first satisfies $(3.28 − \mu)·\sqrt{n} ≥ 0.004$ (n-dependent threshold).

## Records

| record | script | slurm folder | n | **sig-cross** |
|---|---|---|---|---|
| MuonH (no-EMA)    | `rec_muonh.py`         | `slurm/muonh_noema/`    | 4 | 3050 |
| NorMuonH (no-EMA) | `rec_normuonh.py`      | `slurm/normuonh_noema/` | 4 | 3015 |
| AuroraH (no-EMA)  | `rec_aurorah.py`       | `slurm/aurorah_noema/`  | 4 | 2995 |
| SOAP-H (no-EMA)   | `rec27_soaph.py`       | `slurm/soaph_noema/`    | 8 | 2850 |
| **MuonH +PEMA**    | `rec_muonh_pema.py`    | `slurm/muonh_pema/`    | 4 | **2880** |
| **NorMuonH +PEMA** | `rec_normuonh_pema.py` | `slurm/normuonh_pema/` | 8 | **2810** |
| **AuroraH +PEMA**  | `rec_aurorah_pema.py`  | `slurm/aurorah_pema/`  | 8 | **2800** |
| **SOAP-H +PEMA**   | `rec27_soaph_pema.py`  | `slurm/soaph_pema/`    | 8 | **2740** |

## Hyperparameters

|                | MuonH | NorMuonH | SOAP-H | AuroraH |
|---|---|---|---|---|
| **`RHO_SCALE`**  | **0.5** | **0.9** | **0.5** | **0.75** |
| horizon (no-EMA) | 3100 | 3050 | 2925 | 3050 |
| **horizon (+PEMA)** | **2900** | **2850** | **2825** | **2825** |
| **EMA stepsize** | **0.75** | **0.75** | **0.75** | **0.75** |

**Shared (all 8):** base matrix LR `0.018`; hyperball `scale_invariant_update_`, full linear anneal (`train_steps = lr_steps`);

**Projected EMA-Nesterov (the 4 `*_pema`):** `EMA_DECAY=0.99`, `EMA_PREFILL=323`,
`rest_steps = train_steps`.

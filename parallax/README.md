# Parallax mechanism on the Track-3 optimization benchmark

Drop-in **Parallax** mechanism applied on top of the
`track_3_optimization` records, **without changing the optimizer configuration**.

Paper: [https://arxiv.org/abs/2605.29157](https://arxiv.org/abs/2605.29157)</br>
kernels library: [https://github.com/yifei-zuo/Parallax](https://github.com/yifei-zuo/Parallax))

## Results (steps to val < 3.28; lower is better)

Standard Attention steps are the documented record numbers (see
`records/track_3_optimization/README.md`). Parallax's records follow the benchmark's convention: the
first eval-grid step at which the **seed-mean** val_loss first drops below 3.28 (i.e. average seeds, then threshold).

<p align="left">
  <img src="imgs/parallax_all_zoom.png" width="49%" />
  <img src="imgs/attention_all_zoom.png" width="49%" />
</p>

| Algo (Track-3 record) | Script | Attn steps | PLX steps | % boost |
| - | - | - | - | - |
| **SOAP-H** ([#27](https://github.com/KellerJordan/modded-nanogpt/pull/302)) | `rec27_soaph.py` | 3125 | **2880** (n=4) | **7.84%** |
| **DynMuon** ([#28](https://github.com/KellerJordan/modded-nanogpt/pull/304)) | `rec28_dynmuon.py` | 3175 | **2975** (n=3) | **6.30%** |
| Aurora ([#17](https://github.com/KellerJordan/modded-nanogpt/pull/284)) | `rec17_aurora.py` | 3175 | 3025 (n=1) | 4.72% |
| trustlight ([#16](https://github.com/KellerJordan/modded-nanogpt/pull/283)) | `rec16_trustlight.py` | 3125 | 3052 (n=1) | 2.34% |
| SinkSOAP ([#26](https://github.com/KellerJordan/modded-nanogpt/pull/298)) | `rec26_sinksoap.py` | 3090 | 3025 (n=1) | 2.10% |
| SOAP-MLP ([#14](https://github.com/KellerJordan/modded-nanogpt/pull/278)) | `rec14_soap_mlp.py` | 3150 | 3100 (n=1) | 1.59% |
| split-cooldown ([#24](https://github.com/KellerJordan/modded-nanogpt/pull/292)) | `rec24_split_cd.py` | 3175 | 3125 (n=1) | 1.57% |
| Contra-Soft-Muon ([#20](https://github.com/KellerJordan/modded-nanogpt/pull/291)) | `rec20_contra_soft.py` | 3030 | 3000 (n=1) | 0.99% |
| Vanilla Muon (baseline) | `muon_baseline.py` | ~3400 | 3325 (n=1) | ~2% |

SOAP-H + Parallax's **2880** (n=4) is below the best current Track-3 record ([#30](https://github.com/KellerJordan/modded-nanogpt/pull/300), 2930).

## Logs (`results/`)

| variant | logs | per-seed crossing |
|---|---|---|
| `rec27_soaph/`    | `seed{0..3}.txt` | 2910 / 2865 / 2870 / 2860 |
| `rec28_dynmuon/`  | `seed{0..2}.txt` | 2950 / 2975 / 2975 |
| `rec17_aurora/`   | `seed0.txt` | 3025 |
| `rec16_trustlight/` | `seed0.txt` | 3052 |
| `rec26_sinksoap/` | `seed0.txt` | 3025 |
| `rec14_soap_mlp/` | `seed0.txt` | 3100 |
| `rec24_split_cd/` | `seed0.txt` | 3125 |
| `rec20_contra_soft/` | `seed0.txt` | 3000 |
| `muon_baseline/`  | `seed0.txt` | 3325 |

For the **vanilla-attention baseline** trajectory of each, use the corresponding official record log
already in this repo under `records/track_3_optimization/` (the benchmark ID in the table above).

## How to run

Each `rec*.py` / `muon_baseline.py` is a self-contained training script (a copy of the corresponding record with the Parallax patch applied).

```bash
# 1) data (same as the benchmark baseline)
python data/cached_fineweb10B.py 20

# 2) Parallax kernels: clone + point PARALLAX_PATH at it (or pip install it)
git clone https://github.com/yifei-zuo/Parallax /path/to/Parallax
export PARALLAX_PATH=/path/to/Parallax     # parallax_op.py inserts this onto sys.path

# 3) run a variant
SEED=0 torchrun --standalone --nproc_per_node=8 parallax/rec27_soaph.py
```

The first step pays a Triton autotune sweep on the Parallax fwd + bwd kernels plus a Dynamo trace of
the model. Subsequent steps hit the autotune cache and run at full compiled speed. The logs in
`results/` were produced from an earlier eager-only revision of the scripts; loss trajectory is
unchanged under compile, only wall-clock per step.

## The Parallax patch

**Each `rec*.py` / `muon_baseline.py`** adds:
  1. `self.r = Linear(dim, hdim)` in the attention module.
  2. `r = self.r(x).view(...)` in `forward`, then the same RMSNorm + RoPE applied to `q`/`k`.
  3. Replaces the `F.scaled_dot_product_attention(...)` call with `parallax_func(q, r, k, v, scale)`
     (imported from the local `parallax_op.py`).

The optimizer, data, batch size, and architecture are otherwise **unchanged** from the record.

## Param-controlled study: ContraNorMuon + Parallax (identical params via MLP-trim)

The Results table above adds the probe projection `W_R` (~+7M at 124M scale) on top of each
record, so part of the speedup could be "more parameters" rather than "the Parallax mechanism".
This section isolates the mechanism on the **ContraNorMuon (Muown)** optimizer by funding `W_R`
**param-neutrally**, so the total parameter count is byte-for-byte the vanilla baseline.

### Funding `W_R` for free: trim the MLP, don't touch the KV heads

The cheapest place to recover `W_R`'s cost is **optimizer-dependent**:

* **SOAP-H** (PR to this repo): GQA on `k`/`v` (`H_kv=3`) gives back the params and still wins.
* **ContraNorMuon**: GQA-3 **fails** — halving the KV heads costs more than the probe buys, and the
  run never reaches val<3.28 at 2900 steps. The knob that works here is **trimming the MLP**:
  `hdim = 7*dim//2` (3.5× instead of 4×) frees exactly `2·dim·(4dim − 3.5dim)·n_layer = +7.08M`,
  which cancels `W_R`'s `dim·dim·n_layer = +7.08M`. `q`, `k`, `v`, the probe `r`, and the output
  projection all stay at full 6 heads.

### Result — same params, the mechanism still wins

`train_steps = lr_schedule_steps = 2900`; numbers are the seed-mean simple crossing
(first eval-grid step where the averaged val_loss drops below 3.28), same convention as the table above.

| config (identical parameter count) | script | steps to seed-mean val<3.28 |
|---|---|---|
| ContraNorMuon record (vanilla attention) | — | ~2995 |
| ContraNorMuon @2900, **no Parallax** (MLP-4×), n=4 | `rec31_contranormuon_base.py` | **never crosses** (seed-mean 3.2845) |
| ContraNorMuon @2900, **+Parallax, param-neutral** (MLP-3.5× + `W_R`), n=40 | `rec31_contranormuon_plx.py` | **2840** (seed-mean 3.27742) |

At **identical parameters** the same-horizon baseline never reaches the target while the
Parallax variant crosses at 2840 — and against the ~2995 record that is **~155 steps / ~5.2%**,
entirely parameter-free.

### Statistical strength & horizon sweep

The leaderboard's validity test is `(3.28 − seed_mean)·√n ≥ 0.004`. At 2900 (n=40) the margin is
**0.0163 ≈ 4× the bar**. Sweeping the (jointly set) `train_steps = lr_schedule_steps` horizon:

| horizon | n | seed-mean @ horizon | `(3.28−μ)·√n` | seed-mean crossing |
|---|---|---|---|---|
| 2875 | 20 | 3.27881 | 0.0053 | 2838 |
| **2900** | 40 | 3.27742 | **0.0163** | **2839–2840** |
| 2925 | 12 | 3.27608 | 0.0136 | 2843 |
| 2950 | 12 | 3.27481 | 0.0180 | 2851 |

The crossing is **non-monotone** with a minimum near 2900 (longer horizons lower the final loss but
keep the LR higher mid-cooldown, so the threshold is reached later). **2875** is the shortest horizon
that still clears the `0.004` margin (thin, 0.0053); **2900** is the safe, well-separated submission.

Both scripts are self-contained copies of the ContraNorMuon training script with only the Parallax
patch (`self.r` + `parallax_func`) and the 3.5× MLP differing between them; optimizer, data, and batch
size are unchanged. Logs: `results/rec31_contranormuon_plx/seed{0..39}.txt`,
`results/rec31_contranormuon_base/seed{0..3}.txt`.

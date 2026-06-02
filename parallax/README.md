# Parallax attention on the Track-3 optimization benchmark

Drop-in **Parallax** attention ([arXiv 2605.29157](https://arxiv.org/abs/2605.29157), kernels:
[yifei-zuo/Parallax](https://github.com/yifei-zuo/Parallax)) applied on top of the
`track_3_optimization` records, **without changing the optimizer**. The question: how much does a
better *attention* (rather than a better *optimizer*) move the "steps to val < 3.28" metric?

## What Parallax is

Softmax attention computes a *local-constant* value estimate (a weighted mean of $v$). Parallax
upgrades it to a *local-linear* one by adding **one** extra query-like projection
$\rho = x W_R$ (same shape as $q$, RMSNorm'd + RoPE'd like $q$/$k$):

$$o_i = \sum_j p_{ij}\, v_j - \mathrm{Cov}_{p_i}\big(\rho_i^{\top} k,\; v\big), \qquad p_i = \mathrm{softmax}(q_i K^{\top})$$

$\rho \to 0$ recovers softmax. It is **not** a linear-attention method — same $O(L^2)$ flash
structure; it just adds a learned first-order correction term. $W_R$ adds $\approx d_\mathrm{model}^2$
params per layer ($\approx$ +7M / +6% for the 124M model here).

## Results (steps to val < 3.28; lower is better)

`Vanilla attn steps` are the documented record numbers (see
`records/track_3_optimization/README.md`). `Parallax steps` follow the benchmark's convention: the
first eval-grid step (val every 125 steps, then every 25 past 94% of training) at which the
**seed-mean** val_loss first drops below 3.28 (i.e. average seeds, then threshold — as
`make_figures.py` does). **% boost = (vanilla − parallax) / vanilla.** The bracketed **#N** is the
algorithm's Track-3 *leaderboard record index* (its row in `records/track_3_optimization/README.md`,
ordered by acceptance — **not** a PR number); each links to the upstream PR that added that record.

| Algo (Track-3 record) | Script | Vanilla attn steps | Parallax steps | % boost |
| - | - | - | - | - |
| **SOAP-H** ([#27](https://github.com/KellerJordan/modded-nanogpt/pull/302)) | `rec27_soaph.py` | 3125 | **2900** (n=4) | **7.20%** |
| **DynMuon** ([#28](https://github.com/KellerJordan/modded-nanogpt/pull/304)) | `rec28_dynmuon.py` | 3175 | **2975** (n=3) | **6.30%** |
| Aurora ([#17](https://github.com/KellerJordan/modded-nanogpt/pull/284)) | `rec17_aurora.py` | 3175 | 3025 (n=1) | 4.72% |
| trustlight ([#16](https://github.com/KellerJordan/modded-nanogpt/pull/283)) | `rec16_trustlight.py` | 3125 | 3052 (n=1) | 2.34% |
| SinkSOAP ([#26](https://github.com/KellerJordan/modded-nanogpt/pull/298)) | `rec26_sinksoap.py` | 3090 | 3025 (n=1) | 2.10% |
| SOAP-MLP ([#14](https://github.com/KellerJordan/modded-nanogpt/pull/278)) | `rec14_soap_mlp.py` | 3150 | 3100 (n=1) | 1.59% |
| split-cooldown ([#24](https://github.com/KellerJordan/modded-nanogpt/pull/292)) | `rec24_split_cd.py` | 3175 | 3125 (n=1) | 1.57% |
| Contra-Soft-Muon ([#20](https://github.com/KellerJordan/modded-nanogpt/pull/291)) | `rec20_contra_soft.py` | 3030 | 3000 (n=1) | 0.99% |
| Vanilla Muon (baseline) | `muon_baseline.py` | ~3400 | 3325 (n=1) | ~2% |

Notes:
- 124M / seq-1024 is Parallax's *weak* regime (its design edge is recall / long-context / scale), so
  these are conservative numbers.
- A param-matched control (widen MLP by the same +7M, no Parallax) is the right way to confirm the
  gain is structural rather than just "+params" — recommended follow-up.

## Logs (`results/`)

Full val-loss trajectories for the runs behind the table — one subdirectory per variant, one file
per seed: `results/<variant>/seed<N>.txt`. Each line is
`step:N/TOTAL val_loss:X train_time:... step_avg:...`, the **same format as
`records/track_3_optimization/results/`**, so the benchmark's `make_figures.py` regex
(`step:(\d+)/(\d+)\s+val_loss:([0-9.]+)`) parses them directly and averages the seeds in a subdir.

| variant | logs | crossing(s) |
|---|---|---|
| `rec27_soaph/`    | `seed{0..3}.txt` | 2925 / 2875 / 2875 / 2875 |
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
These Parallax runs are eager / single-seed (except SOAP-H n=4, DynMuon n=3), so a vanilla-vs-Parallax
overlay is not a perfectly matched A/B across harness/seed.

## How to run

Each `rec*.py` / `muon_baseline.py` is a self-contained training script (a copy of the corresponding
record with the ~6-line Parallax patch applied). They run **eager** (`torch.compile` disabled) for
compatibility with the upstream Parallax kernel; `torch.compile` support is WIP.

```bash
# 1) data (same as the benchmark baseline)
python data/cached_fineweb10B.py 20

# 2) Parallax kernels: clone + point PARALLAX_PATH at it (or pip install it)
git clone https://github.com/yifei-zuo/Parallax /path/to/Parallax
export PARALLAX_PATH=/path/to/Parallax     # scripts: _sys.path.insert(0, $PARALLAX_PATH)

# 3) run a variant (SEED / SOAP_SEED env selects the seed)
SEED=0 torchrun --standalone --nproc_per_node=8 parallax/rec27_soaph.py
```

Note: the upstream Parallax `parallax/__init__.py` imports a CuTeDSL decode kernel that needs
`cutlass`. In a training-only environment, either install `cutlass` or guard that import
(`try: from parallax.cute import parallax_decode / except Exception: parallax_decode = None`).

## The Parallax patch (how each variant differs from its record)

Relative to the base record script, each variant adds exactly:
1. `self.r = Linear(dim, hdim)` in the attention module (the $W_R$ projection),
2. `r = self.r(x).view(...)` in `forward`, then the same RMSNorm + RoPE applied to `q`/`k`,
3. replaces the `F.scaled_dot_product_attention(...)` call with `parallax_func(q, r, k, v, scale)`.

The optimizer, data, batch size, and architecture are otherwise **unchanged** from the record.

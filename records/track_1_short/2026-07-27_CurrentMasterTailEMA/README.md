# Terminal-blended readout TailEMA

This Track 1 submission reaches the target in **1288 total steps** and averages
**75.655s (1.261 minutes)** on 8xH100. It is based on current training code at
`003ff3e2dd23ff5f7dfd6ada88eb1ff9320bfd3b`, which already contains the merged
PR #317 XSA rewrite. The subsequent `bc1b58e` upstream commit changes only the
leaderboard README.

The change combines three pieces:

- keep an FP32 TailEMA of the sharded `lm_head` and embedding readout from
  logical step 848, with decay `0.9933`;
- blend 65% of that TailEMA into the final readout weights once, immediately
  before terminal validation;
- initialize MLP `c_proj` with standard deviation `0.5 * d_model**-0.5` rather
  than zero, then shorten the schedule from `1285 + 15 = 1300` to
  `1273 + 15 = 1288` steps.

The TailEMA updates and terminal blend are fused into the existing compiled
Adam pointwise update. Each rank stores only its optimizer shard, and the
terminal blended shard uses the optimizer's ordinary all-gather. There is no
new terminal collective and no extra `torch._inductor.config` or
`torch.compile` flag.

## Confirmatory result

The final timing experiment used one dedicated Nebius node with 8 NVIDIA H100
SXM 80GB GPUs. Baseline and candidate alternated in eight prespecified blocks:
`ABBA, BAAB, ABBA, BAAB, ABBA, BAAB, ABBA, BAAB`. Every accepted block has two
observations per arm, giving 16 baseline and 16 candidate runs on the same
machine and software environment.

| Arm | Runs | Steps | Time mean | Time SD | Loss mean | Loss SD | Loss p |
|---|---:|---:|---:|---:|---:|---:|---:|
| current master | 16 | 1300 | 76.3385s | 0.0364s | 3.2776125 | 0.0017029 | — |
| this submission | 16 | 1288 | **75.6554s** | 0.0365s | **3.2777875** | 0.0017712 | **7.97e-05** |

The geometric mean of the eight within-block candidate/baseline runtime ratios
is **0.99105219**, or **0.8948% faster**. Its two-sided 95% confidence interval
is `[0.99064263, 0.99146191]`, corresponding to **0.8538% to 0.9357% faster**.
The prespecified one-sided paired timing test gives `p = 1.378e-10`.

| Block | Order | Baseline mean | Candidate mean | Ratio |
|---:|:---:|---:|---:|---:|
| 1 | ABBA | 76.3480s | 75.6425s | 0.99075953 |
| 2 | BAAB | 76.3390s | 75.6370s | 0.99080419 |
| 3 | ABBA | 76.3940s | 75.6820s | 0.99067987 |
| 4 | BAAB | 76.3225s | 75.6485s | 0.99116903 |
| 5 | ABBA | 76.3330s | 75.6490s | 0.99103923 |
| 6 | BAAB | 76.3580s | 75.6795s | 0.99111426 |
| 7 | ABBA | 76.2835s | 75.6860s | 0.99216732 |
| 8 | BAAB | 76.3300s | 75.6190s | 0.99068491 |

The merged leaderboard record from PR #317 is listed at 1.266 minutes
(75.96s). The candidate mean here is about 0.305s faster. The formal speed
claim above uses only the paired same-hardware current-master comparison.

## Independent accuracy gate

Before the timing confirmation, ten independent candidate runs were frozen and
executed. Mean loss was **3.2781500**, sample SD was `0.0016814`, and the
one-sided one-sample t-test against 3.28 gave **p = 0.003473**. Mean runtime was
75.6297s. The 16 candidate outcomes in the final paired experiment independently
pass the same rule with `p = 7.97e-05`.

## Implementation notes

Adam updates the readout only on alternating outer steps. To match a logical
per-step TailEMA, the fused graph observes both the unchanged pre-update weight
and the post-update weight on each Adam step. At the embedding split, the
existing `lm_head` optimizer state transfer also copies the FP32 EMA shard into
the corresponding embedding shard. The final graph performs the last two EMA
observations, BF16 blend, and ordinary Adam update together.

Tail averaging was inspired by Jesse Clark's Track 3 tail-reference and
TailEMA work in
[`20260520_tail_refinterp_2900`](../../track_3_optimization/results/20260520_tail_refinterp_2900)
and
[`20260611_tailema_2720_submission`](../../track_3_optimization/results/20260611_tailema_2720_submission).
This submission narrows the scope to the readout and integrates it into the
sharded Track 1 optimizer to keep the runtime overhead small.

## Reproduction and audit trail

Each `.full.txt` log embeds the exact `train_gpt.py` source used by that arm.
The final candidate source SHA-256 is
`ee9586a5579a94d135ad77a826538553566ecd398b9ca3c49729ca3107969063`.

Recompute all reported tests with:

```bash
uv run --with scipy python \
  records/track_1_short/2026-07-27_CurrentMasterTailEMA/analyze.py \
  --accuracy records/track_1_short/2026-07-27_CurrentMasterTailEMA/runs/nebius-accuracy-10 \
  --timing records/track_1_short/2026-07-27_CurrentMasterTailEMA/runs/nebius-timing-confirmation
```

Files in this record:

- `runs/nebius-timing-confirmation/`: the 32 final paired timing logs;
- `runs/nebius-accuracy-10/`: the independent ten-run accuracy gate;
- `runs/nebius-abba-screen/`: the exploratory four-run engineering screen;
- `EXPERIMENT_PLAN.md`: protocol frozen before outcomes;
- `EXECUTION_NOTES.md`: chronological execution and infrastructure disclosures;
- `RESULTS.txt`: exact final analyzer output;
- `analyze.py`: loss and paired timing analysis;
- `infra/`: exact paired runner and Python 3.12 wrapper used on the node.

Environment: 8x NVIDIA H100 SXM 80GB, Ubuntu 24.04, Python 3.12.3,
PyTorch `2.10.0+cu128`, Triton `3.6.0`, NVIDIA driver `570.211.01`.

Python 3.12 occasionally delayed Inductor fork-pool cleanup after a terminal
result. Runs had a fixed ten-minute infrastructure watchdog, and rejected
pre-outcome attempts were retained as described in `EXECUTION_NOTES.md`.
Compiler worker count, CPU affinity, I/O priority, compilation flags, model
code, RNG, and measured training were unchanged.

# Record candidate: ANVIL2 + sampled head, n-gram token norm, train-fit copy mixture, retuned schedule

> **Status: rule 2 met on Modal, rule 4 open.** 18 runs of `code/` unchanged on Modal 8xH100:
> **38.211 ± 0.278 s**, mean final val loss **3.27925 ± 0.00077**, one-sided p = 3.3e-4 against
> 3.28 ([`this_pr/statistics.md`](this_pr/statistics.md)). There is no same-machine baseline against
> [#360](https://github.com/KellerJordan/modded-nanogpt/pull/360) yet; see [What is still needed](#what-is-still-needed).

`code/` builds on the ANVIL2 record ([#360](https://github.com/KellerJordan/modded-nanogpt/pull/360))
and three open PRs. It is the exact tree that ran: its 17 Python modules hash to the
`source_sha256` in [`first_run/final_state.json`](first_run/final_state.json), and every validation
run's checkpoint manifest matched the same 17 hashes. The only file left out is the FA3 binary
(871 MB), which the trainer loads from the Hugging Face cache.

## Results

All rows ran on the same Modal host type: 8x H100 80GB HBM3, driver 580.95.05, CUDA 13.1,
torch 2.10.0+cu128. Modal's virtualized host costs more per kernel launch than bare metal, so these
times do not compare directly with the leaderboard.

| Run | runs | train_time | val loss | steps |
| --- | ---: | ---: | ---: | ---: |
| Earlier version of this tree (ANVIL2 port) | 1 | 40.450 s | 3.27960 | 1184 |
| This tree, first run | 1 | 37.590 s | 3.27854 | 1168 |
| **This tree, validation pool** | **18** | **38.211 s** | **3.27925** | **1168** |

The first run is not in the pool: this tree was chosen because of it, so it is a favourable draw.

For context only: ANVIL2's own pool is 39.914 s mean over 18 runs (val loss 3.27731) on a vast.ai
bare-metal 8xH100.

## What changed vs ANVIL2 ([#360](https://github.com/KellerJordan/modded-nanogpt/pull/360))

The PR's second commit is exactly this delta: +1,330 / -318 lines across the trainer, the kernels
and six new modules, plus the bundled GPT-2 tokenizer files and the vendored FA3 Python package.

| Change | Origin |
| --- | --- |
| Sampled LM-head weight gradient in the CE backward (`sampled_head.py`) | [#371](https://github.com/KellerJordan/modded-nanogpt/pull/371) (`approx_backward/head.py`), env-var reads removed |
| Token normalization for the n-gram hashes only (`token_norm.py`); the map's sha256 equals [#375](https://github.com/KellerJordan/modded-nanogpt/pull/375)'s | [#375](https://github.com/KellerJordan/modded-nanogpt/pull/375), reimplemented without `tokenizers` |
| Document-local copy mixture at the final validation (`copy_mixture.py`). The 31 bucket weights are fit inside the timer on the last 8 training batches the run already consumed, instead of a second loader on the last train shard | [#376](https://github.com/KellerJordan/modded-nanogpt/pull/376), adapted |
| n-gram table 84.6M -> 105.75M rows; n-gram gradients aggregated into compact per-row slots before the row Adam (`compact_grad.py`, `deferred_sign_scatter.py`, `bigram_kernels.py`) | this PR |
| Schedule 1194 -> 1168 steps (scheduled 1122 -> 1098, growth 52 -> 36, extension 20 -> 34); tail EMA window 298 -> 154, blend 0.65 -> 0.85 | this PR |
| Narrow (d_v=64) rows in the packed FP8 QKV quantizer, per-program FP8 amax slots, CE kernel reads FP16 logits | this PR |
| No environment reads besides `DATA_PATH` (`KX_STEPS`, `KX_SEED`, `CUDA_HOME` removed), GPT-2 tokenizer files bundled, FA3 package vendored as a fallback | this PR, for an offline run |
| Checkpoint manifest plus one shard per rank after the final validation (`checkpoint.py`, untimed) | this PR; can be dropped |

## Running it

Same stack as ANVIL2: `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` (the CE kernel's nvrtc build reads
`/usr/local/cuda/include`), torch `2.10.0+cu128`, driver >= 580, 8x H100 80GB.

```bash
pip install -r records/track_1_short/2026-10-02_ANVIL2Stack/code/requirements.txt
python data/cached_fineweb10B.py 9
hf download devenpzak/flash-attn3-12864 --revision 64c1e6d1f2780e7931839f41426ddcdb564a7cb9
cd records/track_1_short/2026-10-02_ANVIL2Stack/code
DATA_PATH=$(git rev-parse --show-toplevel) ./run.sh
```

Each run writes `logs/final_state.pt` and eight 21.3 GB rank shards (about 170 GB) after the
timed section. Keep that much disk free, or delete `logs/rank_*.pt` between runs.

## What is still needed

1. **Rule 4:** interleaved runs of current master ([#360](https://github.com/KellerJordan/modded-nanogpt/pull/360)) on the same machine, into `baseline/`
   with its own `statistics.md`.
2. **Loss margin:** the pool's mean sits 0.00075 under 3.28, against ANVIL2's 0.0027 and the
   0.001-0.002 the maintainers keep, and 3 of 18 runs land above 3.28, so discretionary reason 2
   is likely to come up. About 15 more extension steps (21.5 ms each on Modal, about 0.3 s) would
   restore the margin; that changes `code/` and needs a new pool.
3. **Layout:** master now runs the record from the `track_1_short/` package ([#373](https://github.com/KellerJordan/modded-nanogpt/pull/373)), so this
   delta still has to be ported into the package, without `checkpoint.py` and the bundled assets.

## Files

`this_pr/` holds the 18 validation run logs (each embeds the trainer and its kernel modules),
`pip freeze` (identical in every run), and `statistics.md`. `first_run/` holds the console log and
the checkpoint manifest of the first run.

## Credits

Builds on ANVIL2 ([#360](https://github.com/KellerJordan/modded-nanogpt/pull/360)) and the open PRs [#371](https://github.com/KellerJordan/modded-nanogpt/pull/371), [#375](https://github.com/KellerJordan/modded-nanogpt/pull/375) and [#376](https://github.com/KellerJordan/modded-nanogpt/pull/376).

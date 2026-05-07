# Plan: beat WR #80 (1.406 min) on the `xsa-gated-layers` branch

## Where you are now (verified)

- **Branch is correct.** `xsa-gated-layers` is checked out, clean, ahead of `master` by 5 commits. Remote: `origin = djdumpling/modded-nanogpt`. The change vs. master is contained in `train_gpt.py` (24 lines) plus run logs under `records/track_1_short/2026-04-29_XSAGatedLayers/`.
- **Branch idea (per `b55ca5f`, `78f1885`, `b0a3060`).** Per-(layer, head) learnable scalar `xsa_alphas` (zero-init, `tanh`-bounded), applied as a gated XSA head residual on every non-paired attention layer (`{1,3,4,7,8,10}`), trained with Adam (no WD); and `num_scheduled_iterations` reduced 1440 → 1410.
- **You already have evidence it works.** Your 30 committed logs say:

  | group                       | n  | mean val_loss | sd     | t (vs 3.28) | one-sided p(mean<3.28) | mean wall-time |
  |-----------------------------|----|---------------|--------|-------------|-------------------------|----------------|
  | master baseline @ 1410      | 10 | 3.2832        | 0.0009 | +10.80      | ≈1.0 (fails)            | 78.76 s        |
  | master baseline @ 1440      | 10 | 3.2794        | 0.0014 | −1.29       | 0.115 (fails p<0.01)    | 80.40 s        |
  | **xsa-gated PR @ 1410**     | 10 | **3.2786**    | 0.0010 | **−4.09**   | **0.0014 ✓**            | **79.85 s**    |

  → On this RunPod the PR meets the val-loss rule with p<0.01 and is ~0.55 s faster than the master 1440-step config that's the closest passing baseline.

- **Hardware/env on this RunPod.** 8× H100 80GB idle, driver 580.126, CUDA 13.0; Python 3.12.3; torch 2.10.0+cu128; numpy 2.1.2; kernels 0.14.0; triton 3.6.0; tiktoken 0.12.0; datasets 4.8.5; huggingface_hub 1.14.0. `HF_HOME=/workspace/.cache/huggingface`. `HF_HUB_ENABLE_HF_TRANSFER=1`. 100 G workspace, 80 G overlay free, 2 TB RAM.
- **Data.** `data/fineweb10B/` has `fineweb_train_000001..000009.bin` (9 shards × ~200 MB) plus the val shard. README says ≥9 shards is enough — you're set.

## Blocker before any training

`kernels.get_kernel('varunneal/flash-attention-3')` hits `https://huggingface.co/api/kernels/varunneal/flash-attention-3/tree/...` which returns **401 Unauthorized** in this fresh RunPod (no `HF_TOKEN`). Your existing 30 logs imply a token was present in the previous session. Two paths:

1. **Recommended:** export `HF_TOKEN=...` in this shell (and `~/.bashrc`) before any run. I have already pre-cached the 107-file snapshot under model repo type at `/workspace/.cache/huggingface/hub/models--varunneal--flash-attention-3/`, but the kernels lib does its own API call regardless.
2. Alternatively: bypass the kernels lib by switching `train_gpt.py` to load the kernel from a local path. Touches the WR-submission code — avoid unless option 1 is impossible.

## Rules check (README "Rules" section)

1. ✅ Train/val data pipelines untouched on this branch.
2. ✅ p<0.01 with 10 runs already met for `xsa-gated PR @ 1410`.
3. ✅ No `torch._inductor.config` / `torch.compile` flags added (only the `dynamo.config.recompile_limit = 64` already present on master).
4. ⏳ "Faster than the prior record on the same hardware" — need a `master`-branch timing run on this *same* RunPod to make a like-for-like comparison; PrimeIntellect re-timing is the official validation but RunPod side-by-side is what the rule requires for development.

## Execution plan

Tasks 1 → 6 are tracked in the task list. Concretely:

1. **Unblock kernels API auth.** `export HF_TOKEN=…`; smoke-test `python -c "from kernels import get_kernel; get_kernel('varunneal/flash-attention-3')"`.
2. **Sanity smoke run.** `cd /workspace/modded-nanogpt && ./run.sh`. First run will pay torch.compile (~6–8 min) — that's fine, expected.
3. **Re-time master baseline on this RunPod.** From `master`, run ≥10× `./run.sh` with `num_scheduled_iterations=1440` (the one that hits ≤3.28 on master). Capture logs to `logs/master_runpod/`.
4. **Re-validate the PR on this RunPod.** From `xsa-gated-layers`, run ≥10× `./run.sh`. Capture to `logs/xsa-gated-layers_runpod/`.
5. **Compute deltas + p-value** from (3) vs (4). Acceptance gates: PR mean val_loss ≤ 3.28 with p<0.01 and PR mean wall-time strictly faster than master mean wall-time on this hardware (one-sided Welch t at p<0.05 on the timing).
6. **Prepare PR.** Move the existing logs into `records/track_1_short/<date>_XSAGatedLayers/`, write an entry README (method blurb, the `arXiv:2603.09078` reference from the diff comment, the t-test numbers, hardware), and open the PR upstream.

## Critical files to touch later

- `train_gpt.py` (1058–1098, 1252–1301, 1568–1596, 1700–1740): the existing diff is already correct.
- `records/track_1_short/2026-04-29_XSAGatedLayers/`: rename + reorganize when preparing PR.
- `README.md` table row: append after current row #80 once PR is accepted upstream.

## Verification

- After step 1, `python -c "from kernels import get_kernel; k=get_kernel('varunneal/flash-attention-3'); print(k.flash_attn_interface.flash_attn_varlen_func)"` should print without error.
- After step 2 a `logs/xsa-gated-layers/<run-id>.txt` should end with `step:1450/1450 val_loss:<3.28 train_time:~80000ms`.
- After step 5, the recomputed t-statistic on this RunPod's runs must be ≤ −2.821 (df=9, one-sided p=0.01).

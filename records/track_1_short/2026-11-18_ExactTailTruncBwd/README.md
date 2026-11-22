# Track 1: #397 + exact tail/startup speedups + truncated attention backward, 30.245 s (n=18; −0.514 s vs #397 on the same node)

This record stacks two kinds of change on top of open PR #397 (head `11699e3`):

1. **Exact systems changes** (intended to be arithmetic-identical to #397): they remove host-side work from the start and the tail of the timed region. Alone they give −0.354 s on the same node (ablation arm below).
2. **One ML change**: a truncated attention backward window on the short-window layers during stages 1–2. It adds a further −0.16 s. Over all 18 runs its mean val loss (3.27833) equals the interleaved #397 runs' (3.27832, n=9); p ≈ 2e-5 for mean < 3.28.

The code is a single commit on top of #397. All 18 runs of this PR were on one 8×H100 node, interleaved with #397 (and, in rounds 1–6, with the exact-only ablation).

## Changes

### Exact (systems)
- **Async canonical-mask upload** (`canonical_mask.py`, `train_gpt.py`). Every rank builds its own copy of the 316 MB final-validation canonical mask (deterministic, so the bytes are identical) instead of rank 0 building it and broadcasting. Once the builder child exits, the H2D copy into `model.canon_mask` is queued on a side stream in 10 chunks of ~32 MB, one per train step, ordered by an event. The tail no longer has a blocking H2D, a `dist.broadcast`, or a device sync. After the clock stops, `release()` checks across ranks that the masks agree (row and column sums, all_reduce MIN == MAX) and asserts otherwise.
- **Val batch prefetch** (`data.py`, `train_gpt.py`). After train step `total_steps // 2` a background thread reads only the first `val_tokens + 1` tokens of the val shard and builds the 5 final-val batches (inputs, targets, cu_seqlens, n-gram ids) into pinned buffers allocated before the clock. The final-val branch then only issues non-blocking H2Ds. On CPU the batches were checked equal (dtype, shape, values) to #397's generator on all 8 ranks.
- **Lazy first shard + unpinned shards** (`data.py`, `train_gpt.py`). The first training shard is read in the background, starting at clock start. Shards are no longer copied into page-locked memory (this removes a ~200 MB `cudaHostAlloc`). The BOS index is built with `np.flatnonzero`, giving the same int64 positions as before. On CPU the train-stream hash (ranks 0 and 6) and the val-stream hash equal #397's.
- **`posix_spawn` canonical-mask builder** (`canonical_mask_build.py`, new and torch-free). The builder child is started with `os.posix_spawn` (vfork+exec, ~0.5 ms) instead of `os.fork()`. The fork copied the page tables of a GB-scale process on every rank, which took 57–68 ms at clock start. The child writes into a `/dev/shm` file mapped `MAP_SHARED`. On CPU its mask is byte-identical to #397's in-process build (70,889,701 bits set).
- **Tied-gradient fold** (`perf/kernels/transpose.py`, `optim/anvil.py`, `perf/cuda_graphs/step_graphs.py`). On tied-embedding Adam steps, `foreach_add(lm_acc += lm_g, emb_acc += emb_g)` followed by `transpose_add(emb_acc, lm_acc)` (3 kernels; 6 reads and 3 writes of a 77 MB bf16 tensor) becomes one Triton kernel, `tied_grad_fold`: `lm_acc = bf16(bf16(lm_acc + lm_g) + bf16(emb_acc + emb_g).T)`. It does the same three roundings in the same order, with 4 reads and 1 write.

### ML: truncated attention backward on short-window layers (stages 1–2)
`model/attention.py` adds `FlashAttnVarlenTruncBwd`. Its forward is exactly `flash_attn_varlen_func`'s forward with window `(bm_size, 0)`. Its backward calls the pinned build's stock `_flash_attn_backward` with a smaller `window_size_left`, so P is rebuilt from the forward LSE and the distant key tiles' terms are dropped without renormalizing. This is the estimator of #371, but it needs no FA3 rebuild and uses one window per layer instead of per-head windows. Long-window layers (3, 10) are never truncated. The last 79 steps use the exact backward.

| stage | steps | fwd short window | bwd short window |
|---|---|---|---|
| 0 | 0–279 | 128 | 128 (exact) |
| 1 | 280–594 | 384 | **256** |
| 2 | 595–973 | 640 | **384** |
| taper | 974–1032 | 640 | 640 (exact) |
| extension | 1033–1052 | 768 | 768 (exact) |

The backward window is a field of the step-graph key. It changes only at stage boundaries, which already start new graphs, so there are no new keys, recompiles or warmup steps.

## Result

Hardware: one preemptible 8×H100 80GB HBM3 node on GKE (europe-west2), driver 580.178.04, torch 2.10.0+cu128, Python 3.12.3. Run order: rounds 1–6 = [this PR, exact-only ablation, #397, this PR] (08:45–09:58 UTC, 2026-11-18); rounds 7–9 = [this PR, #397, this PR] (10:52–11:20 UTC). Seeds were random (`TRAIN_SEED` unset), as in a normal record run.

| arm | n | train_time (s), mean ± sd | val_loss, mean ± sd | one-sided t-test, mean < 3.28 |
|---|---|---|---|---|
| **this PR** | 18 | **30.2454 ± 0.0146** | **3.27833 ± 0.00127** | t = 5.58, **p = 1.7e-5** |
| #397, interleaved (r1–r9) | 9 | 30.7592 ± 0.0085 | 3.27832 ± 0.00170 | t = 2.97, p = 0.009 |
| exact systems only (ablation, r1–r6) | 6 | 30.4085 ± 0.0113 | 3.27735 ± 0.00203 | t = 3.19, p = 0.012 |
| #397, all runs on this node | 16 | 30.7598 ± 0.0129 | 3.27810 ± 0.00184 | t = 4.14, p = 0.0004 |

- **Time vs #397 on the same node:** −0.514 s (Welch t = −115). Per round (mean of the round's two runs of this PR minus that round's #397 run): −0.522, −0.508, −0.520, −0.504, −0.531, −0.510, −0.534, −0.495, −0.499, so **−0.514 ± 0.014 s** (paired t = −113, n = 9).
- **Split:** the exact systems changes give −0.35 s (ablation, rounds 1–6) and the truncated backward a further −0.16 s.
- **Loss:** p-values are from the one-sided Student t-test against 3.28 (`t = (3.28 − mean) / (sd/√n)`, `p = P(T_{n−1} > t)`), equivalent to `scipy.stats.ttest_1samp(..., alternative='less')`. Over rounds 1–6 the truncated backward looked about +1.3 mnats worse than interleaved #397; with rounds 7–9 included the two means are equal (3.27833 vs 3.27832). Two of the 18 runs ended at 3.2809.
- "#397, all runs" also includes 4 runs before the interleaved rounds (3 stock, 1 path test), one run with tail-profiling hooks (`base_prof`, timestamps only), and two same-seed runs (below).

### Every run

Times are UTC on 2026-11-18. Each log embeds the full source of the run. The `this_pr/` logs' embedded source matches this commit file for file. The `baseline_397*/` logs match `11699e3`, except `base_prof`, which adds profiling hooks.

| run | arm | recorded (UTC) | train_time (s) | val_loss | log |
|---|---|---|---|---|---|
| ems_trunc_r1 | this PR | 08:45 | 30.248 | 3.2809 | `this_pr/3b06ea77-dedc-4b3f-a1cd-3ce0be2fc37d.txt` |
| ems_trunc_b_r1 | this PR | 08:55 | 30.234 | 3.2771 | `this_pr/4ba3f21f-09d3-48b8-ab17-c5d746d963f6.txt` |
| ems_trunc_r2 | this PR | 08:58 | 30.245 | 3.2789 | `this_pr/6baf357b-4bfd-4591-b78b-11cad0edcef8.txt` |
| ems_trunc_b_r2 | this PR | 09:07 | 30.252 | 3.2790 | `this_pr/0c209076-e706-4357-86de-f93b57db3179.txt` |
| ems_trunc_r3 | this PR | 09:10 | 30.252 | 3.2787 | `this_pr/e3252a17-7eef-48a2-8db6-6aa8e34c941b.txt` |
| ems_trunc_b_r3 | this PR | 09:20 | 30.250 | 3.2790 | `this_pr/75a1d8f8-3224-4d05-96cb-fdddafc37341.txt` |
| ems_trunc_r4 | this PR | 09:23 | 30.257 | 3.2787 | `this_pr/649cf140-762a-4bfc-87aa-467b7238cd68.txt` |
| ems_trunc_b_r4 | this PR | 09:32 | 30.248 | 3.2772 | `this_pr/5ac708d2-b98f-4d4c-96dc-808f69aa7101.txt` |
| ems_trunc_r5 | this PR | 09:35 | 30.240 | 3.2809 | `this_pr/7caa670b-771b-4855-8b18-2731a0b987b5.txt` |
| ems_trunc_b_r5 | this PR | 09:45 | 30.246 | 3.2780 | `this_pr/9661ec2c-d1de-4d2f-b296-6b3e904e0d5c.txt` |
| ems_trunc_r6 | this PR | 09:48 | 30.240 | 3.2769 | `this_pr/095128c9-6b75-4c1d-820d-de734c813d24.txt` |
| ems_trunc_b_r6 | this PR | 09:57 | 30.250 | 3.2781 | `this_pr/75a3afa5-c7fd-4d37-8914-e253b50ad6ef.txt` |
| ems_trunc_r7 | this PR | 10:55 | 30.218 | 3.2771 | `this_pr/83a5022b-4da8-4701-91e7-fabfdcfb4da7.txt` |
| ems_trunc_b_r7 | this PR | 11:01 | 30.214 | 3.2772 | `this_pr/4356679e-7949-410b-8a52-d508bd0ec48a.txt` |
| ems_trunc_r8 | this PR | 11:04 | 30.250 | 3.2799 | `this_pr/54bc1748-2c29-460e-b58a-2e231e2de060.txt` |
| ems_trunc_b_r8 | this PR | 11:10 | 30.259 | 3.2777 | `this_pr/c6837617-92a1-4009-9b0a-6b3718f98147.txt` |
| ems_trunc_r9 | this PR | 11:13 | 30.279 | 3.2772 | `this_pr/5b83d962-86a2-4420-ba4e-10d75f2efb4a.txt` |
| ems_trunc_b_r9 | this PR | 11:19 | 30.235 | 3.2775 | `this_pr/66330783-dffe-4d0d-adf6-1aed5fcbbaec.txt` |
| pr397_base_1 | #397 | 07:26 | 30.764 | 3.2776 | `baseline_397_extra/f665c520-ed3e-41b9-9120-df2dcd1c1990.txt` |
| pr397_base_2 | #397 | 07:29 | 30.783 | 3.2801 | `baseline_397_extra/21526eef-f904-4f98-888d-24fb6b25a98b.txt` |
| pr397_base_3 | #397 | 07:33 | 30.785 | 3.2799 | `baseline_397_extra/10de79b2-297c-418f-b7df-cadf308bd2eb.txt` |
| pr397_pathtest | #397 | 07:36 | 30.738 | 3.2770 | `baseline_397_extra/654a5bd9-c3e4-4a72-bf26-12abdfdddc74.txt` |
| base_r1 | #397 | 08:52 | 30.763 | 3.2777 | `baseline_397/7879fa8c-d9f5-4ecb-8506-8f56c075ecaa.txt` |
| base_r2 | #397 | 09:04 | 30.757 | 3.2780 | `baseline_397/bd3889fb-8b65-4afb-a900-43b125fef423.txt` |
| base_r3 | #397 | 09:17 | 30.771 | 3.2762 | `baseline_397/4226c6aa-e141-4ba7-86bc-82588192bfd4.txt` |
| base_r4 | #397 | 09:29 | 30.757 | 3.2764 | `baseline_397/4726b2ad-222d-4c4f-8a2a-ba85bd3a2a01.txt` |
| base_r5 | #397 | 09:42 | 30.774 | 3.2769 | `baseline_397/dba9da14-4383-4676-a776-82d080347236.txt` |
| base_r6 | #397 | 09:54 | 30.755 | 3.2789 | `baseline_397/cf0561df-45db-40c0-9200-c3e20026fce0.txt` |
| base_r7 | #397 | 10:58 | 30.750 | 3.2809 | `baseline_397/2f2149e8-54d9-470c-bf33-0f9a1611c538.txt` |
| base_r8 | #397 | 11:07 | 30.750 | 3.2800 | `baseline_397/32c0d069-98fd-4357-a1a7-df67dfb897b2.txt` |
| base_r9 | #397 | 11:16 | 30.756 | 3.2799 | `baseline_397/6d866bcb-60e3-4155-8841-ec16a862150a.txt` |
| ems_r1 | exact systems only | 08:48 | 30.387 | 3.2762 | `ablation_exact_only/7357bc9f-14aa-4d4b-8237-3beaf18b538f.txt` |
| ems_r2 | exact systems only | 09:01 | 30.420 | 3.2770 | `ablation_exact_only/349af75a-2f91-403a-bfa6-a0d39d6b6318.txt` |
| ems_r3 | exact systems only | 09:13 | 30.409 | 3.2775 | `ablation_exact_only/a3c473ba-5236-4564-bcf2-f1f4793a16ef.txt` |
| ems_r4 | exact systems only | 09:26 | 30.410 | 3.2787 | `ablation_exact_only/a8e0ff99-d94c-4512-bce4-c54f3f2b5f5a.txt` |
| ems_r5 | exact systems only | 09:38 | 30.410 | 3.2744 | `ablation_exact_only/d4dbf5ef-b4d4-4bfb-9b73-9d41c557aeaf.txt` |
| ems_r6 | exact systems only | 09:51 | 30.415 | 3.2803 | `ablation_exact_only/e5cbb064-a94e-4d11-b74c-470398d1a8ab.txt` |
| base_prof | #397 (profiled) | 07:43 | 30.750 | 3.2796 | `baseline_397_extra/3dcef1b9-c996-44c7-b5c6-c77cb79dec19.txt` |
| ems_prof | exact systems only (profiled) | 08:23 | 30.398 | 3.2758 | `ablation_exact_only/94ef4e68-dd8c-4485-8cfa-1de68bb5125d.txt` |
| ems_prof2 | exact systems only (profiled) | 08:29 | 30.418 | 3.2760 | `ablation_exact_only/0185d941-19cc-4eae-94ea-8065ed9986c2.txt` |

The ablation logs (`ablation_exact_only/`) are this commit without the truncated-backward files (`config.py`, `model/attention.py`, `model/gpt.py`, `capture_plan.py`, `step_graphs.py`, `training.py`). `ems_prof*` also carry tail-profiling hooks.

Other variants were run on the same node earlier that morning: trunc-only, systems-tail-only, systems-tail + trunc, a sampled-head taper, a rail cooldown, no-MLP-in-layer-4 and #403's stratified sampling. They are not part of this PR, and their logs are not included here.

## Reproduce

```bash
git checkout <this commit>      # = #397 head 11699e3 + this change
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

There is no toggle. To get the exact-only arm, set `bwd_window_short=None` for stages 1 and 2 in `track_1_short/config.py`. To get #397, check out `11699e3`.

## Caveats

- **Hardware.** The runs used a preemptible GKE node in europe-west2, not the PrimeIntellect/RunPod nodes of earlier records. Absolute times may differ elsewhere; the same-node deltas are the claim.
- **Exactness.** The exact systems changes were checked on CPU (identical val batches, train and val stream hashes, mask bytes) and by reasoning about rounding order (`tied_grad_fold`). On GPU, a same-seed check (below) puts them inside #397's own run-to-run band (±0.0001), but #397 is not bitwise deterministic, so bitwise identity cannot be shown that way.
- **Nondeterminism.** Seeds are random and the stack is not bitwise deterministic run to run, so the loss spread mixes seed variance with kernel nondeterminism. The truncated backward is a biased gradient estimator; its loss effect is measured over these 18 runs only.
- **n.** 18 runs of this PR and 9 interleaved #397 runs. Every run of this exact commit on the node is listed above; none was dropped. (One extra #397 same-seed run was killed by the operator before training started and has no result. Runs of other code, e.g. profiling builds, are not counted.)

## Same-seed check of the exact changes (TRAIN_SEED=7, same node, 10:29–10:39 UTC)

| run | arm | train_time | final val_loss |
|---|---|---|---|
| det_base_s7b | #397 | 30.751 s | 3.2752 |
| det_base_s7c | #397 | 30.752 s | 3.2753 |
| det_ems_s7 | exact systems only | 30.390 s | 3.2754 |
| det_ems_s7b | exact systems only | 30.413 s | 3.2753 |

#397 itself is not bitwise deterministic with a fixed seed: two runs differ by 0.0001. The exact-only arm lands in the same ±0.0001 band, so it is consistent with an arithmetic-identical change. This does not prove bitwise identity, because the base's own nondeterminism hides differences that small.


## Comment cleanup after the runs

After all runs, the comments and docstrings in the code were shortened; no code changed. Every changed `.py` file has the same AST (docstrings stripped) as the commit that produced the logs (`1e508b7`), so the code embedded in the logs is that commit's, with longer comments.

## Where the time goes now (torch.profiler, CUPTI started before graph capture as in #396)

The GPU is 96–98% busy in every stage. Gaps between graph replays total ~0.1 ms/step, and the host is never late. The 3–5 ms/step outside the forward and backward graphs is GPU work:
- optimizer and fp8-refresh graphs: ~1.7 ms;
- eager n-gram and optimizer kernels: ~1.0–1.6 ms;
- NCCL that nothing hides: 0.2–0.8 ms.

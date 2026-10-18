# #379 + #375 on record #92: 34.856 s (0.581 min)

This record combines two open PRs:

- **[PR #379](https://github.com/KellerJordan/modded-nanogpt/pull/379), CPLM copy-sink pointer LM**, by @NathanGodey.
- **[PR #375](https://github.com/KellerJordan/modded-nanogpt/pull/375), normalized token ids in the n-gram hashes**, by @daniel-monroe.

The modeling ideas are theirs. This PR merges the two (they merge without conflicts), re-tunes the step count for the combination, and certifies the result. The README encourages incorporating open PRs, so we hope this helps validate and merge both.

**A separate timing fix, not included here.** While timing these runs we looked at how the clock starts across ranks.
- **The problem:** rank 0 starts the timer as soon as *it* has finished its untimed setup. Any rank still busy with its own setup gets charged to the clock at the first collective: up to 0.83 s in one master run on this node, though usually 0.
- **The fix:** a three-line `dist.barrier()` before `t0`, proposed in its own PR ([#390](https://github.com/KellerJordan/modded-nanogpt/pull/390)).
- **Not included here:** this record's numbers don't depend on it. The PR's code runs 34.856 s with or without it; see the notes on the logs below.

## Result

8×H100 80GB HBM3 (SXM), all runs on the same node, the three arms interleaved over ~90 minutes:

| | runs | train time (mean ± sd) | final val loss (mean ± sd) |
|---|---|---|---|
| **this PR (#379 + #375, 955 scheduled steps)** | **18** | **34.856 ± 0.023 s** | **3.27773 ± 0.00114** |
| master (record #92's code, 1122), same node | 7 | 40.162 ± 0.030 s | 3.27719 ± 0.00134 |
| #379 alone (978), same node | 7 | 35.647 ± 0.040 s | 3.27729 ± 0.00107 |

- **Loss:** one-sided t-test of the mean val loss < 3.28, using the README's method: n = 18, **p = 8.0e-8**. All 18 runs are below 3.28 (range 3.2751–3.2796).
- **Speed (same node):** **−5.305 s (−13.2 %) vs record #92** (Welch p = 2e-20) and −0.790 s (−2.2 %) vs #379 alone (p = 9.5e-11).
- **Loss buffer:** the mean is ≈2.3 millinats below 3.28, about the buffer the README asks records to keep.
- **The node is close to the official timing.** Record #92 measured 40.162 s here, against 39.914 s published, so absolute times should be close to official ones. The speedup above is the same-node comparison (rule 4).
- **Every run counts:** none excluded or rerun. Seeds (`TRAIN_SEED`) were fixed in advance: 101–118 for this PR, 101–107 for each baseline. 14 runs of this PR were planned, then extended to 18 (the n of record #92) at an interim point of 4 runs, before the remaining results were known.
- `python summarize.py` recomputes all of the above from the logs in this folder.

## Change

Both PRs' code is unchanged. On top of the two merges, this PR only sets the default schedule to **955 scheduled steps (1027 total)**:
- `NUM_SCHEDULED_ITERATIONS`: #379 had 978 and #375 had 1107.
- The 72 growth and extension steps are unchanged.
- 955 keeps roughly the usual loss buffer for the combination.

- **#379 (CPLM):** the next-token distribution is a mixture of the LM softmax and a QK-normed pointer over the document's earlier tokens. Its README covers the method, kernels, cost and the validity of the probability model.
- **#375 (n-gram token normalization):**
  - The hashed n-gram table looks up bigram and trigram rows by *normalized* token ids: NFKC, accents stripped, lowercased, whitespace collapsed. For example, " The", "THE" and " the" share rows.
  - Only the n-gram hashes read the mapped ids. Embeddings, lm_head, targets and the data stream keep raw GPT-2 ids, so the token streams are unchanged (rule 1).
  - The mapping is a fixed function of the token, so the model stays causal and valid.
  - It adds the dependency `tokenizers==0.23.2`.
- **Integration:** git merged the two with no conflicts. We checked that every line each PR adds or removes is present in the merge, that the new parameter names are unique, and that the edits don't overlap.

## Also tested: adding #385 (LayerRoPE, @shikhar-srivastava)

On the same node, the three-PR stack at 950 steps reached 3.2753–3.2758 (n=3). #379 + #375 alone at 950 reached 3.2779 (n=1). So #385 looks worth about 2 millinats on top of this record.

We left it out of this certification. Its README reports a few collapsed runs, and we wanted an 18-run certification without that risk. Combining it remains a natural next step.

## Notes on the logs

- **`certification/`** (18 runs): these runs had three extra lines that are **not in this PR**: a `dist.barrier()` before the clock starts. It doesn't touch training or the loss.
- **`no_barrier/`** (3 runs, same seeds as the first three certification runs): the PR code without the barrier. The mean is **34.856 s**, identical to the certification mean (the paired difference is 0.011 s). The barrier is proposed separately as a timing-only change.
- **`baseline_master/`** and **`baseline_379/`** (7 runs each): the two baselines above, without the barrier.
- **Extra code in the logs:** each log embeds the code that ran, which includes a few measurement hooks (`track_1_short/diag.py`) used to time the runs. Inside the timed region they only take wall-clock stamps; files are written outside it. The step count came from the environment (`NUM_SCHEDULED_ITERATIONS=955`).

  This PR's code is the logged code without those hooks (and without the barrier), with 955 as the default.

## Environment

- **Hardware:** 8× NVIDIA H100 80GB HBM3 (SXM5, NVLink), driver 595.91.07. It's a rented single node (RunPod Secure Cloud, US). Master ran 40.162 s here vs 39.914 s published, about 0.6 % slower than the official timing.
- **Software:** PyTorch 2.10.0+cu128, Triton 3.6.0, `kernels` 0.16.1 (FlashAttention-3 `devenpzak/flash-attn3-12864`), Python 3.12.
- **Command:** `torchrun --standalone --nproc_per_node=8 train_gpt.py` (defaults, plus `TRAIN_SEED` per run).

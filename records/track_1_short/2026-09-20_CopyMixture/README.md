# Document-local copy mixture at the final validation (−15 extension steps)

This one leaves training untouched. At the final validation, the model's prediction is mixed with a (very) old trick: if the last few tokens have already appeared earlier in the same document, put some probability on whatever followed them last time. That is worth 0.0031 val loss on identical weights, which I cash in as 15 fewer extension steps (40 → 25, so 1290 → 1275 total).

| 1×H100 | steps | n | val loss | p (mean ≤ 3.28) | train time |
|---|---|---|---|---|---|
| model alone (same runs as the row below) | 1275 | 6 | 3.28028 ± 0.00054 | 0.87 | |
| **with the copy mixture** | **1275** | **6** | **3.27715 ± 0.00053** | **2.3e-5** | 503.8 ± 2.5 s |
| record #91 as is, on the same hardware | 1290 | 1 | 3.2764 | | 508.8 s |
| **with the copy mixture, 8×H100** (Modal node, see caveats) | **1275** | **1** | **3.27746** (model alone 3.28060) | | 179.6 s |

In other words, the step cut fails the bar on its own and clears it comfortably with the mixture. Every run prints both losses from the same forward passes (the way #91 printed `val_loss_unmasked`), so the gain carries no seed noise: +0.00312 ± 0.00004 across all seven runs. The p-values are the README's one-sided t-test.

## What it does

For each position, find the longest suffix of the context (lengths 1, 2, 3, 4, 6, 8, 12, 16, 24, 32) that occurs earlier in the same document, take its most recent occurrence, and call the token that followed it `c`. The validation distribution becomes

    P'(x) = (1 − λ_b) · P_model(x) + λ_b · [x = c]

where `b` buckets the match by its length and by how far back it sits (≤768, ≤2560 or >2560 tokens, i.e., the final short and long attention windows). The properties I care about:

- **It is still a valid probability model.** `c` and `b` depend only on inputs up to the current position, never on the target, and P' sums to one.
- **Nothing is learned from validation.** The 31 weights λ_b are fitted by maximum likelihood on training tokens (the first 2.1M tokens of the last train shard, which a run with the standard 9-shard download never reaches, read by the unmodified loader), using eval-mode forward passes only.
- **The fit is on the clock.** It takes 1.38 s on one H100 (≈2.4 extension steps' worth) and sits after the canonical mask is collected, so it sees the distribution validation will score. Matching itself adds ~0.2 s per validation pass, off the clock.
- **The compiled graph does not change.** Everything runs eagerly on the per-token loss the eval forward already returns. The matcher is exact (it ranks token pairs, à la suffix-array doubling, so there is no hashing) and never crosses a document boundary.
- `COPY_MIX=0` restores the previous behaviour.

## Where the gain comes from

This turned out to be not a story about copying so much as one about context length. 79% of the gain comes from matches more than 2560 tokens back (only 1.6% of validation tokens), where no attention layer can see the source: for length-32 matches out there the copy rule is right 95% of the time while the model's loss is 3.33 nats, against 0.42 with the mixture (!). Within 768 tokens the model already copies well, and the mixture adds ~0.0004. The gain is also additive with #91's canonical masking: +0.00307 on record #89's weights, +0.00312 here.

## What I tried that didn't pay

- **Re-tuning the eval windows.** Could a longer window get this for free? I tried: on one set of weights (record #89's), long windows of 16/26/32 blocks were all worse than 20 (by 0.0004–0.0006), and widening the short window was far worse (+0.006 at 8 blocks, +0.022 at 10). The record's 6/20 looks like a genuine optimum.
- **A fancier mixture.** I rescored designs offline on one run's per-token losses (ranked on train tokens only). Extra distance edges bought ≤0.000005, a "the previous occurrence agrees" feature +0.0002 for 3× the buckets, and fitting the weights on validation itself (an oracle) only +0.00005. The distance split is the one thing that earned its keep (length-only buckets give 0.0024).

## Rules, as I read them

The README defines the target as a probability model of the validation tokens and allows evaluation at any sequence length "so long as we still have a valid probability model of language". I think this is that, plus a lookup. It follows #91's template: eval-only, with the whole cost of the extra machinery on the clock. There are no backward passes and no adaptation to validation data (so it is not the test-time training of PR 205), and unlike PR 367 it never consults the train shards at validation. The only extra context is the document already being scored.

## Caveats and lingering questions

- Timing, measured on both shapes, both on Modal. On 1×H100 an extension step takes 566 ms and the fit 1.38 s, so the cut saves 15 × 566 − 1384 ≈ 7.1 s (1.4%) against a 1290-step run. On 8×H100 the fit takes 0.44 s across the ranks and an extension step 327 ms, so within that run the cut saves 15 × 327 − 440 ≈ 4.5 s (2.4%). That node is a slow one, 140 ms per step against the ~52 ms of the leaderboard's (PCIe-class, I assume), so the absolute times do not transfer; the per-step and fit costs do. With the leaderboard node's ~76 ms extension steps and a 0.3–0.44 s fit I'd expect 0.7–0.85 s (1.0–1.3%), but that is a projection, and the maintainers' own timing is the one that counts. The 8×H100 run also confirms the multi-GPU path end to end (the fit's all-reduce across ranks, the averaged mixture loss, upstream's sharded optimizer) and lands where the 1×H100 runs did: 3.27746 with the mixture, 3.28060 without.
- `copy-mix-1275/` (six 1×H100 runs) and `diagnostic-1290/` are console captures from my launcher, so they lack the code header the repo's own logs carry; `copy-mix-1275-8xH100/` is the 8×H100 run's own log, header included. `copy-mix-1275/` and the 8×H100 run were produced by exactly this `train_gpt.py` with `NUM_EXTENSION_ITERATIONS=25`; `diagnostic-1290/` by an earlier version that also printed the per-bucket table.
- Is the head of an untouched train shard fair game for the fit? I think so (it is training data, read by the unmodified pipeline), but if you'd rather it use batches the run has already consumed, that is a small change.
- The mixture's mean sits 0.0029 under 3.28, so there may be room for a few more steps. I stayed next to the 30–45 extension-step range that #91 measured.
- The gain shrinks as the model sees further: on record #89's weights it was 0.0051 at the step-1250 windows and 0.0031 at the final ones. Anything that lengthens eval context will eat into it.

Thanks to @jvarho, whose canonical masking record is the template for both the evidence format and the on-the-clock accounting here.

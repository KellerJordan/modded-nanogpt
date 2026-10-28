# Track 3: SOAP-Shampoo (#388) + spectral annealing and head eigen-Adam (#384) + 1.5x Adam LR for head and scalar params -- 2450 steps pooled (n=39)

## Result

The trainer `train_gpt_soap_shampoo_adam_lr_2450.py` reaches 3.28 validation loss in **2,450
steps**. We publish all 50 finished runs of it and count the earliest run of each of their 39
seeds. 2,450 is the earliest step at which the rule holds both over all 39 seeds and over the 33
left once the 6 runs that chose this trainer are taken out. Over all 39 seeds the rule already
holds at 2,445; we do not claim 2,445, because without the 6 choosing runs it fails there.

This is a pooled result: most of its runs were made for other experiments, and their number was
not fixed in advance (see "Every run"). The one batch that was planned before any of it ran, 8
seeds, passes at **2,455** on its own.

Mean `val_ema_loss` and the rule's statistic `(3.28 - mean) * sqrt(n)`, which must be at least
0.004:

| Seeds | n | 2,440 | 2,445 | 2,450 | 2,455 | First passing step |
| --- | --- | --- | --- | --- | --- | --- |
| All seeds | 39 | 3.27965, 0.00218 | 3.27927, **0.00454** | 3.27888, **0.00699** | not logged by 13 seeds | 2,445 |
| Without the 6 choosing runs | 33 | 3.27975, 0.00142 | 3.27937, 0.00359 | 3.27898, **0.00585** | not logged by 13 seeds | **2,450** |
| The 8-seed batch fixed before it ran | 8 | 3.27955, 0.00128 | 3.27918, 0.00233 | 3.27878, 0.00345 | 3.27843, **0.00444** | 2,455 |

A step counts only when every seed in the pool logged it, so 2,455 is not used for the first two
pools: 13 of their seeds stopped at 2,450.

Compared with published results:

| Result | Steps | Our step count is | Pairwise statistic (must be at least 0.004) |
| --- | --- | --- | --- |
| #46, accepted record (n=8) | 2,690 | 240 fewer | 0.0264 |
| #384, open (n=14) | 2,540 | 90 fewer | 0.0127 |
| #388, open (n=16) | 2,615 | 165 fewer | 0.0253 |

The pairwise statistic is the README's `(final_loss_diff + exp_stepbased_loss_diff) /
(1/n1 + 1/n2)**0.5`, with our 39 seeds at 2,450 and each result's published mean at its own step.
Over the 33 seeds without the choosing runs it is 0.0257, 0.0121 and 0.0243.

`python3 claim_stats.py` recomputes the first table and the 39-seed pairwise statistics from the
logfiles and the three published means written into it, and exits with an error if any part of
the claim does not hold.

## What changed from #384

We started from our copy of #384's trainer (`train_gpt_spectral_annealing_2540.py` at commit
`eb2aceeb` of jn2clark/modded-nanogpt). Our copy differs from it only in ways that do not touch
training: a guarded import of our evaluation harness's hook (see "Notes"), the three settings #384
read from the environment written in at its defaults (`LOG_DIR`, `OUTPUT_KFAC_POWER`,
`OUTPUT_KFAC_END`), and #384's development checkpoint and resume code removed (it ran only when
`DEV_*` variables were set). This trainer then makes two changes to how it trains. Everything not
listed is #384's.

1. **#388's SOAP-Shampoo step in place of the Newton-Schulz polar step** on the hidden matrices.
   The update is preconditioned on both sides by the inverse fourth roots of exponential averages
   of the residual's two Gram matrices, computed with #388's coupled Newton-Schulz polynomial (8
   iterations, 10 from step 1,450), with #388's residual weight 0.375, factor average 0.8 and
   damping 0.01, and scaled to a norm of sqrt(rows). #384's momentum, SOAP preconditioning,
   spectral annealing and Tail-EMA readout are kept. The hidden-matrix learning rate is #388's
   0.0325 (was 0.0375), with its decay constant scaled by the same ratio. The inverse roots of
   factors with a side over 1,024 use TF32 matrix products; that is our choice, for speed, and #388
   computes them in float32. Validation runs every 5 steps from step 2,400 to 2,800 (#384: from
   2,500), then less often to 2,900.
2. **Head and scalar Adam learning rates ×1.5.** The output head's learning rate (its AdamW, then
   its eigen-Adam from step 1,000) and the three scalar Adam groups (gains, other parameters with
   fewer than 2 dimensions, the attention projection's bias) are multiplied by 1.5, their decay
   constants alike.

On our setup:

- #384 meets the rule at 2,550 on its seeds 0 to 7; its own logs of those seeds also first pass
  at 2,550.
- With change 1 alone, the 8-seed batch that confirmed it first passes at 2,500. That trainer has
  16 runs: the 14 that ran past 2,490 first pass at 2,495, and 2 stopped at 2,490.
- With both changes, 2,450, as above.

These step counts come from different pools of seeds, so they do not measure each change by
itself. At step 2,450 the mean logged loss is 3.28182 over the 16 runs of change 1 alone and
3.27888 over this trainer's 39 counted seeds; those are different pools of seeds too.

## Every run

We publish all 50 finished runs of this trainer, one logfile each, and count the earliest run of
each seed. Every logfile embeds the trainer byte for byte (`claim_stats.py` checks this). They
ran between 2026-10-24 and 2026-10-27, one H100 80GB per run, as part of one automated search:

| Runs | Why they ran | Stop step |
| --- | --- | --- |
| 6 | The runs on whose results this trainer was chosen (the "choosing runs") | 2,530 |
| 8 | A batch planned before any of it ran; all 8 are here | 2,470 |
| 36 | Controls for other experiments in the same search | 16 at 2,450, 20 at 2,530 |

- **The controls were not a batch fixed in advance.** The experiments they belonged to decided
  how many runs to make, and when to stop, from their own comparisons, and those comparisons use
  this trainer's losses. Some also chose which seeds got controls after seeing results. One study
  of same-seed pairs ran its first controls on seeds 0 and 1. It then put controls on six of the
  eight seeds of the other script's confirmation runs, starting with the seed where that script
  had scored best. It then added fresh seeds drawn in advance, and stopped after 12 of a planned 16
  pairs. So the 33-seed pool, too, contains controls whose seeds were chosen this way. This is why
  we call 2,450 a pooled result and show the planned batch's 2,455 beside it.
- **Seeds.** 11 runs repeat a seed an earlier run already used (seed 101601244 ran 5 times). A
  repeated seed counts once, by its earliest run. Runs with the same seed do not give the same
  losses, because GPU arithmetic is not deterministic here; `summary.tsv` marks which run of each
  seed is counted. Counting all 50 runs instead also passes at 2,450 (statistic 0.00834).
- **Stop steps.** The schedules are those of #384's 2,900-step run whatever the stop step: the
  learning rates reach zero at step 2,840 and the loop runs to 2,900. `STOP_STEP` only ends the
  loop early and adds a final validation, so runs with different stop steps are the same training
  up to the earliest stop, and their losses at 2,450 are comparable.
- **Cancelled jobs.** The search's notes record 8 jobs of this trainer cancelled before training
  finished. No partial logfile of these 8 jobs was kept, so they are not counted. None was
  cancelled because of its own loss. The reasons:
  - 5: the job service became unreachable, and the harness then cancels its job. 4 of them were
    run again.
  - 1: its paired comparison job had been lost, so the pair was run again together.
  - 1: the paired study it belonged to had stopped.
  - 1: as it was submitted, because its experiment had decided from its first two paired seeds
    not to run a third.

## Reproduce

From the root of the repository, after the quickstart's data step (`python
data/cached_fineweb10B.py 20` downloads enough tokens):

```bash
STOP_STEP=2450 torchrun --standalone --nproc_per_node=1 \
    records/track_3_optimization/results/20261028_soap_shampoo_adam_lr_2450/train_gpt_soap_shampoo_adam_lr_2450.py --seed <seed>
```

The logs come from one H100 80GB per run with PyTorch 2.11.0 (CUDA 13.0); each logfile's header
records its versions, device and seed. All hyperparameters are written into the trainer; only
`--seed` is a command-line argument, and `STOP_STEP` is read from the environment, as in #384.
Without `STOP_STEP` the run continues to step 2,900. Dataset, batch size, architecture and
validation are those of `records/track_3_optimization/train_gpt_simple.py`; there is one
forward-backward pass per step and no third-party optimizer import.

## Credits

| What | From |
| --- | --- |
| The stack this builds on: spectral annealing, head eigen-Adam, anchor-extrapolated gradient, fp32 embedding master, Bi-Maxwell momentum | [#384](https://github.com/KellerJordan/modded-nanogpt/pull/384), [#377](https://github.com/KellerJordan/modded-nanogpt/pull/377), [#341](https://github.com/KellerJordan/modded-nanogpt/pull/341) (@jn2clark) and [#339](https://github.com/KellerJordan/modded-nanogpt/pull/339) (@orange4664) |
| The SOAP-Shampoo step, its constants and the matrix learning rate | [#388](https://github.com/KellerJordan/modded-nanogpt/pull/388) (@congxie1108) |
| Everything #384 inherits from the accepted record line, including SOAP-Muon on all hidden matrices, the u/w floor, RowFloor, the Tail-EMA readout and cautious weight decay ([arXiv:2510.12402](https://arxiv.org/abs/2510.12402), in Track 3 since [#265](https://github.com/KellerJordan/modded-nanogpt/pull/265)) | #46 / [#328](https://github.com/KellerJordan/modded-nanogpt/pull/328) (@ypwang61), [#321](https://github.com/KellerJordan/modded-nanogpt/pull/321) (@ypwang61 and @nooraovo), [#325](https://github.com/KellerJordan/modded-nanogpt/pull/325) (@jn2clark), and the contributors their write-ups credit |
| Porting #388's step onto #384, the head and scalar learning-rate change, and the measurements | this submission |

This submission came out of an automated research run: AI agents chose the experiments, wrote the
code changes and ran the measurements, and people checked the evidence before submitting.

## Notes

- **Loss.** The claim uses `val_ema_loss`, the loss of the Tail-EMA blend of the weights that the
  trainer validates, as #46 and #384 do. Each log line also gives the raw weights' `val_loss`.
- **Validation grid.** The trainer validates every 125 steps, and every 5 steps from 2,400 to
  2,800, then less often to 2,900. The claimed step is the earliest 5-step boundary at which both pools meet the rule,
  chosen the same way for every run, as #46 and #384 choose theirs.
- **An import that does nothing here.** The trainer tries to import `track3_scorer`, the hook of
  our evaluation harness that hands the validated weights to a separate evaluator. Outside that
  harness the import fails and the hook is skipped; it does not change training. Where the
  separate evaluator measured a run (its last 12 validations), its loss agreed with the logged
  `val_ema_loss` within 0.00002.
- **Text we kept.** The submitted file is byte for byte the one that produced the logs, so it
  keeps text that no longer describes it. A simplified copy of the trainer can follow if wanted.
  - Its docstring is #384's and names an earlier development file.
  - Its comments cite a different arXiv number for cautious weight decay than the one above.
  - Helpers it no longer calls are still there: the output K-FAC of #341 and the Newton-Schulz
    polar step.
  - The header lines every log prints (`Experiment=`, `Intuition=`, `Output KFAC ...`) come from
    #384 and the port: the K-FAC functions are never called, and the header does not name change 2.
- **Time per step.** The SOAP-Shampoo step is slower: about 2.9 s per step on one H100, against
  about 2.35 s for #384. Track 3 counts steps.
- **Open dependencies.** #384 and #388 are open pull requests. This result builds on both.

## Files

- `train_gpt_soap_shampoo_adam_lr_2450.py`: the trainer every log ran (sha256
  `c852286ca93859902ffb8c120fde013e088a1886d7a8293dcf8c2dbf9cbe917f`).
- `H100_seed<seed>_<id>.txt`: the 50 logfiles, each with the full trainer, then a line of `=`, then
  the run's output.
- `summary.tsv`: one row per run: seed, start time, stop step, whether it is counted, whether it is a
  choosing run or in the planned batch, and its `val_ema_loss` at 2,440 to 2,455 (`-` where the run
  stopped earlier).
- `claim_stats.py`: recomputes the first table and the pairwise statistics from the logfiles, and
  checks the claim (CPU only, standard library).

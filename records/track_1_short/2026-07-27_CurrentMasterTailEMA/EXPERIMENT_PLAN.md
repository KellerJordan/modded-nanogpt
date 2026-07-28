# Current-master TailEMA experiment plan

Plan frozen before any result from this source transplant was queried.

## Question

Does terminal-blended readout TailEMA plus nonzero MLP `c_proj`
initialization let current Track-1 master reach the official loss target in
less same-hardware training time?

## Sources

- Baseline: `KellerJordan/modded-nanogpt` master at
  `003ff3e2dd23ff5f7dfd6ada88eb1ff9320bfd3b`.
- Candidate: that exact baseline plus the local TailEMA/readout and `c_proj`
  transplant in this worktree.
- Baseline schedule: current-master defaults, 1285 scheduled + 15 extension
  steps (1300 total).
- Candidate schedule: 1273 scheduled + 15 extension steps (1288 total).
- Candidate TailEMA: readout scope, FP32 shadows, decay 0.9933, start 848,
  every step, terminal blend 0.65.
- Candidate MLP `c_proj` init: normal with std `0.5 * d_model**-0.5`.

The training and validation token streams, validation calculation, compile
flags, and hardware request remain unchanged.

## Stage 1: engineering screen

Run one same-container ABBA block on Modal 8xH100:

1. current master
2. candidate
3. candidate
4. current master

The screen is usable only if all four jobs complete, both candidate jobs report
the terminal-EMA assertion as applied, and no job starts on a dirty GPU node.
Proceed only if the symmetric candidate/baseline runtime ratio is below 1.0
and candidate losses are finite and plausibly near the 3.28 threshold.

This stage is exploratory and will not be represented as record evidence.

## Stage 2: accuracy confirmation

If Stage 1 passes, launch a fixed set of ten independent candidate runs. The
candidate must attain mean validation loss <= 3.28 with a one-sided one-sample
t-test p-value below 0.01. Preserve all launched outcomes and disclose any
pre-outcome infrastructure failures.

## Stage 3: timing confirmation

If accuracy passes, run eight independent same-container blocks: four ABBA and
four BAAB, with two observations per arm per block. Analyze the mean of the
within-block log runtime ratios and retain every accepted block. Infrastructure
rejections must be whole-block, pre-specified, and unrelated to observed time
or loss.

## Decision

Prepare a review PR only if the candidate passes the official loss test and the
confirmatory same-hardware timing ratio is below 1.0. Claims will use measured
paired results; no cross-provider seconds projection will be called a record.

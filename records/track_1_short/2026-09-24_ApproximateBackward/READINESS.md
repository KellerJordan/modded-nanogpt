# Submission review

The measured implementation and evidence are ready for review. Publication is
pending the author's review of the PR description. No new GPU run was needed
for this pass: the training, kernel and dependency-specification files remain
identical to the benchmarked commit.

## Evidence and claims

- All 32 eight-H100 runs are present: four pilots, 18 fixed-cohort runs, eight
  shorter-training controls and two accepted-master references. None was
  omitted or replaced.
- The fixed cohort saves 0.758866 seconds / 1.881% against unmodified PR360 on
  the same node. Mean times are 39.582968 and 40.341833 seconds.
- The fixed 12-candidate cohort independently passes the one-sided loss test
  against 3.28, with p=0.00241937. Including both matching-source candidate
  pilots gives 14 runs, mean CE 3.27868402 and p=0.00114243.
- Every final metric was checked against both the console output and the
  source-containing training log. Candidate full-precision diagnostics agree
  with their saved JSON; rounded console values agree to their printed precision.
- Numerical-check logs contain all eight ranks, unchanged head input gradients,
  checks of the sampled head formula, both attention head shapes, compiled
  autograd and the all-rank calibration reduction. These checks are distinct
  from the training timing evidence.
- The raw logs, frozen source and original provenance remain unchanged. The
  read-only verifier checks their hashes and recomputes the timing and quality
  claims directly from the raw logs.

The current [submission rules](https://github.com/KellerJordan/modded-nanogpt#rules)
require the original token streams, a supported loss result and a same-hardware
speed comparison; they also permit incorporating open PRs. The changes to
PR360 preserve its token streams, validation, training timer and compiler
configuration. Calibration and snapshot work are charged to training, warmup
observations are reset, and no learned state is transferred between runs.

Both ordinary shortening controls have observed mean CE above 3.28. They
support the proposed tradeoff but do not identify an exact equal-loss budget.
Baseline variance is high, so the PR makes no claim of improved or equivalent
quality. Compiler caches were retained in both arms; the timing claim uses
the fixed cohort, with pilot times reported separately. Maintainer reproduction
and the judgment about code complexity remain part of normal review.

## Integration

PR360 remains open at `c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`. Accepted master
was checked at `bc3a0c2d640d0d73dedaef87eae26148d2e32afb`.

The submission branch now descends from both histories. The conflicting
trainer and Triton files retain the exact benchmarked PR360-based runtime.
Current master's record archives, record-history table and unrelated kernel
module are preserved. The merge does not mix in untested changes to the
training algorithm. `codex/approximate-backward-benchmarked` preserves the
original pre-integration submission snapshot locally.

## Reproduction

- The main README now includes dependency preparation and the CUDA development
  toolkit requirement. The container instructions include the same preparation
  step; its package list explicitly includes `patch`, and installs the cu128
  Torch wheel before the remaining requirements.
- Fresh preparation with `huggingface-hub==1.29.0` downloaded the pinned Git and
  Hub sources. All 19 reconstructed headers match the benchmarked archive.
  See [the preparation receipt](provenance/readiness_dependency_check.json).
- That preparation check ran on macOS without a GPU. The GPU correctness and
  performance evidence comes from the completed Linux H100 campaign. A new
  Docker image was not built or benchmarked during this review.
- Use the [current reproduction instructions](../../../approx_backward/README.md).
  `source/` is the historical snapshot, including its pre-run documentation.

From this directory, run `python verify_evidence.py --runtime-root ../../..`
with SciPy installed to verify the packet without rewriting evidence.

Before publication, review `PR_DESCRIPTION.md`, recheck upstream for intervening
changes, and render its relative links as permalinks to the pushed commit. No
branch has been pushed and no PR has been opened during this review.

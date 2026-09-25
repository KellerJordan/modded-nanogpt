# Approximate head and attention backward

Submission candidate built on [PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360).
This directory records completed runs; it does not assert maintainer acceptance.

- [Short PR description](PR_DESCRIPTION.md)
- [Submission review and remaining publication steps](READINESS.md)
- [Results, uncertainty and every individual outcome](RESULTS.md)
- [Current reproduction instructions](../../../approx_backward/README.md)
- [Frozen runtime](source)
- [All training logs and fixed run protocols](runs)
- [Hardware, source hashes and verification records](provenance)

The source under test is commit `c5be2fbf86c918e91b9e283b4044595bb74a350f`.
`source/` preserves that exact snapshot, including its pre-run documentation;
the completed evidence and conclusions are in this directory's results files.
The current root runtime is identical to the measured code. Later documentation
or evidence commits do not represent new training implementations.

Verify hashes, all 32 raw logs, numerical-check results and statistics with
Python and SciPy. From this directory:

```bash
python -m pip install scipy
python verify_evidence.py --runtime-root ../../..
```

The verification command is read-only and needs no GPU. `report_results.py`
can also regenerate the tables, but it rewrites derived `summary.json` files;
run it on a copy of `runs/` to preserve the original manifest hashes.

The quality test includes all 12 fixed-cohort candidate runs and both pilot
candidate runs. Timing compares the interleaved fixed cohort (12 candidates,
6 PR360 controls); pilots are shown separately. Four runs at each of two shorter
PR360 schedules and two accepted-master reference runs are also included.
There are no omitted or replaced runs. Compiler caches were retained, with the
same policy for both arms. Calibration and all training work remain on the
original clock; graph warmup and validation follow the base timing convention.

The complete source/environment archive was downloaded and every member hash
verified before releasing the rented node. `packet_manifest.json` independently
hashes this review packet, including all human-readable logs.

# Track 3: SOAP-Shampoo with damping — 2615 steps (n=16), 2620 pooled (n=24)

Replace the Muon polar step in result #46's SOAP-Muon stack with two-sided temporal-residual Newton-Schulz Shampoo, with mean-eigenvalue damping of 0.01. Keep SOAP preconditioning, RowFloor, radial brake/radius pin, post-pin CWD and EMA-Nesterov. Retune matrix LR to 0.0325 and Tail-EMA start/tau to 2000/200 (blend 0.6).

At **2615 steps**, the 16 claimed seeds (49–64) have mean Tail-EMA validation loss **3.278969375**. The Track 3 significance score is **0.0041225 ≥ 0.004**; the previous evaluated step, 2610, fails. This reduces step count by 75 (2.79%) relative to #46's 2690. The step-adjusted pairwise score against the eight #46 control runs is **0.0063263 ≥ 0.004**.

Eight earlier runs of the identical configuration (seeds 37–44) were used during damping selection and are supplied separately. Including them, all 24 runs first pass at **2620 steps**, with mean **3.278850833** and score **0.005629744**. The 16-seed batch is reported as a confirmation batch; no registration artifact is included and independently documented pre-registration is not claimed. The raw-weight loss at 2615 averages 3.287031250; the claim is for the fixed Tail-EMA readout.

This is an optimizer-specific result improving accepted result #46, rather than a claim to the lowest submitted step count. Other open submissions report [2600 steps (#341)](https://github.com/KellerJordan/modded-nanogpt/pull/341), [2580 steps (#377)](https://github.com/KellerJordan/modded-nanogpt/pull/377), and [2540 steps (#384)](https://github.com/KellerJordan/modded-nanogpt/pull/384), as reviewed on 2026-10-16. The pairwise significance claim here is only against #46.

The submission includes a self-contained trainer, all 32 source-embedded logs (16 claimed, eight tuning, eight controls), a figure, reproduction launcher, and a standard-library statistics checker. Dataset, model dimensions, global batch size (524288 tokens), and causal 1024-token contexts follow Track 3. Training uses one forward/backward traversal per batch with microbatch accumulation. Validation and the fixed Tail-EMA readout perform no backward pass and restore training weights afterward. Existing #46 RMSNorm precision handling is retained.

The logs embed the research trainer rather than the simplified submission trainer. Their resolved configuration uses `--seed S --train-steps 2700 --shampoo-damping 0.01`. The submission trainer hardcodes the same active settings, stops at 2650, and keeps the schedule endpoint at 2900. The supplied write-up reports a 40-step comparison, with no supporting logs included; full-run equivalence has not been established. `reproduce_log.py` extracts the actual logged research source and prepares a launcher passing every resolved option, plus a configuration file and trainer SHA-256. Use that path for direct reproduction. Hardware names, device metadata, and CUDA build/version details are removed from filenames, logs, and trainer logging statements; numerical results and active training settings are unchanged.

Validation performed while preparing this submission:

- Recomputed every claimed mean and the first passing common step for all four batches from the logs using `claim_stats.py`.
- Checked seed coverage, duplicate validation steps, finite losses, and identical parseable embedded source within each batch. All 24 SOAP-Shampoo resolved configurations must include every research-trainer option and match `expected_config.json` apart from seed.
- Parsed both submitted training scripts and the statistics checker; checked launcher syntax and repository-root resolution.
- Reviewed architecture/data settings against the Track 3 baseline and inherited #46 implementation.

Training was not rerun during submission preparation. The submitted logs report PyTorch 2.11.0. Additional ablation claims in the supplied write-up are experimenter reports; only the claimed, tuning, and #46-control batches have logs in this submission.

Reproduce the statistics from the repository root:

```bash
python3 records/track_3_optimization/results/20261015_soap_shampoo_damping/claim_stats.py
```

Reproduce the exact logged research trainer after installing PyTorch 2.11.0 and `huggingface_hub`, and downloading FineWeb data with `python data/cached_fineweb10B.py 20`:

```bash
python3 records/track_3_optimization/results/20261015_soap_shampoo_damping/reproduce_log.py records/track_3_optimization/results/20261015_soap_shampoo_damping/seed49_05026440-e173-4c93-81e1-cb37471336de.txt /tmp/soap-shampoo-seed49
NPROC_PER_NODE=2 bash /tmp/soap-shampoo-seed49/run.sh
```

See [the submission README](https://github.com/congxie1108/modded-nanogpt/blob/codex/track3-soap-shampoo-submission/records/track_3_optimization/results/20261015_soap_shampoo_damping/README.md) for the method, per-seed table, disclosures, and credits. Experiments and write-up were assisted by an AI coding agent.

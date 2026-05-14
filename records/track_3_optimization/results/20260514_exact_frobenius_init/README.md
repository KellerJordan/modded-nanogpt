# I Exact Frobenius Init results

I Exact Frobenius Init is a Track 3 optimizer/init run that keeps the benchmark
dataset, batch size, and model architecture unchanged, and changes the hidden
matrix initialization while retaining the experiment C optimizer path.

The optimizer path directly inherits the outward radial damping and post-step
radius correction introduced by [PR #294](https://github.com/KellerJordan/modded-nanogpt/pull/294)
by [@nilin](https://github.com/nilin). The experimental change here is narrower
than that inherited optimizer stack: initialize non-projection hidden matrices
with speedrun-style variance, then rescale each such matrix to its exact target
Frobenius radius so early angular update speed is less dependent on random
initial norm noise.

## Result

The fixed training endpoint for this record is step 3020. Three seeds were run
sequentially on 4x H100: seeds `0`, `1`, and `2`. No seed was dropped. The runs
also logged dense near-end validation checkpoints from step 2960 through 3020 to
show the cooldown behavior.

This is a three-seed experiment record, not an eight-seed Track 3 submission
record. Under the Track 3 one-sided expression with `sigma=0.0013`, the step
3010 mean is `3.27748333` over `n=3` seeds:

```text
(3.28 - 3.27748333) * sqrt(3) = 0.00435899 >= 0.004
z = 3.3531
```

Step 3010 is the earliest near-end checkpoint in this three-seed run set that
clears the `0.004` margin. The final step 3020 mean is `3.27683000`:

```text
(3.28 - 3.27683000) * sqrt(3) = 0.00549060
z = 4.2235
```

| Seed | 2960 val | 2970 val | 2975 val | 2980 val | 2985 val | 2990 val | 2995 val | 2999 val | 3000 val | 3010 val | 3020 val |
| -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 0 | 3.28025 | 3.27954 | 3.27918 | 3.27885 | 3.27850 | 3.27814 | 3.27782 | 3.27757 | 3.27750 | 3.27684 | 3.27621 |
| 1 | 3.28080 | 3.28008 | 3.27975 | 3.27941 | 3.27903 | 3.27868 | 3.27836 | 3.27813 | 3.27805 | 3.27736 | 3.27670 |
| 2 | 3.28165 | 3.28092 | 3.28058 | 3.28025 | 3.27987 | 3.27952 | 3.27924 | 3.27903 | 3.27895 | 3.27825 | 3.27758 |
| **Mean** | **3.28090000** | **3.28018000** | **3.27983667** | **3.27950333** | **3.27913333** | **3.27878000** | **3.27847333** | **3.27824333** | **3.27816667** | **3.27748333** | **3.27683000** |

At step 3020 the average training time was about 1287.456 seconds on 4x H100,
with average `step_avg` about 426.31 ms. Wallclock is not the target metric for
Track 3, but the timing is included to make the record easier to compare.

## Same-seed convergence comparison

As a sanity check, the first three seeds were compared against the corresponding
seed `0`, `1`, and `2` logs from PR #294. This is not enough seed count to prove
that exact Frobenius initialization is statistically better than PR #294, but it
does give a useful same-seed signal.

Near the end of training, this run is consistently about `0.0004` validation
loss lower on the three-seed mean. The table reports paired improvements as
`PR #294 val - this run val`, so positive numbers mean this run is better at
the same seed and checkpoint:

| Step | Seed 0 diff | Seed 1 diff | Seed 2 diff | Mean diff |
| -: | -: | -: | -: | -: |
| 2875 | +0.00152 | +0.00120 | -0.00146 | +0.00042 |
| 2990 | +0.00149 | +0.00128 | -0.00158 | +0.00040 |
| 2995 | +0.00149 | +0.00131 | -0.00162 | +0.00039 |
| 2999 | +0.00150 | +0.00129 | -0.00164 | +0.00038 |
| 3000 | +0.00150 | +0.00130 | -0.00162 | +0.00039 |
| 3010 | +0.00148 | +0.00130 | -0.00161 | +0.00039 |
| 3020 | +0.00145 | +0.00130 | -0.00158 | +0.00039 |

So the evidence is suggestive rather than conclusive: seeds `0` and `1` improve,
seed `2` regresses, and the three-seed mean improves by about `0.00039`. Using
the Track 3 README's rough conversion of `0.0045` validation loss per 100 steps,
that mean tail improvement corresponds to roughly 8-10 training steps:

```text
0.00039 / 0.0045 * 100 ~= 9 steps
```

This is consistent with the observed threshold behavior: the first-three-seed
PR #294 comparison passes the Track 3 precision condition at step 3020, while
this run passes it at step 3010. Because additional H100 compute was not
available for this follow-up, only seeds `0`, `1`, and `2` were run here. More
non-cherry-picked seeds would be needed to turn this same-seed signal into a
stronger claim about exact Frobenius initialization.

## Architecture and data

The model and data path are the Track 3 baseline setup:

- Dataset: FineWeb token shards from `data/fineweb10B`.
- Global batch size: `8 * 64 * 1024 = 524288` tokens.
- Sequence length: 1024.
- Validation tokens: `20 * 524288`.
- Microbatch size: 64 sequences per rank.
- Model: GPT with vocab size 50304, 12 transformer blocks, model dimension 768.
- Attention: causal self-attention, head dimension 128, 6 heads, Q/K RMS
  normalization, rotary embeddings, and attention scale `0.12`.
- MLP: two linear layers with hidden size `4 * 768 = 3072` and ReLU-squared
  activation.
- Normalization: RMSNorm pre-normalization in each block plus final RMSNorm.
- Output logits: same bounded logit transform as the baseline script.
- Compilation: `model.compile(dynamic=False)`.

## Optimizer

The non-hidden and auxiliary parameters use the usual fused AdamW groups:

- Embedding weight: AdamW, `lr=0.3`.
- Output projection weight: AdamW, `lr=1/320`.
- Biases and norm gains: AdamW, `lr=0.01`.
- AdamW betas: `(0.8, 0.95)`, `eps=1e-10`, `weight_decay=0`.

All optimizer groups use the PR #287-style power cooldown:

```text
lr = min(flat_lr, power_c * (3105 - step) ** 1.2)
train_steps = 3020
schedule_steps = 3105
```

The group constants are:

```text
ADAM_EMBED_POWER_C = 4.976805410800738e-05
ADAM_PROJ_POWER_C  = 5.184172302917436e-07
ADAM_OTHER_POWER_C = 1.6589351369335795e-06
MUON_POWER_C       = 3.3169534699576625e-06
```

Hidden 2D block matrices use a Muon-family optimizer:

- Momentum: `mu=0.95`.
- Muon learning rate: `0.0375`.
- Stored Muon group weight decay: `0.025`.
- Newton-Schulz orthogonalization: 12 iterations of the quintic NS map, with
  Gram-Frobenius input normalization.
- Contra Muon subtraction: starts at `CONTRA_MUON_COEFF=-0.2` and linearly
  ramps to ordinary Muon by step 2000.
- Soft Muon: blends from ordinary Muon to soft Muon over steps 2500 through
  3010, using `SOFT_MUON_P=0.1`, `SOFT_MUON_INPUT_NORM=frobenius_schatten4`,
  `SOFT_MUON_SCALE=none`, and `SOFT_MUON_CEIL=0.80`.
- NorMuon-lite row/column variance normalization: per-long-axis second moment
  with beta2 `0.95`, then Frobenius renormalization.
- PR #294 radial dampening: outward radial update components are scaled by
  `0.5`, inward radial components by `1.0`.
- PR #294 post-step radius correction: after applying the update, the parameter
  is rescaled to the radius implied by the first-order radial component,
  removing finite tangent drift.
- I Exact Frobenius Init update/weight floor: the global target `u/w` is
  `0.3825`, applied only to the tangential component after PR #294-style radial
  dampening and before the post-step radius correction.

SOAP-style preconditioning is active for MLP matrices and attention V matrices:

```text
SOAP_PARAM_MODE = "mlp_plus_v"
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 10
SOAP_DENOM_POWER = 0.50
SOAP_BLEND = 1.00
SOAP_UPDATE_BEFORE_USE = False
ATTN_SOAP_DENOM_FLOOR = 0.55
ATTN_SOAP_BLEND = 1.00
V_SOAP_BLEND = 0.95
V_SOAP_BLEND_RAMP_END_STEP = 0
```

The attention trust-gate constants remain present in the script, but with
`SOAP_PARAM_MODE="mlp_plus_v"` only V is selected among attention matrices, so
the attention projection trust gate is not exercised in this run:

```text
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_STEP = 1375
ATTN_TRUST_FLOOR_FADE_END_STEP = 1625
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
```

## PR #294 versus this run

PR #294 adds a Track 3 result named "Dampen radial gradient component".
Its optimizer forms the Muon-family update, decomposes that update into radial
and tangential components relative to each hidden weight matrix, scales outward
radial movement by `RADIAL_OUTWARD_SCALE=0.5`, preserves inward radial movement
with `RADIAL_INWARD_SCALE=1.0`, applies the `u/w` floor to the full dampened
update, then rescales the post-step weight radius to remove tangent-induced
second-order norm drift.

This experiment inherits the PR #294 radial damping constants and radius
correction, but deliberately differs in three places:

- The `u/w` floor is tangent-only. After the PR #294-style radial dampening,
  this script scales only the tangential component to meet the `u/w` target,
  leaving the dampened radial component intact. In PR #294, the floor scales the
  full dampened update.
- Initialization is changed. PR #294 zero-initializes projection parameters by
  name. This run also zeroes all non-projection biases and initializes
  non-projection hidden matrices with speedrun-style `sqrt(0.33 / fan_in)`
  variance, then rescales each such matrix to its exact target Frobenius radius.
- Validation coverage is denser before the endpoint. PR #294's fixed near-end
  grid is `2990, 2995, 2999, 3000, 3010, 3020`; this run adds
  `2960, 2970, 2975, 2980, 2985` to inspect the approach to the threshold.

PR #294 reported an 11-seed Track 3 result at step 2990. This folder records a
three-seed follow-up experiment focused on exact hidden-matrix Frobenius
initialization, so the result table above should not be confused with PR #294's
11-seed submission record.

## What I Exact Frobenius Init changes

For tangent/radius-controlled optimizers, the effective first-order angular step
is approximately:

```text
lr * ||u_tangent|| / ||W||_F
```

If initial hidden matrix Frobenius radii vary by seed or module, the same
tangent `u/w` floor creates slightly different angular speeds before training
starts. This experiment keeps the optimizer machinery fixed and reduces that
source of variation at initialization.

The initialization rule used here is:

- Residual projections and the LM head projection are zero-initialized.
- All biases are zero-initialized.
- RMSNorm gains and embedding initialization are left at the PyTorch defaults.
- Hidden non-projection block matrices are initialized with
  `std = sqrt(0.33) / sqrt(fan_in)`.
- The exact Frobenius target for each initialized hidden matrix is
  `sqrt(numel) * std`.
- After normal initialization, each hidden matrix is rescaled so its actual
  Frobenius norm equals that exact target.

The affected hidden matrices are the non-projection block weights:

- `blocks.*.attn.q.weight`
- `blocks.*.attn.k.weight`
- `blocks.*.attn.v.weight`
- `blocks.*.mlp.fc.weight`

Projection weights are intentionally excluded by name, because the optimizer
lineage relies on zero-initialized residual projections and a zero LM head
projection:

- `blocks.*.attn.proj.weight`
- `blocks.*.mlp.proj.weight`
- `proj.weight`

Intuition: with the radius fixed exactly, the tangent-only floor spends the same
minimum angular budget from the first optimizer step, instead of inheriting small
per-seed radius differences from random normal initialization.

## Reproduction

This record was run sequentially on 4x H100. The local result directory combines
three sequential one-seed runs: seed `0` from run `20260513_123740`, seed `1`
from run `20260513_130134`, and seed `2` from run `20260513_132405`.

The folder includes the files needed to replay the run set:

- `train_gpt_i_exact_frobenius_init.py`: the train script used for the run.
- `run.sh`: a sequential or parallel seed launcher that installs requirements,
  checks/downloads FineWeb shards, writes per-seed stdout/stderr/log files, and
  records metadata under `runs/<timestamp>/`.
- `runs/*/seed_*/experiment_logs/*.txt`: the original UUID logfiles containing
  the source used by each completed seed.
- `runs/*/seed_*/stdout.txt`: the original full stdout logs.
- `runs/*/seed_*/metadata.txt`: per-seed launch metadata.
- `runs/*/seed_*/val_3010.txt`: captured step 3010 validation line.
- `runs/*/seed_*/final_val_loss.txt`: captured step 3020 validation line.

From a fresh pod, the convenient three-seed command is:

```bash
INSTALL_REQS=1 DOWNLOAD_DATA=1 RUN_MODE=h100 SEEDS="0 1 2" GPUS_PER_RUN=4 \
./records/track_3_optimization/results/20260514_hyperball_radial_experiments/i_exact_frobenius_init/run.sh
```

## Lineage and credits

This run should be read as an initializer experiment on top of a credited
optimizer lineage. The main inherited pieces are:

- [PR #274 / Skylight-001](https://github.com/KellerJordan/modded-nanogpt/pull/274)
  by [@kumarkrishna](https://github.com/kumarkrishna): NorMuon-lite row/column
  variance normalization, `u/w` floor postprocessing, and the `lr=0.0375` Muon
  setup used by this branch of Track 3 optimizers.
- [PR #275 / Contra-Muon](https://github.com/KellerJordan/modded-nanogpt/pull/275)
  by [@nilin](https://github.com/nilin): the Contra-Muon update term.
- [PR #278 / MLP SOAP preconditioning](https://github.com/KellerJordan/modded-nanogpt/pull/278)
  by [@samacqua](https://github.com/samacqua): SOAP-style Shampoo-basis
  preconditioning for MLP matrices. This lineage also applies the same
  SOAP-like machinery to attention V matrices.
- [PR #283 / Trustlight](https://github.com/KellerJordan/modded-nanogpt/pull/283):
  the bounded/trust-gated attention SOAP lineage credited by PR #294. In this
  run the trust-gate code remains present, but attention projection SOAP is not
  selected because `SOAP_PARAM_MODE="mlp_plus_v"`.
- [PR #287 / power-law LR schedule](https://github.com/KellerJordan/modded-nanogpt/pull/287)
  by [@yash-oai](https://github.com/yash-oai): the split power-law cooldown
  schedule constants and form.
- [PR #291](https://github.com/KellerJordan/modded-nanogpt/pull/291) by
  [@nilin](https://github.com/nilin): the Contra-Muon to Soft-Muon setup that
  PR #294 builds on.
- [PR #294 / Dampen radial gradient component](https://github.com/KellerJordan/modded-nanogpt/pull/294)
  by [@nilin](https://github.com/nilin): outward radial damping and post-step
  radius correction, inherited directly here.
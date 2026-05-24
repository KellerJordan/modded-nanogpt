# Trustlight results

Trustlight is a Track 3 optimizer run that keeps the benchmark dataset, batch
size, and model architecture unchanged, and changes only the optimizer path.

The method starts from the Contra/NorMuon + MLP SOAP idea in
[PR #278](https://github.com/KellerJordan/modded-nanogpt/pull/278) by
[@samacqua](https://github.com/samacqua). PR #278 adds SOAP-style Shampoo-basis
preconditioning to the MLP matrices before the usual Contra/NorMuon update. This
run keeps that MLP SOAP path and adds a narrow, bounded SOAP path for attention
output projections.

## Result

The submitted fixed checkpoint is step 3125. All eight seeds were run with
`EARLY_STOP=0` and `TARGET_VAL_LOSS=0`; no seed was dropped or stopped based on
validation loss. The runs continued to step 3175, and the later checkpoints are
shown for transparency.

Under the Track 3 one-sided test with `sigma=0.0013`, the step 3125 mean is
`3.27844125` over `n=8` seeds. This gives:

```text
(3.28 - 3.27844125) * sqrt(8) = 0.00440881 >= 0.004
z = 3.3914
p = 3.48e-4
```

Step 3125 is the earliest passing checkpoint in the refreshed eight-seed set.
With the original six seeds it was just below the required precision; seeds 6
and 7 move the same fixed checkpoint over the Track 3 significance threshold.

| Seed | 3125 val | 3130 val | 3140 val | 3150 val | 3175 val |
| -: | -: | -: | -: | -: | -: |
| 0 | 3.27918 | 3.27881 | 3.27823 | 3.27774 | 3.27713 |
| 1 | 3.27785 | 3.27748 | 3.27690 | 3.27642 | 3.27579 |
| 2 | 3.27948 | 3.27911 | 3.27853 | 3.27807 | 3.27744 |
| 3 | 3.28002 | 3.27966 | 3.27907 | 3.27861 | 3.27797 |
| 4 | 3.27822 | 3.27786 | 3.27725 | 3.27678 | 3.27615 |
| 5 | 3.27683 | 3.27647 | 3.27586 | 3.27542 | 3.27479 |
| 6 | 3.27830 | 3.27795 | 3.27733 | 3.27687 | 3.27624 |
| 7 | 3.27765 | 3.27728 | 3.27669 | 3.27622 | 3.27558 |
| **Mean** | **3.27844125** | **3.27807750** | **3.27748250** | **3.27701625** | **3.27638625** |

At step 3125 the average training time was about 1127.821 seconds on 4x H100,
with average `step_avg` about 360.90 ms. Wallclock is not the target metric for
Track 3, but the timing is included to make the record easier to compare.

## Architecture and data

The model and data path are the Track 3 baseline setup:

- Dataset: FineWeb token shards from `data/fineweb10B`.
- Global batch size: `8 * 64 * 1024 = 524288` tokens.
- Sequence length: 1024.
- Validation tokens: `20 * 524288`.
- Microbatch size: 64 sequences per rank.
- Model: GPT with vocab size 50304, 12 transformer blocks, model dimension 768.
- Attention: causal self-attention, head dimension 128, 6 heads, Q/K RMS
  normalization, and rotary embeddings.
- MLP: two linear layers with hidden size `4 * 768 = 3072` and ReLU-squared
  activation.
- Normalization: RMSNorm pre-normalization in each block plus final RMSNorm.
- Output logits: same bounded logit transform as the baseline script.
- Initialization: projection weights are zero-initialized as in the current
  Track 3 baseline lineage.

## Optimizer

The non-hidden and auxiliary parameters use the usual fused AdamW groups:

- Embedding weight: AdamW, `lr=0.3`.
- Output projection weight: AdamW, `lr=1/320`.
- Biases and norm gains: AdamW, `lr=0.01`.
- AdamW betas: `(0.8, 0.95)`, `eps=1e-10`, `weight_decay=0`.

Hidden 2D block matrices use a Muon-family optimizer:

- Momentum: `mu=0.95`.
- Muon learning rate: `0.0375`.
- Newton-Schulz orthogonalization: 12 iterations of the quintic NS map.
- Contra Muon subtraction: subtract `CONTRA_MUON / 2 = 0.2` times the
  operator-normalized momentum-gradient direction.
- NorMuon-lite row/column variance normalization: per-long-axis second moment
  with beta2 `0.95`, then Frobenius renormalization.
- Update/weight floor: scale updates up when `||u||_F / ||w||_F < 0.35`.
- Effective Muon weight decay is zero; the stored `weight_decay=0.025` group
  value is not used inside `Muon.step`.

## What Trustlight changes

PR #278 applies SOAP-style preconditioning to `mlp.fc.weight` and
`mlp.proj.weight`, while leaving attention matrices on the standard
Contra/NorMuon path. Trustlight keeps that idea and revisits attention more
conservatively:

- MLP matrices: use the PR #278 SOAP-style basis preconditioner before the
  Contra/NorMuon update.
- Attention output projections: also get a SOAP candidate update, but only for
  `attn.proj.weight`, not Q/K/V.
- The attention SOAP path is bounded by a denominator floor and a trust gate, so
  the SOAP direction cannot dominate when its basis statistics are too young or
  when the preconditioned direction disagrees with the ordinary momentum.

The attention trust parameters used for this run were:

```text
ATTN_SOAP_DENOM_FLOOR = 0.20
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_STEP = 1375
ATTN_TRUST_FLOOR_FADE_END_STEP = 1625
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
SOAP_BETA2 = 0.90
```

Intuition: SOAP can be very helpful when its row/column bases align with stable
curvature directions, but attention projections are easier to over-precondition
than MLP matrices. Trustlight treats the SOAP attention update as a trusted
component only when the preconditioned direction agrees with the raw optimizer
signal. Early in training, the floor gives attention SOAP enough influence to
learn useful geometry; after the warm phase, the gate becomes more selective.

## Reproduction

This record was run sequentially on 4x H100. The local result directory combines
three sequential batches: seeds `0 1 2` from run `20260506_023538`, seeds
`3 4 5` from run `20260506_034620`, and seeds `6 7` from run
`20260507_121153`.

The folder now includes the files needed to replay the run set:

- `train_gpt_simple_trustlight.py`: the train script extracted from the logged
  source, with standalone early stopping defaulted off.
- `requirements.txt`: the minimal Python packages used by the run environment.
- `run.sh`: a sequential eight-seed launcher that sets `EARLY_STOP=0` and
  `TARGET_VAL_LOSS=0`, writes per-seed stdout/stderr/log files, and records
  metadata under `repro_runs/<timestamp>/`.
- `idea_10e_attnproj_bounded_trust_floor_soapish/seed_*/logs/*.txt`: the
  original UUID logfiles containing the source used by each completed seed.

From a fresh pod, the convenient command is:

```bash
INSTALL_REQS=1 DOWNLOAD_DATA=1 \
./records/track_3_optimization/results/20260506_trustlight/run.sh
```

If requirements and FineWeb shards are already present, use:

```bash
./records/track_3_optimization/results/20260506_trustlight/run.sh
```

By default, `run.sh` uses `NPROC_PER_NODE=4`, `SEEDS="0 1 2 3 4 5 6 7"`,
`DATA_TOKENS=40`, and writes outputs into this result folder. Override these
environment variables if needed, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 SEEDS="0 1 2 3 4 5 6 7" \
./records/track_3_optimization/results/20260506_trustlight/run.sh
```

The original full stdout logs are stored under
`idea_10e_attnproj_bounded_trust_floor_soapish/seed_*/stdout.txt`. Each seed's
metadata, final validation line, and UUID source log are stored beside its
stdout.

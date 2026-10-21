# HyMUON (Hybrid Momentum MUON): 3,200 updates

This submission evaluates HyMUON on Track 3. It complements fast momentum
with a linearly weighted gradient average spanning the full training history.
The paper is [Hybrid Momentum Methods for Optimizing Deep Learning Models](https://arxiv.org/abs/2610.27674),
§3.2 and Appendix D; Appendix G describes this Track 3 configuration.
This is a Track 3 adaptation, rather than a strict reproduction of the paper's
matrix algorithm and evaluation protocol.

The frozen eight-seed confirmation reaches mean final **raw** validation loss
**3.2783215045928955** at **3,200 updates**, with
`sqrt(8) * (3.28 - mean) = 0.004747501938215692 >= 0.004`.
All prescribed seeds 636–643 are included, with no retries, replacement seeds,
or checkpoint selection. Development runs are excluded. This is a per-optimizer
result, not a global Track 3 or wall-clock record. The reference clean Muon
configuration uses 3,250 updates; a pairwise statistical superiority claim is
not made here.

| Seed | Final raw validation loss | Log |
| ---: | ---: | --- |
| 636 | 3.277712345123291 | [seed636](logs/seed636.txt) |
| 637 | 3.2802491188049316 | [seed637](logs/seed637.txt) |
| 638 | 3.2784745693206787 | [seed638](logs/seed638.txt) |
| 639 | 3.2794086933135986 | [seed639](logs/seed639.txt) |
| 640 | 3.2774875164031982 | [seed640](logs/seed640.txt) |
| 641 | 3.2777516841888428 | [seed641](logs/seed641.txt) |
| 642 | 3.2776706218719482 | [seed642](logs/seed642.txt) |
| 643 | 3.277817487716675 | [seed643](logs/seed643.txt) |

## Update and settings

For the one-indexed update `t`, gradient `G_t`, momentum `mu=0.95`, and
history weight `alpha=0.05`, initialize `F_0=S_0=0`, `w_0=0`, then use:

```text
F_t = mu * F_(t-1) + (1-mu) * G_t
S_t = (t-1)/(t+1) * S_(t-1) + 2/(t+1) * G_t
w_t = mu * w_(t-1) + (1-mu)
D_t = (1-mu) * G_t + mu * ((1-alpha) * F_t + alpha * w_t * S_t)
U_t = NS12(D_t) * sqrt(max(1, rows/columns))
W_t = (1 - lr_t * wd_t) * W_(t-1) - lr_t * U_t
```

The slow state equals `sum(2*i*G_i / (t*(t+1)), i=1..t)`, retaining all past
gradients without a reset or finite window. `w_t=1-mu**t` matches the
initialization mass of the fast EMA. The paper's slow recurrence is retained;
ordinary fast EMA, this stream mixing and extra Nesterov-style blend, NS12,
AdamW auxiliaries, and raw-parameter evaluation are the Track 3 adaptations.

Hidden attention and MLP matrices use peak LR `0.03`, peak weight decay
`0.05`, and FP32 fast/slow states. NS uses BF16 iterates, transpose-on-tall,
Frobenius normalization with `1e-7`, and polynomial `(2, -1.5, 0.5)` twelve times.
The schedule is constant for the first 30%, then linearly decays over 70%.
Both matrix LR and weight decay are multiplied by the schedule factor `eta`,
so their product is `0.03 * 0.05 * eta**2`.

Auxiliary parameters use fused AdamW: embedding LR `0.7`, output-head LR
`0.004`, other 1D/bias/gain LR `0.015`, betas `(0.8, 0.95)`, epsilon `1e-10`,
and constant weight decay `0.001`. Their learning rates follow the same schedule.
Embedding parameters/moments are BF16; the other auxiliary parameters/moments
are FP32. Initialization follows the Track 3 baseline.

Model, sequential data traversal, loss, and batch match the audited baseline:
162,354,816 parameters; 12 layers, width 768, six heads; context 1,024;
524,288 tokens per global update; microbatches of 64 sequences;
10,485,760 validation tokens. There is one global-batch forward/backward
traversal per update, accumulated over microbatches. Total training is
1,677,721,600 token positions. Validation uses raw parameters. A passive
`OnlineXPoint` average is maintained by the archived trainer, but never feeds
the forward/backward pass, optimizer, or reported readout for the admitted
`muon12` path. It is retained in the release to preserve the confirmed path.
The historical internal arm name `muon12` identifies the HyMUON configuration
with nonzero `nested_alpha`; the separate inactive `hymuon12` branch did not
produce these results and is rejected by frozen admission.

## Reproduction

Recorded runs used one H100 80GB GPU, Python 3.12.13, PyTorch 2.11.0, CUDA
12.9, and Triton 3.6.0, with matmul TF32 disabled. Timing is not a benchmark
claim. From the repository root, install the Track 3 dependencies and obtain
the standard data:

```bash
pip install torch==2.11.0 huggingface_hub
python data/cached_fineweb10B.py 20
```

The public trainer defaults to the frozen settings above, including 3,200
updates and seed 636. Its sibling manifest binds the source and configuration.
To reproduce all eight seeds sequentially on a CUDA machine:

```bash
for seed in 636 637 638 639 640 641 642 643; do
  OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  torchrun --standalone --nproc_per_node=1 \
    records/track_3_optimization/results/20261007_hymuon_3200/train_gpt_hymuon.py \
    --seed "$seed"
done
```

To prepare a replay of the **exact archived trainer**, including all original
explicit options and the observational wrapper, use a fresh output directory:

```bash
python3 records/track_3_optimization/results/20261007_hymuon_3200/reproduce_log.py \
  records/track_3_optimization/results/20261007_hymuon_3200/logs/seed636.txt \
  /tmp/hymuon-seed636
# On a CUDA machine, with dependencies/data already available:
bash /tmp/hymuon-seed636/run.sh
```

The extraction helper only writes replay files; it never starts training.
Data entities and model weights are not included. The historical data-admission
manifest hash is retained as provenance, not represented as an independently
verified digest of the organizer's current shards.

## Evidence and release changes

The source-bearing files in `logs/` were **assembled during publication** from
the archived trainer, observational wrapper, exact launch arguments, and
recorded console streams. They are not the original native trainer logfiles.
Every file contains the complete verbatim trainer and wrapper needed for its
run. Runner-specific absolute paths in console warnings are redacted; numerical
records, timestamps, runtime metadata, and all 3,200 update records are retained.
Legacy method labels, a stale 3,225 CLI default, and old paper-section labels
are preserved only inside the verbatim historical source/records for integrity.
Explicit recorded launch options override that historical default to 3,200.

The archived trainer SHA-256 is
`0fc90d63245e50e2e0ddf8fafdea653fedce769cd590472e2e5b164a8088a53f`.
The public trainer changes its method header/study label, corrects stale
endpoint/confirmation metadata, supplies the frozen CLI defaults and run ID,
and discovers the sibling release manifest. All function/class ASTs are
identical except manifest discovery in `validate_run_admission`; active model,
data, update, schedule, and evaluation bodies are retained. The release has
not been rerun on a GPU. `provenance.json` records both source hashes and
original/published log hashes. Guard summaries are reduced to exit and
interference checks; machine inventory and unrelated research material are
excluded.

Development seeds 658–659 compared budgets 3,150/3,175/3,200/3,225 before the
confirmation. The shortest budget whose two-seed mean was at most 3.2780 was
selected, then frozen before seeds 636–643. This describes the supplied
development/confirmation provenance; independently timestamped public
preregistration is not claimed.

Run the CPU and evidence checks from the repository root:

```bash
python3 records/track_3_optimization/results/20261007_hymuon_3200/verify_submission.py
python3 records/track_3_optimization/results/20261007_hymuon_3200/check_optimizer_cpu.py
```

The first check requires only Python's standard library. The second uses
PyTorch for small CPU matrices, including the closed-form history, EMA mass,
NS12, global optimizer clock, and state restoration; it does not train a model.
See [VALIDATION.txt](VALIDATION.txt) for submission-preparation results. The
baseline model/data AST comparison was checked at upstream commit
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`. This package and PR preparation
were assisted by an AI coding agent.

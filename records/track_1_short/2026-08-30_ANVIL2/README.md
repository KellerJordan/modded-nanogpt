# NanoGPT speedrun: record attempt (ANVIL2)

GPT-2 (124M-class) on FineWeb10B, 8×H100-80GB single node, targeting the modded-nanogpt gate:
**final val CE ≤ 3.28, mean over runs, all runs count.**

> **This PR supersedes [PR #349](https://github.com/KellerJordan/modded-nanogpt/pull/349)
> ("ANVIL").** It is based on **record #89**, not on PR #349, and it carries everything PR #349
> introduced. PR #349 will be closed in favour of this one; there is no separate ANVIL record
> folder in this branch.

> **CERTIFIED POOL ATTACHED.** The pool below is **n=18 unseeded runs** of the shipped source,
> every leg fully cold-cache, interleaved with **n=9 baseline runs of record #89 on the same
> machine in the same session** (rule 4). Mean final val CE 3.27731 ± 0.00099 clears the 3.28
> gate with one-sided p = 9.4e-10 (rule 2: p < 0.01). Logs in `this_pr/` and `baseline/`.

## Requirements

Read this section before running. Every item below was hit at least once while reproducing this
record on a fresh node.

**Stack (pinned; the repo-root `Dockerfile` builds exactly this).**

| | |
|---|---|
| base image | `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` |
| torch | `2.10.0+cu128` (pinned wheel, **not** a nightly) |
| loader | `kernels==0.16.1`, `huggingface-hub==1.29.0` (`requirements.txt`) |
| driver | **>= 580** |
| GPUs | 8x H100-80GB, single node |

```bash
python data/cached_fineweb10B.py 9    # ~1.9 GB; the run uses the first 4 train shards + the val shard
bash run.sh                           # no environment variables needed
```

**Five things that will bite you.**

1. **Do not use a torch nightly.** A torch without the cu128 stable ABI fails loud at import
   (*"Cannot find a build variant for this system"*). A cu128 *nightly* is worse: it loads the
   kernel fine and then trains with unpinned numerics. One reproduction attempt produced **NaNs**
   exactly this way. Check `torch.__version__` prints `2.10.0+cu128` before trusting any run.
2. **`get_kernel` can return 401 even though the repo is public.** `kernels>=0.16` resolves
   through the Hub's kernel-repo namespace, which rejects anonymous requests for repos predating
   that repo type. Two verified fixes: `hf auth login` with any HF account, or pre-seed and run
   offline:
   ```bash
   hf download devenpzak/flash-attn3-12864 --revision 64c1e6d1f2780e7931839f41426ddcdb564a7cb9
   cp -rl $HF_HOME/hub/models--devenpzak--flash-attn3-12864 \
          $HF_HOME/hub/kernels--devenpzak--flash-attn3-12864
   HF_HUB_OFFLINE=1 bash run.sh
   ```
   The in-code sha256 assert verifies the artifact either way. Record #89's baseline pulls
   `kernels-community/flash-attn3` separately, so the first baseline run on a node must be online.
3. **Driver below 580 fails silently.** The whole step slows uniformly (~35 % on otherwise
   identical H100 nodes) with no error. The tell is that the stock baseline slows identically.
4. **Bare metal needs `libcudart.so.13`.** The kernel is linked `--cudart shared`. On a CUDA-12.x
   host: install NVIDIA's `cuda-keyring` deb, `apt-get install cuda-cudart-13-1`, and put
   `/usr/local/cuda-13.*/targets/x86_64-linux/lib` on `LD_LIBRARY_PATH`. If the image already
   carries an NVIDIA apt entry for the same repo, `apt-get update` fails with a *Signed-By
   conflict*: delete one of the two duplicate `.list` files under `/etc/apt/sources.list.d/`.
5. **One environment, end to end.** Install `requirements.txt` into the *same* environment whose
   `torchrun` launches the run. A system `torchrun` over a venv install gives
   `ModuleNotFoundError` at startup. `CUDA_HOME` must point at a real CUDA tree (the training CE
   kernel compiles via nvrtc at import and needs `cuda_bf16.h` under `$CUDA_HOME/include`), and
   the node wants >= 10 GB free disk for the Inductor/Triton caches.

**Expected numbers.** Across 26 unseeded runs on two 8xH100 machines, 
single-run final val landed in **3.2756 to 3.2805**, mean **3.27751** (sd 0.00129); 
1 of the 26 printed above 3.28. **The criterion is the mean over runs, not any single 
run.** For calibration, record #89's own baseline measured on the same machine is both
wider and higher: 3.2760 to 3.2825, mean 3.27828, with 2 of its 9 runs above 3.28. A single 
high run is the benchmark's noise, not a failed reproduction. For a variance-reduced check,
`KX_SEED=1234` pins init and data order (not bitwise: the fp16-atomic kernels jitter about
+/-0.5 millinats) and measured 3.2763 and 3.2789 on our two machines.

## Result

`KX_STEPS=1122` scheduled steps plus 52 grown mid-schedule steps and a 20-step extension stage
= **1194 trained steps** (the trainer's default; `bash run.sh` reproduces with no environment).

**Certification pool (2026-08-31).** 18 unseeded runs of the shipped source, every leg with EMPTY
compile caches (fully cold start), interleaved with 9 runs of record #89 on the same 8×H100-80GB
machine in the same session; all runs counted:

| | runs | wall (training section) | final val CE |
|---|---|---|---|
| **this PR** | **18** | **39.914 ± 0.120 s** | **3.27731 ± 0.00099** |
| record #89 (same pod, interleaved) | 9 | 73.889 ± 0.137 s | 3.27828 ± 0.00205 |
| **win** | | **33.975 s (1.8512×)** | |

- one-sided t vs the 3.28 gate: **t = 11.5, p = 9.4e-10** (17 dof)
- 17 of the 18 run logs ship in `this_pr/` (one leg's raw log was lost to the pod expiring
  minutes after it finished; its wall/val row (39.969 s @ 3.2775) was recorded from the
  machine's run ledger by two independent pulls and is included in the statistics above;
  see `this_pr/statistics.md`)

- peak memory 53,424 MiB allocated / 64,706 MiB reserved (certification-pool logs)
- record #89 is 1.23 minutes (73.8 s) as published; measured here same-session, same machine,
  interleaved with the pool legs: **73.889 ± 0.137 s (n=9)**. The delta is **33.975 s / 46.0 %**
  (1.8512×) as a true rule-4 comparison.
- for lineage: the superseded ANVIL PR (#349) measured 64.95 s @ 3.27604 (n=16) / 64.72 s @
  3.27740 on its 12-run logged pool. ANVIL2 is ≈25 s faster than that, again not same-session.
- val CE margin to the 3.28 gate is 2.7 millinats on the pool mean (n=18), one-sided
  p = 9.4e-10 against the gate.

## Where the 34 seconds come from: per-mechanism ablation study

Every mechanism below was removed individually from the shipped trainer (reverted to record
#89's corresponding code, verbatim) and re-run: 26 removal legs interleaved with 7 champion
anchor legs on one 8×H100-80GB node (driver 580.126, the pinned stack above). Most rows are
n=2. The optimizer is split into two rows rather than one because its wall and its val move in
opposite directions. As a whole-mechanism check, record #89's optimizer running exactly as #89
ships it (no weight shipping, eager tail) costs **+21 millinats of val at roughly equal wall**,
consistent with the two rows below. Wall deltas are measured directly against the
interleaved anchors; val deltas are priced at the single exchange rate that makes the whole
table sum to the certified 33.98 s win. That implied rate, 164 ms/millinat, sits near the centre
of the ~120-250 ms/millinat band measured independently during development (the marginal cost of
buying val with extra training steps), which is the coherence check that makes the normalization
honest rather than fitted. **No leg is discarded anywhere in this table**: every completed
ablation run is counted, the same rule the record pool itself is held to.

| group / component | net@r* |
|---|---|
| **ANVIL** | **+5.08** |
| &nbsp;&nbsp;├─ ANVIL algorithm (update rule + anneal/shipping) | +4.16 |
| &nbsp;&nbsp;├─ optimizer-update graph capture (tgr) | +0.92 |
| **TRAINING-STEP GRAPH CAPTURE** | **+2.99** |
| **SAMPLED-SOFTMAX TRAINING LOSS (val pipeline untouched, full-vocab)** | **+8.40** |
| **EMBEDDINGS (hashed n-gram table)** | **+7.05** |
| **FULL-STACK FP8** | **+4.96** |
| **MIXED-WIDTH ATTENTION** | **+3.82** |
| **SPARSE GRAD COMMS & DATA LOADER** | **+1.20** |
| &nbsp;&nbsp;├─ value-embedding sparse gradient exchange | +1.13 |
| &nbsp;&nbsp;├─ parallel shard loader (openpack) | +0.07 |
| **MUDDFORMER RESIDUAL MIX (dynamic dense connections)** | **+0.48** |
| **TOTAL** | **+33.98** |

Per-mechanism measurements behind the groups (removal deltas vs interleaved anchors):

| mechanism | wallΔ | valΔ (millinats) | legs | net@r* |
|---|---|---|---|---|
| CE stack (sampled softmax + prefix-CE + fused kernel) | +8.80 | -2.4 | 2 | **+8.40** |
| n-gram table (84.6M hashed, sparse) | +1.19 | +35.8 | 2 | **+7.05** |
| fp8 extension (MLP+lm_head cache) | +5.02 | -0.4 | 2 | **+4.96** |
| ships / anneal endgame (EMA+tavg+blend+decon) | +0.52 | +20.2 | 2 | **+3.82** |
| attention pkg (mixed-width, dv64, bankc2, fp8 QKV) | +4.95 | -6.9 | 2 | **+3.82** |
| fwd/bwd CUDA-graph runners | +2.99 | +0.0 | 2 | **+2.99** |
| VE sparse path | +1.05 | +0.5 | 2 | **+1.13** |
| optimizer tail graphs (tgr) | +0.84 | +0.5 | 3 | **+0.92** |
| MUDD 10-coef mix | -0.51 | +6.1 | 2 | **+0.48** |
| ANVIL update rule vs 89-Muon | -0.47 | +4.9 | 2 | **+0.33** |
| openpack loader | +0.07 | +0.0 | 2 | **+0.07** |
| **TOTAL** | +24.45 | +58.2 | | **+33.98** |

*anchors: wall n=7, val mean 3.2775 | Σwall=24.45s Σval=58.2 millinats | implied rate r*=164 ms/millinat (measured band ~120-250)*

Notes: (1) ablations are not strictly additive: interactions are absorbed by the implied
rate, which is why it is stated as a normalization convention; (2) the sampled softmax is
training-only: validation is always the full 50,304-way softmax, unchanged from record #89;
(3) the optimizer-update graph row's three legs read +2,090, +55 and +365 ms against their
interleaved anchors; the first ran immediately before another leg that is also high against its
own replicate, so the two of them look like a localised artifact over two adjacent runs. All
three are counted anyway, under the all-runs-count rule; (4) swapping the whole optimizer process for record #89's optimizer exactly as
#89 ships it (no weight shipping, eager tail) costs +21.1 millinats of val at roughly equal
wall. That is the whole-mechanism check quoted above; the two ANVIL rows are measured from their
own legs, not derived from it.

## Files (all at the repo root; this folder holds the record documentation)

Modern record folders in this repo carry documentation and logs only (the code lives at the
repo root), so that convention is followed here.

- `train_gpt.py`: the trainer. All configuration baked in. Env: `KX_STEPS` (default 1122,
  before the +52 growth), optional `KX_SEED` (reproduction only; records run unseeded),
  `DATA_PATH`. The only variable the file *sets* is `PYTORCH_ALLOC_CONF`.
- `triton_kernels.py` (modified): the kernel package, kept at record #89's path. The stock
  symmetric-matmul entry points are carried over; added alongside them: fused softcapped CE with
  the sampled-softmax and prefix-token-CE arms, the fp8 quantize/matmul machinery and the
  optimizer matrix helpers.
- `anvil_attn_kernels.py` (new): the packed dual-layout fp8 QKV op, QK-norm/RoPE/pad fusion,
  and the mixed-width attention plumbing, split out of the kernel package.
- `bigram_kernels.py` (new): hashed n-gram embedding forward/backward, the sparse-gradient
  sink and the row-compacted Adam with exact replay.
- `fuse_tiny_kernels.py` (new): tiny-kernel fusions for the optimizer tail (replicated-Adam
  params and their Adam state).
- `value_embed_op.py` (new): selected-load value-embedding backward, kernel included.
- `run.sh`: the stock launcher, byte-identical to record #89's single line (`torchrun
  --standalone --nproc_per_node=8 train_gpt.py`). Runtime image: see "Runtime requirements".
- **Deleted by this PR:** `dc_triton_kernels.py` (the dual-chunk attention correction, whose
  layer is gone). After the change its only importer was record #89's `train_gpt.py`, which this
  PR replaces; `train_gpt_medium.py` (track 2) does not import it and is untouched.
- The data pipeline, tokenization and the validation path are byte-identical to stock.

Shipped sha256 (first 16 hex) of each file, so a reviewer can pin exactly what the pool
above ran: `train_gpt.py` c6da246b1d87a525, `triton_kernels.py` 71728386ec63079b,
`anvil_attn_kernels.py` 5882b11329fab39d, `bigram_kernels.py` 8cd6986fa68edc18,
`fuse_tiny_kernels.py` 066ae2652648c6ba, `value_embed_op.py` 7f2e7f2e3d548f6e.

## Changes vs record #89

This PR replaces [PR #349](https://github.com/KellerJordan/modded-nanogpt/pull/349), so the
diff it carries is *two* records' worth of change against the #89 tree. Part A is what ANVIL
(PR #349) introduced over record #89. The description is Deven's own from that PR, with the
items ANVIL2 later revised marked as such. Part B is what ANVIL2 adds on top.

### A. Carried over from ANVIL (PR #349): new vs record #89

**Optimizer: ANVIL (Averaged, Normalized Velocity with Isotropic Lanes).** A rework of record
#89's Muon/NorMuon optimizer stack (lineage cited in the trainer docstrings). Twin-rail
velocity: two EMA rails of the reduced gradient (fast rail on a scheduled beta with plateau
0.93, slow rail on a constant long-horizon beta) held in one `[2, *chunk]` fp32 state and
blended into a Nesterov-lookahead velocity. A cascade of six quintic spectral maps on the
velocity Gram, the Polar Express / Newton-Schulz family, with coefficients **re-derived from
scratch** (CEM + minimax polish; the envelope quoted here, [0.9971, 1.0097] on σ ≥ 3e-3 vs ~[0.86,
1.14] for the classical 5-map schedule, is PR #349's schedule; **B re-derived the maps**, and
the shipped `ANVIL_MAPS` give tail gain 643.13 and envelope [0.9951, 1.00041]), whitens the velocity in bf16. Per-lane energy
equalization (the NorMuon rescale) at fixed Frobenius norm; sign-aligned (cautious) decoupled
decay ×1.5; exact-fp32 commits on bf16 storage via a uint16 mantissa sidecar; and a bank
tail-blend ship step (fp32 EMA over the last ~300 steps, blended into the shipped banks).
lm_head/embed on sharded Adam; embedding-table LR multipliers 70.

**FP8 everywhere it pays**: full fp8 MLP forward **and** backward (dx/dW1/dW2) with statically
clip-free scales; lagged comm-overlapped weight-cache quantize; fp8 lm_head cache feeding the
fused CE in both layouts, refreshed by a tiled-transpose kernel. Record #89 has fp8 on the MLP
up- and down-projection forward only.

**Narrow query/key attention**: 8 of 10 attention layers compute Q and K at head width 64 (V
stays 128) with a 64-dim rotating rotary basis; layers 3 and 10 keep full width 128 as
carriers. Attention scale retuned to 0.085 under the YaRN window-switch compounding, 0.12 on
the narrow layers. *(Revised by B: ANVIL2 removes layers outright, runs seven attention
sublayers, halves V/O on two of them and moves the narrow-layer scale to 0.13.)*

**Period-4 embedding cadence**: the value-embedding and bigram-embedding Adam channels update
every 4th step from step 336; gradients accumulate losslessly across the cycle, so update
content is preserved while communication and update cost drop 4×. The channels' Adam betas are
squared and the weight-decay multiplier doubled when the cadence engages: the per-update
equivalents of the per-step settings.

**Sparse gradient path for the embedding channels**: the value-embedding gather backward loads
1 adjoint plane instead of 5, and the bigram channel uses a detached lookup with a zeros-leaf and
a compact segment-sum exchange.

**Virtual sequence cap**: from step 715 the train loader's attention metadata splits documents
longer than 2048 tokens into virtual segments. Attention masking only, the same class as the
block-window schedules. The token stream, the targets and the entire validation pipeline stay
byte-identical to stock. *(Revised by B: a virtual 2560-token attention cap under a 3072-token
document cap.)*

**Schedule**: 3 stages (batch 8/16/24, seq 896/2048/3072) + a terminal batch taper + 23
extension steps at windows (6,13); cooldown fraction 0.80 to LR floor 0.20; windows
1,3 → 3,7 → 5,11 through the stages; final-val long window at the stock 20; 40 steps grown into
the large-batch stage, with the LR cooldown re-anchoring to the grown length. *(Revised by B.)*

**Host-side systems work**: automatic Python garbage collection is frozen and disabled for the
timed loop (the single collect runs after the final timer read); the run log is held in one
buffered handle with step prints thinned. Both are host-only; neither touches numerics.

**Validation cadence**: validation runs once, at the end (`val_loss_every=0`), purely to shorten
machine occupancy. Validation passes are untimed either way, the validation code path is
unchanged from record #89, and setting `val_loss_every=250` reproduces the mid-run curve.

### B. New in ANVIL2 (on top of ANVIL / PR #349)

**A hashed n-gram embedding table of 84,602,880 rows.** Record #89's hashed-bigram channel
carries `bigram_vocab_size = 50304 * 15 // 2 = 377,280` rows (ANVIL keeps that size). This one
is 224.24× that, at `bigram_dim = 768`, and it carries **two** channels rather than one: a
bigram hash and a trigram hash over `x[t-2], x[t-1], x[t]`. The table is sharded one eighth per
rank (10,575,360 rows, 15.13 GiB/rank); each step pulls only the rows its own tokens hash to,
through a pull-fed row cache (cap 528,384 rows, 774 MiB) and a compact segment-sum exchange,
and is optimized by a row-compacted Adam with exact replay so an untouched row's moments are
advanced arithmetically rather than by a dense pass. This is precedented in *kind* by the
stock hashed-bigram embedding and not in *scale*, so it is called out first: see
"Disclosures".

**Sampled (shared-negative) softcapped cross-entropy for the early stages.** Over roughly the
first 93 % of the run the training CE is computed against a candidate set (every target in
the batch, and every prefix target, plus a per-step duplicate-free stride sweep of negatives)
instead of the full 50,304-way softmax: `P = 10,240` from step 0, 14,336 from 681, 24,576
from 961, off at 1101, with logged headroom of 34-61 % on the candidate budget. The batch's
own targets are always in the candidate set. **This biases the training gradient. Validation
is always the full 50,304-way softmax**, so the graded number is unaffected.

**A fixed-max LSE in the training CE kernel.** The softcap bounds the logit at `z ∈ (0, 23]`,
so the online block max in the CE kernel is replaced by that static bound and the sigmoid is
cached in fp16. Training-only; validation takes the eager full-softmax branch.

**Depth reduction and mixed-width attention.** Layer 7 is removed whole and layers 4 and 9
lose their attention sublayer, leaving seven attention sublayers `{0,1,2,3,5,8,10}`. Five of
them run `d_qk = 64` with a fully-rotating 64-dim rotary at attention scale 0.13; two of those
five (`{1,8}`) also halve the value/output heads, so the live head geometries are (64,128) on
`{0,2,5}` and (64,64) on `{1,8}`. The two long-window carriers `{3,10}` keep full-width QK on
their own module and rotary table. All of it runs through one packed dual-layout fp8 QKV op
whose activation quantize is hoisted and shared across the layers reading the same normalized
residual. The (64,128) and (64,64) geometries are why the attention kernel is a patched FA3.

**Four value-embedding planes.** Plane 3 is retired with layer 9's attention: 201,216 × 768 =
154.5 M parameters and a 294.8 MiB sink, against five planes / 193.2 M / 368.4 MiB in
record #89 and in ANVIL. The selected-load backward is carried over.

**A ten-coefficient post-loop MUDD mix, per channel group,** over every residual depth the
layer loop leaves bound and all four live value-embedding planes.

**A five-entry batch / window / seq-len schedule.** Three main stages (batch 8/16/24, document
cap 896/2048/3072, windows (1,3) → (3,7) → (5,11)), a batch-20 terminal taper cut out of the
last 6 % of the third, and a batch-8 extension stage at windows (6,13); +52 steps grown into
the third stage; cooldown fraction 0.80; embed/lm_head split at the extension stage. A
3072-token document cap sits under a virtual 2560-token attention cap on the packed seqlens:
attention masking only, the token stream and every per-rank shape stay stock. YaRN rescaling
is split across the short and long window chains.

**Terminal ships, all on the clock.** TailEMA (window 298, blend 0.65, on embed and lm_head),
tail-average (window 250, every 4th step, realized rate 0.0755, on `mlp_bank` and `vo_bank`),
the value-embedding ship (window 250), the ANVIL bank tail-blend (k=298, blend 0.55 on
`qk_bank` and 0.3493 on `vo_bank`/`mlp_bank`), and the decontraction ship.

**Full CUDA-graph capture of the training step.** The forward+backward of every step is
captured (one graph per distinct trace-time configuration, nine keys covering steps 0..1187)
alongside the ANVIL bank tails and the fp8 weight refresh (three graphs: attention, and the
MLP refresh with and without the lm_head copy). Every pointer a graph bakes is fingerprinted
before the warmup pass's reset and re-verified after it, and every capture owes an eager-vs-
replay self-check that is asserted to have been consumed before the clock starts. All of it
happens inside **one** untimed warmup pass (see "Disclosures").

**Optimizer refinements.** ANVIL as in part A, now with the attention banks carrying live
matrices only, permuted into fp8-batch order so every quantize batch is a contiguous run, and a
fused flat `all_reduce` for the replicated parameters (one NCCL call per gradient dtype).

## Disclosures

These are the visible departures from the shipped record. None of them is a rules problem, but
each should be read here rather than found in the diff.

**37 untimed warmup steps**, against 14 in record #89 (21 in the superseded ANVIL PR). The arithmetic is `29 sampled
configurations + the 8-step capture prefix [0..7], visited LAST`. The log names the sampled set
before `_cf_order` reorders it, so the split is derived here rather than printed. The 29 is a property of the schedule: five batch/window/seq-len stages instead of three,
plus the taper and the extension, plus a "≥ 2 untimed visits per CUDA-graph capture key" rule
that makes every capture checkable off the clock. The 8-step prefix is what the tail captures
need for their cadence (the fp8 bootstrap must be past, and both `refresh_lm` values must
appear), and it is visited last so that the memory-floor gates see the run's real steady-state
free memory. There is exactly **one** warmup pass, followed by the stock reset; there is no
second pass and no pre-clock first-touch warming beyond it. A budget assert caps the total
at 40.

**The capture lifecycle.** Every CUDA graph in this run is captured and proved inside that one
untimed pass, and the pass follows record #89's own shape. `warmup_steps` is built from the
schedule's transitions and unioned with an 8-step prefix; an assert caps the total at 40 and
refuses any configuration that would get fewer than two untimed visits, one to capture on and one
to check on. Each warmup step runs the real training step, capturing on the first visit to each
configuration key and spending the check on a later visit (the train-step graph checks to 5e-4
relative, since the fp16 sink atomics move the loss about 1e-4; both tail captures check
bitwise). The pass then ends in the stock reset: `model.load_state_dict` from the pre-warmup
deepcopy plus `training_manager.reset(initial_state)`, exactly as record #89 does, after which
the fp8 weight caches are rebuilt from the restored weights (record #89 makes the same call in
the same position). Two extra guards that record #89 does not have: every address a captured
graph baked is snapshotted before the reset and re-resolved after it, failing if one moved; and
`_cf_assert_captured()` aborts the run unless every graph exists and every owed check has been
spent. Only then does the clock start.

**Sampled softmax is training-only.** Described above. Validation is the full 50,304-way
softmax on every run.

**The hashed n-gram table is 84,602,880 rows** (bigram + trigram channels), against record
#89's 377,280.

**Parameter count.** The transformer is the track's GPT-2-scale (124M-class) one. Beside it the
model carries 84,602,880 × 768 = **6.50e10 parameters** of hashed n-gram table plus 201,216 ×
768 = 1.55e8 of value embeddings, so the honest total is **≈65.3 billion parameters**, of which
>99 % is the table, sharded one eighth per rank and read one row-set per step.

**Table-size dose grid** (development tree, only the row count changed; 2 unseeded runs each, same 8×H100 box; supporting data for the EMBEDDINGS row of the ablation study above):

| rows | wall (s) | final val CE |
|---|---|---|
| 84,602,880 (this PR) | 39.52 (n=2, knobbed dev tree) | 3.2791 |
| 21,150,720 (¼) | 39.35 / 39.36 | 3.2825 / 3.2863 |
| 377,280 (record #89's size, this PR's two-channel hashing) | 39.00 / 38.98 | 3.3149 / 3.3152 |

The large table costs ~0.17 s of wall against the ¼-size one and is worth ~5 millinats of val there, and ~36 millinats against record #89's row count. The table is a lookup: no FLOPs, sparse row updates, never on the wire as a whole.

**Host-side GC and logging.** Carried over from ANVIL (part A): automatic garbage collection is
frozen and disabled across the timed loop, and the run log goes through one buffered handle
with thinned step prints. Host-only, no effect on numerics.

**Three `last_step` dead-work skips.** At the final validation of a run that saves no
checkpoint, three operations have no remaining consumer and are skipped: (1) the fp8 weight-
cache refresh inside the final `ags_flush`: every consumer is gated on `self.training` and
the run breaks after this eval (the *gathers* still run); (2) the dense bigram replay, whose
only outputs are `exp_avg_sq` and the last-event clock, both read only by a checkpoint or a
next training step, neither of which exists. What validation reads is the row cache, filled
by the charged pull below it; (3) the `value_embeds` replica rebuild, superseded by the
value-embedding ship a few lines later, which writes the same shard and gathers over the whole
replica. These are skips of provably dead work, not work moved off the clock; the argument for
each is written out at the call site. Measured: forcing the bigram replay back on costs
0.26 s (39.79 s vs 39.53 s on the same leg).

**The attention kernel is a patched FA3**, not the community build. `train_gpt.py` loads
`get_kernel('devenpzak/flash-attn3-12864', revision='64c1e6d1f2780e7931839f41426ddcdb564a7cb9')`
(the immutable commit behind tag v1) and sha256-verifies the loaded `.so` in code. Upstream FA3's Arch-90 bf16 pair set
does not instantiate the `(64,128)` or `(128,64)` head-dim pairs that the mixed-width attention
above requires, and adding them is a source patch, not an instantiation-list entry, so it is
called a patched build. The patches (`fa3_12864_from_64x128.patch` and its companion), the
build driver, the base commit, the CUTLASS version and the `.so` sha256 are published in that
HF repo under `src_patches/`. The Python surface (`flash_attn_interface.py`,
`flash_attn_config.py`, `flash_attn3/__init__.py`) is byte-identical to
`kernels-community/flash-attn3`.

**Clean-room replication.** The full path above was verified end-to-end by wiping an 8×H100
node bare and re-running from only this document, scripted with zero manual intervention:
pinned stack → bare-metal `libcudart.so.13` → the no-login kernel recipe → data → `bash run.sh`.
Result: `val_loss 3.2783, train_time 40.34 s`, inside the expected single-run band on that
machine (its interleaved champion runs measured 40.2-40.6 s; wall varies by machine, val does not).

## Timing convention

Standard. Compilation, kernel warmup, CUDA-graph capture and every validation pass (including
the final one) are untimed; the clock runs from the post-warmup `torch.cuda.synchronize()` to
the synchronize at the validation break. The first-batch fetch, shard loading, the prefix-table
build, the validation token read and all terminal weight ships are **inside** the timed
section. Validation runs once, at the end (`val_loss_every=0`); the validation code path is
unchanged from record #89 and setting `val_loss_every=250` reproduces the mid-run curve.

## Known limitation

**FA3 packaging.** The attention kernel is published in one variant, `torch-stable-abi29-cu128`,
which is why this PR carries the runtime bump described under Requirements. Publishing a
`torch-stable-abi29-cu126` variant would let the stock base image run it unchanged. That remains
open.

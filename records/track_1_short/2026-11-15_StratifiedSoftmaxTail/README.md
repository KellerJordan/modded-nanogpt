# Stratified sampled softmax on the widest steps (candidate, not yet run on 8xH100)

**Status: built and CPU-tested only. No 8xH100 run exists yet. Every 8xH100 number below is an
expectation, not a measurement.** Base: record #92 (ANVIL2), commit `4ea6b93`.

## What changes

The record trains steps 0-1106 on a sampled softmax. Its widest setting, 24,576 candidate ids, covers
steps 965-1106. This change runs those 142 steps over 8,192 columns instead. Nothing else changes: the
other steps, the schedule, the optimizer, the data and the validation are the record's.

On those steps each GPU's 49,152-row microbatch is split into 3 chunks of 16,384 rows **for the loss
only** (the model still runs once over the whole microbatch). Each chunk gets its own candidate set:

| Part | Ids | Weight |
| --- | --- | --- |
| Exact | 6,144: every target in the chunk, filled up with the ids this GPU has seen most often as targets so far | 1 |
| Tail | 2,048 drawn uniformly at random from all the remaining ids | (remaining ids) / 2,048, about 21.6 |

The loss normalizes over `sum_exact exp(z) + w * sum_tail exp(z)`, an unbiased estimate of the
full-vocabulary normalizer, so every id is still pushed down with the right strength on average. The CE
kernel is the record's with one log weight per column added inside its three `exp()` calls.

Why chunks: every target must be in its set. A 49,152-row microbatch holds up to 10,397 distinct
targets (census of the run's data, 8-GPU slicing), more than 8,192 columns; a 16,384-row chunk holds at
most about 5,300.

## Files

| File | Change |
| --- | --- |
| `track_1_short/stratified_softmax.py` | New. The constants, the host-side set builder (numpy, on the record's prep thread) and the kernel-source substitution. |
| `track_1_short/sampled_softmax.py` | Steps at P = 24,576 become stratified steps: one upload and one row gather per chunk; `ChunkedSampledLoss`. |
| `track_1_short/perf/kernels/sampled_cross_entropy.py` | The column-weighted CE kernel and `StratifiedSoftcappedCrossEntropy` (per-chunk GEMMs, kernel and densify). |
| `track_1_short/perf/sampled_softmax_overlap.py` | One more pinned vector per slot: the column log weights. |
| `track_1_short/perf/kernels/cross_entropy.py` | `compile_ce_kernel` takes the kernel source. |
| `track_1_short/model/gpt.py`, `perf/cuda_graphs/step_graphs.py`, `train_gpt.py`, `config.py` | Wiring: the loss branch, the graph key and address census, two log lines, the switch's docstring. |
| `tests/test_stratified_softmax.py` | CPU tests (no GPU, no triton). |
| This folder: `node.sh`, `gpu_check.py`, `run_pool.sh`, `statistics.py` | The node procedure, the GPU kernel self-check, the interleaved pool, the pool statistics. |

`STRATIFIED_SOFTMAX=0` turns the change off (the record's sets on every step). The pool script does not
use it: its baseline legs run the record's own commit from a second checkout.

## Evidence so far (one RTX 5090, a one-GPU port of this record)

The port feeds each step as 24 microbatches of 16,384 tokens, which is 8 GPUs x 3 chunks, so its loss
is this change's loss. All runs are pairs: the unmodified port, then the candidate, same seed and
session. The port's validation loss is not on the 3.28 scale (it reads about 3.295).

| Run | Loss change | Time change |
| --- | --- | --- |
| This change, steps 965-1106 (5 paired seeds) | +0.0010 +/- 0.0005 (every seed between +0.0005 and +0.0015) | -2.12% +/- 0.16% |
| Control: the record with 26 fewer steps, no pruning (2 paired seeds) | +0.0043 | -1.85% |
| Control: steps 965-1106 at 24,576 columns, sets built on the GPU (3 paired seeds) | not the point | +0.0% |

So on the port it costs about a millinat, about a quarter of what cutting steps costs for the same
time, and the saving comes from the narrower set, not from where the port builds its sets.

Not carried over from the port: there every microbatch updated one frequency count; here each GPU
counts its own targets (an eighth of the data). The sets are built on the host here, on the GPU there.

## What is not verified

- The 8-GPU code path has never run. The captured step graphs, the compiled model and the upload
  stream cannot run on one RTX 5090 or on a CPU. The first 8xH100 run is the first execution.
- The time on 8xH100. 2.1% of 39.9 s would be about 0.8 s; three kernel launches and three row
  gathers per step instead of one, and two extra adds of the lm_head gradient, take some of it back.
- The loss on the 3.28 scale. The record's pool mean is 3.27731 (n = 18, std 0.00099), 2.7 millinats
  under the gate. A cost of one millinat would leave about 1.7.

## Running it on an 8xH100 node

The node needs an NVIDIA driver >= 580 and Docker with the NVIDIA container toolkit; the root
`Dockerfile` builds the rest (the record's stack: torch 2.10.0+cu128 on a CUDA 13 runtime).
`node.sh` does everything inside the container: it checks the stack and the driver, downloads the data,
pre-seeds the attention kernel and goes offline (anonymous `get_kernel` can return 401, see the
record's README), then runs the mode asked for. Run inside tmux, from the directory holding the clone.

1. Build, then the smoke mode: `gpu_check.py` (the weighted CE kernel against the record's kernel and
   an fp64 reference, about a minute on one GPU) and one candidate run (about 7 minutes of compiling,
   then about 40 seconds of training):

```bash
git clone -b experiment/vocab-prune-h100 https://github.com/x4ahmed/modded-nanogpt.git && cd modded-nanogpt && sudo docker build -t modded-nanogpt . && sudo docker run --rm --name run --gpus all --ipc=host --ulimit memlock=-1:-1 -v $(pwd)/..:/work -w /work/modded-nanogpt modded-nanogpt bash records/track_1_short/2026-11-15_StratifiedSoftmaxTail/node.sh smoke 2>&1 | tee smoke.log
```

`--ipc=host` is required: the run shares a 316 MB buffer with a child process, more than Docker's
default 64 MB `/dev/shm`. The parent directory is mounted so the pool's second checkout and the kernel
cache outlive the container. Healthy output, in order: `stack ok`, `driver ... ok`, `GPU CHECK PASSED`,
`stratified softmax: steps 965-1106 run 3 chunks of 16384 rows at P=8192`, `CUDA graphs sealed`, and a
final `step:1194/1194 val_loss:... train_time:...`. Anything else: stop and fix before buying more time.
`sudo docker kill run` stops a run.

2. The pool, interleaved with the record's own commit on the same node, then the statistics (every log
   in `candidate/` and `baseline/` counts; the smoke run is a functional check and is not in the pool):

```bash
sudo docker run --rm --name run --gpus all --ipc=host --ulimit memlock=-1:-1 -v $(pwd)/..:/work -w /work/modded-nanogpt modded-nanogpt bash records/track_1_short/2026-11-15_StratifiedSoftmaxTail/node.sh pool 10 5 2>&1 | tee pool.log
```

## How many runs

The record's own pool was 18 runs plus 9 same-node baseline runs; run-to-run std was 0.00099 in loss and
0.12 s in time.

- Time (rule 4): with 10 + 5 runs the standard error of the difference is about 0.07 s, enough to
  resolve a saving of 0.3 s or more.
- Loss (rule 2, one-sided p < 0.01 that the mean is <= 3.28): if the mean lands at 3.2783 (one millinat
  above the record), 10 runs give t of about 5 and pass. At 3.2790, 10 runs are borderline and 18 pass.
  At 3.2795 or above no reasonable pool passes; the run then needs steps back, which costs time.

Suggested: the smoke run, then 10 + 5. If both rules pass, top up to 18 + 9 with the same script
(`run_pool.sh 8 4`) before a PR, to match the pool the last record was certified with. Never drop a run.

## Against the rules

| Rule | Status |
| --- | --- |
| 1. Data pipelines untouched | Yes. Only the training loss's candidate sets change. |
| 2. Mean val loss <= 3.28 at p < 0.01 | Open: decided by the pool. This changes the ML, so it is not waived. |
| 3. No extra compile flags | Yes. None added. |
| 4. Faster than the prior record on the same hardware | Open: decided by the interleaved baseline legs. |
| Readability | One new 150-line module, about 200 net lines added to 8 existing files, plus tests. |
| Loss buffer (must beat a simple step decrease) | On the port, 4-5x less loss than cutting steps for the same time. The record's write-up prices a millinat at 120-250 ms; a 0.8 s saving for one millinat would clear that, if the 8xH100 numbers hold. |

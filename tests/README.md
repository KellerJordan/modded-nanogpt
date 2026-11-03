CPU verification before GPU validation
=====================================

Run the complete suite with NumPy, PyTorch 2.10, tiktoken and pytest installed,
and a working C++ toolchain for CPU Inductor. The repository Docker image has
the toolchain. No CUDA or Triton imports are needed for these checks. The first
prefix-table build uses tiktoken's GPT-2 vocabulary download/cache.

```bash
python -m pytest -q
```

The NumPy-only subset needs no PyTorch or Triton:

```bash
python -m pytest tests/test_sampled_candidates.py tests/test_sampled_candidate_edges.py -q
```

`reference_sampled_candidates.py` freezes the original upstream builder at commit
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`. It does not inherit any production
constructor, reset or negative-selection code. Tests compare complete outputs,
dtypes and sweep state against that reference and check literal sentinel results.

Coverage includes exhaustive vocabularies of size 1–5 and batches of length 0–3,
2,000 seeded state transitions, every production P/T geometry, all eight ranks,
resets, sweep wrap and negative-selection fallback, full/empty candidate sets,
repeated and boundary token IDs, strided/read-only arrays, buffer ownership,
recovery after failed builds, and invalid shapes or capacities.

The trainer passes one-dimensional signed-int64 token IDs and an int64 prefix
table containing vocabulary IDs or -1. Differential parity is checked for that
contract. Broader dtype/error compatibility is not promised: for example, NumPy
1.x `take` rejects uint64 target indices that integer indexing accepts. A separate
test checks known uint64 results without claiming old-version error parity.

`test_sampled_candidate_staging.py` exercises the actual `fill_slot`, `submit` and
`take` methods with real CPU tensors, a real prep thread, and completed-upload
event stubs. It checks that copies survive reuse of builder scratch, the host
ring wraps correctly, and worker exceptions propagate and permit recovery.
It does not validate pinned allocation, CUDA event ordering or DMA.

Benchmark the full host build, including the validation guards:

```bash
python -m tests.benchmark_sampled_candidates --shard data/fineweb10B/fineweb_train_000001.bin
```

`reference_training_prep.py` also freezes upstream's BOS planner and n-gram hash.
The hash checks include all five actual train/validation shapes, signed int32
overflow, reserved history rows, input ownership, and no new compiled graphs
after priming under either grad mode. Planner tests compare every rank's spans
across geometry changes and cover exact budgets, exhaustion without consumption,
and the partial-to-full asynchronous index switch. As in upstream, planning uses
the currently available BOS snapshot; tests join the full scan when comparing
training trajectories rather than promising schedule-independent partial scans.

Benchmark both new production paths on actual shard tokens:

```bash
python -m tests.benchmark_training_prep --shard data/fineweb10B/fineweb_train_000001.bin
```

The hash benchmark includes the complete public function after compilation.
`BOS-plan` measures repeated plans with a cached prefix; `BOS-first-plan` includes
rebuilding it on each call, as required when the cap or BOS index changes. Both
report alternating-order median/min/max. Prefix cache bytes are also reported.
This benchmark accepts a shard prefix of at least 262144 tokens and uses up to
6 million tokens, so it does not require loading an entire shard into memory.

Coupled training regression
--------------------------

```bash
python -m tests.training_loop --output work/training-parity --steps 32 \
  --shard data/fineweb10B/fineweb_train_000001.bin
```

This launches two real Gloo processes. Each rank runs baseline and optimized
preprocessing through the same reduced decoder: 50304 vocabulary entries,
document-masked causal SDPA, QK normalization, squared-ReLU MLP, learned hashed
n-gram embeddings/signs, sampled/full softmax with the record's softcap, MTP,
and the actual GPT-2 prefix mapping. Each step performs autograd, gradient
reductions, and parameter updates. The shortened stages change token and
candidate counts, activate the slow ANVIL rail, and change row-update frequency.

Every step compares the packed tokens, document lengths, row hashes, losses,
complete gradients, model parameters/buffers, Adam moments, fast/slow rails,
equalizer energy, and row-RMS moments exactly. A midpoint checkpoint is serialized,
the candidate state is destroyed and restored, and training continues. The tail
averaging/decontraction cycle and a held-out full-softmax loss are compared too.
Per-rank JSON files contain the trajectory and update events. The automated
synthetic-corpus test runs seeds 0 and 387; a deliberately corrupted candidate
label must fail the actual forward-loss comparison.

This fixture validates the preprocessing changes through an actual training
loop. Its small decoder, CPU reference optimizer equations, FP32 master weights,
compact touched-row table, and shortened schedules differ from the record. It
does not execute the production FP8/FA3/Triton kernels, mantissa-split parameters,
NCCL sparse exchanges, pinned transfers, CUDA events, or CUDA graphs. The
experimental paired-Gram optimizer is not used in either branch.

Full record validation on an existing GPU host
---------------------------------------------

Prepare separate, clean baseline and candidate checkouts at different commits,
the repository dependencies, nine training shards and validation data, then run:

```bash
python -m tests.validate_record_training \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --data-path /path/containing/data/fineweb10B --output work/record-validation \
  --repeats 2
```

The runner requires exactly eight H100 GPUs and Torch 2.10. It alternates full
baseline/candidate `train_gpt.py` launches with matched seeds and unique compiler
caches, retains logs/revisions, rejects incomplete or non-finite final metrics,
and reports timing variance and validation loss. The OS page cache is not flushed.
It removes an inherited step-count override rather than shortening training.
These runs exercise the real record trainer and its final validation; accepting
a record still requires assessing time variance, loss and the record rules.

No GPU runs have been performed for this PR. Real pinned H2D/event ordering,
ring reuse under asynchronous execution, CUDA graph capture/replay, numerical
parity, and matched 8×H100 timing remain unverified. CPU component speedups do
not establish the end-to-end training gain.

Attention layout validation
---------------------------

The FA3 layout change adapts #396's copy removal to this baseline, including its
attention gate. Internal Q/K/V stay `[tokens, heads, width]` on both projection
paths; the public attention output remains `[1, tokens, model_dim]`.

```bash
python -m pytest tests/test_attention_layout.py -q
```

The harness executes the actual `Yarn`, `AttnArgs` and `CausalSelfAttention`
definitions. `reference_attention.py` freezes the old forward at `42c9714`.
The GPU-only QKV/RoPE and FA3 boundaries use explicit CPU mathematical references;
the production shape/control code is not rewritten by the test harness.

Checks include exact output/every-input-gradient parity over 160 combinations
of 64/128 QK and V widths, paired heads, auxiliary V, attention gates, partial key
offset, BF16/FP64 and both projection paths. Additional cases cover strided
inputs, zero values/XSA's denominator clamp, missing output gain, 16-token inputs,
padded document boundaries and actual no-grad evaluation. A reduced attention
fixture compares gradients, weights and complete Adam state over 32 updates.
It is separate from the coupled language-model fixture above.

A CPU profiler test observes three 3-D `select_backward` materializations in the
frozen path and zero in the new path. That test failed before the production
change and passed after it. It describes eager CPU autograd behavior, not a
measured compiled GPU speedup. The CPU boundaries do not emulate FP8 rounding,
actual FA3 numerics, GPU memory scheduling or CUDA graph capture.

On an existing H100 with the record's Torch 2.10/FA3 environment, run real-kernel
parity separately in eager and compiled modes, then run the full-record A/B:

```bash
python -m tests.validate_attention_cuda --output work/attention-eager.json
python -m tests.validate_attention_cuda --compile --tokens 16384 \
  --output work/attention-compiled.json
```

This gate uses real FP8 caches and the pinned FA3 build, enables deterministic
FA3 backward in both comparison arms, and requires bitwise output/gradient
equality. Each mode compares old versus new; it does not compare eager versus
compiled. The only compilation flags/limit are those already used by the
trainer. It makes no timing claim and does not capture CUDA graphs. It fails
explicitly without an H100. Only that preflight was exercised locally; CUDA
execution of this gate and the full-record runner remains outstanding.

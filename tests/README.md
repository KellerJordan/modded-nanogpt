CPU verification before GPU validation
=====================================

Run the complete suite with NumPy, PyTorch 2.10 and pytest installed:

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

GPU validation still needs real pinned H2D transfers and reader/writer events,
ring reuse under asynchronous execution, CUDA-graph capture/replay, loss/gradient
parity, and a matched 8×H100 baseline/candidate run. CPU timing does not establish
the end-to-end training gain.

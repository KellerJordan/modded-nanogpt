"""Stratified candidate sets for the widest sampled-softmax steps (see sampled_softmax.py).

What: on the steps where the record's candidate set is 24576 wide (the last third of stage 2: steps
965-1106 of the default run), the training CE normalizes over 8192 columns instead. A rank's 49152-row
microbatch is split into 3 chunks of 16384 rows, each with its own candidate set:
  - the exact part, 6144 ids at weight 1: every class that is a target in the chunk, filled up with
    the classes this rank has seen most often as targets so far in the run;
  - the tail, 2048 ids drawn uniformly without replacement from ALL the remaining classes, each at
    weight w = (remaining classes) / 2048.
The loss normalizes over Z_hat = sum_exact exp(z) + w * sum_tail exp(z), an unbiased estimate of the
full-vocabulary normalizer, so every class is still pushed down with the right strength on average.
The CE kernel gets one log weight per column (0 for exact columns, log w for tail columns).

Why: the logit GEMM, the CE kernel and both gradient GEMMs of these steps run at a third of the
record's width. The record's set at this point needs room for a 49152-row microbatch's distinct
targets (up to ~10.4k) plus enough unweighted negatives to approximate the normalizer; 16384 rows hold
at most ~5.3k distinct targets, and the weighted tail stands in for all the classes left out.

Why chunks of the loss only: the model's forward and backward still run once over the whole
microbatch; only the lm_head GEMMs and the CE kernel run per chunk.

Host-side and numpy-only, like CandidateBuilder: it runs on the same prep thread. The sets of a step
are returned concatenated, chunk after chunk.

STRATIFIED_SOFTMAX=0 turns it off: every step then runs the record's own sets (for same-node baselines).
"""
import os

import numpy as np

ENABLED = os.environ.get("STRATIFIED_SOFTMAX", "1") != "0"
# The record candidate count these sets replace (sampled_softmax.CANDIDATES_BY_STAGE: stage 2's last third).
REPLACED_COUNT = 24576
EXACT_IDS = 6144
TAIL_IDS = 2048
WIDTH = EXACT_IDS + TAIL_IDS  # 8192: a legal CE kernel width (a multiple of 256 * 8)
CHUNK_ROWS = 16384
# A chunk with more distinct targets than EXACT_IDS grows its exact part and shrinks its tail (still
# unbiased); the tail never goes below this.
MIN_TAIL_IDS = 256
# The tail draws have their own generator, seeded per rank: no global RNG stream is touched.
TAIL_SEED = 20011


def stratified_counts(record_counts: list[int]) -> list[int]:
    """The run's candidate count per step: WIDTH where the record's is REPLACED_COUNT."""
    if not ENABLED:
        return list(record_counts)
    assert WIDTH not in record_counts, f"{WIDTH} is already a record candidate count"
    return [WIDTH if count == REPLACED_COUNT else count for count in record_counts]


def weighted_kernel_source(source: str) -> str:
    """The CE kernel source (perf/kernels/cross_entropy.py) with a per-column log weight added inside
    every exp(): the forward sum, the dense gradient and the sparse target correction. With col_logw
    all zero it computes what the unweighted kernel computes."""
    swaps = [
        ("    const float* __restrict__ prefix_weight_ptr,\n",
         "    const float* __restrict__ prefix_weight_ptr,\n    const float* __restrict__ col_logw,\n"),
        ("thread_sum += __expf(tmp - A);", "thread_sum += __expf(tmp - A + col_logw[idx + k]);"),
        ("          float sigmoid_u = __half2float(sigmoid_us[j + t]);\n"
         "          float z = A * sigmoid_u;\n"
         "          float p = __expf(z - lse);\n",
         "          float sigmoid_u = __half2float(sigmoid_us[j + t]);\n"
         "          float z = A * sigmoid_u;\n"
         "          float p = __expf(z - lse + col_logw[idx + j + t]);\n"),
        ("    float sigmoid_u = __half2float(smem[target]);\n"
         "    float z = A * sigmoid_u;\n"
         "    float p = __expf(z - lse);\n",
         "    float sigmoid_u = __half2float(smem[target]);\n"
         "    float z = A * sigmoid_u;\n"
         "    float p = __expf(z - lse + col_logw[target]);\n"),
    ]
    for old, new in swaps:
        assert source.count(old) == 1, f"CE kernel source changed: {old!r}"
        source = source.replace(old, new)
    return source


class StratifiedBuilder:
    """Host build of a step's stratified sets. Scratch is reused, so one build runs at a time."""

    def __init__(self, vocab_size: int, max_rows: int, rank: int):
        assert max_rows % CHUNK_ROWS == 0
        self.vocab_size = V = vocab_size
        self.rank = rank
        max_chunks = max_rows // CHUNK_ROWS
        self.counts = np.zeros(V, dtype=np.int64)       # this rank's targets so far: the ranking's data
        self.mark = np.zeros(V + 1, dtype=bool)         # the chunk's exact part
        self.is_tail = np.zeros(V, dtype=bool)
        # class -> position in the chunk's set; entry V stays -1, the "no prefix target" sentinel.
        self.pos = np.empty(V + 1, dtype=np.int64)
        self.pos[V] = -1
        self.arange = np.arange(WIDTH, dtype=np.int64)
        self.candidates = np.empty(max_chunks * WIDTH, dtype=np.int64)
        self.col_logw = np.empty(max_chunks * WIDTH, dtype=np.float32)
        self.target_pos, self.prefix_pos = np.empty((2, max_rows), dtype=np.int64)
        self.reset()

    def reset(self):
        self.counts.fill(0)
        self.rng = np.random.Generator(np.random.PCG64([TAIL_SEED, self.rank]))
        self.grown_chunks = 0  # chunks whose distinct targets exceeded EXACT_IDS

    def observe(self, targets: np.ndarray):
        """Count a step's targets. Called for every sampled step, after its set is built."""
        self.counts += np.bincount(targets, minlength=self.vocab_size)

    def build(self, targets: np.ndarray, prefix_table: np.ndarray | None):
        """(candidates, target_pos, prefix_pos, col_logw) for one microbatch; views into scratch.

        candidates and col_logw hold the chunks' sets back to back, WIDTH entries each, ids ascending
        within a set. target_pos / prefix_pos are positions in the row's own chunk's set; a prefix
        target outside the exact part is dropped (-1), as the record drops one outside its set.
        prefix_pos is None when there is no prefix table yet (warmup).
        """
        V, mark, is_tail, pos = self.vocab_size, self.mark, self.is_tail, self.pos
        T = targets.shape[0]
        assert T % CHUNK_ROWS == 0, f"{T} rows is not a whole number of {CHUNK_ROWS}-row chunks"
        chunks = T // CHUNK_ROWS
        # Classes by how often this rank has seen them as targets, most frequent first (before this step).
        order = np.argsort(-self.counts, kind="stable")
        for k in range(chunks):
            rows = slice(k * CHUNK_ROWS, (k + 1) * CHUNK_ROWS)
            cols = slice(k * WIDTH, (k + 1) * WIDTH)
            chunk_targets = targets[rows]
            mark.fill(False)
            mark[chunk_targets] = True
            num_targets = int(np.count_nonzero(mark[:V]))
            num_exact = max(EXACT_IDS, num_targets)
            num_tail = WIDTH - num_exact
            assert num_tail >= MIN_TAIL_IDS, f"{num_targets} distinct targets leave a tail of {num_tail}"
            self.grown_chunks += num_targets > EXACT_IDS
            if num_exact != num_targets:
                mark[order[~mark[order]][:num_exact - num_targets]] = True
            rest = np.flatnonzero(~mark[:V])
            tail = self.rng.choice(rest, size=num_tail, replace=False)
            is_tail.fill(False)
            is_tail[tail] = True
            candidates = np.flatnonzero(mark[:V] | is_tail)  # ascending
            self.candidates[cols] = candidates
            self.col_logw[cols] = np.where(is_tail[candidates], np.float32(np.log(rest.size / num_tail)),
                                           np.float32(0.0))
            pos[:V].fill(-1)
            pos[candidates] = self.arange
            np.take(pos, chunk_targets, out=self.target_pos[rows])
            if prefix_table is not None:
                pos[tail] = -1  # prefix targets stay exact: a tail column is no prefix target
                np.take(pos, np.take(prefix_table, chunk_targets), out=self.prefix_pos[rows])
        prefix_pos = None if prefix_table is None else self.prefix_pos[:T]
        return self.candidates[:chunks * WIDTH], self.target_pos[:T], prefix_pos, self.col_logw[:chunks * WIDTH]

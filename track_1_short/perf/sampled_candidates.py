"""Host-only sampled-softmax candidate construction and reusable position buffers."""
import math

import numpy as np

# Coprime with the vocabulary, so the stride sweep is a permutation (record #360).
NEGATIVE_STRIDE = 20011


class CandidateBuilder:
    """Host half of the candidate build: numpy only, no device work, so it can run on a worker thread.

    Scratch arrays are allocated once and reused, so at most one build may run at a time.
    """
    def __init__(self, vocab_size: int, max_candidates: int, max_rows: int):
        assert math.gcd(NEGATIVE_STRIDE, vocab_size) == 1
        self.vocab_size = V = vocab_size
        self.mark = np.zeros(V + 1, dtype=bool)          # candidate bitmap
        self.mark_u8 = self.mark.view(np.uint8)
        # class -> position in C. Entry V is a sentinel that stays -1, so a prefix target of -1
        # ("no prefix") indexes it and reaches the kernel as its no-op.
        self.pos = np.empty(V + 1, dtype=np.int64)
        self.pos[V] = -1
        self.arange = np.arange(max_candidates, dtype=np.int64)
        self.prefix_targets, self.target_pos, self.prefix_pos = np.empty((3, max_rows), dtype=np.int64)
        # The stride sweep, doubled so any window of it is one contiguous slice.
        sweep = (NEGATIVE_STRIDE * np.arange(V, dtype=np.int64)) % V
        self.sweep = np.concatenate((sweep, sweep))
        self.sweep_offset = 0

    def reset(self, rank: int, world_size: int):
        # Ranks start the sweep at staggered offsets, so their negatives differ.
        self.sweep_offset = (self.vocab_size * rank) // world_size

    def _negatives(self, need: int) -> np.ndarray:
        """`need` distinct unmarked classes: the next window of the stride sweep."""
        V, mark = self.vocab_size, self.mark
        draw = min(V, int(need * 1.7) + 256)
        idx = self.sweep[self.sweep_offset:self.sweep_offset + draw]
        self.sweep_offset = (self.sweep_offset + draw) % V
        idx = idx[~mark[idx]]
        if idx.size >= need:
            return idx[:need]
        free = np.flatnonzero(~mark[:V])
        assert free.size >= need, f"only {free.size} free classes for {need}"
        return free[::max(1, free.size // need)][:need]

    def build(self, P: int, targets: np.ndarray, prefix_table: np.ndarray | None):
        """(candidates, target_pos, prefix_pos) for one microbatch; views into scratch.

        Targets are a one-dimensional integer array of vocabulary ids, no longer than max_rows;
        prefix_table holds vocabulary ids or -1 (no prefix). P fits both max_candidates and the
        vocabulary, and may be zero for an empty batch.
        prefix_pos is None when there is no prefix table yet (warmup), leaving the device buffer at -1.
        """
        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        V, mark, pos, T = self.vocab_size, self.mark, self.pos, targets.shape[0]
        if T > self.target_pos.size:
            raise ValueError(f"{T} targets exceed scratch capacity {self.target_pos.size}")
        if P < 0 or P > self.arange.size or P > V:
            raise ValueError(f"candidate count {P} is outside [0, {min(self.arange.size, V)}]")
        mark.fill(False)
        mark[targets] = True
        num_targets = int(np.count_nonzero(mark[:V]))
        assert num_targets <= P, f"{num_targets} distinct targets > P={P}"
        if num_targets != P:
            self.mark_u8[self._negatives(P - num_targets)] = 1
        candidates = np.flatnonzero(mark[:V])  # ascending
        pos[:V].fill(-1)
        pos[candidates] = self.arange[:P]
        target_pos = self.target_pos[:T]
        # Integer indexing gathers faster than np.take(..., out=...) here. Assign into the same
        # scratch views, retaining bounds checks and -1's sentinel lookup for missing prefixes.
        target_pos[:] = pos[targets]
        if prefix_table is None:
            return candidates, target_pos, None
        prefix_targets, prefix_pos = self.prefix_targets[:T], self.prefix_pos[:T]
        prefix_targets[:] = prefix_table[targets]
        prefix_pos[:] = pos[prefix_targets]
        return candidates, target_pos, prefix_pos

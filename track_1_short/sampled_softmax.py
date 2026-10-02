"""Sampled softmax for the training loss: early steps normalize over a shared candidate set, not the vocabulary.

What: through the batch-24 stage (stage 2), each rank's training CE is a softmax over
P candidate classes instead of all 50304. The candidate set C is shared by every token of the rank's
microbatch ("shared negatives"): every class that is a target somewhere in the microbatch, plus enough
negatives to reach P, taken from a fixed stride permutation of the vocabulary. The kernel sees C as a
P-wide vocabulary: logits are x @ lm_head[C].T, and each target (and prefix target) is replaced by its
position in C. A prefix target that is not in C is dropped for that token (position -1, the kernel's
no-op), so prefix targets never force classes into C. Validation always uses the full softmax.

Why: early in training the full vocabulary's logit traffic (the [T, 50304] GEMM, CE pass and two
gradient GEMMs) buys resolution the model cannot use yet. With P = 10k-25k those three GEMMs and the CE
kernel shrink by 2-5x. The loss is biased (the normalizer misses the non-candidate mass), which is why
the count ramps up through the last main stage and the run finishes on the full softmax.

The negatives are the next window of k * NEGATIVE_STRIDE mod V (a permutation, since the stride is
coprime to V), each rank starting at its own offset, so across steps every class is visited uniformly
and no draw needs an RNG. The gradient for lm_head is dense over the vocabulary with zeros outside C.

Schedule (record #360's, per TRAINING_STAGES index): P = 10240 through stages 0 and 1, then stage 2
ramps 14336 -> 14336 -> 24576 in equal thirds; the batch taper (stage 3) and the extension run the full
softmax. For the default 1060 + 52 + 20 steps: P = 10240 on steps 0-642, 14336 on 643-912, 24576 on
913-1047, and the full softmax from step 1048 on.

Requires exactly one microbatch per step: see the race contract in perf/sampled_softmax_overlap.py.

Provenance: record #360 (ANVIL2).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from track_1_short.perf.kernels.transpose import transpose_copy
from track_1_short.perf.sampled_softmax_overlap import CandidateStaging, StagedCandidates
from track_1_short.schedule import TrainingSchedule

# Candidate counts per stage index (TRAINING_STAGES order). A stage with several counts splits its steps
# into equal parts, one count each. Stages not listed run the full softmax. Every count must be a
# multiple of the CE kernel's vector width (256 threads * 8) and cover the microbatch's distinct targets
# (~11k at batch 24). Stage 2 ramps rather than steps: "the count has to reach full-softmax resolution
# before the batch taper" (record #360).
CANDIDATES_BY_STAGE = {
    0: (10240,),
    1: (10240,),
    2: (14336, 14336, 24576),
}
ALL_CANDIDATE_COUNTS = tuple(sorted({p for counts in CANDIDATES_BY_STAGE.values() for p in counts}))
# Coprime with the vocabulary, so k * stride mod V is a permutation (record #360).
NEGATIVE_STRIDE = 20011


def _stage_index(schedule: TrainingSchedule, step: int) -> int:
    """Index of `step`'s stage; the final step (total_steps, validation only) counts as the last stage."""
    return next((i for i, (_, end) in enumerate(schedule.boundaries) if step < end), len(schedule.boundaries) - 1)

def candidate_count_at(schedule: TrainingSchedule, step: int) -> int:
    """Candidate count P for `step`; 0 means the full softmax."""
    stage_index = _stage_index(schedule, step)
    counts = CANDIDATES_BY_STAGE.get(stage_index)
    if counts is None:
        return 0
    start, end = schedule.boundaries[stage_index]
    return counts[(step - start) * len(counts) // (end - start)]


@dataclass(slots=True)
class SampledLoss:
    """What the training loss reads to run over a candidate set; one object per P, reused every step."""
    rows: Tensor        # (P, D) e4m3: lm_head rows of the candidates, ascending class id
    rows_t: Tensor      # (D, P) e4m3: the same rows transposed, the backward's operand
    target_pos: Tensor  # (T_max,) int64: position in C of each token's target (first T rows are live)
    prefix_pos: Tensor  # (T_max,) int64: position in C of each token's prefix target, -1 if absent
    vocab_pos: Tensor   # (V,) int32: position in C of each class, -1 if not a candidate


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

        prefix_pos is None when there is no prefix table yet (warmup), leaving the device buffer at -1.
        """
        V, mark, pos, T = self.vocab_size, self.mark, self.pos, targets.shape[0]
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
        np.take(pos, targets, out=target_pos)
        if prefix_table is None:
            return candidates, target_pos, None
        prefix_targets, prefix_pos = self.prefix_targets[:T], self.prefix_pos[:T]
        np.take(prefix_table, targets, out=prefix_targets)
        np.take(pos, prefix_targets, out=prefix_pos)
        return candidates, target_pos, prefix_pos


class SampledSoftmax:
    """Per-step candidate sets for the training loss: host build, upload, and the lm_head row gather.

    Per step: upload() then gather() (before the forward) -> the forward reads the returned SampledLoss ->
    mark_readers_done() (after the backward is enqueued) -> optionally prefetch() the next step.
    """
    def __init__(self, schedule: TrainingSchedule, vocab_size: int, model_dim: int, max_rows: int,
                 rank: int, world_size: int, device: torch.device):
        # The loss keeps no copy of the gathered rows or the uploaded positions, so the next step's
        # upload/gather must not be enqueued before this one's backward: one microbatch per step
        # (distributed.py requires the 8-GPU world that guarantees it).
        self.schedule = schedule
        self.rank, self.world_size = rank, world_size
        self.counts = [candidate_count_at(schedule, s) for s in range(schedule.total_steps + 1)]
        p_values = sorted(set(self.counts) - {0})
        assert p_values
        max_p = max(p_values)
        self.builder = CandidateBuilder(vocab_size, max_p, max_rows)
        self.staging = CandidateStaging(max_candidates=max_p, max_rows=max_rows, device=device)
        self.prefix_table = None  # host copy of model.prefix_table, set once it is built

        # Device buffers, one identity for the process's life (the compiled graph reads them).
        # prefix_pos starts at -1, the kernel's no-op, for steps that upload no prefix positions.
        self.candidates = torch.zeros(max_p, dtype=torch.int64, device=device)
        target_pos = torch.zeros(max_rows, dtype=torch.int64, device=device)
        prefix_pos = torch.full((max_rows,), -1, dtype=torch.int64, device=device)
        vocab_pos = torch.empty(vocab_size, dtype=torch.int32, device=device)
        self.arange = torch.arange(max_p, dtype=torch.int32, device=device)
        # A (D, P) slice of a wider slab is not a valid _scaled_mm operand, so each P has its own pair.
        e4m3 = torch.float8_e4m3fn
        self.loss_inputs = {
            p: SampledLoss(rows=torch.zeros(p, model_dim, dtype=e4m3, device=device),
                           rows_t=torch.zeros(model_dim, p, dtype=e4m3, device=device),
                           target_pos=target_pos, prefix_pos=prefix_pos, vocab_pos=vocab_pos)
            for p in p_values
        }
        self.reset()

    def reset(self):
        self.builder.reset(self.rank, self.world_size)
        self.staging.reset()
        self.uploaded: tuple[int, StagedCandidates] | None = None  # (step, slot) between upload and gather

    def set_prefix_table(self, prefix_table: np.ndarray):
        self.prefix_table = prefix_table

    def count_change_steps(self) -> list[int]:
        """Steps whose P differs from the previous step's: each P is its own compiled graph."""
        return [s for s in range(1, len(self.counts)) if self.counts[s] != self.counts[s - 1]]

    def assert_warmup_covers(self, warmup_steps):
        """Every (P, stage) pair the run reaches must be compiled during warmup, not on the clock."""
        key = lambda s: (self.counts[s], _stage_index(self.schedule, s))
        missing = {key(s) for s in range(self.schedule.total_steps)} - {key(s) for s in warmup_steps}
        assert not missing, f"warmup never compiles (P, stage) {sorted(missing)}"

    def _stage(self, step: int, targets: np.ndarray) -> StagedCandidates:
        """Build step `step`'s candidate set on the host and copy it into a pinned slot."""
        P = self.counts[step]
        candidates, target_pos, prefix_pos = self.builder.build(P, targets, self.prefix_table)
        return self.staging.fill_slot(candidates, target_pos, prefix_pos)

    def prefetch(self, step: int, targets_cpu: Tensor):
        """Start step `step`'s host build on the prep thread (joined by upload(step))."""
        if self.counts[step]:
            self.staging.submit(step, self._stage, step, targets_cpu.numpy())

    def upload(self, step: int, targets_cpu: Tensor):
        """Start uploading step `step`'s candidate set (async, on the copy stream); gather(step) completes
        it. A no-op on full-softmax steps."""
        assert self.uploaded is None, "the previous upload was never gathered"
        P = self.counts[step]
        if not P:
            return
        staged = self.staging.take(step)
        if staged is None:
            staged = self._stage(step, targets_cpu.numpy())
        inputs = self.loss_inputs[P]
        self.staging.upload(staged, self.candidates, inputs.target_pos, inputs.prefix_pos)
        self.uploaded = (step, staged)

    def gather(self, step: int, lm_head_f8_col: Tensor) -> SampledLoss | None:
        """Step `step`'s candidate set with its lm_head rows gathered; None on full-softmax steps.

        Must run after the last lm_head fp8 refresh, so the gathered rows are the current weights.
        """
        P = self.counts[step]
        if not P:
            return None
        uploaded_step, staged = self.uploaded
        self.uploaded = None
        assert uploaded_step == step, f"uploaded step {uploaded_step}, gathering step {step}"
        inputs = self.loss_inputs[P]
        self.staging.wait_uploaded(staged)
        self._gather_rows(lm_head_f8_col, P, inputs)
        return inputs

    def mark_readers_done(self):
        self.staging.mark_readers_done()

    def _gather_rows(self, lm_head_f8_col: Tensor, P: int, inputs: SampledLoss):
        """Copy the candidates' rows out of the fp8 lm_head cache and rebuild C's inverse map."""
        cand = self.candidates[:P]
        # The cache is (D, V) with strides (1, D), so its transpose is a contiguous (V, D) row table;
        # the uint8 views move identical bytes.
        torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, cand, out=inputs.rows.view(torch.uint8))
        transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])

"""Overlap for the sampled-softmax candidate set (see sampled_softmax.py): a host prep thread and a side copy stream.

What it replaces: building step s's candidate set on the main thread right before its forward, and
uploading the index vectors on the compute stream. Why it is faster: (1) the host build for step s+1
(a bitmap over the vocabulary plus a few np.take over the batch) runs on a worker thread while the
main thread is still launching step s's backward and optimizer, so only the join is on the critical path;
(2) the three H2D copies run on a private stream, so they overlap the tail of step s's work instead of
queueing behind it on the compute stream.

Race contract (the ordering on the device buffers the loss reads):
  - WAR: step s's upload waits on an event the compute stream records after step s-1's last reader of
    those buffers (its backward) is enqueued -- mark_readers_done().
  - RAW: the compute stream waits on the copy stream's event before its first reader (the row gather).
  - The loss keeps no copy of the gathered rows or positions, which is legal only because exactly one
    microbatch runs per step: its backward is enqueued before the next step's upload and gather.
  - Pinned slots rotate through a ring; before the host rewrites a slot it waits for that slot's
    previous H2D to finish.
  - The prep thread runs pure host numpy work on frozen inputs, and at most one build is in flight, so
    the builder's scratch arrays have one writer.

Provenance: record #360 (ANVIL2): the `_hprep` pool, `sns_war_record` / `sns_upload_async`.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Deeper than the host's run-ahead of the device, so a slot is rarely still being copied when reused.
PINNED_SLOTS = 16


@dataclass(slots=True)
class PinnedSlot:
    candidates: Tensor
    target_pos: Tensor
    prefix_pos: Tensor
    uploaded: torch.cuda.Event  # recorded on the copy stream after this slot's H2Ds


@dataclass(slots=True)
class StagedCandidates:
    """A pinned slot plus how many entries of each vector are live (prefix count 0: skip that upload)."""
    slot: PinnedSlot
    num_candidates: int
    num_rows: int
    num_prefix_rows: int


class CandidateStaging:
    def __init__(self, max_candidates: int, max_rows: int, device: torch.device):
        pinned = lambda n: torch.empty(n, dtype=torch.int64, pin_memory=True)
        self.slots = [PinnedSlot(pinned(max_candidates), pinned(max_rows), pinned(max_rows), torch.cuda.Event())
                      for _ in range(PINNED_SLOTS)]
        self.next_slot = 0
        # One worker; the no-op submit starts its thread here, off the clock.
        self.prep = ThreadPoolExecutor(max_workers=1)
        self.prep.submit(int).result()
        self.pending: tuple[int, Future] | None = None
        self.copy_stream = torch.cuda.Stream(device=device)
        self.readers_done = torch.cuda.Event()
        self.readers_done.record()  # vacuous first stamp

    def reset(self):
        assert self.pending is None

    def fill_slot(self, candidates: np.ndarray, target_pos: np.ndarray, prefix_pos: np.ndarray | None) -> StagedCandidates:
        """Copy host arrays into the next pinned slot (may run on the prep thread)."""
        slot = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % PINNED_SLOTS
        slot.uploaded.synchronize()
        slot.candidates[:candidates.shape[0]].copy_(torch.from_numpy(candidates))
        slot.target_pos[:target_pos.shape[0]].copy_(torch.from_numpy(target_pos))
        num_prefix_rows = 0
        if prefix_pos is not None:
            num_prefix_rows = prefix_pos.shape[0]
            slot.prefix_pos[:num_prefix_rows].copy_(torch.from_numpy(prefix_pos))
        return StagedCandidates(slot, candidates.shape[0], target_pos.shape[0], num_prefix_rows)

    def submit(self, step: int, build, *args):
        """Run `build(*args)` on the prep thread; take(step) joins it."""
        assert self.pending is None
        self.pending = (step, self.prep.submit(build, *args))

    def take(self, step: int) -> StagedCandidates | None:
        """The prefetched build for `step`, or None if nothing was prefetched."""
        if self.pending is None:
            return None
        pending_step, future = self.pending
        self.pending = None
        assert pending_step == step, f"prefetched step {pending_step}, loading step {step}"
        return future.result()

    def upload(self, staged: StagedCandidates, candidates: Tensor, target_pos: Tensor, prefix_pos: Tensor):
        """Async H2Ds on the copy stream, behind the previous step's last reader."""
        slot = staged.slot
        self.copy_stream.wait_event(self.readers_done)
        with torch.cuda.stream(self.copy_stream):
            candidates[:staged.num_candidates].copy_(slot.candidates[:staged.num_candidates], non_blocking=True)
            target_pos[:staged.num_rows].copy_(slot.target_pos[:staged.num_rows], non_blocking=True)
            if staged.num_prefix_rows:
                prefix_pos[:staged.num_prefix_rows].copy_(slot.prefix_pos[:staged.num_prefix_rows], non_blocking=True)
        slot.uploaded.record(self.copy_stream)

    def wait_uploaded(self, staged: StagedCandidates):
        torch.cuda.current_stream().wait_event(staged.slot.uploaded)

    def mark_readers_done(self):
        self.readers_done.record()

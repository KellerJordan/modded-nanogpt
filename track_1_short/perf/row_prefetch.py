"""Overlap for the row pulls of the two tables that change only at their update events -- the n-gram
table (ngram_table.py) and value_embeds (perf/value_embed_pull.py), which share the n-gram table's
cycles: lookahead batches, a host prep thread, pinned uploads, and pulls spread across the event step so
no stage waits.

What it replaces: pulling a cycle's rows synchronously -- building its sorted-unique want lists on the
main thread and waiting on each exchange before the next forward (doing that every step, instead of once
per cycle, costs +4.7 s over a run).

Why it is faster:
  1. Building the want lists for the cycle after event E needs that cycle's batches, E+1 .. E+k
     (k <= MAX_CYCLE_STEPS), before step E runs. The loader therefore runs ahead (ScheduledBatches
     holds the future batches), and both tables' lists are built in one job on a prep thread: posted at
     the end of a step once E is at most PREP_SLACK_STEPS away (in practice at the end of the previous
     event step), joined at the top of E. Only the join is on the critical path.
  2. The pulls are split across step E, each stage where it hides under work already queued:
       top of E             upload the want lists (pinned rings, H2D on the compute stream); count
                            exchanges (gloo, CPU tensors, async)
       after E's forward    wait for the counts (long done); id all_to_alls (NCCL stream, under the backward)
       E's optimizer step   at value_embeds' place in the scatter order: E's gradient exchanges (NCCL,
                            queued behind the bank reduce-scatters);
                            at value_embeds' place in the work order: value_embeds' shard merge + Adam, the
                            n-gram per-row merge + Adam, then serve the post-update rows (row all_to_alls,
                            async: queued ahead of the mlp_bank gather);
                            Phase 3, after the lm_head gather: the n-gram cache write
       top of E+1           value_embeds' replica write, with the deferred bank gathers
                            (perf/deferred_gathers.py)
  3. Steps E+1 .. E+k do no exchange at all: one searchsorted maps their n-gram row ids to cache slots,
     and the value_embeds replica already holds their rows.

Lookahead window: at the end of step s the next event E is at most MAX_CYCLE_STEPS away and its prep
reads batches up to E + MAX_CYCLE_STEPS, so the loader runs at most PREP_SLACK_STEPS + MAX_CYCLE_STEPS
batches ahead of the step being trained.

Ordering and race contracts:
  - Collectives are issued only by the main thread, in the same order on every rank (every rank runs
    the same step logic over the same step sequence), value_embeds' before the n-gram table's at each
    stage. The prep thread runs pure numpy on arrays the loader created and never mutates, and at most
    one build is in flight.
  - The n-gram cache is overwritten only by a land: in Phase 3 of an event's optimizer step -- the
    event's forward and backward, the last readers of the old cycle, are queued before it on the same
    stream -- or by an eval pull, after which the next training step re-lands the live cycle
    (NgramTable.restore_cache) before its forward. An eval never writes the value_embeds replica.
  - The pulls serve rows after the event's Adam updates, on the same stream, so they are post-update.
  - A started land is completed before anything reads the rows it writes: the n-gram cache in Phase 3
    of the same optimizer step, the value_embeds replica by DeferredGathers.flush before the next forward.
  - A pinned upload slot is rewritten only after its previous H2D retired (one event per slot).
  - Every step of a cycle indexes the same n-gram want list, restored or not, so the gradient slots of
    a cycle all refer to one row order.

Deviations from record #360 (the rest follows its schedule and kernels):
  - The event's gradients reuse the pulls' routes: the gradient rows of a cycle are exactly the rows
    it pulled, so the counts and ids exchanged for the pull already say who sends what to whom. #360
    re-ran the count and id exchanges for the gradient (grad_start / grad_share): fewer collectives here.
  - The n-gram owner merges each row's gradient entries through a 42 MB row -> entry claim map into an
    [entries, 768] buffer (perf/kernels/ngram_adam.py), where #360 scatters into a dense 16.2 GB
    [V/world, 768] gradient buffer: the same launches and atomics, without the memory this run's
    peak cannot spare. The lazy second-moment replay, its pass over the next cycle's own rows, and the
    fused Triton row update are #360's. The n-gram table is not in the checkpoint, so #360's dense replay
    before a checkpointing validation (bgpull_dense_gather) has no counterpart.
  - After an eval pull the whole live cycle is re-landed along its existing routes, instead of
    #360's fresh synchronous fill of the cycle's remaining steps. Only an intermediate validation
    (config.val_loss_every > 0) evaluates mid-cycle: the record run validates only at the end, as #360.
  - No fixed-address row-slot ring or pointer fingerprints. The CUDA-graph side of #360's pull (the
    graph sink ring that holds each step's n-gram gradient until the event) lives in
    perf/cuda_graphs/step_graphs.py (StepGraphs.hold_ngram_grad).
  - Its own prep thread (the sampled-softmax build has another), where #360 shares one.
  - Warmup runs this same armed path over the warmup steps (with the tables updating on every Adam
    step, so every warmup cycle fits the cache), where #360 warms an unarmed synchronous variant.

Provenance: record #360 (ANVIL2): `_la_fifo` / `_bgpull_fill`, `_la_window`, `_hprep`, `_BigramPull`,
`_vebg_want_build`, `bgpull_request` / `bgpull_post` / `vepull_request`, PREP_SLACK_STEPS,
`_BGPULL_PIN_RING`.
"""
import bisect
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from track_1_short.data import Batch, ScheduledBatches
from track_1_short.ngram_table import MAX_CYCLE_STEPS, NgramTable, sorted_unique_rows
from track_1_short.perf.value_embed_pull import ValueEmbedPull
from track_1_short.sharded_rows import RowPull

# The prep for event E is posted once E is at most this many steps away (record #360: K=1 left the
# GPU idle 10-21 ms per event waiting for the join; 4 hides it).
PREP_SLACK_STEPS = 4
# Deeper than the host's run-ahead of the device, so a slot's previous upload is long done when reused.
PINNED_SLOTS = 4


class PinnedUploadRing:
    """Want lists go to the device through pinned slots, each with the event of its last H2D."""

    def __init__(self, max_rows: int, device: torch.device):
        self.device = device
        self.slots = [(torch.empty(max_rows, dtype=torch.int32, pin_memory=True), torch.cuda.Event())
                      for _ in range(PINNED_SLOTS)]
        self.next_slot = 0

    def upload(self, rows: np.ndarray) -> Tensor:
        buf, uploaded = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % PINNED_SLOTS
        uploaded.synchronize()
        n = rows.shape[0]
        buf[:n].copy_(torch.from_numpy(rows))
        out = torch.empty(n, dtype=torch.int32, device=self.device)
        out.copy_(buf[:n], non_blocking=True)
        uploaded.record()
        return out


class PrefetchStaging:
    """What outlives a run: one pinned ring per table and the prep thread. Created once in main(), off the clock."""

    def __init__(self, max_ngram_rows: int, max_value_embed_rows: int, device: torch.device):
        self.ngram_ring = PinnedUploadRing(max_ngram_rows, device)
        self.value_embed_ring = PinnedUploadRing(max_value_embed_rows, device)
        self.prep = ThreadPoolExecutor(max_workers=1)
        self.prep.submit(int).result()  # starts the thread now


def cycle_wants(batches: list[Batch], value_embed_rows: Callable[[list[np.ndarray]], np.ndarray]):
    """The prep job: a cycle's n-gram want list and value_embeds want list. Pure numpy."""
    return (sorted_unique_rows([b.ngram_ids_cpu for b in batches]),
            value_embed_rows([b.inputs_cpu for b in batches]))


@dataclass(slots=True)
class NextPull:
    """The pulls requested at an event step, for the cycle at `cycle` (positions in the step sequence)."""
    event: int
    ngram: RowPull
    value_embeds: RowPull
    cycle: list[int]


class RowPrefetch:
    """One run's row pulls (warmup and the timed run each get their own) over `steps`, with the tables
    updating on the steps `is_update` names. The step function calls, in order: forward_slots,
    after_forward, (backward, ngram.accumulate_grad), post_prep, then the optimizer runs
    optimizer_event's hooks."""

    def __init__(self, ngram: NgramTable, value_embeds: ValueEmbedPull, batches: ScheduledBatches,
                 steps: Sequence[int], is_update: Callable[[int], bool], staging: PrefetchStaging):
        self.ngram = ngram
        self.value_embeds = value_embeds
        self.batches = batches
        self.steps = list(steps)
        self.position = {step: i for i, step in enumerate(self.steps)}
        self.is_update = is_update
        self.staging = staging
        self.update_positions = [i for i, step in enumerate(self.steps) if is_update(step)]
        cycle_ends = [*self.update_positions, len(self.steps) - 1]
        cycle_lengths = np.diff([-1, *cycle_ends])
        assert cycle_lengths.max() <= MAX_CYCLE_STEPS, f"a cycle of {cycle_lengths.max()} steps overflows the cache"
        self.prep_pending: tuple[int, Future] | None = None  # (event position, want lists)
        self.next_pull: NextPull | None = None
        self.live_cycle: list[int] | None = None

    def _cycle_from(self, i: int) -> list[int]:
        """Positions i .. the next update at or after i (or the end of the run)."""
        k = bisect.bisect_left(self.update_positions, i)
        last = self.update_positions[k] if k < len(self.update_positions) else len(self.steps) - 1
        return list(range(i, last + 1))

    def _batches(self, cycle: list[int]) -> list[Batch]:
        return [self.batches.peek(self.steps[j]) for j in cycle]

    def forward_slots(self, step: int, row_ids: Tensor) -> Tensor:
        """Top of the step: make sure the tables hold this step's cycle, start the next cycle's pulls at
        an event, and return this step's n-gram cache slots."""
        i = self.position[step]
        if self.ngram.live is None:  # first step of the run: nothing to hide the pulls under
            assert self.value_embeds.live is None
            cycle = self._cycle_from(i)
            ngram_want, value_embed_want = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)
            self.ngram.start_cycle(self.ngram.fill(ngram_want))
            self.value_embeds.fill(value_embed_want)
            self.live_cycle = cycle
        elif self.ngram.cached is not self.ngram.live:  # an eval pull overwrote the cache
            self.ngram.restore_cache()
        assert i in self.live_cycle, f"step {step} is outside the cycle the tables hold"
        if self.is_update(step):
            self._request(i)
        return self.ngram.slots(self.ngram.live, row_ids)

    def _request(self, i: int):
        cycle = self._cycle_from(i + 1)
        if not cycle:
            return  # the run's last step
        if self.prep_pending is not None:
            event, future = self.prep_pending
            self.prep_pending = None
            assert event == i, f"prep built for position {event}, consumed at {i}"
            ngram_want, value_embed_want = future.result()
        else:
            ngram_want, value_embed_want = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)
        # value_embeds first at every stage, on every rank (the collective order).
        value_embed_pull = self.value_embeds.request(self.staging.value_embed_ring.upload(value_embed_want),
                                                     value_embed_want)
        ngram_pull = self.ngram.request(self.staging.ngram_ring.upload(ngram_want), ngram_want)
        self.next_pull = NextPull(event=i, ngram=ngram_pull, value_embeds=value_embed_pull, cycle=cycle)

    def after_forward(self, step: int):
        if self.next_pull is not None and self.next_pull.event == self.position[step]:
            self.value_embeds.exchange_ids(self.next_pull.value_embeds)
            self.ngram.exchange_ids(self.next_pull.ngram)

    def post_prep(self, step: int):
        """End of the step: post the next event's want-list build to the prep thread, once it is close enough."""
        if self.prep_pending is not None:
            return
        i = self.position[step]
        k = bisect.bisect_right(self.update_positions, i)
        if k == len(self.update_positions) or self.update_positions[k] - i > PREP_SLACK_STEPS:
            return
        event = self.update_positions[k]
        cycle = self._cycle_from(event + 1)
        if cycle:
            self.prep_pending = (event, self.staging.prep.submit(cycle_wants, self._batches(cycle),
                                                                 self.value_embeds.want_rows))

    def serve_next_cycle(self):
        """Right after the event's table updates: serve the post-update rows and start the row exchanges."""
        if self.next_pull is None:
            return
        self.value_embeds.start_land(self.next_pull.value_embeds)
        self.ngram.start_land(self.next_pull.ngram)
        # Behind the serves, so it delays neither exchange; it writes no row they read.
        self.ngram.bring_current(self.next_pull.ngram)

    def land_next_cycle(self):
        """Phase 3 of an event's optimizer step: the pulled n-gram rows go into the cache and become live.
        value_embeds' replica write is left in flight until right before the next forward
        (perf/deferred_gathers.py)."""
        if self.next_pull is None:
            return
        pull, cycle = self.next_pull, self.next_pull.cycle
        self.next_pull = None
        self.ngram.complete_land()
        self.ngram.start_cycle(pull.ngram)
        self.live_cycle = cycle

    def optimizer_event(self, step: int, lr: float, weight_decay: float, eps: float) -> "CycleEvent | None":
        """The optimizer hooks of this step's table updates; None unless the tables update this step.
        lr, weight_decay and eps are the n-gram table's (value_embeds' come from its optimizer config)."""
        if not self.is_update(step):
            return None
        return CycleEvent(self, step, lr, weight_decay, eps)

    def close(self):
        """End of the run: nothing may be left in flight."""
        if self.prep_pending is not None:
            self.prep_pending[1].result()
            self.prep_pending = None
        assert self.next_pull is None and self.ngram.grads_in_flight is None and self.ngram.landing is None
        assert self.value_embeds.grads_in_flight is None and self.value_embeds.landing is None


@dataclass(slots=True)
class CycleEvent:
    """Both tables' updates, slotted into AnvilAndAdam.step's comms schedule (optim/anvil.py SparseUpdate)."""
    prefetch: RowPrefetch
    step: int
    lr: float
    weight_decay: float
    eps: float

    def launch(self):
        self.prefetch.value_embeds.send_grads()
        self.prefetch.ngram.send_grads()
        self.prefetch.value_embeds.zero_sent_grads()

    def update(self):
        self.prefetch.value_embeds.adam_update()
        self.prefetch.ngram.adam_update(self.step, self.lr, self.weight_decay, self.eps)
        self.prefetch.serve_next_cycle()

    def finish(self):
        self.prefetch.land_next_cycle()

"""Row-sparse updates of value_embeds: the gradient travels as rows, and each rank's replica pulls only the
rows its next update cycle reads.

What it replaces: value_embeds as a plain sharded Adam parameter -- a dense [4 * 50304, 768] gradient
reduce-scattered at every update (309 MB) and the updated shards all-gathered back into every rank's
replica (309 MB), although a cycle reads only a few ten thousand distinct tokens.

Why it is faster: value_embeds updates on the n-gram table's events (ngram_table.is_update_step), so
it shares its cycles, and a cycle touches only the rows of its own tokens (4 planes x its distinct tokens).
  - The backward adds each step's row gradients into one persistent fp16 buffer, `grad_accum`
    (perf/kernels/value_embed.py): no dense gradient is ever allocated or reduced.
  - At the event (send_grads, adam_update), the cycle's rows go to their owners along the routes of the
    pull that fetched them (sharded_rows.py); each owner rebuilds its dense gradient shard from them --
    zero, then scatter-add its own rows and every peer's, then / world: exactly the AVG reduce-scatter's
    output -- and runs the ordinary Adam on its shard (AnvilAndAdam, value_embeds' own config).
  - Instead of the all-gather, each rank pulls exactly the rows the NEXT cycle reads, post-update (land).
The stages run where the n-gram table's do (perf/row_prefetch.py drives both).

Invariants:
  - Adam moves every row of the shard at every event (momentum and weight decay), so a replica row a
    cycle does not read goes stale. Nothing reads it before the next pull fetches it again: the training
    forward reads only its cycle's rows, and the replica is gathered whole (gather_replica) before
    every validation; the tail-average ship gathers it too.
  - This rank's own shard of the replica is the parameter Adam updates in place; it never lands from
    the wire, so the tail average, which reads the own shard, never races a land.
  - The land is split (start_land in the event's optimizer step, complete_land right before the next
    forward, perf/deferred_gathers.py): every reader of the peers' rows -- the forward, gather_replica,
    the tail ship's gather -- runs after complete_land.
  - grad_accum is nonzero only at the live cycle's rows (the backward adds only at its tokens); the
    event compacts exactly those rows and zeroes them. reset() zeroes it whole.
  - fp16 accumulation with atomics: reproducible within fp16 rounding, not bitwise.

The owner merge and the replica land are row-scatter kernels (perf/kernels/row_scatter.py), as in #360.
Deviations from record #360: the gradient reuses the pull's routes (as the n-gram table does, instead of
#360's second count/id exchange: fewer collectives); the taken rows are zeroed at the event, once every
gradient exchange is launched, rather than at the next step's top; the warmup runs this same path instead
of an unarmed dense one (untimed).

Provenance: record #360 (ANVIL2): `_VembPull`, `vesp_rows_from_mask`, `vesp_merge_gradients`,
`vepull_land`, `vepull_dense_gather`, `_velite_buf` / `ve_lite_prepare`.
"""
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from track_1_short.optim.anvil import AnvilAndAdam
from track_1_short.perf.kernels.row_scatter import scatter_add_rows, scatter_copy_rows
from track_1_short.perf.kernels.value_embed import GRAD_DTYPE
from track_1_short.sharded_rows import RowOwners, RowPull


def value_embed_rows(token_arrays: list[np.ndarray], vocab_size: int, num_planes: int) -> np.ndarray:
    """The sorted-unique value_embeds rows the given tokens read: p * vocab_size + t for every plane p
    and distinct token t, int32. Pure numpy (runs on the prep thread)."""
    seen = np.zeros(vocab_size, dtype=bool)
    for tokens in token_arrays:
        seen[tokens] = True
    distinct = np.flatnonzero(seen).astype(np.int32)
    return np.concatenate([distinct + np.int32(p * vocab_size) for p in range(num_planes)])


class ValueEmbedPull:
    """value_embeds' gradient buffer, its row-sparse update event and its replica pulls. Created in main();
    every method that exchanges rows is a collective."""

    def __init__(self, param: nn.Parameter, vocab_size: int, optimizer: AnvilAndAdam, rank: int, world_size: int,
                 device: torch.device):
        assert param.shape[0] % vocab_size == 0
        assert world_size & (world_size - 1) == 0, "the gradient's / world_size must be exact in fp16"
        self.param = param
        self.vocab_size = vocab_size
        self.num_planes = param.shape[0] // vocab_size
        self.optimizer = optimizer
        self.world_size = world_size
        self.owners = RowOwners(param.shape[0], rank, world_size)
        first = self.owners.first_row
        self.own_rows = slice(first, first + self.owners.rows_per_rank)
        # 309 MB fp16: the gradient of the live cycle, written by the backward (perf/kernels/value_embed.py).
        self.grad_accum = torch.zeros(param.shape, dtype=GRAD_DTYPE, device=device)
        # The dense gradient of this rank's shard, rebuilt at each event from the rows the cycle touched.
        self.grad_shard = torch.zeros(self.owners.rows_per_rank, param.shape[1], dtype=GRAD_DTYPE, device=device)
        self.landing = None  # a started land: (pull, received, work, served), until complete_land
        self.reset()

    def reset(self):
        """Back to no cycle and no gradient (the post-warmup reset)."""
        self.grad_accum.zero_()
        assert self.landing is None, "reset with a replica land in flight"
        self.live: RowPull | None = None  # the pull of the cycle being trained: its rows are its gradient rows
        self.grads_in_flight = None

    def want_rows(self, token_arrays: list[np.ndarray]) -> np.ndarray:
        return value_embed_rows(token_arrays, self.vocab_size, self.num_planes)

    # ---- pulling rows into the replica ----

    def request(self, want: Tensor, want_host: np.ndarray) -> RowPull:
        return self.owners.request(want, want_host)

    def exchange_ids(self, pull: RowPull):
        self.owners.exchange_ids(pull)

    def start_land(self, pull: RowPull):
        """Serve our (just updated) rows to the peers and start receiving theirs (async). The replica
        write waits for complete_land, which perf/deferred_gathers.py runs right before the next
        forward: nothing reads the peers' rows of the replica in between."""
        assert self.landing is None
        self.landing = (pull, *self.owners.start_fetch(pull, self.param.data[self.own_rows]))

    @torch.no_grad()
    def complete_land(self):
        """Write the received rows into the replica; the pull becomes the live cycle. No-op if none is in flight."""
        if self.landing is None:
            return
        pull, received, work, _served = self.landing
        self.landing = None
        work.wait()
        scatter_copy_rows(received, pull.peer_want, self.param.data)
        self.live = pull

    def fill(self, want_host: np.ndarray):
        """All three stages inline (cold start: the replica is whole, but the cycle needs its routes)."""
        pull = self.request(torch.from_numpy(want_host).to(self.grad_accum.device), want_host)
        self.exchange_ids(pull)
        self.start_land(pull)
        self.complete_land()

    @torch.no_grad()
    def gather_replica(self):
        """Make every rank's replica whole and current (before a validation reads arbitrary tokens)."""
        # A land still in flight would later overwrite the gathered rows with the same values, but
        # after the gather: the caller flushes first (DeferredGathers.flush).
        assert self.landing is None, "gather_replica with a replica land in flight"
        dist.all_gather_into_tensor(self.param.data, self.param.data[self.own_rows])

    # ---- the update event ----

    @torch.no_grad()
    def send_grads(self):
        """Event, part 1: take the cycle's gradient rows out of grad_accum and send the peers' rows to
        their owners (async)."""
        assert self.param.grad is None, "value_embeds got a dense gradient"
        pull = self.live
        compact = self.grad_accum.index_select(0, pull.want)
        wire = torch.cat([compact[:pull.lo], compact[pull.hi:]])
        received, work = self.owners.send_grads(pull, wire)
        self.grads_in_flight = (compact[pull.lo:pull.hi], received, work, wire)

    @torch.no_grad()
    def zero_sent_grads(self):
        """Event, after every gradient exchange is launched (so it delays none of them): zero the rows
        send_grads took out of grad_accum, for the next cycle."""
        self.grad_accum.index_fill_(0, self.live.want.long(), 0)

    @torch.no_grad()
    def adam_update(self):
        """Event, part 2 (collective with send_grads): rebuild this rank's dense gradient shard and run
        value_embeds' Adam on it."""
        own_grad, received, work, _ = self.grads_in_flight
        self.grads_in_flight = None
        work.wait()
        pull, first = self.live, self.owners.first_row
        grad = self.grad_shard.zero_()
        scatter_add_rows(own_grad, pull.want[pull.lo:pull.hi] - first, grad)
        scatter_add_rows(received, pull.asked - first, grad)
        grad.mul_(1.0 / self.world_size)  # exact in fp16: world_size is a power of two
        self.optimizer.adam_update_shard(self.param, grad)

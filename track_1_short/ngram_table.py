"""The hashed n-gram embedding: an 84.6M-row table, sharded across ranks, with its own sparse Adam.

What: every token position t reads two rows of one [NGRAM_VOCAB_SIZE, 768] table -- a bigram row
(hash of x[t-1], x[t]) from the first half and a trigram row (hash of x[t-2], x[t-1], x[t]) from the
second half. Each row is multiplied by a +-1 sign row drawn from a small shared pool by an independent
hash of the same tokens (the sign trick of #299: rows shared by several n-grams separate by sign), and
the two signed rows are summed into x0_bigram, which the model injects into the residual stream.

Why sharded: the table is 65e9 elements (130 GB in bf16), so no rank can hold a replica. Rank r owns
rows [r * V/world, (r + 1) * V/world) -- 16.2 GB -- and only a small, data-dependent set of rows is
live on a step, so every cost below scales with tokens, never with the table.

Update cycles: the table only changes at its Adam events (is_update_step: every odd step, then every
4th step from NGRAM_ADAM_PERIOD4_START). A *cycle* is the run of steps up to and including one event.
Every row a cycle reads is fetched ONCE, right after the previous event, into the model's
`ngram_cache` -- that is exact, since nothing changes the table in between. The fetch is a row pull
(sharded_rows.py): request (the cycle's sorted-unique row ids, "want"; cache slot i holds row want[i]),
exchange_ids, then land -- owners serve the asked rows from their (just updated) shard into the cache;
the rows this rank owns itself are read locally and never go on the wire.

The forward reads the cache at `slots` (searchsorted of each token's row ids in want) and adds a
zeros "sink" leaf, so each step's row gradient lands on the sink ([2T, 768]), never on a table-sized
tensor. accumulate_grad keeps (slots, sink grad) per step; at the event the gradient is summed per
cache slot (in fp16) and travels back along the pull's routes in reverse (send_grads, bf16 on the wire),
and each owner sums what every rank sent per row (in bf16) and runs Adam on exactly those rows
(adam_update).

The optimizer is Adam with beta1 = 0 and ONE fp32 second moment per row (the row's mean squared
gradient), so its state is 4 bytes per row instead of 2 * 768 * 4, and a row no gradient touched
does not move: with no momentum its update is exactly zero, and the cautious weight decay is gated on
the update. Its second moment still decays by beta2 every event; that decay is applied lazily, when a
row is next touched (perf/kernels/ngram_adam.py replays the missed events' beta2 from a history), so
an event costs only the rows it touches -- including the owner-side merge, which needs no sort.

When each stage of a pull runs relative to the forward, backward and optimizer (and the host prep
that builds the want list steps ahead) is perf/row_prefetch.py's job; this file is correct called
in order, synchronously.

Provenance: record #360 (ANVIL2): table size, two-channel layout, hashes, beta1 = 0 row-RMS Adam,
the per-cycle row cache and pull, the period-4 cadence.
"""
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.perf.kernels.ngram_adam import adam_rows_, bring_rows_current, claim_rows
from track_1_short.perf.kernels.row_scatter import scatter_add_rows
from track_1_short.sharded_rows import RowOwners, RowPull, all_to_all_rows

# 224x the 377,280-row bigram table of earlier records. Rows [0, V/2) are the bigram channel,
# [V/2, V) the trigram channel. Divisible by 8 (the shard) and < 2**31 (row ids are int32).
NGRAM_VOCAB_SIZE = 84_602_880
NGRAM_DIM = 768
# The shared +-1 sign pool. A power of two, so `& (rows - 1)` is the non-negative remainder of the
# wrapped int32 hash (values ~500-15000 gave similar results in the bigram record).
NGRAM_SIGN_POOL_ROWS = 8192

# Row hashes (record #360): multiply each token by a large odd constant, xor, reduce mod the channel
# size. int32 products wrap; torch's `%` is Python-signed, so the result is non-negative.
BIGRAM_ROW_MULS = (36313, 27191)            # x[t], x[t-1]
TRIGRAM_ROW_MULS = (17351, 60961, 45259)    # x[t], x[t-1], x[t-2]
# Sign-pool hashes (record #360), independent of the row hashes; computed in the embedding kernel
# (perf/kernels/ngram_embed.py). Position t reads pool row
#   bigram:  (m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 1
#   trigram: (m[2] * x[t-2] ^ m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 2
BIGRAM_SIGN_MULS = (48271, 30011)           # x[t], x[t-1]
TRIGRAM_SIGN_MULS = (58699, 39779, 26801)   # x[t], x[t-1], x[t-2]

# Adam for the table (record #360). lr and weight decay are Adam's scheduled base values times these
# multipliers; beta1 is 0 for the whole run.
NGRAM_LR_MUL = 70.0
NGRAM_WD_MUL = 5.0
NGRAM_ADAM_BETA2 = 0.95
# From this step the table updates every 4th step instead of every 2nd (record #360). One event then
# stands for two: beta2 is squared, so the second moment decays at the same rate per step, and the
# weight-decay multiplier doubles.
NGRAM_ADAM_PERIOD4_START = 336
NGRAM_WD_MUL_PERIOD4 = 10.0
# The longest cycle, in steps: the cache holds one cycle's rows (two per token per step).
MAX_CYCLE_STEPS = 4
# A cycle's gradient is summed per cache slot in fp16, travels between ranks in bf16 (the table's dtype),
# and is summed per row by the owner in bf16 (record #360's precisions).
GRAD_ACCUM_DTYPE = torch.float16
GRAD_WIRE_DTYPE = torch.bfloat16
GRAD_MERGE_DTYPE = torch.bfloat16

assert NGRAM_VOCAB_SIZE % 2 == 0 and NGRAM_VOCAB_SIZE < 2 ** 31
assert NGRAM_SIGN_POOL_ROWS & (NGRAM_SIGN_POOL_ROWS - 1) == 0
# The last period-2 event (an odd step) is then followed by a whole period-4 cycle.
assert NGRAM_ADAM_PERIOD4_START % 4 == 0


# -----------------------------------------------------------------------------
# Hashing

def ngram_row_ids(x: Tensor) -> Tensor:
    """Token ids [T] (host) -> table row ids [2T] int32: out[:T] bigram rows, out[T:] trigram rows.

    Positions without enough history get the channel's reserved last row. Note the bigram hash at
    t = 1 reads the reserved id, not x[0], as its previous token (it is computed in place after
    out[0] is set, exactly as in record #360).
    """
    x = x.to(torch.int32)
    half = NGRAM_VOCAB_SIZE // 2
    bigram_mod, trigram_mod = half - 1, NGRAM_VOCAB_SIZE - half - 1
    n = x.numel()
    out = torch.empty(2 * n, dtype=torch.int32)
    bigram, trigram = out[:n], out[n:]
    bigram.copy_(x)
    bigram[0] = bigram_mod
    bigram[1:] = torch.bitwise_xor(BIGRAM_ROW_MULS[0] * bigram[1:], BIGRAM_ROW_MULS[1] * bigram[:-1]) % bigram_mod
    trigram[0] = trigram[1] = half + trigram_mod
    trigram[2:] = torch.bitwise_xor(
        torch.bitwise_xor(TRIGRAM_ROW_MULS[0] * x[2:], TRIGRAM_ROW_MULS[1] * x[1:-1]),
        TRIGRAM_ROW_MULS[2] * x[:-2],
    ) % trigram_mod + half
    return out


def sorted_unique_rows(row_id_arrays: list[np.ndarray]) -> np.ndarray:
    """The sorted-unique row ids of several steps' [2T] arrays (a cycle's want list), int32.

    A sort and a neighbour compare rather than np.unique: numpy 2 routes integer unique through a
    hash set, which record #360 measured at 15-60 ms on a cycle's ids against 2-6 ms for the sort.
    """
    rows = np.sort(np.concatenate(row_id_arrays))
    if rows.size:
        rows = rows[np.r_[True, rows[1:] != rows[:-1]]]
    return rows.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Update cadence

def is_update_step(step: int) -> bool:
    """Every odd step (with the other Adam params) until NGRAM_ADAM_PERIOD4_START, then steps = 3 mod 4."""
    if step >= NGRAM_ADAM_PERIOD4_START:
        return step % 4 == 3
    return step % 2 == 1


def adam_beta2_and_wd_mul(step: int) -> tuple[float, float]:
    if step >= NGRAM_ADAM_PERIOD4_START:
        return NGRAM_ADAM_BETA2 ** 2, NGRAM_WD_MUL_PERIOD4
    return NGRAM_ADAM_BETA2, NGRAM_WD_MUL


# -----------------------------------------------------------------------------
# The table

class NgramTable:
    """This rank's shard of the n-gram table, its Adam state, and the model's row cache.

    Created in main(); `cache` is the model's `ngram_cache` buffer. Every method that exchanges rows
    is a collective: all ranks call it at the same point.
    """

    def __init__(self, cache: Tensor, max_step_tokens: int, max_events: int, rank: int, world_size: int,
                 device: torch.device):
        assert NGRAM_VOCAB_SIZE % world_size == 0
        assert cache.shape[1] == NGRAM_DIM and cache.dtype == torch.bfloat16
        # A cycle reads at most two rows per token of MAX_CYCLE_STEPS steps.
        max_cycle_rows = 2 * MAX_CYCLE_STEPS * max_step_tokens
        assert cache.shape[0] >= max_cycle_rows, "ngram_cache cannot hold a cycle"
        self.cache = cache
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.owners = RowOwners(NGRAM_VOCAB_SIZE, rank, world_size)
        self.first_row = self.owners.first_row
        rows = self.owners.rows_per_rank
        # 10.6M x 768 bf16 = 16.2 GB. Zero init, as record #360.
        self.shard = torch.zeros(rows, NGRAM_DIM, dtype=torch.bfloat16, device=device)
        # Adam's second moment, one fp32 scalar per row (beta1 = 0: no first moment), current as of the
        # row's last_event; beta2_history[e] is event e's beta2, for replaying the events a row missed.
        self.exp_avg_sq = torch.zeros(rows, dtype=torch.float32, device=device)
        self.last_event = torch.zeros(rows, dtype=torch.int32, device=device)
        self.beta2_history = torch.zeros(max_events + 1, dtype=torch.float32, device=device)
        # The owner merge's row -> claimed entry map (perf/kernels/ngram_adam.py); only the entries of the
        # current event are ever read, so it is never cleared.
        self.entry_of_row = torch.zeros(rows, dtype=torch.int32, device=device)
        # The live cycle's gradient summed per cache slot, rebuilt at each event.
        self.grad_per_slot = torch.empty(max_cycle_rows, NGRAM_DIM, dtype=GRAD_ACCUM_DTYPE, device=device)
        self.sinks: dict[int, Tensor] = {}  # tokens per step -> the zeros leaf, allocated once
        self.reset()

    def reset(self):
        """Back to the initial state, in place: the post-warmup reset must not copy 16 GB."""
        self.shard.zero_()
        self.exp_avg_sq.zero_()
        self.last_event.zero_()
        self.adam_events = 0
        self.live: RowPull | None = None    # the pull of the cycle being trained
        self.cached: RowPull | None = None  # the pull whose rows the cache holds now
        self.pending: list[tuple[Tensor, Tensor]] = []  # (slots, sink grad) per step of the live cycle
        self.grads_in_flight = None
        self.landing = None  # a started land: (pull, received, work, served), until complete_land

    # ---- pulling rows into the cache ----

    def request(self, want: Tensor, want_host: np.ndarray) -> RowPull:
        """Stage 1 (sharded_rows.RowOwners.request), into a cache that must hold the rows."""
        assert want_host.shape[0] <= self.cache.shape[0], f"{want_host.shape[0]} rows overflow ngram_cache"
        return self.owners.request(want, want_host)

    def exchange_ids(self, pull: RowPull):
        self.owners.exchange_ids(pull)

    def land(self, pull: RowPull):
        """Stage 3: fetch the peers' rows from their shards as they are now, and write the pulled rows
        into the cache in want order: peers' rows around our own block [lo:hi)."""
        self.start_land(pull)
        self.complete_land()

    def start_land(self, pull: RowPull):
        """Stage 3, first half: serve our rows and start receiving the peers' (async). The optimizer
        step serves right after the event's update and lands later, so the row exchange is queued on
        the NCCL stream ahead of the large bank gathers (optim/anvil.py SparseUpdate)."""
        assert self.landing is None
        self.landing = (pull, *self.owners.start_fetch(pull, self.shard))

    @torch.no_grad()
    def complete_land(self):
        """Stage 3, second half: wait for the peers' rows and write the cache."""
        pull, received, work, _served = self.landing
        self.landing = None
        work.wait()
        lo, hi, n = pull.lo, pull.hi, pull.num_rows
        self.cache[:lo].copy_(received[:lo])
        self.cache[hi:n].copy_(received[lo:])
        torch.index_select(self.shard, 0, pull.want[lo:hi] - self.first_row, out=self.cache[lo:hi])
        self.cached = pull

    def fill(self, want_host: np.ndarray) -> RowPull:
        """All three stages inline (cold start)."""
        pull = self.request(torch.from_numpy(want_host).to(self.device), want_host)
        self.exchange_ids(pull)
        self.land(pull)
        return pull

    def restore_cache(self):
        """Re-land the live cycle's rows after an eval pull overwrote the cache. No event ran since
        they were pulled, so the shard serves the same rows; the routes are reused, not re-exchanged."""
        self.land(self.live)

    def slots(self, pull: RowPull, row_ids: Tensor) -> Tensor:
        """Cache slots of `row_ids` ([2T] int32, device), all of which `pull` fetched."""
        return torch.searchsorted(pull.want, row_ids, out_int32=True)

    def eval_pulls(self, row_id_batches: list[Tensor]) -> list[RowPull]:
        """Pulls for eval batches (device row ids): the want lists are built on the device and all
        batches' counts go in ONE exchange (record #360's batched validation fill). Not yet landed."""
        wants = [torch.unique(ids) for ids in row_id_batches]  # sorted
        bounds = torch.from_numpy(self.owners.bounds).to(self.device)
        cuts = torch.stack([torch.searchsorted(w.to(torch.int64), bounds) for w in wants]).cpu().numpy()
        send = cuts[:, 1:] - cuts[:, :-1]              # [batches, world]
        send[:, self.rank] = 0
        send_t = torch.from_numpy(np.ascontiguousarray(send.T))  # row r: what we ask rank r, per batch
        recv_t = torch.empty_like(send_t)
        dist.all_to_all_single(recv_t, send_t)
        recv = recv_t.numpy().T
        pulls = []
        for b, want in enumerate(wants):
            assert want.shape[0] <= self.cache.shape[0], f"{want.shape[0]} rows overflow ngram_cache"
            pull = RowPull(want=want.to(torch.int32), lo=int(cuts[b, self.rank]), hi=int(cuts[b, self.rank + 1]),
                           send_counts=send[b].tolist(), recv_counts=recv[b].tolist())
            pull.peer_want = pull.peer_rows()
            pull.asked = all_to_all_rows(pull.peer_want, pull.send_counts, pull.recv_counts)
            pulls.append(pull)
        return pulls

    def load_eval_batch(self, row_ids: Tensor) -> Tensor:
        """Pull one eval batch's rows into the cache; returns its slots."""
        pull, = self.eval_pulls([row_ids])
        self.land(pull)
        return self.slots(pull, row_ids)

    # ---- the training cycle ----

    def start_cycle(self, pull: RowPull):
        """Make a landed pull the live cycle: the forward reads its rows, and its event's gradient rows are its rows."""
        assert self.cached is pull and not self.pending
        self.live = pull

    def grad_sink(self, num_tokens: int) -> Tensor:
        """The zeros leaf the forward adds to its looked-up rows; its gradient is the rows' gradient.
        One per step size, allocated once (the step graphs bake it)."""
        if num_tokens not in self.sinks:
            self.sinks[num_tokens] = torch.zeros(2 * num_tokens, NGRAM_DIM, dtype=torch.bfloat16,
                                                 device=self.device, requires_grad=True)
        return self.sinks[num_tokens]

    def accumulate_grad(self, slots: Tensor, sink_grad: Tensor):
        """Hold one step's (slots, sink gradient) until the cycle's event. `sink_grad` must not be
        rewritten before then (with CUDA graphs: StepGraphs.hold_ngram_grad)."""
        assert self.cached is self.live, "the forward read a cache that holds no training cycle"
        self.pending.append((slots, sink_grad))

    @torch.no_grad()
    def send_grads(self):
        """Event, part 1: sum the cycle's gradient per cache slot and send each row's sum to its owner (async)."""
        pull = self.live
        assert self.pending, "Adam event with no gradient since the last one"
        assert pull.num_rows <= self.grad_per_slot.shape[0], f"{pull.num_rows} cycle rows > gradient buffer"
        per_slot = self.grad_per_slot[:pull.num_rows].zero_()
        for slots, grad in self.pending:
            scatter_add_rows(grad, slots, per_slot)
        self.pending = []
        # Everything but our own block, in want order: the reverse of the rows the pull received.
        lo, hi = pull.lo, pull.hi
        wire = torch.empty(pull.num_rows - (hi - lo), NGRAM_DIM, dtype=GRAD_WIRE_DTYPE, device=self.device)
        wire[:lo].copy_(per_slot[:lo])
        wire[lo:].copy_(per_slot[hi:])
        received, work = self.owners.send_grads(pull, wire)
        self.grads_in_flight = (per_slot[lo:hi], received, work, wire)

    @torch.no_grad()
    def adam_update(self, step: int, lr: float, weight_decay: float, eps: float):
        """Event, part 2 (collective with send_grads): sum the received gradient entries per row, then one
        Adam event over every row the cycle touched. `lr` and `weight_decay` are Adam's scheduled base values.

        Several ranks may send the same row: its entries are summed, then averaged over ranks (the dense
        equivalent is an AVG reduce-scatter; the 1/world rides grad_mul / sq_mul). The update itself, and
        its plain-torch equivalent, are in perf/kernels/ngram_adam.py.
        """
        own_grad, received, work, _ = self.grads_in_flight
        self.grads_in_flight = None
        work.wait()
        pull = self.live
        own_rows = pull.want[pull.lo:pull.hi] - self.first_row
        asked_rows = pull.asked - self.first_row
        entries = torch.cat([own_rows, asked_rows])

        beta2, wd_mul = adam_beta2_and_wd_mul(step)
        lr = lr * NGRAM_LR_MUL
        self.adam_events += 1
        event = self.adam_events
        # Before the update reads it; a fill, not an H2D, so there is no pinned buffer to race.
        self.beta2_history[event].fill_(beta2)
        # One merge slot per entry: each row's gradient lands in its claimed entry's slot.
        claim_rows(entries, self.entry_of_row)
        grad = torch.zeros(entries.numel(), NGRAM_DIM, dtype=GRAD_MERGE_DTYPE, device=self.device)
        scatter_add_rows(own_grad, own_rows, grad, row_map=self.entry_of_row)
        scatter_add_rows(received, asked_rows, grad, row_map=self.entry_of_row)
        adam_rows_(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history, grad, entries, self.entry_of_row,
                   event, beta2=beta2, eps=eps,
                   step_size=lr * (1 - beta2 ** event) ** 0.5,  # beta1 = 0: no first-moment bias
                   decay=lr * lr * weight_decay * wd_mul,
                   grad_mul=1 / self.world_size, sq_mul=(1 - beta2) / self.world_size ** 2)

    @torch.no_grad()
    def bring_current(self, pull: RowPull):
        """Replay the missed second-moment decays of the rows of `pull` this rank owns (the next cycle's):
        they are the rows its next event touches, so its replay stays short."""
        if self.adam_events:
            bring_rows_current(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history,
                               pull.want[pull.lo:pull.hi] - self.first_row, self.adam_events)

"""Tables sharded by row range across ranks: which rank owns a row, pulling rows from their owners, and
sending row gradients back.

Two tables use this: the hashed n-gram table (ngram_table.py), which has no replica and pulls a cycle's
rows into a compact cache, and the value embeddings (perf/value_embed_pull.py), which keep a replica and
pull the rows they will read into it. Rank r owns rows [r * N/world, (r + 1) * N/world).

A pull (RowPull) fetches a sorted-unique row list ("want") in three stages, each a collective:

  1. request: cut want by owner and send each owner the count of rows asked of it (gloo, CPU tensors,
     async). This rank's own rows want[lo:hi] are read locally and never go on the wire.
  2. exchange_ids: the ids go to their owners (all_to_all, async).
  3. start_fetch: owners index_select the asked rows from their shard and send them back; the receive
     is left in flight, for a caller that has other work to queue before it reads them. With
     start_fetch_into the receive lands directly in the caller's want-ordered buffer (the n-gram cache),
     through per-peer views of it, so there is no receive buffer and no copy out of one.

The same routes, reversed, carry the gradient of exactly those rows back to their owners (send_grads):
a cycle's gradient rows are the rows it pulled. The payload is the caller's compact [want, D] buffer
itself, each peer's block a view of it (this rank's own block is simply not sent), and the receive buffer is
packed peer after peer: no send copy, no dtype cast on the wire.

The collectives take an optional process group: the n-gram table runs its exchanges on a communicator of
its own (distributed.py DistEnv.ngram_group).

Provenance: record #360 (ANVIL2): `_RowPull`, `sparse_comms_start`, `sparse_comms_share_indexes`,
`sparse_value_a2a`. The per-peer views and the list all_to_all that uses them are this trainer's
n-gram exchange.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def all_to_all_rows(send: Tensor, send_counts: list[int], recv_counts: list[int], async_op: bool = False,
                    group: dist.ProcessGroup | None = None):
    """Rows send[sum(send_counts[:r]) : sum(send_counts[:r+1])] go to rank r; the rows received come
    back concatenated in rank order. Returns (received, work) with async_op, else received."""
    out = send.new_empty((sum(recv_counts), *send.shape[1:]))
    work = dist.all_to_all_single(out, send, output_split_sizes=recv_counts, input_split_sizes=send_counts,
                                  group=group, async_op=async_op)
    return (out, work) if async_op else out


def _segment_views(buf: Tensor, counts: list[int], rank: int, hi: int, own_gap: bool) -> list[Tensor]:
    """Per-peer row views of `buf` [n, D] in rank order: counts[j] rows belong to peer j (counts[rank] is 0).
    With own_gap the buffer also holds this rank's own block, ending at row `hi`, at its sorted place
    (a want-ordered buffer: the compact gradient, the cache); without it the peers' blocks are packed
    back to back (a flat send or receive buffer)."""
    views, off = [], 0
    for j, count in enumerate(counts):
        if j == rank:
            views.append(buf[off:off])
            if own_gap:
                off = hi
            continue
        views.append(buf[off:off + count])
        off += count
    return views


def send_views(compact: Tensor, lo: int, hi: int, send_counts: list[int], rank: int) -> list[Tensor]:
    """The peers' blocks of a want-ordered [n, D] payload, around the own block [lo:hi): all_to_all's inputs."""
    assert 0 <= lo <= hi <= compact.shape[0] and send_counts[rank] == 0, \
        f"bad own block [{lo}:{hi}] of {compact.shape[0]} rows / own count {send_counts[rank]}"
    assert sum(send_counts) == compact.shape[0] - (hi - lo), \
        f"{sum(send_counts)} peer rows vs a payload of {compact.shape[0]} - {hi - lo}"
    return _segment_views(compact, send_counts, rank, hi, True)


def packed_views(buf: Tensor, counts: list[int], rank: int) -> list[Tensor]:
    """Per-peer views of a buffer [m, D] holding the peers' rows back to back (own count is 0)."""
    assert counts[rank] == 0 and sum(counts) <= buf.shape[0]
    return _segment_views(buf, counts, rank, 0, False)


def cache_views(dest: Tensor, lo: int, hi: int, send_counts: list[int], rank: int, n: int) -> list[Tensor]:
    """Per-peer destination views INSIDE a want-ordered buffer: the peers' rows land exactly where the
    pull's want order puts them (rows [0:lo) and [hi:n)), so the all_to_all writes the buffer directly."""
    assert send_counts[rank] == 0 and sum(send_counts) == n - (hi - lo) and n <= dest.shape[0], \
        f"{n} rows ([{lo}:{hi}) own) vs counts {sum(send_counts)} / buffer {tuple(dest.shape)}"
    return _segment_views(dest, send_counts, rank, hi, True)


@dataclass(slots=True)
class RowPull:
    """One fetch of a sorted-unique row set, and the routes its gradient takes back."""
    want: Tensor            # [n] int32 sorted-unique global row ids, device
    lo: int                 # want[lo:hi] are the rows this rank owns: read locally, never on the wire
    hi: int
    send_counts: list[int]  # rows asked of each rank (0 for ourselves)
    counts_exchange: tuple | None = None  # in flight: (work, received counts, sent counts)
    recv_counts: list[int] | None = None  # rows each rank asks of us
    peer_want: Tensor | None = None       # peer_rows(), as sent to the owners: the order the fetch returns rows in
    asked: Tensor | None = None           # [sum(recv_counts)] int32 global ids asked of us, by rank
    ids_exchange: object | None = None    # in-flight work filling `asked`

    @property
    def num_rows(self) -> int:
        return self.want.shape[0]

    def peer_rows(self) -> Tensor:
        """The wanted rows other ranks own, in want order: the order start_fetch receives them in."""
        return torch.cat([self.want[:self.lo], self.want[self.hi:]])


class RowOwners:
    """Ownership of a [num_rows, D] table sharded by row range, and the collectives of a pull."""

    def __init__(self, num_rows: int, rank: int, world_size: int):
        assert num_rows % world_size == 0
        self.rank = rank
        self.rows_per_rank = num_rows // world_size
        self.first_row = rank * self.rows_per_rank
        # np.searchsorted(want, bounds) cuts a sorted want list by owner.
        self.bounds = np.arange(world_size + 1, dtype=np.int64) * self.rows_per_rank

    def request(self, want: Tensor, want_host: np.ndarray) -> RowPull:
        """Stage 1: cut the sorted want list by owner and start the count exchange (async, gloo)."""
        cuts = np.searchsorted(want_host, self.bounds)
        send_counts = np.diff(cuts)
        lo, hi = int(cuts[self.rank]), int(cuts[self.rank + 1])
        send_counts[self.rank] = 0
        send = torch.from_numpy(send_counts.astype(np.int64))
        recv = torch.empty_like(send)
        work = dist.all_to_all_single(recv, send, async_op=True)
        return RowPull(want=want, lo=lo, hi=hi, send_counts=send_counts.tolist(), counts_exchange=(work, recv, send))

    def exchange_ids(self, pull: RowPull):
        """Stage 2: wait for the counts, then send each owner the ids we want from it (async)."""
        work, recv, _ = pull.counts_exchange
        work.wait()
        pull.counts_exchange = None
        pull.recv_counts = recv.tolist()
        pull.peer_want = pull.peer_rows()
        pull.asked, pull.ids_exchange = all_to_all_rows(pull.peer_want, pull.send_counts, pull.recv_counts,
                                                        async_op=True)

    def _serve(self, pull: RowPull, shard: Tensor) -> Tensor:
        """The rows asked of us, from `shard` (this rank's rows, as they are now), in pull.asked order."""
        if pull.ids_exchange is not None:
            pull.ids_exchange.wait()
            pull.ids_exchange = None
        return shard.index_select(0, pull.asked - self.first_row)

    @torch.no_grad()
    def start_fetch(self, pull: RowPull, shard: Tensor, group: dist.ProcessGroup | None = None
                    ) -> tuple[Tensor, object, Tensor]:
        """Stage 3, async: serve the rows asked of us from `shard` and start receiving the rows we asked
        for, in pull.peer_rows() order. Returns (received, work, served); the caller holds all three
        until work is waited, which keeps the buffers alive."""
        served = self._serve(pull, shard)
        received, work = all_to_all_rows(served, pull.recv_counts, pull.send_counts, async_op=True, group=group)
        return received, work, served

    @torch.no_grad()
    def start_fetch_into(self, pull: RowPull, shard: Tensor, dest: Tensor, group: dist.ProcessGroup | None = None
                         ) -> tuple[object, Tensor]:
        """start_fetch with the receive written straight into `dest`, a want-ordered buffer (row i = want[i]):
        the peers' rows land in [0:lo) and [hi:n), this rank's own block [lo:hi) is left for the caller to
        read locally. One list all_to_all between per-peer views, so no receive buffer and no copy.
        Returns (work, served). `dest`'s rows are overwritten from the moment the collective runs."""
        served = self._serve(pull, shard)
        ins = packed_views(served, pull.recv_counts, self.rank)
        outs = cache_views(dest, pull.lo, pull.hi, pull.send_counts, self.rank, pull.num_rows)
        return dist.all_to_all(outs, ins, group=group, async_op=True), served

    @torch.no_grad()
    def send_grads(self, pull: RowPull, compact: Tensor, group: dist.ProcessGroup | None = None):
        """The gradient of the pulled rows to their owners, along the pull's routes reversed. `compact` is
        [pull.num_rows, D] in want order, own block [lo:hi) included: the peers' blocks go out as views of
        it, in its dtype (the receiver rounds). Returns (received, work): received is packed in rank order,
        so received[i] is the gradient of row pull.asked[i]. `compact` must not change until work is waited."""
        received = compact.new_empty((sum(pull.recv_counts), compact.shape[1]))
        ins = send_views(compact, pull.lo, pull.hi, pull.send_counts, self.rank)
        outs = packed_views(received, pull.recv_counts, self.rank)
        return received, dist.all_to_all(outs, ins, group=group, async_op=True)

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
     is left in flight, for a caller that has other work to queue before it reads them.

The same routes, reversed, carry the gradient of exactly those rows back to their owners (send_grads):
a cycle's gradient rows are the rows it pulled.

Provenance: record #360 (ANVIL2): `_RowPull`, `sparse_comms_start`, `sparse_comms_share_indexes`,
`sparse_value_a2a`.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def all_to_all_rows(send: Tensor, send_counts: list[int], recv_counts: list[int], async_op: bool = False):
    """Rows send[sum(send_counts[:r]) : sum(send_counts[:r+1])] go to rank r; the rows received come
    back concatenated in rank order. Returns (received, work) with async_op, else received."""
    out = send.new_empty((sum(recv_counts), *send.shape[1:]))
    work = dist.all_to_all_single(out, send, output_split_sizes=recv_counts, input_split_sizes=send_counts,
                                  async_op=async_op)
    return (out, work) if async_op else out


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

    @torch.no_grad()
    def start_fetch(self, pull: RowPull, shard: Tensor) -> tuple[Tensor, object, Tensor]:
        """Stage 3, async: serve the rows asked of us from `shard` (this rank's rows, as they are now)
        and start receiving the rows we asked for, in pull.peer_rows() order. Returns (received, work,
        served); the caller holds all three until work is waited, which keeps the buffers alive."""
        if pull.ids_exchange is not None:
            pull.ids_exchange.wait()
            pull.ids_exchange = None
        served = shard.index_select(0, pull.asked - self.first_row)
        received, work = all_to_all_rows(served, pull.recv_counts, pull.send_counts, async_op=True)
        return received, work, served

    def send_grads(self, pull: RowPull, wire: Tensor):
        """Gradient rows of pull.peer_rows() (in that order) to their owners, along the pull's routes
        reversed. Returns (received, work): received[i] is the gradient of row pull.asked[i]."""
        return all_to_all_rows(wire, pull.send_counts, pull.recv_counts, async_op=True)

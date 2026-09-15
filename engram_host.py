"""Engram host table: bf16 rows of hashed n-grams in host RAM (one shared mmap per node) with a
sync-free training step. A worker thread hashes the loader's CPU ids, gathers the rows and
uploads them; the row update runs as one Triton kernel and a second worker scatters the new rows
back; both copies run under the forward. Gathers that overlapped a write are patched on the GPU
from a device ring of recent writes. Nothing in the timed loop waits on the host or the device.
"""

import collections
import concurrent.futures
import os

import torch

from engram_triton import row_update

_PRIMES = (1_000_003, 998_244_353, 911_382_323, 805_306_457)


def ngram_hash(idx, table_size, ngram, salt=0):
    """(B, T) buckets in [0, table_size) of each position's trailing `ngram` ids; `salt` gives each slot its own hash."""
    assert 1 <= ngram <= len(_PRIMES)
    acc = idx * (_PRIMES[0] + salt * 2_654_435_761)
    for j in range(1, ngram):
        prev = torch.zeros_like(idx)
        if j < idx.size(1):
            prev[:, j:] = idx[:, :-j]
        acc = acc + prev * (_PRIMES[j] + salt * 40_503)
    return acc.remainder(table_size)


class EngramHost:
    def __init__(self, rows, n_site, per_slot, decay, accum_beta, orders, weights, shared_path, rank,
                 pf_depth=2, ring=4, max_tokens=0):
        self.rows, self.n_site, self.per_slot = rows, n_site, per_slot
        self.LD = n_site * per_slot
        self.decay, self.accum_beta = decay, accum_beta
        self.orders, self.weights = orders, weights
        self.G = sum(weights)
        self.shared_path, self.rank = shared_path, rank
        n = rows * self.LD
        if shared_path is not None:
            fresh = not os.path.exists(shared_path) or os.path.getsize(shared_path) == 0
            self.weight = torch.from_file(shared_path, shared=True, size=n, dtype=torch.bfloat16).view(rows, self.LD)
            if rank == 0 and not fresh:
                self.weight.zero_()
        else:
            self.weight = torch.zeros(rows, self.LD, dtype=torch.bfloat16)
        self.accum = torch.zeros(rows, n_site, device="cuda")      # RMSprop accumulator per row and site
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.side = torch.cuda.Stream(device=self.device)            # write-back D2H
        self.upl = torch.cuda.Stream(device=self.device)             # row upload H2D
        self._fwd_ev = None                                          # start of the current forward
        self.gather_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-gather")
        self.write_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-write")
        self.ring = ring
        assert pf_depth < ring, "the gather ring must be deeper than the prefetch depth"
        self.pf_depth = pf_depth
        self._g_slots = [None] * ring
        self._g_next = 0
        self._pending = collections.deque()          # prefetches issued, not yet consumed
        self._w_slots = [None] * ring
        self._w_next = 0
        self._writes = collections.deque()           # (id, uniq_gpu, new_rows) of recent writes
        self._write_id = 0
        self._write_futures = {}
        self._deferred = None                        # the write prepared by step(), awaiting its copy
        self._prow_ring = None                       # (R, cap, LD) device ring of new rows; write wid lives in slot wid % R
        self._prow_R = ring + 6
        self._stage_sync = None
        self._max_ids = self.G * max_tokens
        if max_tokens > 0:                           # preallocate the rings: a pinned alloc is a device sync
            for s_ in range(ring):
                self._g_slots[s_] = self._new_gather_slot(self._max_ids)
                self._w_slots[s_] = dict(cap=self._max_ids, stage=torch.empty(self._max_ids, self.LD, dtype=torch.bfloat16).pin_memory(),
                                         event=torch.cuda.Event(), future=None)

    # ---- hashing ---------------------------------------------------------
    def buckets(self, ids):
        """ids: (T,) int on any device -> (G, T) int64 bucket ids, one row per hash slot."""
        ids = ids.reshape(1, -1).to(torch.int64)
        hs = [ngram_hash(ids, self.rows, order, salt=1 + oi * 97 + w * 7919)
              for oi, order in enumerate(self.orders) for w in range(self.weights[oi])]
        return torch.stack(hs).reshape(self.G, -1)

    # ---- prefetch --------------------------------------------------------
    def mark_fwd(self):
        """Record the start of the forward; the row copies wait on it."""
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self._fwd_ev = ev

    def prefetch(self, ids_cpu):
        """Queue the gather for a future training forward from the loader's CPU token ids."""
        n_ids = self.G * ids_cpu.numel()
        self._max_ids = max(self._max_ids, n_ids)
        s = self._g_next
        self._g_next = (s + 1) % self.ring
        slot = self._g_slots[s]
        if slot is None or slot["cap"] < n_ids:
            slot = self._new_gather_slot(n_ids)
            self._g_slots[s] = slot
        # writes not landed when the gather runs are patched in at take()
        p = dict(ids=ids_cpu, n_ids=n_ids, slot=slot, floor=self._oldest_unlanded_write(), fwd_ev=self._fwd_ev)
        p["future"] = self.gather_exec.submit(self._gather_task, p)
        self._pending.append(p)

    def _new_gather_slot(self, cap):
        dev = self.device
        return dict(cap=cap, uniq=torch.empty(cap, dtype=torch.int64).pin_memory(), inv=torch.empty(cap, dtype=torch.int64).pin_memory(),
                    rows=torch.empty(cap, self.LD, dtype=torch.bfloat16).pin_memory(),
                    d_rows=torch.empty(cap, self.LD, dtype=torch.bfloat16, device=dev),
                    d_uniq=torch.empty(cap, dtype=torch.int64, device=dev), d_inv=torch.empty(cap, dtype=torch.int64, device=dev),
                    event=torch.cuda.Event(), consumed=torch.cuda.Event())

    def _oldest_unlanded_write(self):
        pend = self._deferred[0] if self._deferred is not None else None   # prepared, not yet issued
        for wid, f in list(self._write_futures.items()):   # snapshot: the main thread mutates the dict
            if not f.done():
                return wid if pend is None else min(wid, pend)
        if pend is not None:
            return pend
        return self._write_id

    def _gather_task(self, p):
        slot, ev = p["slot"], p["slot"]["event"]
        p["floor"] = min(p["floor"], self._oldest_unlanded_write())
        if ev.query() is False:
            ev.synchronize()                         # the previous upload from this slot is done
        b = self.buckets(p["ids"]).reshape(-1)
        uniq, inv = torch.unique(b, return_inverse=True)
        U = uniq.numel()
        torch.index_select(self.weight, 0, uniq, out=slot["rows"][:U])
        slot["uniq"][:U].copy_(uniq)
        slot["inv"][:p["n_ids"]].copy_(inv)
        p["uniq_cpu"] = uniq
        torch.cuda.set_device(self.device)
        n = p["n_ids"]
        with torch.no_grad(), torch.cuda.stream(self.upl):
            self.upl.wait_event(slot["consumed"])    # the slot's previous leaf is done with
            if p["fwd_ev"] is not None:
                self.upl.wait_event(p["fwd_ev"])
            slot["d_rows"][:U].copy_(slot["rows"][:U], non_blocking=True)
            slot["d_uniq"][:U].copy_(slot["uniq"][:U], non_blocking=True)
            slot["d_inv"][:n].copy_(slot["inv"][:n], non_blocking=True)
            ev.record(self.upl)
        p["rows"], p["uniq"], p["inv"] = slot["d_rows"][:U], slot["d_uniq"][:U], slot["d_inv"][:n]
        return p

    def take(self):
        """The oldest prefetch as (uniq_gpu, uniq_cpu, inv_gpu, rows_gpu, slot), patched with the writes
        in flight when it was gathered; pass `slot` back to step()."""
        p = self._pending.popleft()
        p["future"].result()
        cur = torch.cuda.current_stream()
        cur.wait_event(p["slot"]["event"])           # the uploads have landed
        rows, uniq = p["rows"], p["uniq"]
        live = [(wid, w_uniq) for wid, w_uniq, _ in self._writes if wid >= p["floor"] and w_uniq.numel() > 0]
        if live:
            # newest write wins: (ring slot, position) of the last hit per row, then one gather
            U = uniq.numel()
            best_slot = torch.full((U,), -1, dtype=torch.int64, device=uniq.device)
            best_pos = torch.zeros(U, dtype=torch.int64, device=uniq.device)
            for wid, w_uniq in live:
                pos = torch.searchsorted(w_uniq, uniq).clamp_(max=w_uniq.numel() - 1)
                hit = w_uniq[pos] == uniq
                best_slot = torch.where(hit, torch.full_like(best_slot, wid % self._prow_R), best_slot)
                best_pos = torch.where(hit, pos, best_pos)
            cap = self._prow_ring.shape[1]
            src = self._prow_ring.view(-1, self.LD)[best_slot.clamp_min(0) * cap + best_pos]
            rows.copy_(torch.where((best_slot >= 0).unsqueeze(1), src, rows))
        return uniq, p["uniq_cpu"], p["inv"], rows, p["slot"]

    # ---- synchronous gather (eval, first warmup step) --------------------
    def gather_sync(self, ids_gpu):
        self.flush()
        b = self.buckets(ids_gpu).reshape(-1)
        uniq, inv = torch.unique(b, return_inverse=True)
        uniq_cpu = uniq.to("cpu")
        U = uniq_cpu.numel()
        if self._stage_sync is None or self._stage_sync.shape[0] < U:
            self._stage_sync = torch.empty(U, self.LD, dtype=torch.bfloat16).pin_memory()
        stage = self._stage_sync[:U]
        torch.index_select(self.weight, 0, uniq_cpu, out=stage)
        return uniq, uniq_cpu, inv, stage.to(self.device, non_blocking=True), None

    # ---- update ----------------------------------------------------------
    def step(self, uniq, uniq_cpu, leaf, lr, slot):
        """Row update for one step; the write-back is issued by issue_writes() after the next forward."""
        U = uniq.numel()
        cur = torch.cuda.current_stream()
        grad = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
        new_rows = row_update(grad.contiguous(), leaf.detach(), self.accum, uniq, lr, self.n_site, self.per_slot,
                              self.decay, self.accum_beta, seed=self._write_id * 8 + self.rank, out=self._prow_slot(U))
        if slot is not None:
            slot["consumed"].record(cur)             # the gather slot may be refilled
        ready = torch.cuda.Event()
        ready.record(cur)                            # new_rows is final
        wid = self._write_id
        self._write_id += 1
        self._writes.append((wid, uniq.clone(), new_rows))   # uniq is a ring view: clone it
        self._deferred = (wid, U, new_rows, uniq_cpu, ready)

    def _prow_slot(self, U):
        cap = max(U, self._max_ids)
        if self._prow_ring is None or self._prow_ring.shape[1] < cap:
            self._prow_ring = torch.empty(self._prow_R, cap, self.LD, dtype=torch.bfloat16, device=self.device)
        s = self._write_id % self._prow_R
        assert all(w[0] % self._prow_R != s for w in self._writes), "patch ring too shallow for the retained writes"
        return self._prow_ring[s, :U]

    def issue_writes(self):
        """D2H the last update's rows under the forward just enqueued, then scatter them on the host."""
        if self._deferred is None:
            return
        wid, U, new_rows, uniq_cpu, ready = self._deferred
        self._deferred = None
        s = self._w_next
        self._w_next = (s + 1) % self.ring
        slot = self._w_slots[s]
        if slot is None or slot["cap"] < U:
            cap = max(U, self._max_ids)
            slot = dict(cap=cap, stage=torch.empty(cap, self.LD, dtype=torch.bfloat16).pin_memory(),
                        event=torch.cuda.Event(), future=None)
            self._w_slots[s] = slot
        if slot["future"] is not None and not slot["future"].done():
            slot["future"].result()
        stage, ev = slot["stage"][:U], slot["event"]
        self.side.wait_event(ready)
        if self._fwd_ev is not None:
            self.side.wait_event(self._fwd_ev)
        with torch.cuda.stream(self.side):
            stage.copy_(new_rows, non_blocking=True)
            ev.record(self.side)
        new_rows.record_stream(self.side)
        weight = self.weight

        def _write(_keep=(new_rows, uniq_cpu)):
            ev.synchronize()
            weight.index_copy_(0, uniq_cpu, stage)

        slot["future"] = self.write_exec.submit(_write)
        self._write_futures[wid] = slot["future"]
        # forget writes no pending or future prefetch can need
        low = min([q["floor"] for q in self._pending] + [self._oldest_unlanded_write()])
        while self._writes and self._writes[0][0] < low:
            self._writes.popleft()
        for k in [k for k in self._write_futures if k < low]:
            del self._write_futures[k]

    # ---- lifecycle -------------------------------------------------------
    def flush(self):
        self.issue_writes()
        for p in list(self._pending):
            p["future"].result()
        for f in list(self._write_futures.values()):
            f.result()

    def reset(self):
        self.flush()
        self._pending.clear()
        if self.rank == 0:
            self.weight.zero_()
        self.accum.zero_()
        self._writes.clear()
        self._write_futures.clear()

    def shutdown(self, unlink_shared=False):
        self.flush()
        self.gather_exec.shutdown(wait=True)
        self.write_exec.shutdown(wait=True)
        if unlink_shared and self.shared_path:
            self.weight = torch.empty(0)
            try:
                os.unlink(self.shared_path)
            except FileNotFoundError:
                pass

"""Exact corpus counts of the grams ending at the validation and fit positions (orders 2..8 over the training shards
the loader never reads, counted on the GPUs after training) and the chain mixture they feed: from P = p_model(y),
P = (1 - lam) P + lam C / N per order (shortest first), then the hint rows' and the in-document links; lam by
(order, log2 N)."""
import os
import threading

import numpy as np
import torch
import torch.distributed as dist
import triton
import triton.language as tl


MAX_ORDER, NBINS, BOS = 8, 25, 50256
ORDERS = list(range(2, MAX_ORDER + 1))
HB, FILT, FW, DUMMY = 27, (1 << 2) | (1 << 3) | (1 << 4), 21, 1 << 16  # home buckets, filtered gram lengths, filter
_B = tl.constexpr(0x5851F42D4C957F2D)
_SALT = tl.constexpr(0x1B873593)
_SIGN = tl.constexpr(-9223372036854775808)
# SCAN_HINTS (default on; SCAN_HINTS=0: #380's 20-shard validation rows): the scan also takes the corpus's next token wherever a validation position's 2- or 3-token context occurs
# (at most SCAN_HINT_CAP per context, an equal part on each rank; the 4-token context too, its row in the 3-token rows' cells), and the validation hint rows of those two lengths are
# built from these samples of all the scanned shards, in place of the 20-shard indexes' rows
SCAN_HINTS = os.environ.get("SCAN_HINTS", "1") == "1"
SCAN_HINT_CAP = 96
# SCAN_HINTS_SKIP (default on): a context whose 20-shard row already holds a large sample (ValidationCache.lookup's marks) keeps
# that row: the scan takes no samples of it, and none are sent or tallied
SCAN_HINTS_SKIP = SCAN_HINTS and os.environ.get("SCAN_HINTS_SKIP", "1") == "1"
SCAN_HINT_KEYS = (4, 3, 2)  # the sampled context lengths; a position's row is its longest context's with a match
NK = len(SCAN_HINT_KEYS)  # the sampled context lengths: 2 .. NK + 1 (bit length - 2 of a mark)
SCAN_HINT_MARKS = (1 << NK) - 1  # every context's mark
ROW_WIDTHS = (2, 8)  # a context counted at most this often over all the ranks is sent and tallied at this width per rank


@triton.jit
def _fmix(h):
    h = h ^ ((h >> 33) & 0x7FFFFFFF)
    h = h * 0x62A9D9ED799705F5
    h = h ^ ((h >> 29) & 0x7FFFFFFFF)
    h = h * 0x4BE98134A5976FD3
    h = h ^ ((h >> 32) & 0xFFFFFFFF)
    return h | 1


@triton.jit
def _bhome(key, HB: tl.constexpr):
    return ((key ^ _SIGN) >> (64 - HB)) & ((1 << HB) - 1)


@triton.jit
def _need(key, BLOCK: tl.constexpr):
    one = tl.full([BLOCK], 1, tl.int64)
    return (one << (key & 63)) | (one << ((key >> 6) & 63)) | (one << ((key >> 12) & 63))


@triton.jit
def _insert(tab_ptr, key, live, lane, dummy_base, HB: tl.constexpr):
    b = _bhome(key, HB)
    k0 = tl.load(tab_ptr + 4 * b + 1, mask=live, other=0, cache_modifier=".cg")
    k1 = tl.load(tab_ptr + 4 * b + 3, mask=live, other=0, cache_modifier=".cg")
    todo = live & (k0 != key) & (k1 != key)
    s = 2 * b + tl.where(k0 == 0, 0, tl.where(k1 == 0, 1, 2))
    while tl.max(todo.to(tl.int32), axis=0) > 0:
        p = tl.where(todo, 2 * s + 1, dummy_base + lane)
        old = tl.atomic_cas(tab_ptr + p, tl.where(todo, 0, 1).to(tl.int64), key)
        todo = todo & (old != 0) & (old != key)
        s = tl.where(todo, s + 1, s)
    return tl.where(k0 == key, 2 * b, tl.where(k1 == key, 2 * b + 1, s))  # the key's slot


@triton.jit
def _lookup(tab_ptr, key, live, HB: tl.constexpr):
    b = _bhome(key, HB)
    k0 = tl.load(tab_ptr + 4 * b + 1, mask=live, other=0)
    k1 = tl.load(tab_ptr + 4 * b + 3, mask=live, other=0)
    found = live & ((k0 == key) | (k1 == key))
    widx = 4 * b + tl.where(k0 == key, 0, 2)
    todo = live & (k0 != key) & (k1 != key) & (k0 != 0) & (k1 != 0)
    while tl.max(todo.to(tl.int32), axis=0) > 0:
        b = tl.where(todo, b + 1, b)
        k0 = tl.load(tab_ptr + 4 * b + 1, mask=todo, other=0)
        k1 = tl.load(tab_ptr + 4 * b + 3, mask=todo, other=0)
        hit = todo & ((k0 == key) | (k1 == key))
        found = found | hit
        widx = tl.where(hit, 4 * b + tl.where(k0 == key, 0, 2), widx)
        todo = todo & (k0 != key) & (k1 != key) & (k0 != 0) & (k1 != 0)
    return found, widx


@triton.jit(do_not_specialize=["q0", "q1", "dummy_base", "nv"])
def _ins_kernel(tok_ptr, tgt_ptr, start_ptr, q0, q1, tab_ptr, dummy_base, filt_ptr, mark_ptr, nv, MAXO: tl.constexpr,
                HB: tl.constexpr, FILT: tl.constexpr, FW: tl.constexpr, HINTS: tl.constexpr, NK: tl.constexpr, BLOCK: tl.constexpr):
    offs = q0 + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = offs < q1
    lane = tl.arange(0, BLOCK).to(tl.int64)
    start = tl.load(start_ptr + offs, mask=live, other=0).to(tl.int64)
    y = tl.load(tgt_ptr + offs, mask=live, other=0).to(tl.int64)
    if HINTS:  # SCAN_HINTS: the next query (a validation position: its token is this one's target) and its mark
        nq = live & (offs + 1 < nv)
        nstart = tl.load(start_ptr + offs + 1, mask=nq, other=0).to(tl.int64)
        nmark = tl.load(mark_ptr + offs + 1, mask=nq, other=0).to(tl.int64)
    h = tl.zeros([BLOCK], dtype=tl.int64)
    pw = tl.full([BLOCK], 1, dtype=tl.int64)
    for m in tl.static_range(1, MAXO + 1):
        ok = live & (offs - (m - 1) >= start)
        t = tl.load(tok_ptr + offs - (m - 1), mask=ok, other=0).to(tl.int64)
        h = h + (t + 1) * pw
        pw = pw * _B
        key = _fmix(h * _B + (y + 1) + (m + 1) * _SALT)
        slot = _insert(tab_ptr, key, ok, lane, dummy_base, HB)
        if HINTS and m <= NK:
            # this gram is the next query's (m + 1)-token context: if that query marks it (bit m - 1), the high half of
            # the slot's count word takes nv - the first such query (the same on every rank; 0: no query marks it)
            tl.atomic_max(tab_ptr + 2 * slot, (nv - (offs + 1)) << 32,
                          mask=ok & nq & (((nmark >> (m - 1)) & 1) != 0) & (offs + 1 - m >= nstart), sem="relaxed")
        if (FILT >> (m + 1)) & 1:
            tl.atomic_or(filt_ptr + ((key >> 20) & ((1 << FW) - 1)), _need(key, BLOCK), mask=ok, sem="relaxed")


@triton.jit(do_not_specialize=["n", "nv"])
def _scan_kernel(corpus_ptr, n, tab_ptr, filt_ptr, buf_ptr, nv, CAP: tl.constexpr, NK: tl.constexpr, MAXLEN: tl.constexpr,
                 HB: tl.constexpr, FILT: tl.constexpr, FW: tl.constexpr, BLOCK: tl.constexpr):
    # int64 positions: a rank's corpus share can be over 2^31 tokens
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    alive = offs < n
    h = tl.zeros([BLOCK], dtype=tl.int64)
    pw = tl.full([BLOCK], 1, dtype=tl.int64)
    for L in tl.static_range(1, MAXLEN + 1):
        alive = alive & (offs - (L - 1) >= 0)
        t = tl.load(corpus_ptr + offs - (L - 1), mask=alive, other=0).to(tl.int64) & 0xFFFF
        h = h + (t + 1) * pw
        pw = pw * _B
        if L >= 2:
            key = _fmix(h + L * _SALT)
            if (FILT >> L) & 1:
                wv = tl.load(filt_ptr + ((key >> 20) & ((1 << FW) - 1)), mask=alive, other=0)
                need = _need(key, BLOCK)
                alive = alive & ((wv & need) == need)
            found, widx = _lookup(tab_ptr, key, alive, HB)
            old = tl.atomic_add(tab_ptr + widx, 1, mask=found, sem="relaxed")
            if CAP and L <= NK + 1:
                # a marked context's first CAP occurrences: the count word before the add holds the sample's index (low
                # half) and the context's sample row (high half: nv - its first query; 0: not marked); the sample, the
                # corpus's next token, is read only then
                seen, mine = old & 0xFFFFFFFF, old >> 32
                put = found & (offs + 1 < n) & (mine > 0) & (seen < CAP)
                tl.store(buf_ptr + (NK * (nv - mine) + (L - 2)) * CAP + seen, tl.load(corpus_ptr + offs + 1, mask=put, other=0), mask=put)
            alive = found


@triton.jit(do_not_specialize=["n", "nv"])
def _find_kernel(tok_ptr, tgt_ptr, start_ptr, n, tab_ptr, out_ptr, ctx_ptr, nv, MAXO: tl.constexpr, HB: tl.constexpr,
                 HINTS: tl.constexpr, NK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = offs < n
    start = tl.load(start_ptr + offs, mask=live, other=0).to(tl.int64)
    y = tl.load(tgt_ptr + offs, mask=live, other=0).to(tl.int64)
    h = tl.zeros([BLOCK], dtype=tl.int64)
    pw = tl.full([BLOCK], 1, dtype=tl.int64)
    for m in tl.static_range(1, MAXO + 1):
        ok = live & (offs - (m - 1) >= start)
        t = tl.load(tok_ptr + offs - (m - 1), mask=ok, other=0).to(tl.int64)
        h = h + (t + 1) * pw
        pw = pw * _B
        if m >= 2:  # the context's count (column m - 2), then the context and target's (column MAXO + m - 3)
            found, widx = _lookup(tab_ptr, _fmix(h + m * _SALT), ok, HB)
            word = tl.load(tab_ptr + widx, mask=found, other=0)  # the count in its low half
            tl.store(out_ptr + offs * (2 * MAXO - 2) + (m - 2), word.to(tl.int32), mask=live)
            if HINTS and m <= NK + 1:  # SCAN_HINTS, a validation position: its context's sample row (-1: not marked, or no such context)
                tl.store(ctx_ptr + offs * NK + (m - 2), tl.where(word >> 32 > 0, NK * (nv - (word >> 32)) + (m - 2), -1).to(tl.int32),
                         mask=offs < nv)
            found, widx = _lookup(tab_ptr, _fmix(h * _B + (y + 1) + (m + 1) * _SALT), ok, HB)
            tl.store(out_ptr + offs * (2 * MAXO - 2) + (MAXO + m - 3), tl.load(tab_ptr + widx, mask=found, other=0).to(tl.int32), mask=live)


@triton.jit(do_not_specialize=["base", "stride"])
def _row_kernel(s_ptr, base, stride, out_ptr, WORLD: tl.constexpr, W: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr):
    # one context per program: its samples (0xFFFF: unused), W from each of WORLD ranks (rank r's at base + r * stride +
    # W * the context) -> its row as the index's row() makes it at the key's own length: the cell, the top 8 tokens by
    # count (ties: lower token). The WORLD * W samples are tallied in two parts of W1 and (up to) W2, powers of 2 (64 +
    # 32 at 96, not one block of 128), each against itself and the other
    c = tl.program_id(0).to(tl.int64)
    i, j = tl.arange(0, W1), W1 + tl.arange(0, W2)
    a = tl.load(s_ptr + base + (i // W) * stride + c * W + i % W).to(tl.int32) & 0xFFFF
    b = tl.load(s_ptr + base + (j // W) * stride + c * W + j % W, mask=j < WORLD * W, other=-1).to(tl.int32) & 0xFFFF
    ab = (a[:, None] == b[None, :]).to(tl.int32)
    cnt_a = tl.sum((a[:, None] == a[None, :]).to(tl.int32), axis=1) + tl.sum(ab, axis=1)
    cnt_b = tl.sum((b[:, None] == b[None, :]).to(tl.int32), axis=1) + tl.sum(ab, axis=0)
    key_a = tl.where(a != 0xFFFF, cnt_a * 65536 + (65535 - a), -1)  # a token's samples all have its key
    key_b = tl.where(b != 0xFFFF, cnt_b * 65536 + (65535 - b), -1)
    total = tl.sum((a != 0xFFFF).to(tl.int32), axis=0) + tl.sum((b != 0xFFFF).to(tl.int32), axis=0)
    top = tl.maximum(tl.max(key_a, axis=0), tl.max(key_b, axis=0))
    purity = (top >> 16).to(tl.float64) / tl.maximum(total, 1).to(tl.float64)
    count_bin = ((total >= 2).to(tl.int32) + (total >= 3).to(tl.int32) + (total >= 5).to(tl.int32) + (total >= 9).to(tl.int32)
                 + (total >= 17).to(tl.int32) + (total >= 33).to(tl.int32))
    f64 = tl.zeros([1], dtype=tl.float64)  # the bin edges as float64 (as the index compares them)
    purity_bin = tl.sum((purity >= f64 + 0.3).to(tl.int32) + (purity >= f64 + 0.5).to(tl.int32) + (purity >= f64 + 0.7).to(tl.int32)
                        + (purity >= f64 + 0.85).to(tl.int32) + (purity >= f64 + 0.95).to(tl.int32), axis=0)
    tl.store(out_ptr + c * 9, tl.where(total > 0, (7 + count_bin) * 6 + purity_bin, 0))
    for i in tl.static_range(8):  # the largest key left; all its copies go at once
        m = tl.maximum(tl.max(key_a, axis=0), tl.max(key_b, axis=0))
        tl.store(out_ptr + c * 9 + 1 + i, tl.where(m >= 0, (65535 - (m & 0xFFFF)) | ((m >> 16) << 16), 0))
        key_a = tl.where(key_a == m, -1, key_a)
        key_b = tl.where(key_b == m, -1, key_b)


def segment_starts(tokens, chunk_starts):  # per position the first index its context may reach (a blocked cummax)
    n = len(tokens)
    s = torch.maximum(torch.where(tokens == BOS, torch.arange(n, device=tokens.device), 0), chunk_starts)
    loc = torch.cummax(torch.cat([s, s.new_full((-n % 4096,), -1)]).view(-1, 4096), 1).values
    pre = torch.cat([s.new_full((1,), -1), torch.cummax(loc[:-1, -1], 0).values])
    return torch.maximum(loc, pre[:, None]).view(-1)[:n]


class ExactCounts:  # one rank's corpus share on its GPU and the query table (buckets of two [count, key] slots)
    def __init__(self, train_files, exclude, rank, world, device, q_chunk, pinned=False, n_val=0):  # q_chunk: each query's segment start
        self.device, self.files = device, sorted(train_files)[exclude:][rank::world]
        self.n = sum((os.path.getsize(f) - 1024) // 2 for f in self.files)
        self.bufs = [torch.empty(max((os.path.getsize(f) - 1024) // 2 for f in self.files), dtype=torch.int16, pin_memory=True)
                     for _ in range(2)] if pinned else None
        self.corpus = torch.empty(self.n, dtype=torch.int16, device=device)
        self.tab = torch.zeros(4 * ((1 << HB) + (1 << 12)) + DUMMY, dtype=torch.int64, device=device)
        self.filt = torch.zeros(1 << FW, dtype=torch.int64, device=device)
        self.q_chunk, (self.q_tok, self.q_tgt) = q_chunk, torch.zeros(2, len(q_chunk), dtype=torch.int32, device=device)
        self.out, self.loaded = torch.zeros(len(q_chunk), 2 * MAX_ORDER - 2, dtype=torch.int32, device=device), threading.Event()
        self.queries = threading.Event()  # set once q_tok / q_tgt hold the queries (launch_marked waits for it)
        # SCAN_HINTS (n_val: the first n_val queries are the validation positions). A context's id is the first query
        # that marks it, kept in the table slot's count word (the high half; the counts are below 2^32). Per context this
        # rank's cap samples (row NK * id + length - 2); per validation position its NK contexts' sample rows and its mark:
        # the contexts it has sampled (bit 0: the 2-token one, bit 1: the 3-token one, bit 2: the
        # 4-token one; all, unless SCAN_HINTS_SKIP sets the marks before the count)
        self.nv, self.cap, self.rank = n_val, SCAN_HINT_CAP // world if n_val else 0, rank
        if n_val:
            self.marks = torch.full((n_val,), SCAN_HINT_MARKS, dtype=torch.int8, device=device)
            assert NK * n_val <= 1 << 25  # hint_rows packs a sample row and its width's index in an int32
            self.buf = torch.full((NK * n_val, self.cap), -1, dtype=torch.int16, device=device)
            self.ctx = torch.zeros(n_val, NK, dtype=torch.int32, device=device)
        # compile the kernels: the scan as launch() calls it, with the share's own length (an int64 argument above 2^31 tokens,
        # which is another kernel), over its first blocks only
        tok = torch.randint(0, 50000, (4096,), device=device, dtype=torch.int32)
        self._count(tok, tok, segment_starts(tok, torch.zeros_like(tok)), torch.zeros_like(self.out[:4096]), self.n, blocks=16)
        if n_val:  # as new again: no sample is taken; and the row kernel at each width
            self.buf.fill_(-1)
            self.widths = [w for w in ROW_WIDTHS if w < self.cap] + [self.cap]
            for w in self.widths:
                self._rows(torch.full((world * w,), -1, dtype=torch.int16, device=device), 0, w, torch.empty(1, 9, dtype=torch.int32, device=device), world, w)
        torch.cuda.synchronize(device)

    def load_corpus(self):  # a thread: this rank's corpus share into GPU memory, on a side stream
        def run():
            torch.cuda.set_device(self.device)
            stream, off, done = torch.cuda.Stream(self.device), 0, [torch.cuda.Event(), torch.cuda.Event()]
            with torch.cuda.stream(stream):
                for i, f in enumerate(self.files):
                    if self.bufs is None:
                        a = torch.from_numpy(np.fromfile(f, dtype=np.int16, offset=1024))
                    else:  # one read into a pinned buffer (two, in turn) and an asynchronous copy
                        a = self.bufs[i % 2][:(os.path.getsize(f) - 1024) // 2]
                        done[i % 2].synchronize()
                        view, got = memoryview(a.numpy()).cast("B"), 0
                        with open(f, "rb", buffering=0) as fh:
                            while got < view.nbytes:
                                got += os.preadv(fh.fileno(), [view[got:]], 1024 + got)
                    self.corpus[off:off + len(a)].copy_(a, non_blocking=self.bufs is not None)
                    done[i % 2].record(stream)
                    off += len(a)
            stream.synchronize()
            self.loaded.set()
        threading.Thread(target=run, daemon=True).start()

    def _count(self, tok, tgt, start, out, n_corpus, blocks=None):  # the query table, the scan of the corpus share, the lookups
        n = tok.numel()
        self.tab.zero_()
        self.filt.zero_()
        nv, marks, buf, ctx = (min(self.nv, n), self.marks, self.buf, self.ctx) if self.nv else (0, tok, tok, tok)  # not SCAN_HINTS: unused
        for q0, q1 in ((0, -(-n // 2)), (-(-n // 2), n)):
            _ins_kernel[(max(1, triton.cdiv(q1 - q0, 256)),)](tok, tgt, start, q0, q1, self.tab, self.tab.numel() - DUMMY,
                                                              self.filt, marks, nv, MAXO=MAX_ORDER, HB=HB, FILT=FILT, FW=FW,
                                                              HINTS=nv > 0, NK=NK, BLOCK=256)
        grid = (min(blocks or n_corpus, triton.cdiv(n_corpus, 256)),)
        _scan_kernel[grid](self.corpus, n_corpus, self.tab, self.filt, buf, nv, CAP=self.cap, NK=NK, MAXLEN=MAX_ORDER + 1, HB=HB,
                           FILT=FILT, FW=FW, BLOCK=256)
        _find_kernel[(triton.cdiv(n, 256),)](tok, tgt, start, n, self.tab, out, ctx, nv, MAXO=MAX_ORDER, HB=HB, HINTS=nv > 0,
                                             NK=NK, BLOCK=256)

    @torch.no_grad()
    def launch(self):  # after training, the queries in q_tok / q_tgt: this rank's counts into self.out
        self.loaded.wait()
        self._count(self.q_tok, self.q_tgt, segment_starts(self.q_tok, self.q_chunk), self.out, self.n)

    def launch_marked(self, marks):  # SCAN_HINTS_SKIP, on the thread that looked the marks up: the launch, once the queries are in
        self.queries.wait()
        torch.cuda.set_device(self.device)
        self.marks.copy_(marks)
        self.launch()

    @staticmethod
    def _rows(samples, base, stride, rows, world, w):  # rows.shape[0] contexts' rows from each rank's w samples of them
        w1 = 1 << (world * w - 1).bit_length() - 1  # the largest power of 2 below world * w
        _row_kernel[(rows.shape[0],)](samples, base, stride, rows, WORLD=world, W=w, W1=w1, W2=1 << (world * w - w1 - 1).bit_length())

    @torch.no_grad()
    def hint_rows(self, own, want, tags):
        """SCAN_HINTS, after the counts' all-reduce: the hint rows of this rank's validation positions `own` from the
        contexts they `want` (bits as the marks; a position with a longer match wants none), zeros where there is none:
        the row of the longest context that is wanted and has a match (3-token, else 2-token; the
        4-token one first); cells offset by tags = (the 2-token rows' tag, the 3-token rows', which the 4-token rows
        take too). Each distinct context's row is built once, from
        every rank's samples of it: the ranks tell each other the sample rows they need, and each sends every other its
        samples of them -- only its first 2 or 8 where the context's count over all the ranks is no more than that (no
        rank has more samples of it), and those rows are tallied at that width."""
        world, dev = dist.get_world_size(), self.device
        widths = self.widths
        ctx, cnt = self.ctx[own], self.out[own, :NK]
        has = ((want[:, None] >> torch.arange(NK, device=dev)) & 1 > 0) & (cnt > 0) & (ctx >= 0)
        k = torch.where(has[:, 1], 1, torch.where(has[:, 0], 0, -1))  # the row's context: 1: 3-token, 0: 2-token, -1: no row
        if NK == 3:
            k = torch.where(has[:, 2], 2, k)  # 2: 4-token
        row, n_all = (x.gather(1, k.clamp_min(0)[:, None])[:, 0] for x in (ctx, cnt))
        tier = sum((n_all > w).int() for w in widths[:-1])  # by the context's count: the index of the width its samples fit
        need, inv = torch.unique(torch.where(k >= 0, row | (tier << 25), 0), return_inverse=True)  # the tiers in turn
        mine = torch.stack([(need >> 25 == i).sum() for i in range(len(widths))])
        sizes = torch.zeros(world, len(widths), dtype=torch.int64, device=dev)
        wanted = torch.zeros(world, len(own), dtype=torch.int32, device=dev)
        dist.all_gather_into_tensor(sizes.view(-1), mine)
        dist.all_gather_into_tensor(wanted.view(-1), torch.cat([need, need.new_zeros(len(own) - len(need))]))
        sizes = sizes.tolist()
        # the samples travel as bytes (NCCL has no int16): to each rank its tiers in turn, each [contexts, width], the
        # message padded to whole pages (one whose size is not a multiple of 16 bytes took NCCL 5 times as long)
        send, cuts, flat = [], [], self.buf.view(-1)
        for r in range(world):
            a = 0
            for n, w in zip(sizes[r], widths):
                at = (wanted[r, a:a + n] & 0x1FFFFFF).long() * self.buf.shape[1]
                send.append(flat[at[:, None] + torch.arange(w, device=dev)].view(-1))
                a += n
            cuts.append(2 * sum(n * w for n, w in zip(sizes[r], widths)))
            send.append(flat.new_zeros(-cuts[-1] % 4096 // 2))
            cuts[-1] += -cuts[-1] % 4096
        got = torch.empty(world * cuts[self.rank], dtype=torch.uint8, device=dev)
        dist.all_to_all_single(got, torch.cat(send).view(torch.uint8), [cuts[self.rank]] * world, cuts)
        torch.cuda.synchronize(dev)
        rows, a, base = torch.empty(len(need), 9, dtype=torch.int32, device=dev), 0, 0
        for n, w in zip(sizes[self.rank], widths):  # rank r's samples of this tier at base + r * its whole message
            if n:
                self._rows(got.view(torch.int16), base, cuts[self.rank] // 2, rows[a:a + n], world, w)
            a, base = a + n, base + n * w
        rows = rows[inv] * (k >= 0)[:, None]
        rows[:, 0] += torch.tensor(tags, dtype=torch.int32, device=dev)[k.clamp(0, 1)] * (rows[:, 0] > 0)
        torch.cuda.synchronize(dev)
        return rows


def context_counts(out, offs):  # N, C at query positions offs (the kernel leaves 0 where the context leaves the segment)
    N = out[offs, :len(ORDERS)].float()
    return N, torch.minimum(out[offs, len(ORDERS):].float(), N)


def components(N, C, extra=()):  # Q [n, K], G [n, K] (absent: the dump group), the group count; innermost first
    J, has = N.shape[1], N > 0
    Q = [torch.where(has, C / N.clamp_min(1), 0.0).double()]
    G = [torch.where(has, torch.clamp(torch.floor(torch.log2(N.clamp_min(1))), 0, NBINS - 1).long() + NBINS * torch.arange(J, device=N.device), -1)]
    total = J * NBINS
    for q, g, n in extra:
        Q.append(q.double()[:, None])
        G.append(torch.where(g >= 0, g.long() + total, -1)[:, None])
        total += n
    Q, G = torch.cat(Q, 1), torch.cat(G, 1)
    return Q, torch.where(G >= 0, G, total), total


def chain_prob(p, Q, L):  # the chain's probability of the target, with the components' weights L [n, K]
    P = p.double()
    for j in range(Q.shape[1]):
        P = (1 - L[:, j]) * P + L[:, j] * Q[:, j]
    return P


def gate_L(h, G, theta, W, b, n_groups):  # the weights from the hidden state, sigmoid(theta[G] + h W + b)
    return torch.sigmoid(torch.cat([theta, theta.new_full((1,), -30.0)])[G] + (h.float() @ W + b).double()) * (G < n_groups)


@triton.jit(do_not_specialize=["n", "K"])
def _gate_chain_kernel(L_ptr, q_ptr, p_ptr, dL_ptr, n, inv_n, K, BLOCK: tl.constexpr):
    # d(-inv_n * log P) / dL_j per row, each P_{j-1} recomputed from p (a scratch store read back raced on the GPU)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    ok = rows < n
    p0 = tl.load(p_ptr + rows, mask=ok, other=1.0)
    P = p0
    for j in range(K):
        lam = tl.load(L_ptr + rows * K + j, mask=ok, other=0.0)
        q = tl.load(q_ptr + rows * K + j, mask=ok, other=0.0)
        P = (1 - lam) * P + lam * q
    gP = tl.where(P > 1e-30, -inv_n / tl.maximum(P, 1e-30), 0.0)
    for jj in range(K):
        j = K - 1 - jj
        prev = p0
        for i in range(j):
            lam_i = tl.load(L_ptr + rows * K + i, mask=ok, other=0.0)
            q_i = tl.load(q_ptr + rows * K + i, mask=ok, other=0.0)
            prev = (1 - lam_i) * prev + lam_i * q_i
        lam = tl.load(L_ptr + rows * K + j, mask=ok, other=0.0)
        q = tl.load(q_ptr + rows * K + j, mask=ok, other=0.0)
        tl.store(dL_ptr + rows * K + j, gP * (q - prev), mask=ok)
        gP = gP * (1 - lam)


def _chain_grad_L(p, Q, L, inv_n):
    dL = torch.empty_like(Q)
    _gate_chain_kernel[(triton.cdiv(Q.shape[0], 256),)](L, Q, p, dL, Q.shape[0], inv_n, Q.shape[1], BLOCK=256)
    return dL


class ChainFit:
    """The chain's weights: 10 diagonal Newton steps from lam 0.01 (float32) on static buffers, the group sums
    all-reduced over the ranks each step. The hidden-state gate: 100 Adam steps on random batches of the fit
    positions, gradients written out by hand and averaged over the ranks."""

    def __init__(self, F, K, n_groups, device, d=0, world=1, rank=0):
        self.ng, self.device, self.K, self.d, self.world, self.rank = n_groups, device, K, d, world, rank
        self.G, self.Gs = torch.zeros(F, K, dtype=torch.int64, device=device), torch.zeros(F * K, dtype=torch.int64, device=device)
        self.Q, self.p = torch.zeros(F, K, device=device), torch.zeros(F, 1, device=device)
        self.lam, self.sums = torch.zeros(n_groups + 1, device=device), torch.zeros(2, n_groups + 1, device=device)
        if d:
            self.B, z = min(65536, F), lambda *s, dtype=torch.float64: torch.zeros(*s, dtype=dtype, device=device)
            self.h, self.gp, self.gQ, self.gG = z(F, d, dtype=torch.bfloat16), z(F), z(F, K), z(F, K, dtype=torch.int64)
            self.x, self.m, self.v, self.g = (z(n_groups + d * K + K) for _ in range(4))
            self.idx, self.gspread = z(self.B, dtype=torch.int64), ((torch.arange(self.B, device=device) % 64) * (n_groups + 1))[:, None]
            self._gate_body()  # compiles the chain kernel before the clock

    def _local(self):
        L = self.lam[self.G]
        keep = torch.flip(torch.cumprod(torch.flip(1 - L, [1]), 1), [1])
        after = torch.cat([keep[:, 1:], torch.ones_like(keep[:, :1])], 1)
        inner, P = [], self.p[:, 0]
        for j in range(self.Q.shape[1]):
            inner.append(P)
            P = (1 - L[:, j]) * P + L[:, j] * self.Q[:, j]
        d = after * (self.Q - torch.stack(inner, 1)) / P.clamp_min(1e-15)[:, None]
        s = torch.zeros(2, 64 * (self.ng + 1), device=self.device)
        s[0].index_add_(0, self.Gs, d.flatten())
        s[1].index_add_(0, self.Gs, (d * d).flatten())
        self.sums.copy_(s.view(2, 64, self.ng + 1).sum(1))

    def _update(self):
        sums, lam = self.sums, self.lam
        lam.copy_(torch.where(sums[1] > 0, (lam + sums[0] / sums[1].clamp_min(1e-30)).clamp(0, 0.9999), torch.zeros_like(lam)))
        lam[-1:].zero_()

    @torch.no_grad()
    def em(self, p, Q, G):
        self.G.copy_(G)
        self.Gs.copy_((G + (torch.arange(G.shape[0], device=G.device) % 64)[:, None] * (self.ng + 1)).flatten())
        self.Q.copy_(Q)
        self.p.copy_(p[:, None])
        self.lam.fill_(0.01)
        self.lam[-1:].zero_()
        for _ in range(10):
            self._local()
            dist.all_reduce(self.sums)
            self._update()
        return self.lam[:-1].float().clone()

    def _gate_body(self):
        ng, d, K, B = self.ng, self.d, self.K, self.B
        idx, x = self.idx, self.x
        hb, Gb = self.h[idx].float(), self.gG[idx]
        theta_e = torch.cat([x[:ng], x.new_full((1,), -30.0)])
        W, b = x[ng:ng + d * K].view(d, K).float(), x[ng + d * K:].float()
        s = torch.sigmoid(theta_e[Gb] + (hb @ W + b).double())
        live = Gb < ng
        dgl = _chain_grad_L(self.gp[idx], self.gQ[idx].contiguous(), (s * live).contiguous(), 1.0 / B) * (s * (1 - s) * live)
        gt = torch.zeros(64 * (ng + 1), dtype=torch.float64, device=self.device)
        gt.index_add_(0, (Gb + self.gspread).flatten(), dgl.flatten())
        self.g.copy_(torch.cat([gt.view(64, ng + 1).sum(0)[:ng], (hb.t() @ dgl.float()).double().flatten(), dgl.sum(0)]))

    @torch.no_grad()
    def gate(self, h, p, Q, G, lam):
        ng, d, K = self.ng, self.d, self.K
        self.h.copy_(h)
        self.gp.copy_(p)
        self.gQ.copy_(Q)
        self.gG.copy_(G)
        x, m, v, g = self.x, self.m, self.v, self.g
        x[:ng].copy_(torch.logit(lam.double().clamp(1e-4, 1 - 1e-4)))
        x[ng:].zero_()
        m.zero_()
        v.zero_()
        gen = torch.Generator(device=self.device).manual_seed(4321 + self.rank)
        idx_all = [torch.randint(0, h.shape[0], (self.B,), device=self.device, generator=gen) for _ in range(100)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, 101):
            self.idx.copy_(idx_all[t - 1])
            self._gate_body()
            dist.all_reduce(g)
            g /= self.world
            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            denom = (v.sqrt() / (1 - beta2 ** t) ** 0.5).add_(eps)
            x.addcdiv_(m, denom, value=-2e-3 / (1 - beta1 ** t))
        return x[:ng].clone(), x[ng:ng + d * K].view(d, K).float(), x[ng + d * K:].float()

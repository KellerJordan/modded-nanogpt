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


@triton.jit(do_not_specialize=["q0", "q1", "dummy_base"])
def _ins_kernel(tok_ptr, tgt_ptr, start_ptr, q0, q1, tab_ptr, dummy_base, filt_ptr, MAXO: tl.constexpr,
                HB: tl.constexpr, FILT: tl.constexpr, FW: tl.constexpr, BLOCK: tl.constexpr):
    offs = q0 + (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    live = offs < q1
    lane = tl.arange(0, BLOCK).to(tl.int64)
    start = tl.load(start_ptr + offs, mask=live, other=0).to(tl.int64)
    y = tl.load(tgt_ptr + offs, mask=live, other=0).to(tl.int64)
    h = tl.zeros([BLOCK], dtype=tl.int64)
    pw = tl.full([BLOCK], 1, dtype=tl.int64)
    for m in tl.static_range(1, MAXO + 1):
        ok = live & (offs - (m - 1) >= start)
        t = tl.load(tok_ptr + offs - (m - 1), mask=ok, other=0).to(tl.int64)
        h = h + (t + 1) * pw
        pw = pw * _B
        key = _fmix(h * _B + (y + 1) + (m + 1) * _SALT)
        _insert(tab_ptr, key, ok, lane, dummy_base, HB)
        if (FILT >> (m + 1)) & 1:
            tl.atomic_or(filt_ptr + ((key >> 20) & ((1 << FW) - 1)), _need(key, BLOCK), mask=ok, sem="relaxed")


@triton.jit(do_not_specialize=["n"])
def _scan_kernel(corpus_ptr, n, tab_ptr, filt_ptr, MAXLEN: tl.constexpr, HB: tl.constexpr,
                 FILT: tl.constexpr, FW: tl.constexpr, BLOCK: tl.constexpr):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
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
            tl.atomic_add(tab_ptr + widx, 1, mask=found, sem="relaxed")
            alive = found


@triton.jit(do_not_specialize=["n"])
def _find_kernel(tok_ptr, tgt_ptr, start_ptr, n, tab_ptr, out_ptr, MAXO: tl.constexpr, HB: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
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
            tl.store(out_ptr + offs * (2 * MAXO - 2) + (m - 2), tl.load(tab_ptr + widx, mask=found, other=0).to(tl.int32), mask=live)
            found, widx = _lookup(tab_ptr, _fmix(h * _B + (y + 1) + (m + 1) * _SALT), ok, HB)
            tl.store(out_ptr + offs * (2 * MAXO - 2) + (MAXO + m - 3), tl.load(tab_ptr + widx, mask=found, other=0).to(tl.int32), mask=live)


def segment_starts(tokens, chunk_starts):  # per position the first index its context may reach (a blocked cummax)
    n = len(tokens)
    s = torch.maximum(torch.where(tokens == BOS, torch.arange(n, device=tokens.device), 0), chunk_starts)
    loc = torch.cummax(torch.cat([s, s.new_full((-n % 4096,), -1)]).view(-1, 4096), 1).values
    pre = torch.cat([s.new_full((1,), -1), torch.cummax(loc[:-1, -1], 0).values])
    return torch.maximum(loc, pre[:, None]).view(-1)[:n]


class ExactCounts:  # one rank's corpus share on its GPU and the query table (buckets of two [count, key] slots)
    def __init__(self, train_files, exclude, rank, world, device, q_chunk, pinned=False):  # q_chunk: each query's segment start
        self.device, self.files = device, sorted(train_files)[exclude:][rank::world]
        self.n = sum((os.path.getsize(f) - 1024) // 2 for f in self.files)
        self.bufs = [torch.empty(max((os.path.getsize(f) - 1024) // 2 for f in self.files), dtype=torch.int16, pin_memory=True)
                     for _ in range(2)] if pinned else None
        self.corpus = torch.empty(self.n, dtype=torch.int16, device=device)
        self.tab = torch.zeros(4 * ((1 << HB) + (1 << 12)) + DUMMY, dtype=torch.int64, device=device)
        self.filt = torch.zeros(1 << FW, dtype=torch.int64, device=device)
        self.q_chunk, (self.q_tok, self.q_tgt) = q_chunk, torch.zeros(2, len(q_chunk), dtype=torch.int32, device=device)
        self.out, self.loaded = torch.zeros(len(q_chunk), 2 * MAX_ORDER - 2, dtype=torch.int32, device=device), threading.Event()
        tok = torch.randint(0, 50000, (4096,), device=device, dtype=torch.int32)  # compile the kernels
        self._count(tok, tok, segment_starts(tok, torch.zeros_like(tok)), torch.zeros_like(self.out[:4096]), min(self.n, 4096))
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

    def _count(self, tok, tgt, start, out, n_corpus):  # the query table, the scan of the corpus share, the lookups
        n = tok.numel()
        self.tab.zero_()
        self.filt.zero_()
        for q0, q1 in ((0, -(-n // 2)), (-(-n // 2), n)):
            _ins_kernel[(max(1, triton.cdiv(q1 - q0, 256)),)](tok, tgt, start, q0, q1, self.tab, self.tab.numel() - DUMMY,
                                                              self.filt, MAXO=MAX_ORDER, HB=HB, FILT=FILT, FW=FW, BLOCK=256)
        _scan_kernel[(triton.cdiv(n_corpus, 256),)](self.corpus, n_corpus, self.tab, self.filt, MAXLEN=MAX_ORDER + 1, HB=HB,
                                                    FILT=FILT, FW=FW, BLOCK=256)
        _find_kernel[(triton.cdiv(n, 256),)](tok, tgt, start, n, self.tab, out, MAXO=MAX_ORDER, HB=HB, BLOCK=256)

    @torch.no_grad()
    def launch(self):  # after training, the queries in q_tok / q_tgt: this rank's counts into self.out
        self.loaded.wait()
        self._count(self.q_tok, self.q_tgt, segment_starts(self.q_tok, self.q_chunk), self.out, self.n)


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
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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

"""Kernels of the hashed bigram / trigram side channel.  Channel 1 of the shared table is the bigram
(x[t-1], x[t]), channel 2 the TRIGRAM (x[t-2], x[t-1], x[t]) in rows [half, V), sharing every buffer, cap,
payload and the Adam; only a small data-dependent row set is live on a step and all three cost only that set.
Row IDS stay int32, but the shard may exceed 2**31 ELEMENTS, so every offset here is built int64 before it
meets a base pointer."""
import numpy as np
import torch
import triton
import triton.language as tl

BGX_TRI_CH2, BGX_NCH = True, 2   # channel 2 carries the trigram; a third channel would move through every cap and payload


# 1. The x0_bigram producer and its sink gradient: two inductor-OPAQUE custom ops over one launch shape, the
# combine op absorbing the `+ bg_sink` add too.  Opacity is load-bearing -- aten.index is in AOTAutograd's
# recomputable set, so an aten spelling is inlined into all ~21 consumers at 3.402 ms/step; sign rows are
# exactly +-1, so this differs from that chain by one rounding to bf16, both ways.  Channel 2's row id is the
# trigram hash, so its SIGN row is the trigram mix too: the same `ss` in both kernels.
_BGX_XBLOCK, _BGX_WARPS = 1024, 4    # inductor's own `triton_poi_*` geometry at this tensor size

def _bgx_prep(tab, inp, d, mul):
    """Both ops' contract, output and grid: a [mul*T, D] bf16 output of static shape out of the
    caching allocator (so both are CUDA-graph safe), the T*D grid, and R-1 -- `tab` needs a
    power-of-two row count for the kernels' `% R` -> `& (R-1)` fold to be torch's remainder."""
    r, n = tab.shape[0], inp.shape[0] * d
    assert tab.dtype == torch.bfloat16 and tab.is_contiguous() and tab.shape[1] == d
    assert r & (r - 1) == 0, "bgx needs a power-of-two sign table (bigram_sign_table_rows)"
    assert inp.dtype == torch.int32 and inp.is_contiguous() and inp.ndim == 1
    return (torch.empty((mul * inp.shape[0], d), device=inp.device, dtype=torch.bfloat16),
            n, (triton.cdiv(n, _BGX_XBLOCK),), r - 1)

@triton.jit
def _bgx_combine_kernel_tri(w_ptr, bidx_ptr, tab_ptr, inp_ptr, out_ptr, T, xnumel,
                            D: tl.constexpr, RMASK: tl.constexpr, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex % D          # channel
    x1 = xindex // D         # token t in [0, T)
    # Value-identical to the 4-way embedding+where merge: every idx here is in range.
    br = tl.load(bidx_ptr + x1, xmask, eviction_policy="evict_last").to(tl.int64)
    sr = tl.load(bidx_ptr + (T + x1), xmask, eviction_policy="evict_last").to(tl.int64)
    cur = tl.load(inp_ptr + x1, xmask, eviction_policy="evict_last")
    pm1 = tl.load(inp_ptr + (x1 - 1), xmask & (x1 >= 1), other=0, eviction_policy="evict_last")
    pm2 = tl.load(inp_ptr + (x1 - 2), xmask & (x1 >= 2), other=0, eviction_policy="evict_last")
    si = tl.where(x1 >= 1, ((30011 * pm1) ^ (48271 * cur)) & RMASK, 0)
    ss = tl.where(x1 >= 2, ((26801 * pm2) ^ (39779 * pm1) ^ (58699 * cur)) & RMASK, 0)
    a = tl.load(w_ptr + br * D + x0, xmask).to(tl.float32)
    b = tl.load(w_ptr + sr * D + x0, xmask).to(tl.float32)
    sa = tl.load(tab_ptr + si * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    sb = tl.load(tab_ptr + ss * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    tl.store(out_ptr + xindex, (a * sa + b * sb).to(tl.bfloat16), xmask)

@triton.jit
def _bgx_sink_grad_kernel_tri(g_ptr, tab_ptr, inp_ptr, out_ptr, xnumel,
                              D: tl.constexpr, RMASK: tl.constexpr, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex % D
    x1 = xindex // D
    cur = tl.load(inp_ptr + x1, xmask, eviction_policy="evict_last")
    pm1 = tl.load(inp_ptr + (x1 - 1), xmask & (x1 >= 1), other=0, eviction_policy="evict_last")
    pm2 = tl.load(inp_ptr + (x1 - 2), xmask & (x1 >= 2), other=0, eviction_policy="evict_last")
    si = tl.where(x1 >= 1, ((30011 * pm1) ^ (48271 * cur)) & RMASK, 0)
    ss = tl.where(x1 >= 2, ((26801 * pm2) ^ (39779 * pm1) ^ (58699 * cur)) & RMASK, 0)
    g = tl.load(g_ptr + xindex, xmask).to(tl.float32)
    sa = tl.load(tab_ptr + si * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    sb = tl.load(tab_ptr + ss * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    # d_bg_sink == d_bge because `+ bg_sink` is an add; each half gets its channel's signs.
    tl.store(out_ptr + xindex, (g * sa).to(tl.bfloat16), xmask)
    tl.store(out_ptr + (xnumel + xindex), (g * sb).to(tl.bfloat16), xmask)

@torch.library.custom_op("nanogpt::bgx_combine", mutates_args=())
def bgx_combine_op(weight: torch.Tensor, bidx: torch.Tensor,
                   tab: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """x0_bigram [T, D] = W[bidx[:T]] * tab[h1(inp)] + W[bidx[T:]] * tab[h2(inp)]."""
    T, D = inp.shape[0], weight.shape[1]
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert bidx.dtype == torch.int32 and bidx.is_contiguous() and bidx.shape == (2 * T,)
    out, n, grid, rmask = _bgx_prep(tab, inp, D, 1)
    _bgx_combine_kernel_tri[grid](weight, bidx, tab, inp, out, T, n, D=D, RMASK=rmask,
                                  XBLOCK=_BGX_XBLOCK, num_warps=_BGX_WARPS, num_stages=1)
    return out

@bgx_combine_op.register_fake
def _(weight, bidx, tab, inp):
    return _bgx_prep(tab, inp, weight.shape[1], 1)[0]

@torch.library.custom_op("nanogpt::bgx_sink_grad", mutates_args=())
def bgx_sink_grad_op(g: torch.Tensor, tab: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """d(bg_sink) [2T, D] from d(x0_bigram) [T, D]: apply the two sign rows."""
    D = g.shape[-1]
    g = g.contiguous().reshape(-1, D)   # no-op when already contiguous
    assert g.shape[0] == inp.shape[0] and g.dtype == torch.bfloat16
    out, n, grid, rmask = _bgx_prep(tab, inp, D, 2)
    _bgx_sink_grad_kernel_tri[grid](g, tab, inp, out, n, D=D, RMASK=rmask,
                                    XBLOCK=_BGX_XBLOCK, num_warps=_BGX_WARPS, num_stages=1)
    return out

@bgx_sink_grad_op.register_fake
def _(g, tab, inp):
    return _bgx_prep(tab, inp, g.shape[-1], 2)[0]

class _BGXCombine(torch.autograd.Function):
    """`gather(W, bidx).detach() + bg_sink` routes the whole gradient to `bg_sink` and NOTHING to `W`: that
    detach is what stops a dense [V, 768] embedding grad ever being built.  `sink` is exact zeros the forward
    never reads, its only purpose being that autograd slot."""
    @staticmethod
    def forward(ctx, weight, bidx, tab, inp, sink):
        ctx.save_for_backward(tab, inp)
        return torch.ops.nanogpt.bgx_combine(weight, bidx, tab, inp)

    @staticmethod
    def backward(ctx, g):
        tab, inp = ctx.saved_tensors
        return None, None, None, None, torch.ops.nanogpt.bgx_sink_grad(g, tab, inp)

def bgx_x0_bigram(weight, bidx, tab, inp, sink):
    """[T, D] x0_bigram; `sink` is None under eval, where no gradient is wanted."""
    if sink is None:
        return torch.ops.nanogpt.bgx_combine(weight, bidx, tab, inp)
    return _BGXCombine.apply(weight, bidx, tab, inp, sink)


# 2. Fused cast+scatter for the sparse row exchange: one Triton kernel per scatter instead of an aten
# dtype-cast temporary plus a per-element indexing kernel, one program per SOURCE ROW so loads vectorise and
# 16-bit atomics pack.  Only atomic landing order changes, unordered in aten too; the cast is
# load -> fp32 (exact for fp16/bf16) -> atomic_add's RNE, aten's `.to` addend.
MAX_BLOCK = 1024      # largest tile the row splitter will use; 768 -> (256, 3)

@triton.jit
def _bgfuse_scatter_add_kernel(SRC, IDX, DST, D: tl.constexpr, BLOCK: tl.constexpr,
                               NCHUNK: tl.constexpr):
    """DST.index_add_(0, IDX, SRC.to(DST.dtype)) -- one program per source row."""
    pid = tl.program_id(0)
    o64 = tl.arange(0, BLOCK).to(tl.int64)
    drow = tl.load(IDX + pid)
    sbase = pid.to(tl.int64) * D
    dbase = drow.to(tl.int64) * D
    for k in tl.static_range(NCHUNK):
        koff = k * BLOCK
        val = tl.load(SRC + (sbase + koff + o64)).to(tl.float32)
        tl.atomic_add(DST + (dbase + koff + o64), val, sem="relaxed")

@triton.jit
def _bgfuse_scatter_copy_kernel(SRC, IDX, DST, D: tl.constexpr, BLOCK: tl.constexpr,
                                NCHUNK: tl.constexpr):
    """DST.index_copy_(0, IDX, SRC), same dtype both sides: no fp32 round trip, so the value
    moves as raw bits -- a bitwise index_copy_."""
    pid = tl.program_id(0)
    o64 = tl.arange(0, BLOCK).to(tl.int64)
    drow = tl.load(IDX + pid)
    sbase = pid.to(tl.int64) * D
    dbase = drow.to(tl.int64) * D
    for k in tl.static_range(NCHUNK):
        koff = k * BLOCK
        val = tl.load(SRC + (sbase + koff + o64))
        tl.store(DST + (dbase + koff + o64), val)

def _launch(kernel, src, idx, dst):
    """Shared contract and launch of both scatter kernels.  The tile is the largest power of two at or below
    MAX_BLOCK dividing the row width exactly, so a row needs no mask; destination rows are not bounds-checked,
    as in aten's release path."""
    assert src.ndim == 2 and dst.ndim == 2 and src.shape[1] == dst.shape[1] and src.is_contiguous() \
        and dst.is_contiguous() and src.numel() < 2 ** 31, \
        f"[bgfuse] want 2-D contiguous src/dst of one width: {tuple(src.shape)} -> {tuple(dst.shape)}"
    assert idx.ndim == 1 and idx.is_contiguous() and idx.numel() == src.shape[0] \
        and idx.dtype in (torch.int32, torch.int64), \
        f"[bgfuse] index {tuple(idx.shape)} {idx.dtype} for {src.shape[0]} source rows"
    n_rows, d = src.shape
    if n_rows:
        block = MAX_BLOCK
        while block > 1 and d % block:
            block //= 2
        assert d % block == 0 and block >= 32, f"[bgfuse] row width {d} has no usable tile"
        kernel[(n_rows,)](src, idx, dst, D=d, BLOCK=block, NCHUNK=d // block,
                          num_warps=max(1, min(8, block // 64)), num_stages=1)
    return dst

def bgfuse_scatter_add(src, idx, dst):
    """`dst.index_add_(0, idx, src.to(dst.dtype))` in one kernel.  `idx` MAY repeat (the merge's
    recv list concatenates peers'): every contribution lands through an atomic, as index_add_."""
    return _launch(_bgfuse_scatter_add_kernel, src, idx, dst)

def bgfuse_scatter_copy(src, idx, dst):
    """`dst.index_copy_(0, idx, src)`; a bitwise copy, so the dtypes must match."""
    assert src.dtype == dst.dtype, \
        f"[bgfuse] scatter_copy is a bitwise copy: {src.dtype} != {dst.dtype}"
    return _launch(_bgfuse_scatter_copy_kernel, src, idx, dst)

# The exchange's three host-side steps live here with the two kernels they drive: the want list the
# request is cut from, the send cast that drops this rank's own block, and the land that puts it back.
def _want_unique(idx_arrays, cap, vocab):
    """THE want list, armed cycle and cold fill alike: the sorted-unique of the concatenation is
    element for element what a zeroed V-length mask + flatnonzero gives, without the scan over all
    84.6M rows. SORT, not np.unique -- numpy 2.x routes integer unique through a hash set, which
    on a real cycle's ids measures 15.4 / 59.6 ms at the two batch geometries where a sort and a
    neighbour compare take 1.8 / 5.6 ms for the identical array. The asserted bounds are the ones
    that mask's indexing would have checked."""
    # No next cycle at the last step, so the caller hands us NO arrays: np.concatenate([]) raises.
    # The mask spelling this replaced returned an empty want list here for free (an all-zero mask
    # flatnonzeros to nothing), and every consumer already handles a zero-row want.
    if not idx_arrays:
        return np.empty(0, dtype=np.int32)
    want = np.sort(idx_arrays[0] if len(idx_arrays) == 1 else np.concatenate(idx_arrays))
    if want.size:
        want = want[np.r_[True, want[1:] != want[:-1]]]
    want = want.astype(np.int32, copy=False)
    assert want.shape[0] <= cap and (not want.shape[0] or
                                     (0 <= int(want[0]) and int(want[-1]) < vocab)), \
        f"[ngram-cache] {want.shape[0]} rows (cap {cap}), ids vs V={vocab}"
    return want

def bgpull_want_build(mask, idx_arrays, cap):
    """The rows the next bigram cycle reads on this rank, sorted-unique. `mask` names the row
    space they must lie in and is no longer scanned: sorting the cycle's ids costs 1.8 / 5.6 ms
    at the two batch geometries against the 10.6 / 13.9 ms a memset plus a four-thread flatnonzero
    over all 84.6M rows took, on one thread instead of four."""
    return _want_unique(idx_arrays, cap, mask.shape[0])

@torch.no_grad
def bgcache_land(cache, shard, lo, hi, n, own, recv_vals):
    """Write one event's rows into the cache IN WANT ORDER, in three writes: `want` is sorted, so
    this rank's own rows are one block [lo:hi) and the peers' the two slices around it, which is
    the order bgfuse_send_cast's dropped block and the value all_to_all's transposed splits
    deliver. Rows [n, cap) keep the last fill's bytes and are never addressed: every slot indexes
    want[:n]."""
    d = cache.shape[1]
    npeer = n - (hi - lo)
    assert d == shard.shape[1] and n <= cache.shape[0] and recv_vals.numel() == npeer * d, \
        f"[ngram-cache] {n} rows x {d} into {tuple(cache.shape)} off {tuple(shard.shape)}, " \
        f"payload {recv_vals.numel()} for {npeer} peer rows"
    rv = recv_vals.view(npeer, d)
    if lo:
        cache[:lo].copy_(rv[:lo])
    if n > hi:
        cache[hi:n].copy_(rv[lo:])
    if hi > lo:
        torch.index_select(shard, 0, own, out=cache[lo:hi])

def bgfuse_send_cast(compact, lo, hi, dtype):
    """`torch.cat([compact[:lo], compact[hi:]]).to(dtype)` without the intermediate: both steps are pure
    copies, so letting each `copy_` convert into the wire-dtype buffer gives NCCL the same bytes (the same
    RNE cast) one pass and ~400 MB cheaper."""
    n, d = compact.shape
    assert 0 <= lo <= hi <= n, f"[bgfuse] bad own-block [{lo}:{hi}] of {n} rows"
    out = torch.empty((n - (hi - lo), d), dtype=dtype, device=compact.device)
    if lo > 0:
        out[:lo].copy_(compact[:lo])
    if hi < n:
        out[lo:].copy_(compact[hi:])
    return out


# 3. Row-compacted bigram Adam with exact replay (why, and what the dense form costs: see the record README).
# `_adam_update_step`'s update restricted to the rows that moved, plus an EXACT in-register replay of the
# events a row slept through -- that update statement for statement, up to fma/rsqrt ulps.  COMPRESSED
# MOMENTS, a CONTRACT asserted in bgadam_rows_launch and compiled into the kernel: `V` is ONE fp32 scalar per
# row (the gradient's mean square) and `M` a bf16 stub, so adaptivity is per-row-RMS.  THE beta1 == 0 COLLAPSE:
# on a replayed event m *= 0 makes u exactly 0, so `p -= u` is the identity and p is REPLAY-INVARIANT; with
# b1 == 0 at EVERY event (checked in bgadam_hist_push, the scalar column's single writer) the collapsed replay
# is bit-identical to an fp32 uncollapsed per-event one.
H_B1, H_B2, H_S, H_W = 0, 1, 2, 3   # per-event scalars, one column per bigram Adam event
HIST_ROWS = 4

@triton.jit
def _bgadam_rows_kernel_b0(M, V, P, G, OUT, IDX, LE, HIST,
                           t_ref,
                           beta1, beta2, eps, s_t, w_t, a1, a2,
                           D_MODEL: tl.constexpr, HIST_STRIDE: tl.constexpr,
                           HAS_G: tl.constexpr, COMMIT: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row: carry the row's missed second-moment decay, then if HAS_G apply
    this event's real Adam step.  COMMIT writes v/p back and stamps the shared `last_event`
    table LE (idempotent); else the row's parameter goes to OUT.  `M` is never dereferenced."""
    pid_r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)                       # int32
    mask = offs < D_MODEL
    o64 = offs.to(tl.int64)
    row = tl.load(IDX + pid_r)                       # int32 scalar
    base = row.to(tl.int64) * D_MODEL                # int64 scalar
    aoff = base + o64                                # int64 tensor
    if COMMIT:
        if HAS_G:
            # Pass 1, grad rows: IDX concatenates every peer's list, so DUPLICATES ARE A RACE.
            # atomic_max elects one committer; a loser's loads are predicated off (768 or 0 lanes).
            t_new = t_ref + 1
            le = tl.atomic_max(LE + row, t_new)
            own = le < t_new
            omask = offs < D_MODEL * own.to(tl.int32)
            v = tl.load(V + row.to(tl.int64)).to(tl.float32)
            # the collapsed replay: v's decay product over (le, t_ref], left to right, the same
            # multiplies a per-event loop performs (n_rep <= 0 for a loser).
            n_rep = t_ref - le
            for i in range(n_rep):
                v = v * tl.load(HIST + 1 * HIST_STRIDE + (le + 1 + i))
            g = tl.load(G + aoff, mask=omask, other=0.0).to(tl.float32)
            p = tl.load(P + aoff, mask=omask, other=0.0).to(tl.float32)
            m = g * a1                               # == m * beta1 + g * a1 at beta1 = 0
            # The row's MEAN square, so sqrt(v) keeps the scale a per-element sqrt(g*g) had
            # and neither eps nor s_t needs retuning; masked lanes came in 0 and add nothing.
            v = v * beta2 + (tl.sum(g * g, axis=0) / D_MODEL) * a2
            u = (m / (tl.sqrt(v) + eps)) * s_t
            u = u + tl.where((u * p) > 0, p * w_t, 0.0)
            p = p - u
            if own:
                tl.store(V + row.to(tl.int64), v)
                tl.store(P + aoff, p.to(P.dtype.element_ty), mask=mask)
        else:
            # Pass 2, own-read / flush / dense restore: p is replay-invariant, so the row owes
            # only v's decay; IDX is duplicate-free here, nothing to claim.
            le = tl.load(LE + row)
            v = tl.load(V + row.to(tl.int64)).to(tl.float32)
            n_rep = t_ref - le
            for i in range(n_rep):
                v = v * tl.load(HIST + 1 * HIST_STRIDE + (le + 1 + i))
            tl.store(V + row.to(tl.int64), v)
            tl.store(LE + row, t_ref)                # t_new, with HAS_G False
    else:
        # Pass 3, the read-only serve: the replay cannot move p, so this is a gather; LE does
        # not advance, so the row's next real touch replays from the same anchor.
        p = tl.load(P + aoff, mask=mask, other=0.0).to(tl.float32)
        ooff = pid_r.to(tl.int64) * D_MODEL + o64
        tl.store(OUT + ooff, p.to(OUT.dtype.element_ty), mask=mask)

def bgadam_rows_launch(m, v, p, g, idx, le, hist, t_ref,
                       beta1, beta2, eps, s_t, w_t, a1, a2, out=None):
    """Replay over `idx`, an int32 device tensor of LOCAL shard rows.  With a gradient (pass 1) `idx` may
    repeat and the atomic claim picks one committer; without one (g=None) that claim is compiled out and `idx`
    MUST be duplicate-free.  `out` non-None selects the read-only send form; `t_ref` is the anchor -- the grad
    pass replays to t-1 and stamps t.  The asserts are the collapse's preconditions (a per-event replay rounds
    p to bf16 once per slept event, the identity only for a bf16 p) and the int32 row space of IDX and LE."""
    n = idx.numel()
    if n == 0:
        return out
    d_model = p.shape[1]
    assert float(beta1) == 0.0 and p.dtype is torch.bfloat16 and m.dtype is torch.bfloat16, \
        f"[ngram-adam] the row kernel is the beta1 == 0 bf16 collapse; got {beta1}, {p.dtype}"
    assert p.shape[0] < 2 ** 31 and m.shape == (1, d_model) and v.shape == (p.shape[0], 1) \
        and v.dtype is torch.float32 and m.is_contiguous() and v.is_contiguous(), \
        f"[ngram-adam] moment layout {tuple(m.shape)}/{tuple(v.shape)} vs shard {tuple(p.shape)}"
    assert idx.dtype == torch.int32 and le.dtype == torch.int32 \
        and hist.shape[0] == HIST_ROWS and hist.is_contiguous(), \
        "[ngram-adam] int32 row ids and a contiguous [HIST_ROWS, events] scalar history"
    _bgadam_rows_kernel_b0[(n,)](
        m, v, p, g if g is not None else p, out if out is not None else p,
        idx, le, hist, int(t_ref),
        float(beta1), float(beta2), float(eps), float(s_t), float(w_t), float(a1), float(a2),
        D_MODEL=d_model, HIST_STRIDE=hist.shape[1],
        HAS_G=(g is not None), COMMIT=(out is None),
        BLOCK=triton.next_power_of_2(d_model), num_warps=4,
    )
    return out

def bgadam_hist_alloc(max_events, dev):
    """[4, max_events+2] fp32 device table of the per-event Adam scalars, plus its pinned host staging column."""
    return (torch.zeros(HIST_ROWS, max_events + 2, dtype=torch.float32, device=dev),
            torch.zeros(HIST_ROWS, 1, dtype=torch.float32, pin_memory=True))

def bgadam_hist_push(hist, stage, j, beta1, beta2, s_t, w_t):
    """Event j's scalars: one 16-byte H2D per event, no sync.  Only H_B2 is read back, by the
    replay; dropping the rest would change the kernel's HIST_STRIDE."""
    assert float(beta1) == 0.0, \
        f"the bigram row replay is the beta1 == 0 collapse: event {j} pushed beta1={beta1}"
    stage[H_B1, 0], stage[H_B2, 0] = beta1, beta2
    stage[H_S, 0], stage[H_W, 0] = s_t, w_t
    hist[:, j:j + 1].copy_(stage, non_blocking=True)

import os
import torch
import triton
import triton.language as tl

# Softcapped-logit statistics for the CPLM loss.
# Given raw lm_head logits x [N, V] and per-row gather columns idx [N, M] (-1 = skip),
# with the record's softcap z = A * sigmoid((x + B) / C), returns
#   lse [N]    = logsumexp_v z[n, v]
#   zg  [N, M] = z[n, idx[n, m]]  (0 where idx < 0)
# Any loss built from (lse, zg) backprops through one fused pass over the logits:
#   dL/dx[n, v] = dz/dx * (g_lse[n] * softmax(z)[n, v] + sum_m g_zg[n, m] * [v == idx[n, m]])
# With idx = the MTP targets this reproduces FusedSoftcappedCrossEntropy exactly; the extra
# column carries the <copy> slot logit for the copy-sink mixture.
# raw_col (>= 0) exempts one column from the softcap: z[raw_col] = x + raw_bias (uncapped), so the
# <copy> gate is not bounded by the cap that the confident LM tokens saturate against.
# add_col (>= 0) adds a per-row term add[n] to that column's raw logit before the transform (used for a
# copy-gate vector trained outside the LM head); its grad is the logit grad at that column.


@triton.jit
def softcap_lse_gather_fwd_kernel(
    logits_ptr, idx_ptr, lse_ptr, zg_ptr, add_ptr,
    stride_logits_n,
    n_cols,
    A, B, C, raw_col, raw_bias, add_col,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    add_val = 0.0
    if add_col >= 0:
        add_val = tl.load(add_ptr + row_idx)

    max_val = -float('inf')
    sum_exp = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        val = val + tl.where(cols == add_col, add_val, 0.0)
        z = A * tl.sigmoid((val + B) / C)
        z = tl.where(cols == raw_col, val + raw_bias, z)
        z = tl.where(mask, z, -float('inf'))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max
    tl.store(lse_ptr + row_idx, max_val + tl.log(sum_exp))

    for m in tl.static_range(M):
        col = tl.load(idx_ptr + row_idx * M + m).to(tl.int32)
        zg = 0.0
        if col >= 0 and col < n_cols:
            val = tl.load(logits_row_ptr + col).to(tl.float32)
            if col == add_col:
                val = val + add_val
            zg = A * tl.sigmoid((val + B) / C)
            if col == raw_col:
                zg = val + raw_bias
        tl.store(zg_ptr + row_idx * M + m, zg)


@triton.jit
def softcap_lse_gather_bwd_kernel(
    grad_input_ptr, g_lse_ptr, g_zg_ptr, lse_ptr, logits_ptr, idx_ptr, add_ptr,
    stride_logits_n, stride_grad_n,
    n_cols,
    A, B, C, raw_col, raw_bias, add_col,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_input_ptr + row_idx * stride_grad_n
    add_val = 0.0
    if add_col >= 0:
        add_val = tl.load(add_ptr + row_idx)

    lse = tl.load(lse_ptr + row_idx)
    g_lse = tl.load(g_lse_ptr + row_idx)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = val + tl.where(cols == add_col, add_val, 0.0)
        sigmoid_u = tl.sigmoid((val + B) / C)
        is_raw = cols == raw_col
        z = tl.where(is_raw, val + raw_bias, A * sigmoid_u)
        grad_z = g_lse * tl.exp(z - lse)
        for m in tl.static_range(M):
            col = tl.load(idx_ptr + row_idx * M + m).to(tl.int32)
            g = tl.load(g_zg_ptr + row_idx * M + m)
            grad_z += tl.where(cols == col, g, 0.0)
        dz_dx = tl.where(is_raw, 1.0, (1.0 / C) * z * (1.0 - sigmoid_u))
        tl.store(grad_row_ptr + cols, (grad_z * dz_dx).to(tl.bfloat16), mask=mask)


class SoftcapLSEGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, idx, raw_col=-1, add_col=-1, add=None, A=23.0, B=5.0, C=7.5):
        n_rows, n_cols = logits.shape
        M = idx.shape[1]
        raw_bias = A * (1.0 / (1.0 + 2.718281828459045 ** (-B / C)))  # softcap(0): same init as capped
        logits = logits.contiguous()
        idx = idx.contiguous()
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        zg = torch.empty((n_rows, M), dtype=torch.float32, device=logits.device)
        add = lse if add is None else add.float().contiguous()  # dummy pointer when add_col < 0
        softcap_lse_gather_fwd_kernel[(n_rows,)](
            logits, idx, lse, zg, add,
            logits.stride(0),
            n_cols,
            A, B, C, raw_col, raw_bias, add_col,
            M=M,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=4
        )
        ctx.save_for_backward(logits, idx, lse, add)
        ctx.params = (A, B, C, raw_col, raw_bias, add_col)
        return lse, zg

    @staticmethod
    def backward(ctx, g_lse, g_zg):
        logits, idx, lse, add = ctx.saved_tensors
        A, B, C, raw_col, raw_bias, add_col = ctx.params
        n_rows, n_cols = logits.shape
        M = idx.shape[1]
        if g_lse is None:
            g_lse = torch.zeros_like(lse)
        if g_zg is None:
            g_zg = torch.zeros((n_rows, M), dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, n_cols), dtype=torch.bfloat16, device=logits.device)
        softcap_lse_gather_bwd_kernel[(n_rows,)](
            grad_input, g_lse.contiguous(), g_zg.contiguous(), lse, logits, idx, add,
            logits.stride(0), grad_input.stride(0),
            n_cols,
            A, B, C, raw_col, raw_bias, add_col,
            M=M,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=4
        )
        grad_add = grad_input[:, add_col].float() if add_col >= 0 else None
        return grad_input, None, None, None, grad_add, None, None, None


def copy_sink_probs(q, k, k_sink, input_seq, target_seq, seqlens, L, max_lookback=None, return_stats=False):
    """Copy-sink pointer probabilities on a packed 1D stream.

    q, k [N, d] copy queries/keys; k_sink [d]; input_seq/target_seq [N]; seqlens = cu_seqlens
    (padded with N). Query i attends to strictly-past positions j of its own document plus a
    learned sink. Queries in block b of L tokens see key blocks b-1 and b, which covers whole
    documents whenever documents are <= L tokens (training) and >= L tokens of lookback otherwise.
    Returns p_copy [N] = sum_j attn[i, j] * [input_seq[j] == target_seq[i]] and a_sink [N].
    """
    N, d = q.shape
    nb = N // L
    dev = input_seq.device
    scale = d ** -0.5
    doc_start = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    doc_start.index_put_((seqlens.long().clamp(max=N),), torch.ones_like(seqlens, dtype=torch.int32), accumulate=True)
    doc = doc_start[:N].cumsum(0)
    kb = k.view(nb, L, d)
    kk = torch.cat([torch.cat([kb.new_zeros(1, L, d), kb[:-1]]), kb], dim=1)  # [nb, 2L, d]
    scores = torch.bmm(q.view(nb, L, d), kk.transpose(1, 2)).float() * scale  # [nb, L, 2L]
    doc_k = torch.cat([doc.new_full((L,), -1), doc]).unfold(0, 2 * L, L)  # [nb, 2L]
    tok_k = torch.cat([input_seq.new_full((L,), -1), input_seq]).unfold(0, 2 * L, L)
    qpos = torch.arange(L, device=dev)[:, None] + L  # query position inside its 2L key window
    kpos = torch.arange(2 * L, device=dev)[None, :]
    valid = (kpos < qpos)[None] & (doc_k[:, None, :] == doc.view(nb, L, 1))
    if max_lookback is not None:  # e.g. cap eval lookback to what training ever sees
        valid = valid & (qpos - kpos <= max_lookback)[None]
    match = valid & (tok_k[:, None, :] == target_seq.view(nb, L, 1))
    sink = (q.float() @ k_sink.float()).view(nb, L) * scale
    scores = scores.masked_fill(~valid, float('-inf'))
    m = torch.maximum(scores.amax(dim=-1), sink).detach()
    e = torch.exp(scores - m[..., None])
    e_sink = torch.exp(sink - m)
    Z = e.sum(dim=-1) + e_sink
    p_copy, a_sink = ((e * match).sum(dim=-1) / Z).view(N), (e_sink / Z).view(N)
    if not return_stats:
        return p_copy, a_sink
    # score magnitude (post-scale) over rows with >=1 valid source: bf16 spacing is 2^-7 of the magnitude
    row_max = scores.amax(dim=-1).detach()
    has = torch.isfinite(row_max)
    stats = torch.stack([torch.where(has, row_max, 0).sum() / has.sum().clamp_min(1), torch.where(has, row_max, -1e9).amax()])
    # per-token context for eval breakdowns: is the target copyable (present earlier in-doc, within the band),
    # position within its document, and whether the token is in a chunk-leading segment that lacks its BOS
    copyable = match.any(dim=-1).view(N)
    ar = torch.arange(N, device=dev)
    start_idx = torch.cummax(torch.where(doc_start[:N] > 0, ar, 0), dim=0).values
    pos_in_doc = ar - start_idx
    lead = (doc == doc[0]) & (input_seq[0] != 50256)
    return p_copy, a_sink, stats, copyable, pos_in_doc, lead


# ---------------------------------------------------------------------------------------------
# Fused (flash-style) copy-sink pointer: same math and masking as copy_sink_probs, but scores are
# computed tile by tile on-chip and tiles outside the query's own document are skipped entirely
# (documents average a few hundred tokens vs the 2L-wide band). Needs copy dim D in {64, 128, 256}.


@triton.jit
def _copy_fwd_kernel(Q, K, S0, DOC, TOK, TGT, DSTART, P_OUT, A_OUT, LSE_OUT, N, L, LB, scale,
                     D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    # NK shifts share one attention: shift kk copies the token kk after the source (TOK row kk = input[j + kk]) for
    # target row kk (= the target kk steps ahead), counting only sources with j + kk <= query (no future tokens).
    m0 = tl.program_id(0) * BM
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    s0 = tl.load(S0 + offs_m)
    doc_q = tl.load(DOC + offs_m)
    band_lo = (m0 // L) * L - L  # all queries of the tile share an L-block (BM divides L)
    lo = tl.maximum(tl.maximum(tl.load(DSTART + m0), band_lo), m0 - LB)
    lo = (tl.maximum(lo, 0) // BN) * BN
    m_i = s0  # the sink is the running max's starting point, so rows without sources are well defined
    l_i = tl.full([BM], 1.0, tl.float32)
    acc = tl.zeros([BM, 4], tl.float32)
    col = tl.arange(0, 4)
    for n0 in range(lo, m0 + BM, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        doc_k = tl.load(DOC + offs_n)
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        s = tl.where(valid, s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            acc += tl.where(col[None, :] == kk, tl.sum(tl.where(hit, p, 0.0), axis=1)[:, None], 0.0)
        m_i = m_new
    for kk in tl.static_range(NK):
        tl.store(P_OUT + kk * N + offs_m, tl.sum(tl.where(col[None, :] == kk, acc, 0.0), axis=1) / l_i)
    tl.store(A_OUT + offs_m, tl.exp(s0 - m_i) / l_i)
    tl.store(LSE_OUT + offs_m, m_i + tl.log(l_i))


@triton.jit
def _copy_bwd_dq_kernel(Q, K, DOC, TOK, TGT, DSTART, LSE, GP, DROW, DQ, N, L, LB, scale,
                        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    m0 = tl.program_id(0) * BM
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    doc_q = tl.load(DOC + offs_m)
    lse = tl.load(LSE + offs_m)
    drow = tl.load(DROW + offs_m)
    band_lo = (m0 // L) * L - L
    lo = tl.maximum(tl.maximum(tl.load(DSTART + m0), band_lo), m0 - LB)
    lo = (tl.maximum(lo, 0) // BN) * BN
    dq = tl.zeros([BM, D], tl.float32)
    for n0 in range(lo, m0 + BM, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        doc_k = tl.load(DOC + offs_n)
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        a = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        gm = tl.zeros([BM, BN], tl.float32)
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            gp = tl.load(GP + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            gm += tl.where(hit, gp[:, None], 0.0)
        ds = a * (gm - drow[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + offs_m[:, None] * D + offs_d[None, :], dq * scale)


@triton.jit
def _copy_bwd_dk_kernel(Q, K, DOC, TOK, TGT, DEND, LSE, GP, DROW, DK, N, L, LB, scale,
                        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    n0 = tl.program_id(0) * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
    doc_k = tl.load(DOC + offs_n)
    # queries that can see these keys: later positions of the same document, inside the band
    # (query block qb sees keys from block qb-1 onwards, so keys of block kb are seen up to block kb+1)
    hi = tl.minimum(tl.load(DEND + n0 + BN - 1), (n0 // L + 2) * L)
    hi = tl.minimum(tl.minimum(hi, N), n0 + BN + LB)
    lo = (n0 // BM) * BM
    dk = tl.zeros([BN, D], tl.float32)
    for m0 in range(lo, hi, BM):
        offs_m = m0 + tl.arange(0, BM)
        q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
        doc_q = tl.load(DOC + offs_m)
        lse = tl.load(LSE + offs_m)
        drow = tl.load(DROW + offs_m)
        band_lo = (m0 // L) * L - L
        s = tl.dot(q, tl.trans(k)) * scale
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        a = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        gm = tl.zeros([BM, BN], tl.float32)
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            gp = tl.load(GP + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            gm += tl.where(hit, gp[:, None], 0.0)
        ds = a * (gm - drow[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + offs_n[:, None] * D + offs_d[None, :], dk * scale)


def _copy_tiles(D):
    """BM, BN, num_warps, num_stages (CPLM_COPY_TILES="BM,BN,warps,stages" overrides, for tuning)."""
    if os.environ.get("CPLM_COPY_TILES"):
        return tuple(int(v) for v in os.environ["CPLM_COPY_TILES"].split(","))
    return (64, 64, 4, 3) if D <= 128 else (64, 32, 8, 2)  # 64,64,4,3: best-or-tied in the bench sweep (bench_overhead.py)


class CopySinkFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, s0, doc, tok, tgt, dstart, dend, L, LB):
        N, D = q.shape
        NK = tok.shape[0]  # tok, tgt: [NK, N] (see _copy_fwd_kernel)
        BM, BN, nw, ns = _copy_tiles(D)
        assert N % L == 0 and L % BM == 0 and L % BN == 0 and D in (64, 128, 256) and 1 <= NK <= 4
        q, k = q.contiguous(), k.contiguous()
        P = torch.empty((NK, N), dtype=torch.float32, device=q.device)
        A = torch.empty(N, dtype=torch.float32, device=q.device)
        LSE = torch.empty_like(A)
        _copy_fwd_kernel[(N // BM,)](q, k, s0.contiguous(), doc, tok, tgt, dstart, P, A, LSE, N, L, LB, D ** -0.5,
                                     D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
        ctx.save_for_backward(q, k, doc, tok, tgt, dstart, dend, LSE, P, A)
        ctx.L, ctx.LB = L, LB
        return P, A

    @staticmethod
    def backward(ctx, gP, gA):
        q, k, doc, tok, tgt, dstart, dend, LSE, P, A = ctx.saved_tensors
        L, LB = ctx.L, ctx.LB
        N, D = q.shape
        NK = tok.shape[0]
        BM, BN, nw, ns = _copy_tiles(D)
        gP = torch.zeros_like(P) if gP is None else gP.float().contiguous()
        gA = torch.zeros_like(A) if gA is None else gA.float().contiguous()
        drow = ((gP * P).sum(0) + gA * A).contiguous()  # d loss / d (log-normalizer), per row
        ds0 = A * (gA - drow)                   # sink score gradient
        dq = torch.empty((N, D), dtype=torch.float32, device=q.device)
        dk = torch.empty_like(dq)
        _copy_bwd_dq_kernel[(N // BM,)](q, k, doc, tok, tgt, dstart, LSE, gP, drow, dq, N, L, LB, D ** -0.5,
                                        D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
        _copy_bwd_dk_kernel[(N // BN,)](q, k, doc, tok, tgt, dend, LSE, gP, drow, dk, N, L, LB, D ** -0.5,
                                        D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
        return dq.to(q.dtype), dk.to(k.dtype), ds0, None, None, None, None, None, None, None


def copy_sink_probs_fused(q, k, k_sink, input_seq, target_seq, seqlens, L, doc_starts=None, shifts=1, max_lookback=None):
    """Drop-in for copy_sink_probs (training path, no stats/lookback cap) using the fused kernels.
    shifts > 1 also returns, for each kk < shifts, the copy probability of the target kk steps ahead (target_seq[i + kk])
    under the same attention, copying the token kk after each source (input_seq[j + kk], j + kk <= i): p_copy is then
    [shifts, N] (the MTP terms' copy branch). max_lookback (tokens): sources at most this far back (None = the band)."""
    N, d = q.shape
    dev = input_seq.device
    if doc_starts is None:  # boundaries from cu_seqlens: one searchsorted instead of a chain of scans
        cu = seqlens.long().clamp(max=N)
        j = torch.searchsorted(cu, torch.arange(N, device=dev), right=True)  # boundaries <= each position
        cu_ext = torch.cat([cu.new_zeros(1), cu, cu.new_full((1,), N)])
        doc, dstart, dend = j.to(torch.int32), cu_ext[j].to(torch.int32), cu_ext[j + 1].to(torch.int32)
    else:  # explicit [N] bool mask of document starts (e.g. BOS tokens); position 0 always starts one
        starts = torch.zeros(N + 1, dtype=torch.int32, device=dev)
        starts[:N] = doc_starts.to(torch.int32)
        starts[0] = 1
        doc = starts[:N].cumsum(0).to(torch.int32)
        ar = torch.arange(N, device=dev, dtype=torch.int32)
        dstart = torch.cummax(torch.where(starts[:N] > 0, ar, 0), dim=0).values.to(torch.int32)
        nxt = torch.where(starts[1:N + 1] > 0, ar + 1, N)  # exclusive end if the doc ended right after this position
        dend = torch.flip(torch.cummin(torch.flip(nxt, [0]), dim=0).values, [0]).to(torch.int32)
    s0 = (q.float() @ k_sink.float()) * d ** -0.5
    inp, tgt = input_seq.to(torch.int32), target_seq.to(torch.int32)
    if shifts == 1:  # no shifted copies to build
        tok, tgt = inp.view(1, N), tgt.view(1, N)
    else:
        tok = torch.stack([torch.cat([inp[kk:], inp.new_full((kk,), -1)]) for kk in range(shifts)])
        tgt = torch.stack([torch.cat([tgt[kk:], tgt.new_full((kk,), -2)]) for kk in range(shifts)])
    P, A = CopySinkFused.apply(q, k, s0, doc, tok.contiguous(), tgt.contiguous(), dstart, dend, L,
                               N if max_lookback is None else int(max_lookback))
    return (P[0] if shifts == 1 else P), A

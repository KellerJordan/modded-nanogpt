"""CPLM (copy-sink pointer mixture) on top of the fused fp8 CE kernel.

The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens:

    p(y) = p_lm(y) * (1 - alpha * (1 - a_sink)) + alpha * p_copy(y)

alpha is the softmax mass of the <copy> slot (vocab id 50257, one of lm_head's padding rows); p_copy is the attention
mass a one-head (d = 128) QK-normed pointer puts on previous tokens equal to y inside the same document (a 2048-token
band in training, 8192 at validation); a_sink is the mass on the learned sink key and goes back to the LM.

This module holds the two kernels of PR #379, adapted to this trainer's own fused CE:
  1. the flash-style Triton pointer (cplm_copy.py of PR #379), behind torch.library custom ops (Triton launches inside
     compiled code go through custom ops: a bare launch gave NaN under torch.compile);
  2. the mixture NLL inside the fp8 CE CUDA kernels (perf/kernels/cplm_cross_entropy.py of PR #379): cross_entropy.py's
     CE_KERNEL_SOURCE (the plain kernel, the fallback when n_rows % 8 != 0) and CE_LUT_KERNEL_SOURCE (the 256-entry-table
     kernel the trainer runs) plus anchored patches, each asserted. Only the patches of PR #379's FINAL configuration are
     ported (no EXT_GATE / GATE_FLOOR / RAW_COPY / MTP_COPY / LM_AUX / AUX_NOCOPY / GATE_OFFSET / STATS; the lambda
     warmup scalars are the constant 1, as in PR #379 with CPLM_WARMUP=0).

The logit gradient is still  w0 * [g_lse * softmax(z)_v + g_t * [v == y] + g_c * [v == copy]]  with closed-form row
scalars; the kernel also returns d nll / d p_copy and d nll / d a_sink for the pointer. The sampled-softmax steps run
the same kernels at width P (the <copy> class is always a candidate, sampled_softmax.py) and sample the lm_head dW
(sampled_head.py).

Provenance: PR #379 (NathanGodey), the sampled dW from PRs #371/#382.
"""
import torch
import triton
import triton.language as tl
from torch import Tensor

from track_1_short.config import COPY_TOKEN_ID
from track_1_short.perf.kernels import sampled_head
from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE,
    CE_KERNEL_SOURCE,
    CE_KERNEL_VOCAB_SIZE,
    CE_LUT_KERNEL_SOURCE,
    CE_LUT_ROWS,
    CE_LUT_SMEM,
    CUDA_INCLUDE_DIRS,
    SOFTCAP_A,
    SOFTCAP_B,
    SOFTCAP_C,
    _ce_backward_gemms,
    _ce_logit_gemm,
    _ce_lut_minb,
)
from track_1_short.perf.kernels.sampled_cross_entropy import sampled_densify

COPY_BLOCK = 2048      # training copy band: one to two blocks (queries of block b see key blocks b-1 and b)


# ---------------------------------------------------------------------------------------------
# Fused (flash-style) copy-sink pointer: scores are computed tile by tile on-chip and tiles outside the
# query's own document are skipped entirely. Needs the copy dim D in {64, 128, 256}.

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
    """BM, BN, num_warps, num_stages of the copy kernels."""
    return (64, 64, 4, 3) if D <= 128 else (64, 32, 8, 2)


@torch.library.custom_op("nanogpt::cplm_copy_fwd", mutates_args=())
def cplm_copy_fwd(q: Tensor, k: Tensor, s0: Tensor, doc: Tensor, tok: Tensor, tgt: Tensor, dstart: Tensor,
                  dend: Tensor, L: int, LB: int) -> tuple[Tensor, Tensor, Tensor]:
    """dend is read by the backward only; it rides through the op so setup_context can save it."""
    N, D = q.shape
    NK = tok.shape[0]  # tok, tgt: [NK, N] (see _copy_fwd_kernel)
    BM, BN, nw, ns = _copy_tiles(D)
    assert N % L == 0 and L % BM == 0 and L % BN == 0 and D in (64, 128, 256) and 1 <= NK <= 4
    q, k, s0 = q.contiguous(), k.contiguous(), s0.contiguous()
    P = torch.empty((NK, N), dtype=torch.float32, device=q.device)
    A = torch.empty(N, dtype=torch.float32, device=q.device)
    LSE = torch.empty_like(A)
    _copy_fwd_kernel[(N // BM,)](q, k, s0, doc.contiguous(), tok.contiguous(), tgt.contiguous(), dstart.contiguous(),
                                 P, A, LSE, N, L, LB, D ** -0.5, D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    return P, A, LSE


@cplm_copy_fwd.register_fake
def _(q, k, s0, doc, tok, tgt, dstart, dend, L, LB):
    N, NK = q.shape[0], tok.shape[0]
    return (q.new_empty((NK, N), dtype=torch.float32), q.new_empty(N, dtype=torch.float32),
            q.new_empty(N, dtype=torch.float32))


@torch.library.custom_op("nanogpt::cplm_copy_bwd", mutates_args=())
def cplm_copy_bwd(q: Tensor, k: Tensor, doc: Tensor, tok: Tensor, tgt: Tensor, dstart: Tensor, dend: Tensor,
                  LSE: Tensor, gP: Tensor, drow: Tensor, L: int, LB: int) -> tuple[Tensor, Tensor]:
    N, D = q.shape
    NK = tok.shape[0]
    BM, BN, nw, ns = _copy_tiles(D)
    q, k, gP, drow = q.contiguous(), k.contiguous(), gP.contiguous(), drow.contiguous()
    doc, tok, tgt, dstart, dend = (t.contiguous() for t in (doc, tok, tgt, dstart, dend))
    dq = torch.empty((N, D), dtype=torch.float32, device=q.device)
    dk = torch.empty_like(dq)
    _copy_bwd_dq_kernel[(N // BM,)](q, k, doc, tok, tgt, dstart, LSE, gP, drow, dq, N, L, LB, D ** -0.5,
                                    D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    _copy_bwd_dk_kernel[(N // BN,)](q, k, doc, tok, tgt, dend, LSE, gP, drow, dk, N, L, LB, D ** -0.5,
                                    D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    return dq, dk


@cplm_copy_bwd.register_fake
def _(q, k, doc, tok, tgt, dstart, dend, LSE, gP, drow, L, LB):
    return q.new_empty(q.shape, dtype=torch.float32), q.new_empty(q.shape, dtype=torch.float32)


def _copy_setup_context(ctx, inputs, output):
    q, k, s0, doc, tok, tgt, dstart, dend, L, LB = inputs
    P, A, LSE = output
    ctx.save_for_backward(q, k, doc, tok, tgt, dstart, dend, LSE, P, A)
    ctx.L, ctx.LB = L, LB
    ctx.set_materialize_grads(False)


def _copy_backward(ctx, gP, gA, gLSE):
    q, k, doc, tok, tgt, dstart, dend, LSE, P, A = ctx.saved_tensors
    gP = torch.zeros_like(P) if gP is None else gP.float()
    gA = torch.zeros_like(A) if gA is None else gA.float()
    drow = (gP * P).sum(0) + gA * A  # d loss / d (log-normalizer), per row
    ds0 = A * (gA - drow)            # sink score gradient
    dq, dk = torch.ops.nanogpt.cplm_copy_bwd(q, k, doc, tok, tgt, dstart, dend, LSE, gP, drow, ctx.L, ctx.LB)
    return dq.to(q.dtype), dk.to(k.dtype), ds0, None, None, None, None, None, None, None


cplm_copy_fwd.register_autograd(_copy_backward, setup_context=_copy_setup_context)


def copy_sink_probs(q, k, k_sink, input_seq, target_seq, seqlens, L):
    """Copy-sink pointer on a packed 1-D stream (PR #379's fused path: no shifts, no lookback cap).

    q, k [N, d] copy queries/keys (RMS-normed); k_sink [d]; input_seq/target_seq [N]; seqlens = cu_seqlens
    ([0, doc ends..., padded with N]). Query i attends to the strictly-past positions j of its own document inside the
    band (key blocks b-1 and b of L tokens for a query in block b) plus the learned sink. Returns
    p_copy [N] = sum_j attn[i, j] * [input_seq[j] == target_seq[i]] and a_sink [N]."""
    N, d = q.shape
    dev = input_seq.device
    cu = seqlens.long().clamp(max=N)
    j = torch.searchsorted(cu, torch.arange(N, device=dev), right=True)  # boundaries <= each position
    cu_ext = torch.cat([cu.new_zeros(1), cu, cu.new_full((1,), N)])
    doc, dstart, dend = j.to(torch.int32), cu_ext[j].to(torch.int32), cu_ext[j + 1].to(torch.int32)
    s0 = (q.float() @ k_sink.float()) * d ** -0.5
    tok = input_seq.to(torch.int32).view(1, N).contiguous()
    tgt = target_seq.to(torch.int32).view(1, N).contiguous()
    P, A, _ = torch.ops.nanogpt.cplm_copy_fwd(q, k, s0, doc, tok, tgt, dstart, dend, L, N)
    return P[0], A


# ---------------------------------------------------------------------------------------------
# The mixture NLL inside the fused fp8 CE kernel. The next-token (k = 0) cross-entropy of the CE kernel becomes
#
#     p(y) = p_lm(y) * (1 + alpha * a_sink / (1 - alpha)) + alpha * p_copy,   alpha = p_lm(<copy>)
#
# (p_lm is the softmax over all 50304 columns, so p_lm / (1 - alpha) is the LM over the real vocabulary). Its logit
# gradient is  w0 * [g_lse * softmax(z)_v + g_t * [v == y] + g_c * [v == copy]]  with the closed-form row scalars
#     g_t = -pl (1 + al as r) / (pmix + 1e-9),   g_c = -al (pl as r^2 + pc) / (pmix + 1e-9),   g_lse = -(g_t + g_c),
# r = 1 / max(1 - al, 1e-6). The copy branch (p_copy, a_sink) does not read the logits, so it is computed first and
# passed in; the kernel also returns d nll / d p_copy and d nll / d a_sink (both times w0) for the pointer.
# The <copy> column is read from a device int (its position in the candidate set changes per step, and the step is
# CUDA-graph captured).

def _patch(src, old, new):
    assert src.count(old) == 1, old[:80]
    return src.replace(old, new)


def cplm_source(src=None):
    """cross_entropy.CE_KERNEL_SOURCE plus the anchored CPLM patches (each asserted to match exactly once)."""
    s = CE_KERNEL_SOURCE if src is None else src
    s = _patch(s, "__global__ void ce_fwd_bwd_kernel(", "__global__ void cplm_ce_fwd_bwd_kernel(")
    s = _patch(s, """    double grad_s_param)
{""", """    double grad_s_param,
    const float* __restrict__ p_copy_ptr,
    const float* __restrict__ a_sink_ptr,
    float* __restrict__ d_pc_ptr,
    float* __restrict__ d_as_ptr,
    const int* __restrict__ copy_col_ptr)
{""")
    s = _patch(s, """  float lse = A + __logf(block_sum);
""", """  float lse = A + __logf(block_sum);

  // CPLM: copy-sink mixture on the next-token (k = 0) term. g_* = d nll / d (z_target, z_copy, lse).
  int copy_col = copy_col_ptr[0];
  float w0 = mtp_weights[0];
  int64_t t0 = targets[blockIdx.x];
  float pl = __expf(A * __half2float(smem[t0]) - lse);
  float al = __expf(A * __half2float(smem[copy_col]) - lse);
  float pc = p_copy_ptr[blockIdx.x];
  float as_ = a_sink_ptr[blockIdx.x];
  float r = 1.0f / fmaxf(1.0f - al, 1e-6f);
  float pmix = pl * (1.0f + al * as_ * r) + al * pc;
  float inv = 1.0f / (pmix + 1e-9f);
  float g_t = -pl * (1.0f + al * as_ * r) * inv;
  float g_c = -al * (pl * as_ * r * r + pc) * inv;
  float g_lse = -(g_t + g_c);
  float nll0 = -__logf(pmix + 1e-9f);
""")
    s = _patch(s, """          total_loss += weight * (lse - z_target);""",
               """          total_loss += weight * (k == 0 ? nll0 : (lse - z_target));""")
    s = _patch(s, """    losses[blockIdx.x] = total_loss;
""", """    losses[blockIdx.x] = total_loss;
    d_pc_ptr[blockIdx.x] = w0 * (-al * inv);
    d_as_ptr[blockIdx.x] = w0 * (-pl * al * r * inv);
""")
    s = _patch(s, """  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }
""", """  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }
  S_w += w0 * (g_lse - 1.0f);  // the k = 0 softmax coefficient is w0 * g_lse instead of w0
""")
    s = _patch(s, """  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && (prefix_weight > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];""",
               """  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && (prefix_weight > 0.0f);
  bool is_copy = (threadIdx.x == n_predict + 1);  // one more thread corrects the <copy> column
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx || is_copy) {
    int i = threadIdx.x;
    int64_t target = is_copy ? (int64_t)copy_col : (is_pfx ? ptgt_row : targets[blockIdx.x + i]);""")
    s = _patch(s, """        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }""", """        if (targets[target_idx] == target) {
          term2 += (k == 0) ? (-w0 * g_t) : mtp_weights[k];
        }""")
    s = _patch(s, """    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += prefix_weight;
    }
""", """    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += prefix_weight;
    }
    if (target == copy_col) {
      term2 += -w0 * g_c;
    }
""")
    return s


CPLM_CE_KERNELS: dict = {}   # vocabulary width (50304, or a sampled-softmax candidate count P) -> compiled kernel


def _compile(vocab: int):
    decls = f"\nconstexpr int VOCAB_SIZE = {vocab};\nconstexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n"
    kernel = torch.cuda._compile_kernel(decls + cplm_source(), "cplm_ce_fwd_bwd_kernel", compute_capability="90",
                                        cuda_include_dirs=CUDA_INCLUDE_DIRS, nvcc_options=["-lineinfo", "--use_fast_math"])
    kernel.set_shared_memory_config(vocab * 2)
    return kernel


# ---------------------------------------------------------------------------------------------
# PHASE 2: the mixture inside the LUT CE kernel (cross_entropy.py: ce_fwd_bwd_lut_kernel).
#
# Per element the CPLM gradient is still  w0 * [g_lse * softmax(z)_v + g_t * [v == y] + g_c * [v == copy]]: the part that
# depends on the element depends only on its e4m3 code byte (through the fp16 sigmoid cache) and on row scalars, so the
# 256-entry gradient table of a row is built exactly as before, with the row's softmax coefficient
#     S_w' = S_w + w0 * (g_lse - 1)
# in place of S_w, and the two sparse corrections (target column: term2 = -w0 g_t, <copy> column: term2 = -w0 g_c) are
# two more lanes of the warp-0 correction block. The row scalars (pl, al, pc, a_sink, r, pmix, inv, g_t, g_c, g_lse,
# nll0) need only FOUR numbers of the row beyond lse -- the e4m3 bytes of the k = 0 target logit and of the <copy> logit
# (warp 0 holds both in `cur_tbyte`, lane 0 and lane n_predict + 1, prefetched a row ahead like every other lane's) and
# p_copy[row] / a_sink[row] (prefetched with the targets) -- which warp 0 publishes through shared memory in the barrier
# the row sums already use. The arithmetic of every scalar is the plain CPLM kernel's, statement for statement, so the
# results should be bit-identical to cplm_ce_fwd_bwd_kernel (checked byte for byte against it in offline tests; that
# comparison is not repeated at start-up).

# The LUT kernels are the production path: they run whenever n_rows % 8 == 0 and are compiled by cplm_init. The flag is
# a constant main() prints and asserts; the plain kernel stays as the n_rows % 8 != 0 fallback.
CPLM_LUT = True
CPLM_LUT_KERNELS: dict = {}


def cplm_lut_source(src=None):
    """CE_LUT_KERNEL_SOURCE plus the anchored CPLM patches (each asserted to match exactly once)."""
    s = CE_LUT_KERNEL_SOURCE if src is None else src
    s = _patch(s, "ce_fwd_bwd_lut_kernel(\n", "cplm_ce_fwd_bwd_lut_kernel(\n")
    s = _patch(s, """    double grad_s_param)
{
  static_assert(BLOCK_SIZE == 256, "one table entry per thread");""", """    double grad_s_param,
    const float* __restrict__ p_copy_ptr,
    const float* __restrict__ a_sink_ptr,
    float* __restrict__ d_pc_ptr,
    float* __restrict__ d_as_ptr,
    const int* __restrict__ copy_col_ptr)
{
  static_assert(BLOCK_SIZE == 256, "one table entry per thread");""")
    s = _patch(s, """  const unsigned long long pol_stream = ce_policy_evict_first();
""", """  const unsigned long long pol_stream = ce_policy_evict_first();
  const int copy_col = copy_col_ptr[0];
""")
    s = _patch(s, """  __shared__ float block_sums[BLOCK_SIZE / 32];
""", """  __shared__ float block_sums[BLOCK_SIZE / 32];
  __shared__ float sh_row[4];   // CPLM: [k = 0 target byte, <copy> byte, p_copy, a_sink] of the row, published by warp 0
""")
    # lane n_predict + 1 carries the <copy> column (its logit byte is prefetched like every lane's target byte)
    s = _patch(s, """    else if (lane == n_predict) { t = prefix_targets[row]; }
    return t;""", """    else if (lane == n_predict) { t = prefix_targets[row]; }
    else if (lane == n_predict + 1) { t = copy_col; }
    return t;""")
    s = _patch(s, """  unsigned int cur_tbyte = 0;
  {""", """  unsigned int cur_tbyte = 0;
  float cur_pcv = 0.f, cur_asv = 0.f;
  {""")
    s = _patch(s, """    cur_ptgt = prefix_targets[row0];
""", """    cur_ptgt = prefix_targets[row0];
    cur_pcv = p_copy_ptr[row0]; cur_asv = a_sink_ptr[row0];
""")
    s = _patch(s, """    long long ntg = -1, nptgt = -1;
""", """    long long ntg = -1, nptgt = -1;
    float npc = 0.f, nas = 0.f;
""")
    s = _patch(s, """      nptgt = prefix_targets[nrow];
""", """      nptgt = prefix_targets[nrow];
      npc = p_copy_ptr[nrow]; nas = a_sink_ptr[nrow];
""")
    s = _patch(s, """    if (lane == 0) block_sums[warp] = thread_sum;
    __syncthreads();""", """    if (lane == 0) block_sums[warp] = thread_sum;
    if (warp == 0) {   // CPLM: warp 0 holds the row's k = 0 target byte (lane 0) and <copy> byte (lane n_predict + 1)
      unsigned int tb0 = __shfl_sync(0xFFFFFFFF, cur_tbyte, 0);
      unsigned int cbb = __shfl_sync(0xFFFFFFFF, cur_tbyte, n_predict + 1);
      if (lane == 0) { sh_row[0] = (float)tb0; sh_row[1] = (float)cbb; sh_row[2] = cur_pcv; sh_row[3] = cur_asv; }
    }
    __syncthreads();""")
    s = _patch(s, """    if (cur_ptgt >= 0) S_w += pw;

""", """    if (cur_ptgt >= 0) S_w += pw;

    // CPLM row scalars: the plain kernel's statements (lut_sf[b] is the fp16-rounded sigmoid cache entry of byte b)
    const float w0_ = w0;
    const float pl = __expf(A * lut_sf[(unsigned int)sh_row[0]] - lse);
    const float al = __expf(A * lut_sf[(unsigned int)sh_row[1]] - lse);
    const float pc = sh_row[2];
    const float as_ = sh_row[3];
    const float r = 1.0f / fmaxf(1.0f - al, 1e-6f);
    const float pmix = pl * (1.0f + al * as_ * r) + al * pc;
    const float inv = 1.0f / (pmix + 1e-9f);
    const float g_t = -pl * (1.0f + al * as_ * r) * inv;
    const float g_c = -al * (pl * as_ * r * r + pc) * inv;
    const float g_lse = -(g_t + g_c);
    const float nll0 = -__logf(pmix + 1e-9f);
    S_w += w0_ * (g_lse - 1.0f);  // the k = 0 softmax coefficient is w0 * g_lse instead of w0

""")
    s = _patch(s, """      float diff = lse - z_t;
""", """      float diff = (lane == 0) ? nll0 : (lse - z_t);
""")
    s = _patch(s, """      if (lane == 0) losses[row] = total_loss;
""", """      if (lane == 0) {
        losses[row] = total_loss;
        d_pc_ptr[row] = w0_ * (-al * inv);
        d_as_ptr[row] = w0_ * (-pl * al * r * inv);
      }
""")
    s = _patch(s, """      bool is_pfx = (lane == n_predict) && (cur_ptgt >= 0) && (pw > 0.0f);
      if (((lane < n_predict) && in_b) || is_pfx) {""", """      bool is_pfx = (lane == n_predict) && (cur_ptgt >= 0) && (pw > 0.0f);
      bool is_copy = (lane == n_predict + 1);   // one more lane corrects the <copy> column
      if (((lane < n_predict) && in_b) || is_pfx || is_copy) {""")
    s = _patch(s, "(row + 0) < batch_size && t0 == target) term2 += w0;", "(row + 0) < batch_size && t0 == target) term2 += (-w0_ * g_t);")
    s = _patch(s, """        if (cur_ptgt >= 0 && cur_ptgt == target) term2 += pw;
""", """        if (cur_ptgt >= 0 && cur_ptgt == target) term2 += pw;
        if (target == copy_col) term2 += -w0_ * g_c;
""")
    s = _patch(s, """      cur_tg = ntg; cur_ptgt = nptgt; cur_tbyte = ntbyte;""", """      cur_tg = ntg; cur_ptgt = nptgt; cur_tbyte = ntbyte; cur_pcv = npc; cur_asv = nas;""")
    return s


def _compile_lut(vocab: int):
    _pre = CE_KERNEL_SOURCE.split('extern "C"')[0]     # the sigmoid / fp8 helpers, shared verbatim with the stock kernels
    src = (f"\n#define CE_MINB {_ce_lut_minb(vocab)}\nconstexpr int VOCAB_SIZE = {vocab};\n"
           f"constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n" + _pre + cplm_lut_source())
    k = torch.cuda._compile_kernel(src, "cplm_ce_fwd_bwd_lut_kernel", compute_capability="90",
                                   cuda_include_dirs=CUDA_INCLUDE_DIRS, nvcc_options=["-lineinfo", "--use_fast_math"])
    k.set_shared_memory_config(CE_LUT_SMEM)
    return k


def cplm_init(p_values):
    """Compile the CPLM CE kernels for the full vocabulary and for every sampled-softmax candidate count. Called ONCE
    from main() BEFORE torch.compile (nvrtc is untimed)."""
    for v in (CE_KERNEL_VOCAB_SIZE, *p_values):
        if v not in CPLM_CE_KERNELS:
            CPLM_CE_KERNELS[v] = _compile(v)
        if v not in CPLM_LUT_KERNELS:
            CPLM_LUT_KERNELS[v] = _compile_lut(v)


@torch.library.custom_op("nanogpt::cplm_ce_fwd_bwd", mutates_args={"losses", "grad_input", "d_pc", "d_as"})
def cplm_ce_fwd_bwd(
    logits: Tensor, targets: Tensor, mtp_weights: Tensor, prefix_targets: Tensor, prefix_weight: Tensor,
    p_copy: Tensor, a_sink: Tensor, copy_col: Tensor, losses: Tensor, grad_input: Tensor, d_pc: Tensor, d_as: Tensor,
    n_rows: int, n_predict: int, A: float, B: float, C: float, grad_s: float,
) -> None:
    """ce_fwd_bwd + the copy-sink mixture, with the kernel of width logits.shape[1] (the LUT one when n_rows % 8 == 0).
    prefix_weight is a 1-element fp32 device tensor the kernel reads (as in cross_entropy.ce_fwd_bwd); copy_col is a
    1-element int32 device tensor holding the <copy> column."""
    V = logits.shape[1]
    args = (logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
            n_rows, n_predict, A, B, C, grad_s, p_copy, a_sink, d_pc, d_as, copy_col)
    if n_rows % CE_LUT_ROWS == 0:
        CPLM_LUT_KERNELS[V]((n_rows // CE_LUT_ROWS, 1, 1), (CE_KERNEL_BLOCK_SIZE, 1, 1), args, shared_mem=CE_LUT_SMEM)
        return
    CPLM_CE_KERNELS[V]((n_rows, 1, 1), (CE_KERNEL_BLOCK_SIZE, 1, 1), args, shared_mem=V * 2)


def _run(logits, targets, mtp_weights, prefix_targets, prefix_weight, p_copy, a_sink, copy_col, grad_s):
    n_rows, n_cols = logits.shape
    losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    d_pc, d_as = torch.empty_like(losses), torch.empty_like(losses)
    grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)
    cplm_ce_fwd_bwd(logits.contiguous(), targets.contiguous(), mtp_weights.contiguous(), prefix_targets.contiguous(),
                    prefix_weight.reshape(1).to(torch.float32).contiguous(), p_copy.float().contiguous(),
                    a_sink.float().contiguous(), copy_col.to(torch.int32).contiguous(),
                    losses, grad_input, d_pc, d_as, n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
    return losses, grad_input, d_pc, d_as


class FusedCPLMCrossEntropy(torch.autograd.Function):
    """FusedSoftcappedCrossEntropy with the k = 0 term replaced by the copy-sink mixture NLL."""
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, prefix_targets, prefix_weight, lm_head_weight, w_f8_col, w_f8_row,
                x_s, w_s, grad_s, p_copy, a_sink, copy_col):
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        # w_f8_col.T is contiguous, so this is a zero-copy column-major view.
        logits = _ce_logit_gemm(x_f8, w_f8_col.T.contiguous().T, x_s, w_s)
        losses, grad_input, d_pc, d_as = _run(logits, targets, mtp_weights, prefix_targets, prefix_weight, p_copy,
                                              a_sink, copy_col, grad_s)
        # The backward's mat2 needs w_f8.T column-major, so it takes the ROW-major cache.
        ctx.save_for_backward(x_f8, w_f8_row, grad_input, d_pc, d_as)
        ctx.scales = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8_row, grad_input, d_pc, d_as = ctx.saved_tensors
        x_s, w_s, grad_s = ctx.scales
        grad_x, grad_w = _ce_backward_gemms(grad_input, w_f8_row, x_f8, x_s, w_s, grad_s)
        # The logit gradients assume d(loss.sum())/d(loss_i) = 1 (folded in by the forward); the copy branch uses
        # grad_output. One slot per forward input; grad_w lands on lm_head_weight.
        return (grad_x, None, None, None, None, grad_w, None, None, None, None, None,
                d_pc * grad_output, d_as * grad_output, None)


class SampledCPLMCrossEntropy(torch.autograd.Function):
    """SampledSoftcappedCrossEntropy with the copy-sink mixture; <copy> is forced into the candidate set
    (CandidateBuilder.build) and copy_col is its position there (SampledLoss.vocab_pos[COPY_TOKEN_ID])."""
    @staticmethod
    def forward(ctx, x, mtp_weights, target_pos, prefix_pos, prefix_weight, lm_head_weight, rows, rows_t, vocab_pos,
                x_s, w_s, grad_s, p_copy, a_sink, copy_col):
        n_rows = x.shape[0]
        assert target_pos.shape[0] == n_rows and prefix_pos.shape[0] == n_rows
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, rows.T, x_s, w_s)   # rows.T is column-major
        losses, grad_input, d_pc, d_as = _run(logits, target_pos, mtp_weights, prefix_pos, prefix_weight, p_copy,
                                              a_sink, copy_col, grad_s)
        ctx.save_for_backward(x_f8, rows_t, grad_input, vocab_pos, d_pc, d_as, copy_col, target_pos, prefix_pos)
        ctx.scales = (x_s, w_s, grad_s)
        ctx.n_predict = mtp_weights.shape[0]
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, rows_t, grad_input, vocab_pos, d_pc, d_as, copy_col, target_pos, prefix_pos = ctx.saved_tensors
        if grad_input.shape[0] >= sampled_head.MIN_ROWS:
            # sampled dW (positive part 1 row in GROUP, negative part and <copy> exact); dX exact
            grad_x, grad_w_c = sampled_head.ce_backward(grad_input, rows_t, x_f8, *ctx.scales, ctx.n_predict, copy_col,
                                                        target_pos, prefix_pos)
        else:
            grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)
        # Freshly allocated (AccumulateGrad may adopt it as .grad) and never zeroed: densify writes every element.
        grad_w = torch.empty((x_f8.shape[1], vocab_pos.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w)
        # One slot per forward input; grad_w lands on lm_head_weight.
        return (grad_x, None, None, None, None, grad_w, None, None, None, None, None, None,
                d_pc * grad_output, d_as * grad_output, None)


# ---------------------------------------------------------------------------------------------
# The model-side loss (GPT._loss delegates here).

def cplm_loss(m, x, input_seq, target_seq, seqlens, mtp_weights, prefix_weight, sampled_loss, slab_rows):
    """Per-token loss with the next-token CE replaced by the copy-sink mixture.

    m: the GPT module (copy_qk_bank [2, 128, D] bf16, copy_k_sink [128], copy_qk_gain [1], copy_eval_block,
    copy_col_full, cplm_diag, canon_mask, lm_head, the fp8 lm_head caches, prefix_table, .training); x: the final normed
    hidden state; prefix_weight: the prefix-loss weight, a 1-element tensor (training); sampled_loss: this step's
    candidate set, None = full softmax (training only: the validation always runs the full softmax).
    Training: the mixture runs inside the fused fp8 CE kernel (cplm_ce_fwd_bwd), full-vocabulary or sampled.
    Validation: the same mixture over the full-softmax softcapped logits, <copy> being one of the vocab columns, with the
    canonical-token mask applied."""
    xf = x.view(-1, x.size(-1))
    bank = m.copy_qk_bank.type_as(xf)
    q, k = torch.nn.functional.linear(xf, bank.flatten(0, 1)).chunk(2, dim=-1)
    F = torch.nn.functional
    q = (F.rms_norm(q.float(), (q.size(-1),)) * m.copy_qk_gain).type_as(xf)
    k = F.rms_norm(k.float(), (k.size(-1),)).type_as(xf)
    block = COPY_BLOCK if m.training else m.copy_eval_block
    p_copy, a_sink = copy_sink_probs(q, k, m.copy_k_sink, input_seq, target_seq, seqlens, block)
    if m.training:
        if sampled_loss is not None:
            # <copy> is forced into the candidate set (CandidateBuilder.build); its position in C is vocab_pos[<copy>],
            # refilled by SampledSoftmax.gather
            n = target_seq.size(0)
            copy_col = sampled_loss.vocab_pos[COPY_TOKEN_ID:COPY_TOKEN_ID + 1]
            return SampledCPLMCrossEntropy.apply(
                xf, mtp_weights, sampled_loss.target_pos[:n], sampled_loss.prefix_pos[:n], prefix_weight,
                m.lm_head.weight, sampled_loss.rows, sampled_loss.rows_t, sampled_loss.vocab_pos,
                m.lm_head.x_s, m.lm_head.w_s, m.lm_head.grad_s, p_copy, a_sink, copy_col)
        return FusedCPLMCrossEntropy.apply(
            xf, target_seq, mtp_weights, m.prefix_table[target_seq], prefix_weight, m.lm_head.weight,
            m._lm_head_f8_col, m._lm_head_f8_row, m.lm_head.x_s, m.lm_head.w_s, m.lm_head.grad_s,
            p_copy, a_sink, m.copy_col_full)
    # Validation, over slab_rows-row slabs.
    rows = xf.shape[0]
    parts = []
    shifts = torch.arange(8, dtype=torch.uint8, device=xf.device)
    for lo in range(0, rows, slab_rows):
        hi = min(lo + slab_rows, rows)
        logits = m.lm_head(xf[lo:hi])
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        logits = logits.float()
        # Drop the tokens the tokenizer would never emit after input_seq (canonical_mask.py; -60 is far below the softcap's
        # 0..23); the <copy> gate is not a token and is never masked. The pointer's mass on a dropped token is not lost:
        # it goes to that (impossible) token, so the mixture below stays a normalized distribution.
        dropped = (m.canon_mask[input_seq[lo:hi], :, None] >> shifts & 1).view(logits.shape).bool()
        dropped[:, COPY_TOKEN_ID] = False
        logits = logits.masked_fill(dropped, -60.0)
        lse = logits.logsumexp(-1)
        p_lm = torch.exp(logits.gather(1, target_seq[lo:hi, None])[:, 0] - lse)
        alpha = torch.exp(logits[:, COPY_TOKEN_ID] - lse)      # the <copy> slot's softmax mass
        p_t = p_lm / (1.0 - alpha).clamp_min(1e-6)             # the LM over the real vocabulary
        pc, asink = p_copy[lo:hi], a_sink[lo:hi]
        p = p_t * (1.0 - alpha * (1.0 - asink)) + alpha * pc
        nll = -torch.log(p.clamp_min(1e-30))   # exact fp32 NLL: p is normalized, so only floor it (+1e-9 per class would sum to 1 + 5e-5)
        lm_nll = -torch.log(p_t + 1e-30)
        m.cplm_diag.add_(torch.stack([torch.ones_like(nll).sum(), nll.sum(), lm_nll.sum(), alpha.sum(),
                                      asink.sum(), (alpha * pc / p.clamp_min(1e-9)).sum()]))
        parts.append(nll)
    return torch.cat(parts) if len(parts) > 1 else parts[0]

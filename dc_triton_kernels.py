
"""Triton implementation for the best-config post-only no-DD DC correction.

This file is intentionally trimmed to the path used by machine2_exps/train_mudd.py:
layer-10 post-only DC, H=6, D=128, W<=128, split FA3 base output plus fused
Triton correction, with native Triton backward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


DC_POSTONLY_CORR_FWD_WSMALL_BLOCK_K = 128
DC_POSTONLY_CORR_BWD_WSMALL_BLOCK_K = 128
DC_POSTONLY_CORR_BWD_WSMALL_PRE_WARPS = 8
DC_POSTONLY_CORR_BWD_WSMALL_QK_WARPS = 4
DC_POSTONLY_CORR_BWD_WSMALL_PRE_STAGES = 1
DC_POSTONLY_CORR_BWD_WSMALL_QK_STAGES = 2


@triton.jit
def _dc_postonly_doc_bounds(
    CU_SEQLENS,
    q_offs,
    q_mask,
    T: tl.constexpr,
    N_DOCS: tl.constexpr,
    CU_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_safe = tl.minimum(q_offs, T - 1)
    lo = tl.zeros((BLOCK_M,), dtype=tl.int32)
    hi = tl.full((BLOCK_M,), N_DOCS, dtype=tl.int32)
    for _ in tl.static_range(0, CU_SEARCH_ITERS):
        mid = (lo + hi) // 2
        boundary = tl.load(CU_SEQLENS + mid, mask=mid <= N_DOCS, other=T + 1).to(tl.int32)
        go_right = boundary <= q_safe
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    doc_idx = tl.maximum(lo - 1, 0)
    doc_start = tl.load(CU_SEQLENS + doc_idx, mask=q_mask, other=0).to(tl.int64)
    doc_end = tl.load(CU_SEQLENS + doc_idx + 1, mask=q_mask, other=0).to(tl.int64)
    return doc_start, doc_end

@triton.jit
def _dc_postonly_build_doc_bounds_table_kernel(
    CU_SEQLENS,
    DOC_START_TABLE,
    DOC_END_TABLE,
    T: tl.constexpr,
    N_DOCS: tl.constexpr,
    CU_SEARCH_ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    q_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    q_mask = q_offs < T
    doc_start, doc_end = _dc_postonly_doc_bounds(
        CU_SEQLENS, q_offs, q_mask, T, N_DOCS, CU_SEARCH_ITERS, BLOCK_T
    )
    tl.store(DOC_START_TABLE + q_offs, doc_start.to(tl.int32), mask=q_mask)
    tl.store(DOC_END_TABLE + q_offs, doc_end.to(tl.int32), mask=q_mask)


def _build_doc_bounds_tables(
    cu_seqlens: torch.Tensor,
    T: int,
    n_docs: int,
    cu_search_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    doc_start_table = torch.empty((T,), device=cu_seqlens.device, dtype=torch.int32)
    doc_end_table = torch.empty((T,), device=cu_seqlens.device, dtype=torch.int32)
    block_t = 256
    _dc_postonly_build_doc_bounds_table_kernel[(triton.cdiv(T, block_t),)](
        cu_seqlens,
        doc_start_table,
        doc_end_table,
        T=T,
        N_DOCS=n_docs,
        CU_SEARCH_ITERS=cu_search_iters,
        BLOCK_T=block_t,
        num_warps=8,
        num_stages=1,
    )
    return doc_start_table, doc_end_table

@triton.jit
def _dc_postonly_probs_wsmall_head(
    Q,
    K,
    POST_W1,
    q_offs,
    k_offs,
    d_offs,
    q_mask,
    d_mask,
    valid,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wt,
    stride_wh,
    scaling,
    T: tl.constexpr,
    H: tl.constexpr,
):
    q = tl.load(
        Q + q_offs[:, None].to(tl.int64) * stride_qt + H * stride_qh
        + d_offs[None, :].to(tl.int64) * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_load_offs = tl.minimum(k_offs, T - 1)
    k = tl.load(
        K + k_load_offs[:, None].to(tl.int64) * stride_kt + H * stride_kh
        + d_offs[None, :].to(tl.int64) * stride_kd,
        mask=d_mask[None, :],
        other=0.0,
    )
    score = tl.dot(q, tl.trans(k), input_precision="tf32") * (scaling * 1.4426950408889634)
    score = tl.where(valid, score, -3.4028234663852886e38)
    m = tl.max(score, axis=1)
    m = tl.where(q_mask, m, 0.0)
    p = tl.exp2(score - m[:, None])
    p = tl.where(valid, p, 0.0)
    denom = tl.sum(p, axis=1)
    denom_safe = tl.where(denom > 0.0, denom, 1.0)
    probs = p / denom_safe[:, None]
    lse = m * 0.6931471805599453 + tl.log(denom_safe)
    lse = tl.where(q_mask, lse, 0.0)
    w1 = tl.load(
        POST_W1 + q_offs.to(tl.int64) * stride_wt + H * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    return probs, w1, lse

@triton.jit
def _dc_postonly_probs_wsmall_head_no_lse(
    Q,
    K,
    POST_W1,
    q_offs,
    k_offs,
    d_offs,
    q_mask,
    d_mask,
    valid,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wt,
    stride_wh,
    scaling,
    T: tl.constexpr,
    H: tl.constexpr,
):
    q = tl.load(
        Q + q_offs[:, None].to(tl.int64) * stride_qt + H * stride_qh
        + d_offs[None, :].to(tl.int64) * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_load_offs = tl.minimum(k_offs, T - 1)
    k = tl.load(
        K + k_load_offs[:, None].to(tl.int64) * stride_kt + H * stride_kh
        + d_offs[None, :].to(tl.int64) * stride_kd,
        mask=d_mask[None, :],
        other=0.0,
    )
    score = tl.dot(q, tl.trans(k), input_precision="tf32") * (scaling * 1.4426950408889634)
    score = tl.where(valid, score, -3.4028234663852886e38)
    m = tl.max(score, axis=1)
    m = tl.where(q_mask, m, 0.0)
    p = tl.exp2(score - m[:, None])
    p = tl.where(valid, p, 0.0)
    denom = tl.sum(p, axis=1)
    denom_safe = tl.where(denom > 0.0, denom, 1.0)
    probs = p / denom_safe[:, None]
    w1 = tl.load(
        POST_W1 + q_offs.to(tl.int64) * stride_wt + H * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    return probs, w1

@triton.jit
def _dc_postonly_store_corr_wsmall_head(
    V,
    POST_W2,
    BASE,
    OUT,
    a_acc,
    q_offs,
    k_offs,
    d_offs,
    q_mask,
    d_mask,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_wt,
    stride_wh,
    stride_bh,
    stride_bt,
    stride_bd,
    stride_oh,
    stride_ot,
    stride_od,
    T: tl.constexpr,
    H: tl.constexpr,
    ADD_BASE: tl.constexpr,
):
    k_load_offs = tl.minimum(k_offs, T - 1)
    vv = tl.load(
        V + k_load_offs[:, None].to(tl.int64) * stride_vt + H * stride_vh
        + d_offs[None, :].to(tl.int64) * stride_vd,
        mask=d_mask[None, :],
        other=0.0,
    )
    w2 = tl.load(
        POST_W2 + q_offs.to(tl.int64) * stride_wt + H * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    corr_weight = w2[:, None] * a_acc
    acc = tl.dot(corr_weight.to(vv.dtype), vv, input_precision="tf32")
    if ADD_BASE:
        base = tl.load(
            BASE + H * stride_bh + q_offs[:, None].to(tl.int64) * stride_bt
            + d_offs[None, :].to(tl.int64) * stride_bd,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += base
    tl.store(
        OUT + H * stride_oh + q_offs[:, None].to(tl.int64) * stride_ot
        + d_offs[None, :].to(tl.int64) * stride_od,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )

@triton.jit
def _dc_postonly_corr_fwd_wsmall_cached_kernel(
    Q,
    K,
    V,
    POST_W1,
    POST_W2,
    DOC_START_TABLE,
    DOC_END_TABLE,
    A_BUF,
    LSE,
    BASE,
    OUT,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_w1t,
    stride_w1h,
    stride_w2t,
    stride_w2h,
    stride_at,
    stride_aw,
    stride_lt,
    stride_lh,
    stride_bh,
    stride_bt,
    stride_bd,
    stride_oh,
    stride_ot,
    stride_od,
    scaling,
    T: tl.constexpr,
    WINDOW: tl.constexpr,
    N_DOCS: tl.constexpr,
    CU_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_AUX: tl.constexpr,
    STORE_A_BUF: tl.constexpr,
    ADD_BASE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    k_start = q_start - WINDOW + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + tl.arange(0, BLOCK_K)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < T
    d_mask = d_offs < BLOCK_D
    doc_start = tl.load(DOC_START_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    doc_end = tl.load(DOC_END_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    rel = q_offs[:, None] - k_offs[None, :]
    valid = (
        q_mask[:, None]
        & (k_offs[None, :] < T)
        & (rel >= 0)
        & (rel < WINDOW)
        & (k_offs[None, :] >= doc_start[:, None])
        & (k_offs[None, :] < doc_end[:, None])
    )

    if STORE_AUX:
        p0, w10, lse0 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=0,
        )
        p1, w11, lse1 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=1,
        )
        p2, w12, lse2 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=2,
        )
        p3, w13, lse3 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=3,
        )
        p4, w14, lse4 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=4,
        )
        p5, w15, lse5 = _dc_postonly_probs_wsmall_head(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=5,
        )
    else:
        p0, w10 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=0,
        )
        p1, w11 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=1,
        )
        p2, w12 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=2,
        )
        p3, w13 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=3,
        )
        p4, w14 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=4,
        )
        p5, w15 = _dc_postonly_probs_wsmall_head_no_lse(
            Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid,
            stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
            stride_w1t, stride_w1h, scaling, T=T, H=5,
        )
    a_acc = (
        w10[:, None] * p0
        + w11[:, None] * p1
        + w12[:, None] * p2
        + w13[:, None] * p3
        + w14[:, None] * p4
        + w15[:, None] * p5
    )
    if STORE_A_BUF:
        tl.store(
            A_BUF + q_offs[:, None].to(tl.int64) * stride_at + rel.to(tl.int64) * stride_aw,
            a_acc,
            mask=valid,
        )
    if STORE_AUX:
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 0 * stride_lh, lse0, mask=q_mask)
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 1 * stride_lh, lse1, mask=q_mask)
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 2 * stride_lh, lse2, mask=q_mask)
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 3 * stride_lh, lse3, mask=q_mask)
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 4 * stride_lh, lse4, mask=q_mask)
        tl.store(LSE + q_offs.to(tl.int64) * stride_lt + 5 * stride_lh, lse5, mask=q_mask)
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=0, ADD_BASE=ADD_BASE,
    )
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=1, ADD_BASE=ADD_BASE,
    )
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=2, ADD_BASE=ADD_BASE,
    )
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=3, ADD_BASE=ADD_BASE,
    )
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=4, ADD_BASE=ADD_BASE,
    )
    _dc_postonly_store_corr_wsmall_head(
        V, POST_W2, BASE, OUT, a_acc, q_offs, k_offs, d_offs, q_mask, d_mask,
        stride_vt, stride_vh, stride_vd, stride_w2t, stride_w2h,
        stride_bh, stride_bt, stride_bd, stride_oh, stride_ot, stride_od,
        T=T, H=5, ADD_BASE=ADD_BASE,
    )

@triton.jit
def _dc_postonly_probs_loop_head(
    Q,
    K,
    POST_W1,
    q_offs,
    k_offs,
    d_offs,
    q_mask,
    d_mask,
    valid,
    lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wt,
    stride_wh,
    scaling,
    T: tl.constexpr,
    H: tl.constexpr,
):
    q = tl.load(
        Q + q_offs[:, None].to(tl.int64) * stride_qt + H * stride_qh
        + d_offs[None, :].to(tl.int64) * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_load_offs = tl.minimum(k_offs, T - 1)
    k = tl.load(
        K + k_load_offs[:, None].to(tl.int64) * stride_kt + H * stride_kh
        + d_offs[None, :].to(tl.int64) * stride_kd,
        mask=d_mask[None, :],
        other=0.0,
    )
    score = tl.dot(q, tl.trans(k), input_precision="tf32") * (scaling * 1.4426950408889634)
    probs = tl.exp2(score - (lse * 1.4426950408889634)[:, None])
    probs = tl.where(valid, probs, 0.0)
    w1 = tl.load(
        POST_W1 + q_offs.to(tl.int64) * stride_wt + H * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    return probs, w1

@triton.jit
def _dc_postonly_corr_bwd_pre_wsmall_dm_head(
    V,
    DO,
    POST_W2,
    DV,
    GPOST_W2,
    da_acc,
    a_hidden,
    q_offs,
    k_offs,
    rel,
    valid,
    q_mask,
    d_offs,
    d_mask,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_w2t,
    stride_w2h,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    stride_gwt,
    stride_gwh,
    T: tl.constexpr,
    H: tl.constexpr,
):
    do = tl.load(
        DO + H * stride_doh + q_offs[:, None].to(tl.int64) * stride_dot
        + d_offs[None, :].to(tl.int64) * stride_dod,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_load_offs = tl.minimum(k_offs, T - 1)
    vv = tl.load(
        V + k_load_offs[:, None].to(tl.int64) * stride_vt + H * stride_vh
        + d_offs[None, :].to(tl.int64) * stride_vd,
        mask=d_mask[None, :],
        other=0.0,
    )
    dm = tl.dot(do, tl.trans(vv), input_precision="tf32").to(tl.float32)
    w2 = tl.load(
        POST_W2 + q_offs.to(tl.int64) * stride_w2t + H * stride_w2h,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    a_hidden = tl.where(valid, a_hidden, 0.0)
    dv_weight = w2[:, None] * a_hidden
    dv_blk = tl.dot(tl.trans(dv_weight.to(do.dtype)), do, input_precision="tf32")
    tl.atomic_add(
        DV + k_offs[:, None].to(tl.int64) * stride_dv_t + H * stride_dv_h
        + d_offs[None, :].to(tl.int64) * stride_dv_d,
        dv_blk,
        sem="relaxed",
        mask=(k_offs[:, None] < T) & d_mask[None, :],
    )
    gpost_w2 = tl.sum(dm * a_hidden, axis=1)
    tl.store(
        GPOST_W2 + q_offs.to(tl.int64) * stride_gwt + H * stride_gwh,
        gpost_w2,
        mask=q_mask,
    )
    return da_acc + w2[:, None] * dm

@triton.jit
def _dc_postonly_corr_bwd_pre_wsmall_soft_head(
    POST_W1,
    SOFT_DOT,
    GPOST_W1,
    probs,
    da_acc,
    q_offs,
    q_mask,
    stride_wt,
    stride_wh,
    stride_sdt,
    stride_sdh,
    stride_gwt,
    stride_gwh,
    H: tl.constexpr,
):
    w1 = tl.load(
        POST_W1 + q_offs.to(tl.int64) * stride_wt + H * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    dp = w1[:, None] * da_acc
    soft_dot = tl.sum(dp * probs, axis=1)
    gpost_w1 = tl.sum(da_acc * probs, axis=1)
    tl.store(
        SOFT_DOT + q_offs.to(tl.int64) * stride_sdt + H * stride_sdh,
        soft_dot,
        mask=q_mask,
    )
    tl.store(
        GPOST_W1 + q_offs.to(tl.int64) * stride_gwt + H * stride_gwh,
        gpost_w1,
        mask=q_mask,
    )

@triton.jit
def _dc_postonly_corr_bwd_pre_wsmall_kernel(
    Q,
    K,
    V,
    DO,
    POST_W1,
    POST_W2,
    DOC_START_TABLE,
    DOC_END_TABLE,
    LSE,
    DA_BUF,
    SOFT_DOT,
    DV,
    GPOST_W1,
    GPOST_W2,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_w1t,
    stride_w1h,
    stride_w2t,
    stride_w2h,
    stride_at,
    stride_aw,
    stride_lt,
    stride_lh,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    stride_gwt,
    stride_gwh,
    stride_sdt,
    stride_sdh,
    scaling,
    T: tl.constexpr,
    WINDOW: tl.constexpr,
    N_DOCS: tl.constexpr,
    CU_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    k_start = q_start - WINDOW + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + tl.arange(0, BLOCK_K)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < T
    d_mask = d_offs < BLOCK_D
    doc_start = tl.load(DOC_START_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    doc_end = tl.load(DOC_END_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    rel = q_offs[:, None] - k_offs[None, :]
    valid = (
        q_mask[:, None]
        & (k_offs[None, :] < T)
        & (rel >= 0)
        & (rel < WINDOW)
        & (k_offs[None, :] >= doc_start[:, None])
        & (k_offs[None, :] < doc_end[:, None])
    )

    lse0 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 0 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    lse1 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 1 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    lse2 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 2 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    lse3 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 3 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    lse4 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 4 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    lse5 = tl.load(LSE + q_offs.to(tl.int64) * stride_lt + 5 * stride_lh, mask=q_mask, other=0.0).to(tl.float32)
    p0, _w10 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse0,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=0,
    )
    p1, _w11 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse1,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=1,
    )
    p2, _w12 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse2,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=2,
    )
    p3, _w13 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse3,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=3,
    )
    p4, _w14 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse4,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=4,
    )
    p5, _w15 = _dc_postonly_probs_loop_head(
        Q, K, POST_W1, q_offs, k_offs, d_offs, q_mask, d_mask, valid, lse5,
        stride_qt, stride_qh, stride_qd, stride_kt, stride_kh, stride_kd,
        stride_w1t, stride_w1h, scaling, T=T, H=5,
    )

    a_hidden = (
        _w10[:, None] * p0
        + _w11[:, None] * p1
        + _w12[:, None] * p2
        + _w13[:, None] * p3
        + _w14[:, None] * p4
        + _w15[:, None] * p5
    )
    da_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=0,
    )
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=1,
    )
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=2,
    )
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=3,
    )
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=4,
    )
    da_acc = _dc_postonly_corr_bwd_pre_wsmall_dm_head(
        V, DO, POST_W2, DV, GPOST_W2, da_acc, a_hidden, q_offs, k_offs, rel, valid, q_mask, d_offs, d_mask,
        stride_vt, stride_vh, stride_vd, stride_dot, stride_doh, stride_dod,
        stride_w2t, stride_w2h, stride_dv_t, stride_dv_h, stride_dv_d,
        stride_gwt, stride_gwh, T=T, H=5,
    )
    tl.store(
        DA_BUF + q_offs[:, None].to(tl.int64) * stride_at + rel.to(tl.int64) * stride_aw,
        da_acc,
        mask=valid,
    )

    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p0, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=0,
    )
    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p1, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=1,
    )
    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p2, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=2,
    )
    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p3, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=3,
    )
    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p4, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=4,
    )
    _dc_postonly_corr_bwd_pre_wsmall_soft_head(
        POST_W1, SOFT_DOT, GPOST_W1, p5, da_acc, q_offs, q_mask,
        stride_w1t, stride_w1h, stride_sdt, stride_sdh, stride_gwt, stride_gwh, H=5,
    )

@triton.jit
def _dc_postonly_corr_bwd_qk_wsmall_kernel(
    Q,
    K,
    POST_W1,
    DOC_START_TABLE,
    DOC_END_TABLE,
    DA_BUF,
    SOFT_DOT,
    LSE,
    DQ,
    DK,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wt,
    stride_wh,
    stride_at,
    stride_aw,
    stride_sdt,
    stride_sdh,
    stride_lt,
    stride_lh,
    stride_dq_t,
    stride_dq_h,
    stride_dq_d,
    stride_dk_t,
    stride_dk_h,
    stride_dk_d,
    scaling,
    T: tl.constexpr,
    WINDOW: tl.constexpr,
    N_DOCS: tl.constexpr,
    CU_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    k_start = q_start - WINDOW + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + tl.arange(0, BLOCK_K)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < T
    d_mask = d_offs < BLOCK_D
    doc_start = tl.load(DOC_START_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    doc_end = tl.load(DOC_END_TABLE + q_offs, mask=q_mask, other=0).to(tl.int64)
    rel = q_offs[:, None] - k_offs[None, :]
    valid = (
        q_mask[:, None]
        & (k_offs[None, :] < T)
        & (rel >= 0)
        & (rel < WINDOW)
        & (k_offs[None, :] >= doc_start[:, None])
        & (k_offs[None, :] < doc_end[:, None])
    )
    q = tl.load(
        Q + q_offs[:, None].to(tl.int64) * stride_qt + h * stride_qh
        + d_offs[None, :].to(tl.int64) * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_load_offs = tl.minimum(k_offs, T - 1)
    k = tl.load(
        K + k_load_offs[:, None].to(tl.int64) * stride_kt + h * stride_kh
        + d_offs[None, :].to(tl.int64) * stride_kd,
        mask=d_mask[None, :],
        other=0.0,
    )
    lse = tl.load(
        LSE + q_offs.to(tl.int64) * stride_lt + h * stride_lh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    score = tl.dot(q, tl.trans(k), input_precision="tf32") * (scaling * 1.4426950408889634)
    probs = tl.exp2(score - (lse * 1.4426950408889634)[:, None])
    probs = tl.where(valid, probs, 0.0)
    post_w1 = tl.load(
        POST_W1 + q_offs.to(tl.int64) * stride_wt + h * stride_wh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    da = tl.load(
        DA_BUF + q_offs[:, None].to(tl.int64) * stride_at + rel.to(tl.int64) * stride_aw,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    soft_dot = tl.load(
        SOFT_DOT + q_offs.to(tl.int64) * stride_sdt + h * stride_sdh,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    dp = post_w1[:, None] * da
    ds = probs * (dp - soft_dot[:, None])
    dq_blk = tl.dot(ds.to(k.dtype), k, input_precision="tf32") * scaling
    dk_blk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="tf32") * scaling
    tl.store(
        DQ + q_offs[:, None].to(tl.int64) * stride_dq_t + h * stride_dq_h
        + d_offs[None, :].to(tl.int64) * stride_dq_d,
        dq_blk,
        mask=q_mask[:, None] & d_mask[None, :],
    )
    tl.atomic_add(
        DK + k_offs[:, None].to(tl.int64) * stride_dk_t + h * stride_dk_h
        + d_offs[None, :].to(tl.int64) * stride_dk_d,
        dk_blk,
        sem="relaxed",
        mask=(k_offs[:, None] < T) & d_mask[None, :],
    )

def _is_supported_postonly_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    post_w1: torch.Tensor,
    post_w2: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    return (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and post_w1.is_cuda
        and post_w2.is_cuda
        and seq_lens.is_cuda
        and q.ndim == 4
        and k.shape == q.shape
        and v.shape == q.shape
        and post_w1.shape == q.shape[:3]
        and post_w2.shape == q.shape[:3]
        and q.shape[0] == 1
        and seq_lens.numel() >= 2
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and post_w1.dtype == q.dtype
        and post_w2.dtype == q.dtype
    )

def _normalize_block_m(block_m: int) -> int:
    block_m = max(8, min(64, int(block_m)))
    if block_m & (block_m - 1):
        block_m = 1 << (block_m - 1).bit_length()
    return block_m


def _dc_attention_postonly_correction_add_base_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    post_w1: torch.Tensor,
    post_w2: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
    window: int,
    block_m: int,
    return_aux: bool = False,
    base_out: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    B, T, H, D = q.shape
    assert B == 1 and H == 6 and D == 128
    assert int(window) > 0 and int(window) <= 128
    assert base_out is not None

    n_docs = int(cu_seqlens.numel() - 1)
    cu_search_iters = max(1, n_docs.bit_length())
    doc_start_table, doc_end_table = _build_doc_bounds_tables(
        cu_seqlens, T, n_docs, cu_search_iters
    )
    block_m = _normalize_block_m(block_m)
    block_d = triton.next_power_of_2(D)

    if base_out.ndim == 3:
        assert base_out.shape == (T, H, D)
        base_view = base_out
        base_head_stride, base_t_stride, base_d_stride = base_view.stride(1), base_view.stride(0), base_view.stride(2)
        out = torch.empty_like(base_view)
        out_head_stride, out_t_stride, out_d_stride = out.stride(1), out.stride(0), out.stride(2)
    else:
        assert base_out.shape == (B, T, H, D)
        base_view = base_out
        base_head_stride, base_t_stride, base_d_stride = base_view.stride(2), base_view.stride(1), base_view.stride(3)
        out = torch.empty_like(base_view)
        out_head_stride, out_t_stride, out_d_stride = out.stride(2), out.stride(1), out.stride(3)

    block_k = int(DC_POSTONLY_CORR_FWD_WSMALL_BLOCK_K)
    min_block_k = 1 << (int(window) + block_m - 2).bit_length()
    block_k = max(128, min(256, block_k))
    if block_k & (block_k - 1):
        block_k = 1 << (block_k - 1).bit_length()
    block_k = max(block_k, min_block_k)

    lse_for_bwd = torch.empty((B, T, H), device=q.device, dtype=torch.float32) if return_aux else out
    stride_lt, stride_lh = (lse_for_bwd.stride(1), lse_for_bwd.stride(2)) if return_aux else (0, 0)
    a_buf_placeholder = out
    grid_m = (triton.cdiv(T, block_m),)
    _dc_postonly_corr_fwd_wsmall_cached_kernel[grid_m](
        q,
        k,
        v,
        post_w1,
        post_w2,
        doc_start_table,
        doc_end_table,
        a_buf_placeholder,
        lse_for_bwd,
        base_view,
        out,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        post_w1.stride(1),
        post_w1.stride(2),
        post_w2.stride(1),
        post_w2.stride(2),
        0,
        0,
        stride_lt,
        stride_lh,
        base_head_stride,
        base_t_stride,
        base_d_stride,
        out_head_stride,
        out_t_stride,
        out_d_stride,
        float(scaling),
        T=T,
        WINDOW=int(window),
        N_DOCS=n_docs,
        CU_SEARCH_ITERS=cu_search_iters,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        STORE_AUX=return_aux,
        STORE_A_BUF=False,
        ADD_BASE=True,
        num_warps=(8 if block_m >= 32 else 4),
        num_stages=3,
    )
    if return_aux:
        return out, lse_for_bwd, doc_start_table, doc_end_table
    return out


def _dc_attention_postonly_correction_add_base_backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    post_w1: torch.Tensor,
    post_w2: torch.Tensor,
    cu_seqlens: torch.Tensor,
    doc_start_table: torch.Tensor,
    doc_end_table: torch.Tensor,
    softmax_lse: torch.Tensor,
    grad_out: torch.Tensor,
    scaling: float,
    window: int,
    skip_qk_grad_requested: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    doc_start_table = doc_start_table.contiguous()
    doc_end_table = doc_end_table.contiguous()
    softmax_lse = softmax_lse.contiguous()

    B, T, H, D = q.shape
    assert B == 1 and H == 6 and D == 128
    assert int(window) > 0 and int(window) <= 128
    if grad_out.ndim == 3:
        assert grad_out.shape == (T, H, D)
        stride_dot, stride_doh, stride_dod = grad_out.stride(0), grad_out.stride(1), grad_out.stride(2)
    else:
        assert grad_out.shape == (B, T, H, D)
        stride_dot, stride_doh, stride_dod = grad_out.stride(1), grad_out.stride(2), grad_out.stride(3)
    if softmax_lse.ndim == 2:
        stride_lt, stride_lh = softmax_lse.stride(1), softmax_lse.stride(0)
    else:
        stride_lt, stride_lh = softmax_lse.stride(-2), softmax_lse.stride(-1)

    n_docs = int(cu_seqlens.numel() - 1)
    cu_search_iters = max(1, n_docs.bit_length())
    small_block_m = 16
    small_block_k = int(DC_POSTONLY_CORR_BWD_WSMALL_BLOCK_K)
    dv = torch.zeros_like(v)
    da_buf = torch.empty((B, T, int(window)), device=q.device, dtype=torch.float32)
    soft_dot = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    gpost_w1 = torch.empty_like(post_w1, memory_format=torch.contiguous_format)
    gpost_w2 = torch.empty_like(post_w2, memory_format=torch.contiguous_format)

    grid_m = (triton.cdiv(T, small_block_m),)
    _dc_postonly_corr_bwd_pre_wsmall_kernel[grid_m](
        q,
        k,
        v,
        grad_out,
        post_w1,
        post_w2,
        doc_start_table,
        doc_end_table,
        softmax_lse,
        da_buf,
        soft_dot,
        dv,
        gpost_w1,
        gpost_w2,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        stride_dot,
        stride_doh,
        stride_dod,
        post_w1.stride(1),
        post_w1.stride(2),
        post_w2.stride(1),
        post_w2.stride(2),
        da_buf.stride(1),
        da_buf.stride(2),
        stride_lt,
        stride_lh,
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        gpost_w1.stride(1),
        gpost_w1.stride(2),
        soft_dot.stride(1),
        soft_dot.stride(2),
        float(scaling),
        T=T,
        WINDOW=int(window),
        N_DOCS=n_docs,
        CU_SEARCH_ITERS=cu_search_iters,
        BLOCK_M=small_block_m,
        BLOCK_K=small_block_k,
        BLOCK_D=128,
        num_warps=int(DC_POSTONLY_CORR_BWD_WSMALL_PRE_WARPS),
        num_stages=int(DC_POSTONLY_CORR_BWD_WSMALL_PRE_STAGES),
    )

    if skip_qk_grad_requested:
        return None, None, dv, gpost_w1, gpost_w2

    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)
    grid_qk_small = (triton.cdiv(T, small_block_m), H)
    _dc_postonly_corr_bwd_qk_wsmall_kernel[grid_qk_small](
        q,
        k,
        post_w1,
        doc_start_table,
        doc_end_table,
        da_buf,
        soft_dot,
        softmax_lse,
        dq,
        dk,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        post_w1.stride(1),
        post_w1.stride(2),
        da_buf.stride(1),
        da_buf.stride(2),
        soft_dot.stride(1),
        soft_dot.stride(2),
        stride_lt,
        stride_lh,
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        float(scaling),
        T=T,
        WINDOW=int(window),
        N_DOCS=n_docs,
        CU_SEARCH_ITERS=cu_search_iters,
        BLOCK_M=small_block_m,
        BLOCK_K=small_block_k,
        BLOCK_D=128,
        num_warps=int(DC_POSTONLY_CORR_BWD_WSMALL_QK_WARPS),
        num_stages=int(DC_POSTONLY_CORR_BWD_WSMALL_QK_STAGES),
    )
    return dq, dk, dv, gpost_w1, gpost_w2


class _DCPostOnlyCorrectionAddBaseTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        post_w1: torch.Tensor,
        post_w2: torch.Tensor,
        cu_seqlens: torch.Tensor,
        softmax_lse: torch.Tensor,
        scaling: float,
        window: int,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        needs_bwd = any(ctx.needs_input_grad[1:6])
        ctx.scaling = float(scaling)
        ctx.window = int(window)
        ctx.base_ndim = int(base_out.ndim)
        ctx.needs_bwd = needs_bwd
        out_aux = _dc_attention_postonly_correction_add_base_forward_impl(
            q,
            k,
            v,
            post_w1,
            post_w2,
            cu_seqlens,
            scaling,
            window,
            block_m,
            return_aux=needs_bwd,
            base_out=base_out,
        )
        if needs_bwd:
            out, lse_for_bwd, doc_start_table, doc_end_table = out_aux
            ctx.save_for_backward(
                q, k, v, post_w1, post_w2, cu_seqlens,
                doc_start_table, doc_end_table, lse_for_bwd,
            )
            return out
        ctx.save_for_backward()
        return out_aux

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_base = grad_out if ctx.needs_input_grad[0] else None
        if not ctx.needs_bwd:
            return (grad_base, None, None, None, None, None, None, None, None, None, None, None)
        q, k, v, post_w1, post_w2, cu_seqlens, doc_start_table, doc_end_table, softmax_lse = ctx.saved_tensors
        grad_corr = grad_out
        grads = _dc_attention_postonly_correction_add_base_backward_impl(
            q,
            k,
            v,
            post_w1,
            post_w2,
            cu_seqlens,
            doc_start_table,
            doc_end_table,
            softmax_lse,
            grad_corr,
            ctx.scaling,
            ctx.window,
            not (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]),
        )
        return (grad_base, *grads, None, None, None, None, None, None)


def dc_attention_postonly_nodd_correction_add_base_triton(
    base_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dc_weights: tuple[torch.Tensor, torch.Tensor],
    softmax_lse: torch.Tensor | None,
    scaling: float,
    window: int = None,
    seq_lens: torch.Tensor = None,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    post_w1, post_w2 = dc_weights
    B, T, H, D = q.shape
    if window is None:
        window = 112
    if seq_lens is None:
        seq_lens = torch.tensor([0, T], device=q.device, dtype=torch.int32) if B == 1 else None
    elif seq_lens.numel() == B and B == 1:
        seq_lens = torch.tensor([0, int(seq_lens[0].item())], device=q.device, dtype=torch.int32)

    base_shape_ok = base_out.shape == (T, H, D) or base_out.shape == (B, T, H, D)
    if (
        seq_lens is None
        or not base_shape_ok
        or not _is_supported_postonly_triton(q, k, v, post_w1, post_w2, seq_lens)
        or H != 6
        or D != 128
        or int(window) <= 0
        or int(window) > 128
        or (softmax_lse is not None and softmax_lse.numel() != 0)
    ):
        raise RuntimeError("unsupported fused base+post-only no-DD DC correction Triton shape")
    if softmax_lse is None:
        softmax_lse = torch.empty(0, device=q.device, dtype=torch.float32)

    return _DCPostOnlyCorrectionAddBaseTriton.apply(
        base_out,
        q,
        k,
        v,
        post_w1,
        post_w2,
        seq_lens,
        softmax_lse,
        float(scaling),
        int(window),
        int(block_m),
        int(block_n),
    )
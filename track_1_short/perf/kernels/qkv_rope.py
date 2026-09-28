"""Fused QK-norm + RoPE for attention, and the packed FP8 QKV projection.

What it replaces: the aten chain  qkv = x @ W.T -> split -> rms_norm(q), rms_norm(k) -> rotary -> FA3.
Why it is faster: Q, K and V come from ONE fp8 GEMM against a packed [Q; K; V] weight cache
(perf/kernels/fp8_attention_quant.py), and the QK rms-norm (@Grad62304977) and rotary run as one
Triton pass that writes the separate Q and K tensors FA3 reads. In the backward, the norm/rotary
gradient, the V gradient and their dual-layout e4m3 quantize are one kernel.
Invariants: the norm/rotary math runs in fp32 with eps 1.1920928955078125e-7 (F.rms_norm's fp32 eps),
and the QK gradient is rounded to bf16 before the fp8 cast, matching the aten chain these replace.
Q and K keep the layer's own head width (64 or 128): FA3 reads them unpadded.

Provenance: record #344 (packed fp8 QKV, fused norm/rotary); record #360 (mixed head widths without
padding; the activation quantize hoisted out of the op so layers reading the same input share it).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_forward_kernel(
    qk, factor1, factor2, out_q, out_k,
    rows: tl.constexpr,
    num_heads: tl.constexpr,
    heads2: tl.constexpr,
    qk_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    stride_qkt: tl.constexpr,
    stride_qkh: tl.constexpr,
    factor_stride_t: tl.constexpr,
    stride_out_t: tl.constexpr,
    PAIRED: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < rows) & (offs_d[None, :] < qk_dim)

    token = offs_m // heads2
    input_head = offs_m % heads2
    x = tl.load(
        qk + token[:, None] * stride_qkt + input_head[:, None] * stride_qkh + offs_d[None, :],
        mask=mask, other=0.0,
    )
    # RoPE swaps adjacent lanes. The full Q/K tile is already resident, so form the swapped view
    # in registers instead of reading it from HBM again.
    x_flip = tl.reshape(
        tl.flip(tl.reshape(x, (BLOCK_M, BLOCK_D // 2, 2)), dim=2),
        (BLOCK_M, BLOCK_D),
    )
    x = x.to(tl.float32)
    x_flip = x_flip.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7)
    logical_head = input_head % num_heads
    if PAIRED:
        # Paired heads: head h of token t lands at token 2t + h // (num_heads/2), so adjacent heads
        # attend to each other's keys; odd heads read the second half of the paired rotary row.
        output_token = 2 * token + logical_head // (num_heads // 2)
        output_head = logical_head % (num_heads // 2)
        factor_offset = ((logical_head % 2) * qk_dim)[:, None]
    else:
        output_token = token
        output_head = logical_head
        factor_offset = 0
    factor_ptrs = token[:, None] * factor_stride_t + factor_offset + offs_d[None, :]
    f1 = tl.load(factor1 + factor_ptrs, mask=mask, other=0.0).to(tl.float32)
    f2 = tl.load(factor2 + factor_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = f1 * (x * rstd[:, None]) + f2 * (x_flip * rstd[:, None])

    if KEY_OFFSET:
        # Partial key offset: a key's stationary (non-rotating) dims come from the previous token.
        shift_row = (input_head >= num_heads) & (token > 0)
        previous_token = tl.maximum(token - 1, 0)
        x_previous = tl.load(
            qk + previous_token[:, None] * stride_qkt + input_head[:, None] * stride_qkh + offs_d[None, :],
            mask=mask & shift_row[:, None], other=0.0,
        ).to(tl.float32)
        previous_rstd = tl.rsqrt(tl.sum(x_previous * x_previous, axis=1) / qk_dim + 1.1920928955078125e-7)
        shift = shift_row[:, None] & (offs_d[None, :] >= rotary_dim)
        y = tl.where(shift, x_previous * previous_rstd[:, None], y)

    output_ptrs = output_token[:, None] * stride_out_t + output_head[:, None] * qk_dim + offs_d[None, :]
    tl.store(out_q + output_ptrs, y, mask=mask & (input_head[:, None] < num_heads))
    tl.store(out_k + output_ptrs, y, mask=mask & (input_head[:, None] >= num_heads))


def qk_norm_rope_forward(qk, factor1, factor2, num_heads, rotary_dim, paired, key_offset, num_warps=2):
    """(q, k), each [tokens, heads, qk_dim] ([2 * tokens, heads / 2, qk_dim] when paired), from the
    packed [tokens, 2 * heads, qk_dim] QK projection. Dims from `rotary_dim` on are stationary (they are
    what the key offset shifts). Forward only: validation calls it directly, and the fp8 training path
    (PackedFP8QKVFunction) has its own fused backward."""
    tokens, heads2, qk_dim = qk.shape
    assert heads2 == 2 * num_heads
    assert factor1.shape == factor2.shape == (tokens, qk_dim * (2 if paired else 1))
    assert not (paired and key_offset)
    # Shifted keys require an extra token-row load and favor smaller row tiles.
    block_m = 4 if key_offset else 8
    output_shape = (tokens * 2, num_heads // 2, qk_dim) if paired else (tokens, num_heads, qk_dim)
    # Keep Q and K in independent allocations. Returning views into one shared allocation makes
    # functionalization clone both views before this mutating Triton call and copy them back after.
    out_q = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    out_k = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    rows = tokens * heads2
    _qk_norm_rope_forward_kernel[(triton.cdiv(rows, block_m),)](
        qk, factor1, factor2, out_q, out_k,
        rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=rotary_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_out_t=out_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_M=block_m, BLOCK_D=triton.next_power_of_2(qk_dim), num_warps=num_warps,
    )
    return out_q, out_k


# The QK-norm/rotary backward, the V gradient and the dual-layout e4m3 quantize of both, in one kernel
# per attention layer. The grid is head-fastest so factor1/factor2 are read about once instead of once
# per head. The axis=1 reductions are bit-identical under any BLOCK_T: BLOCK_T only grows dim 0.
@triton.jit
def _qkv_norm_rope_pack_fp8_backward_kernel(
    grad_q, grad_k, grad_v, qk, factor1, factor2,
    grad_row, grad_transposed, grad_scale_ptr,
    tokens: tl.constexpr,
    num_heads: tl.constexpr,
    qk_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    v_dim: tl.constexpr,
    qkv_dim: tl.constexpr,
    stride_qkt: tl.constexpr,
    stride_qkh: tl.constexpr,
    factor_stride_t: tl.constexpr,
    stride_grad_q_t: tl.constexpr,
    stride_grad_q_h: tl.constexpr,
    stride_grad_k_t: tl.constexpr,
    stride_grad_k_h: tl.constexpr,
    stride_grad_v_t: tl.constexpr,
    stride_grad_v_h: tl.constexpr,
    PAIRED: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    logical_head = tl.program_id(0)
    token = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_QK)
    token_mask = token < tokens
    qk_mask = token_mask[:, None] & (offs_d[None, :] < qk_dim)
    offs_flip = offs_d ^ 1

    if PAIRED:
        output_token = 2 * token + logical_head // (num_heads // 2)
        output_head = logical_head % (num_heads // 2)
        factor_offset = (logical_head % 2) * qk_dim
    else:
        output_token = token
        output_head = logical_head
        factor_offset = 0

    f1 = tl.load(factor1 + token[:, None] * factor_stride_t + factor_offset + offs_d[None, :],
                 mask=qk_mask, other=0.0).to(tl.float32)
    f2_flip = tl.load(factor2 + token[:, None] * factor_stride_t + factor_offset + offs_flip[None, :],
                      mask=qk_mask, other=0.0).to(tl.float32)
    scale = tl.load(grad_scale_ptr)

    for qk_kind in tl.static_range(2):
        input_head = logical_head + qk_kind * num_heads
        x = tl.load(qk + token[:, None] * stride_qkt + input_head * stride_qkh + offs_d[None, :],
                    mask=qk_mask, other=0.0).to(tl.float32)
        if qk_kind == 0:
            grad_ptr = grad_q + output_token[:, None] * stride_grad_q_t + output_head * stride_grad_q_h + offs_d[None, :]
        else:
            grad_ptr = grad_k + output_token[:, None] * stride_grad_k_t + output_head * stride_grad_k_h + offs_d[None, :]
        grad = tl.load(grad_ptr, mask=qk_mask, other=0.0)
        grad_flip = tl.reshape(tl.flip(tl.reshape(grad, (BLOCK_T, BLOCK_QK // 2, 2)), dim=2), (BLOCK_T, BLOCK_QK))
        grad = grad.to(tl.float32)
        grad_flip = grad_flip.to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7)
        normalized = x * rstd[:, None]
        grad_normalized = f1 * grad + f2_flip * grad_flip
        if KEY_OFFSET and qk_kind == 1:
            next_token = tl.minimum(token + 1, tokens - 1)
            grad_next = tl.load(
                grad_k + next_token[:, None] * stride_grad_k_t + output_head * stride_grad_k_h + offs_d[None, :],
                mask=qk_mask & (token[:, None] < tokens - 1), other=0.0,
            ).to(tl.float32)
            stationary_grad = tl.where(token[:, None] == 0, grad + grad_next, grad_next)
            grad_normalized = tl.where(offs_d[None, :] >= rotary_dim, stationary_grad, grad_normalized)
        correction = tl.sum(grad_normalized * normalized, axis=1) / qk_dim
        dx = rstd[:, None] * (grad_normalized - normalized * correction[:, None])
        # Round to bf16 first: the aten chain materialized the QK gradient in bf16 before the fp8 cast.
        q = (dx.to(tl.bfloat16).to(tl.float32) / scale).to(tl.float8e4nv)
        feature = input_head * qk_dim + offs_d
        tl.store(grad_row + token[:, None] * qkv_dim + feature[None, :], q, mask=qk_mask)
        tl.store(grad_transposed + feature[:, None] * tokens + token[None, :], tl.trans(q),
                 mask=(offs_d[:, None] < qk_dim) & token_mask[None, :])

    offs_v = tl.arange(0, BLOCK_V)
    v_mask = token_mask[:, None] & (offs_v[None, :] < v_dim)
    grad_v_value = tl.load(grad_v + token[:, None] * stride_grad_v_t + logical_head * stride_grad_v_h + offs_v[None, :],
                           mask=v_mask, other=0.0).to(tl.float32)
    qv = (grad_v_value / scale).to(tl.float8e4nv)
    v_feature = 2 * num_heads * qk_dim + logical_head * v_dim + offs_v
    tl.store(grad_row + token[:, None] * qkv_dim + v_feature[None, :], qv, mask=v_mask)
    tl.store(grad_transposed + v_feature[:, None] * tokens + token[None, :], tl.trans(qv),
             mask=(offs_v[:, None] < v_dim) & token_mask[None, :])


def _scaled_mm_bf16(a, b, scale_a, scale_b, fast_accum):
    """The one per-tensor-scaled fp8 GEMM spelling PackedFP8QKV uses, forward and backward."""
    return torch._scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, use_fast_accum=fast_accum)


class PackedFP8QKVFunction(torch.autograd.Function):
    """q, k, v = norm + rotary of x @ (sa_lambda * [Wq; Wk; Wv]).T, as one fp8 GEMM off the cached packed weight.

    The activation arrives already quantized (`x_f8`, `x_f8_t`): layers that read the same attention
    input quantize it once and share it (see GPT.forward). `x` itself only receives the gradient.
    """
    @staticmethod
    def forward(ctx, x, qk_weight, v_weight, weight_f8, weight_f8_t, weight_scale, qkv_scale,
                x_scale, grad_scale, factor1, factor2, num_heads, paired, key_offset, rotary_dim, x_f8, x_f8_t):
        qk_dim = qk_weight.shape[0] // (2 * num_heads)
        v_dim = v_weight.shape[0] // num_heads
        qk_features = 2 * num_heads * qk_dim
        scaled_weight_scale = weight_scale * qkv_scale
        if paired:
            # A packed QKV result leaves V a strided suffix of every token row, which paired attention
            # would have to materialize to merge the token and head axes. Two column-sliced GEMMs read
            # x once more but make V dense at birth.
            qk_flat = _scaled_mm_bf16(x_f8, weight_f8[:qk_features].T, x_scale, scaled_weight_scale, True)
            v_flat = _scaled_mm_bf16(x_f8, weight_f8[qk_features:].T, x_scale, scaled_weight_scale, True)
        else:
            qkv = _scaled_mm_bf16(x_f8, weight_f8.T, x_scale, scaled_weight_scale, True)
            qk_flat, v_flat = qkv[:, :qk_features], qkv[:, qk_features:]
        qk = qk_flat.view(-1, 2 * num_heads, qk_dim)
        v = v_flat.view(-1, num_heads, v_dim)
        q, k = qk_norm_rope_forward(qk, factor1, factor2, num_heads, rotary_dim, paired, key_offset)
        ctx.save_for_backward(qk_weight, v_weight, x_f8_t, weight_f8_t, scaled_weight_scale,
                              qkv_scale, x_scale, grad_scale, qk, factor1, factor2)
        ctx.input_shape = x.shape
        ctx.num_heads, ctx.v_dim = num_heads, v_dim
        ctx.paired, ctx.key_offset, ctx.rotary_dim = paired, key_offset, rotary_dim
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        (qk_weight, v_weight, x_f8_t, weight_f8_t, scaled_weight_scale, qkv_scale,
         x_scale, grad_scale, qk, factor1, factor2) = ctx.saved_tensors
        num_heads, v_dim = ctx.num_heads, ctx.v_dim
        grad_v = grad_v.reshape(-1, num_heads, v_dim)
        tokens, heads2, qk_dim = qk.shape
        qkv_dim = 2 * num_heads * qk_dim + num_heads * v_dim
        assert heads2 == 2 * num_heads and grad_v.shape == (tokens, num_heads, v_dim)
        assert not (ctx.paired and ctx.key_offset)
        grad_f8 = torch.empty((tokens, qkv_dim), device=qk.device, dtype=torch.float8_e4m3fn)
        grad_f8_t = torch.empty((qkv_dim, tokens), device=qk.device, dtype=torch.float8_e4m3fn)
        # Feature rows run BLOCK_T bytes; 32 keeps every transposed store on a full 32 B sector.
        block_t = 32
        _qkv_norm_rope_pack_fp8_backward_kernel[(num_heads, triton.cdiv(tokens, block_t))](
            grad_q, grad_k, grad_v, qk, factor1, factor2, grad_f8, grad_f8_t, grad_scale,
            tokens=tokens, num_heads=num_heads, qk_dim=qk_dim, rotary_dim=ctx.rotary_dim,
            v_dim=v_dim, qkv_dim=qkv_dim,
            stride_qkt=qk.stride(0), stride_qkh=qk.stride(1), factor_stride_t=factor1.stride(0),
            stride_grad_q_t=grad_q.stride(0), stride_grad_q_h=grad_q.stride(1),
            stride_grad_k_t=grad_k.stride(0), stride_grad_k_h=grad_k.stride(1),
            stride_grad_v_t=grad_v.stride(0), stride_grad_v_h=grad_v.stride(1),
            PAIRED=ctx.paired, KEY_OFFSET=ctx.key_offset, BLOCK_T=block_t,
            BLOCK_QK=triton.next_power_of_2(qk_dim), BLOCK_V=triton.next_power_of_2(v_dim),
            num_warps=4 if qk_dim <= 64 else 8,
        )
        grad_weight = _scaled_mm_bf16(grad_f8_t, x_f8_t.T, grad_scale, x_scale, False)
        grad_input = _scaled_mm_bf16(grad_f8, weight_f8_t.T, grad_scale, scaled_weight_scale, False)
        # .to(qkv_scale.dtype): qkv_scale is an fp32 scalar while the weights are bf16, and autograd
        # checks the outgoing gradient's dtype.
        grad_qkv_scale = (grad_weight * torch.cat((qk_weight, v_weight))).sum().to(qkv_scale.dtype)
        grad_qk_weight, grad_v_weight = grad_weight.split((2 * num_heads * qk_dim, num_heads * v_dim))
        # One slot per forward argument; only x, the two weights and qkv_scale carry gradients.
        return (grad_input.view(ctx.input_shape), grad_qk_weight * qkv_scale, grad_v_weight * qkv_scale,
                None, None, None, grad_qkv_scale, None, None, None, None, None, None, None, None, None, None)


PackedFP8QKV = PackedFP8QKVFunction.apply

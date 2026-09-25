"""Attention stack for the ANVIL run: fused QK-norm/RoPE/pad and packed FP8 QKV.

Q, K and V come from ONE packed fp8 weight bank in a single _scaled_mm, and the QK rms-norm
(@Grad62304977), RoPE (half-truncate, @YouJiacheng) and head padding that follow are one Triton kernel,
in that order, so the QK stream is never round-tripped through HBM between them.

NUMERICS: the norm/RoPE math runs in fp32 with the literal eps 1.1920928955078125e-7 and the QK gradient is
materialized in bf16 before the fp8 cast, matching the aten chain these replace.
"""

import inspect

import torch
import triton
import triton.language as tl


# `grad_scale_ptr` is a 1-element fp32 slice whose 16 B alignment varies by layer; suppressing that
# specialization key makes every JIT cache key a pure function of the shape, so no layer forces a compile on
# the clock.  Safe: no INTEGER arg reaches this kernel and no other pointer varies.
_jit_kw = next((_k for _k in ("do_not_specialize_on_alignment", "do_not_specialize")
                if _k in inspect.signature(triton.jit).parameters), None)
_QKV_BWD_JIT_KW = {_jit_kw: ["grad_scale_ptr"]} if _jit_kw else {}

@triton.jit
def _packed_batched_matrix_amax_partial_kernel(
    first, second, partial_amax, FIRST_ELEMENTS: tl.constexpr, SECOND_ELEMENTS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr, SECOND_STRIDE_B: tl.constexpr, NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial amax of the row-pack per (batch, part); quantize_dual_layout_packed_batched reduces."""
    batch = tl.program_id(0)
    part = tl.program_id(1)
    total_elements = FIRST_ELEMENTS + SECOND_ELEMENTS
    offsets = tl.arange(0, BLOCK_SIZE)
    local_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in tl.range(part * BLOCK_SIZE, total_elements, NUM_PARTS * BLOCK_SIZE):
        indices = start + offsets
        valid = indices < total_elements
        in_first = indices < FIRST_ELEMENTS
        first_values = tl.load(first + batch * FIRST_STRIDE_B + indices,
                               mask=valid & in_first, other=0.0)
        second_indices = tl.maximum(indices - FIRST_ELEMENTS, 0)
        second_values = tl.load(second + batch * SECOND_STRIDE_B + second_indices,
                                mask=valid & ~in_first, other=0.0)
        values = first_values.to(tl.float32) + second_values.to(tl.float32)
        local_amax = tl.maximum(local_amax, tl.abs(values))
    tl.store(partial_amax + batch * NUM_PARTS + part, tl.max(local_amax, axis=0))

@triton.jit
def _quantize_dual_layout_packed_batched_kernel(
    first, second, row, transposed, scale_ptr, M, N, FIRST_ROWS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr, SECOND_STRIDE_B: tl.constexpr, ROW_STRIDE_B: tl.constexpr,
    TRANSPOSED_STRIDE_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Quantize that same row-pack in place of a cat: ONE read of each source, TWO fp8 writes (row-major and
    transposed, via a register-tile tl.trans), so the packed QKV GEMM's two operand layouts cost one pass."""
    batch = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_first = offs_m[:, None] < FIRST_ROWS
    first_offsets = offs_m[:, None] * N + offs_n[None, :]
    second_rows = tl.maximum(offs_m - FIRST_ROWS, 0)
    second_offsets = second_rows[:, None] * N + offs_n[None, :]
    source_ptrs = tl.where(in_first, first + batch * FIRST_STRIDE_B + first_offsets,
                           second + batch * SECOND_STRIDE_B + second_offsets)
    values = tl.load(source_ptrs, mask=valid, other=0.0).to(tl.float32)
    scaled = values / tl.load(scale_ptr + batch)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :],
             quantized, mask=valid)
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(transposed + batch * TRANSPOSED_STRIDE_B + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(quantized), mask=transposed_mask)

def quantize_dual_layout_packed_batched(first, second, scales, partial_amax, row, transposed):
    """Refresh the packed QKV fp8 weight cache: one launch per batch of layers sharing (B, rows, N), which is
    what the asserts below check.  The scale is EXACT (this step's own amax, not a lagged one) and the +-448
    clamp is explicit, so the fp8 bytes are what a cat + amax + quantize + transpose chain in aten would
    have written."""
    B, first_rows, N = first.shape
    B2, second_rows, N2 = second.shape
    total_rows = first_rows + second_rows
    num_parts = partial_amax.shape[-1]
    assert (B2, N2) == (B, N) and first.device == second.device and first.dtype == second.dtype
    # Row-major sources (a strided dim-0 view of a weight bank is fine, a row view is not).
    assert first.stride(1) == second.stride(1) == N and first.stride(2) == second.stride(2) == 1
    assert scales.shape == (B,) and partial_amax.shape == (B, num_parts)
    assert row.shape == (B, total_rows, N) and transposed.shape == (B, N, total_rows)
    assert row.dtype == transposed.dtype == torch.float8_e4m3fn
    assert row.is_contiguous() and transposed.is_contiguous()
    _packed_batched_matrix_amax_partial_kernel[(B, num_parts)](
        first, second, partial_amax, FIRST_ELEMENTS=first_rows * N,
        SECOND_ELEMENTS=second_rows * N, FIRST_STRIDE_B=first.stride(0),
        SECOND_STRIDE_B=second.stride(0), NUM_PARTS=num_parts, BLOCK_SIZE=2048,
        num_stages=1, num_warps=8)
    # Same exact-scale arithmetic as quantize_mlp_weights_dual in triton_kernels.
    torch.mul(partial_amax.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=scales)
    block_m, block_n = 64, 128
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(N, block_n), B)
    _quantize_dual_layout_packed_batched_kernel[grid](
        first, second, row, transposed, scales, total_rows, N, FIRST_ROWS=first_rows,
        FIRST_STRIDE_B=first.stride(0), SECOND_STRIDE_B=second.stride(0),
        ROW_STRIDE_B=row.stride(0), TRANSPOSED_STRIDE_B=transposed.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, num_stages=2, num_warps=8)
    return row, transposed

@triton.jit
def _qk_norm_rope_pad_forward_kernel(
    qk, factor1, factor2, out_q, out_k, rows: tl.constexpr, num_heads: tl.constexpr,
    heads2: tl.constexpr, qk_dim: tl.constexpr, rotary_dim: tl.constexpr, padded_dim: tl.constexpr,
    stride_qkt: tl.constexpr, stride_qkh: tl.constexpr, factor_stride_t: tl.constexpr,
    stride_out_t: tl.constexpr, PAIRED: tl.constexpr, KEY_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """QK rms-norm + RoPE + head-dim zero padding in one pass over the packed QK stream, split
    into the separate Q and K tensors FA3 wants.  Norm and rotary in fp32 with the literal eps
    below; the pad columns are exact zeros (inert where padded_dim == qk_dim)."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < rows) & (offs_d[None, :] < qk_dim)

    token = offs_m // heads2
    input_head = offs_m % heads2
    x_ptrs = (qk + token[:, None] * stride_qkt + input_head[:, None] * stride_qkh
              + offs_d[None, :])
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # RoPE swaps adjacent lanes; the tile is resident, so swap in registers not HBM.
    x_flip = tl.reshape(tl.flip(tl.reshape(x, (BLOCK_M, BLOCK_D // 2, 2)), dim=2),
                        (BLOCK_M, BLOCK_D))
    x = x.to(tl.float32)
    x_flip = x_flip.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7)
    if PAIRED:
        logical_head = input_head % num_heads
        head_parity = logical_head % 2
        output_token = 2 * token + logical_head // (num_heads // 2)
        output_head = logical_head % (num_heads // 2)
        factor_offset = head_parity * qk_dim
    else:
        logical_head = input_head % num_heads
        output_token = token
        output_head = logical_head
        factor_offset = 0
    f1 = tl.load(factor1 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_d[None, :]
                 if PAIRED else factor1 + token[:, None] * factor_stride_t + offs_d[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    f2 = tl.load(factor2 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_d[None, :]
                 if PAIRED else factor2 + token[:, None] * factor_stride_t + offs_d[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    normalized = x * rstd[:, None]
    normalized_flip = x_flip * rstd[:, None]
    y = f1 * normalized + f2 * normalized_flip

    if KEY_OFFSET:
        shift_row = (input_head >= num_heads) & (token > 0)
        previous_token = tl.maximum(token - 1, 0)
        x_previous = tl.load(qk + previous_token[:, None] * stride_qkt
                             + input_head[:, None] * stride_qkh + offs_d[None, :],
                             mask=mask & shift_row[:, None], other=0.0).to(tl.float32)
        previous_rstd = tl.rsqrt(tl.sum(x_previous * x_previous, axis=1) / qk_dim
                                 + 1.1920928955078125e-7)
        shift = shift_row[:, None] & (offs_d[None, :] >= rotary_dim)
        y = tl.where(shift, x_previous * previous_rstd[:, None], y)

    output_ptrs = (output_token[:, None] * stride_out_t
                   + output_head[:, None] * padded_dim + offs_d[None, :])
    tl.store(out_q + output_ptrs, y, mask=mask & (input_head[:, None] < num_heads))
    tl.store(out_k + output_ptrs, y, mask=mask & (input_head[:, None] >= num_heads))
    padding = padded_dim - qk_dim
    padding_ptrs = output_ptrs + qk_dim
    padding_mask = (offs_m[:, None] < rows) & (offs_d[None, :] < padding)
    tl.store(out_q + padding_ptrs, 0.0, mask=padding_mask & (input_head[:, None] < num_heads))
    tl.store(out_k + padding_ptrs, 0.0, mask=padding_mask & (input_head[:, None] >= num_heads))

# Both call sites run under torch.no_grad(), so this needs no autograd wrapper; training's
# QK-norm/RoPE backward is fused into the packed QKV backward below.
def QKNormRoPEPad(qk, factor1, factor2, num_heads, padded_dim, paired, key_offset):
    """Launch the kernel above: (out_q, out_k), each [tokens, heads, padded_dim]."""
    tokens, heads2, qk_dim = qk.shape
    rows = tokens * heads2
    assert heads2 == 2 * num_heads and qk_dim <= padded_dim and not (paired and key_offset)
    assert factor1.shape == factor2.shape == (tokens, qk_dim * (2 if paired else 1))
    # Shifted keys require an extra token-row load and favor smaller row tiles.
    block_m = 4 if key_offset else 8
    # Q and K need independent allocations: views into one buffer make functionalization clone
    # and copy back around this mutating Triton call to preserve alias semantics.
    output_shape = (tokens * (2 if paired else 1),
                    num_heads // 2 if paired else num_heads, padded_dim)
    out_q = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    out_k = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    _qk_norm_rope_pad_forward_kernel[(triton.cdiv(rows, block_m),)](
        qk, factor1, factor2, out_q, out_k, rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=qk_dim // 2, padded_dim=padded_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_out_t=out_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_M=block_m, BLOCK_D=triton.next_power_of_2(qk_dim), num_warps=2)
    return out_q, out_k


# QK-norm/RoPE backward, the V gradient and the dual-layout e4m3 quantize of both in ONE kernel, once per
# attention layer.  The grid is head-fastest so factor1/factor2 are read ~once instead of once per head
# (~63 MB/call).  The axis=1 reductions stay BIT-IDENTICAL under any BLOCK_T: dim1 is BLOCK_QK, so BLOCK_T
# grows only the dim0 terms and `tl.sum(..., axis=1)` keeps its per-thread chunk and fp32 summation order.
@triton.jit(**_QKV_BWD_JIT_KW)
def _qkv_norm_rope_pack_fp8_backward_kernel(
    grad_q, grad_k, grad_v, qk, factor1, factor2, grad_row, grad_transposed, grad_scale_ptr,
    tokens: tl.constexpr, num_heads: tl.constexpr, qk_dim: tl.constexpr, rotary_dim: tl.constexpr,
    head_dim: tl.constexpr, qkv_dim: tl.constexpr, stride_qkt: tl.constexpr,
    stride_qkh: tl.constexpr, factor_stride_t: tl.constexpr, stride_grad_q_t: tl.constexpr,
    stride_grad_q_h: tl.constexpr, stride_grad_k_t: tl.constexpr, stride_grad_k_h: tl.constexpr,
    stride_grad_v_t: tl.constexpr, stride_grad_v_h: tl.constexpr, PAIRED: tl.constexpr,
    KEY_OFFSET: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_QK: tl.constexpr,
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

    factor_ptrs = (token[:, None] * factor_stride_t + factor_offset + offs_d[None, :]
                   if PAIRED else token[:, None] * factor_stride_t + offs_d[None, :])
    factor_flip_ptrs = (token[:, None] * factor_stride_t + factor_offset + offs_flip[None, :]
                        if PAIRED else token[:, None] * factor_stride_t + offs_flip[None, :])
    f1 = tl.load(factor1 + factor_ptrs, mask=qk_mask, other=0.0).to(tl.float32)
    f2_flip = tl.load(factor2 + factor_flip_ptrs, mask=qk_mask, other=0.0).to(tl.float32)
    scale = tl.load(grad_scale_ptr)

    for qk_kind in tl.static_range(2):
        input_head = logical_head + qk_kind * num_heads
        x = tl.load(qk + token[:, None] * stride_qkt + input_head * stride_qkh + offs_d[None, :],
                    mask=qk_mask, other=0.0).to(tl.float32)
        if qk_kind == 0:
            grad_ptr = (grad_q + output_token[:, None] * stride_grad_q_t
                        + output_head * stride_grad_q_h + offs_d[None, :])
        else:
            grad_ptr = (grad_k + output_token[:, None] * stride_grad_k_t
                        + output_head * stride_grad_k_h + offs_d[None, :])
        grad = tl.load(grad_ptr, mask=qk_mask, other=0.0)
        grad_flip = tl.reshape(tl.flip(tl.reshape(grad, (BLOCK_T, BLOCK_QK // 2, 2)), dim=2),
                               (BLOCK_T, BLOCK_QK))
        grad = grad.to(tl.float32)
        grad_flip = grad_flip.to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7)
        normalized = x * rstd[:, None]
        grad_normalized = f1 * grad + f2_flip * grad_flip
        if KEY_OFFSET and qk_kind == 1:
            next_token = tl.minimum(token + 1, tokens - 1)
            grad_next = tl.load(grad_k + next_token[:, None] * stride_grad_k_t
                                + output_head * stride_grad_k_h + offs_d[None, :],
                                mask=qk_mask & (token[:, None] < tokens - 1), other=0.0).to(tl.float32)
            stationary_grad = tl.where(token[:, None] == 0, grad + grad_next, grad_next)
            grad_normalized = tl.where(offs_d[None, :] >= rotary_dim, stationary_grad,
                                       grad_normalized)
        correction = tl.sum(grad_normalized * normalized, axis=1) / qk_dim
        dx = rstd[:, None] * (grad_normalized - normalized * correction[:, None])
        q = (dx.to(tl.bfloat16).to(tl.float32) / scale).to(tl.float8e4nv)
        feature = input_head * qk_dim + offs_d
        tl.store(grad_row + token[:, None] * qkv_dim + feature[None, :], q, mask=qk_mask)
        tl.store(grad_transposed + feature[:, None] * tokens + token[None, :], tl.trans(q),
                 mask=(offs_d[:, None] < qk_dim) & token_mask[None, :])

    offs_v = tl.arange(0, BLOCK_V)
    v_mask = token_mask[:, None] & (offs_v[None, :] < head_dim)
    grad_v_value = tl.load(grad_v + token[:, None] * stride_grad_v_t
                           + logical_head * stride_grad_v_h + offs_v[None, :],
                           mask=v_mask, other=0.0).to(tl.float32)
    qv = (grad_v_value / scale).to(tl.float8e4nv)
    v_feature = 2 * num_heads * qk_dim + logical_head * head_dim + offs_v
    tl.store(grad_row + token[:, None] * qkv_dim + v_feature[None, :], qv, mask=v_mask)
    tl.store(grad_transposed + v_feature[:, None] * tokens + token[None, :], tl.trans(qv),
             mask=(offs_v[:, None] < head_dim) & token_mask[None, :])

# The packed fp8 QKV projection with the activation quantize HOISTED OUT: layers taking the same
# `norm(cache[...])` as their attention input pass in x_f8 / x_f8_t, quantized once and shared, while each call
# still returns its own grad_input.  Both are fp8 leaves with no grad edge either way, so autograd is untouched.
class PackedFP8QKVPreFunction(torch.autograd.Function):
    @staticmethod
    def mm8(a, b, scale_a, scale_b, fast_accum):
        """The one per-tensor-scaled fp8 GEMM spelling this class uses, forward and backward."""
        return torch._scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a,
                                scale_b=scale_b, use_fast_accum=fast_accum)

    @staticmethod
    def forward(ctx, x, qk_weight, v_weight, weight_f8, weight_f8_t, weight_scale, qkv_scale,
                x_scale, grad_scale, factor1, factor2, num_heads, paired, key_offset,
                x_f8, x_f8_t):
        """The fp8 QKV GEMM off the cached packed weight, the fused norm/RoPE/pad, and the save-set."""
        qk_dim = qk_weight.shape[0] // (2 * num_heads)
        head_dim = v_weight.shape[0] // num_heads
        attn_qk_dim = qk_dim if qk_dim <= 64 else head_dim
        qk_features = 2 * num_heads * qk_dim
        scaled_weight_scale = weight_scale * qkv_scale
        mm8 = PackedFP8QKVPreFunction.mm8
        if paired:
            # A packed QKV result leaves V a strided suffix of every token row, which paired attention
            # would have to materialize; two column-sliced GEMMs read X once more but make V dense at birth.
            qk_flat = mm8(x_f8, weight_f8[:qk_features].T, x_scale, scaled_weight_scale, True)
            v_flat = mm8(x_f8, weight_f8[qk_features:].T, x_scale, scaled_weight_scale, True)
        else:
            qkv = mm8(x_f8, weight_f8.T, x_scale, scaled_weight_scale, True)
            qk_flat, v_flat = qkv[:, :qk_features], qkv[:, qk_features:]
        qk = qk_flat.view(-1, 2 * num_heads, qk_dim)
        v = v_flat.view(-1, num_heads, head_dim)
        q, k = QKNormRoPEPad(qk, factor1, factor2, num_heads, attn_qk_dim, paired, key_offset)
        ctx.save_for_backward(qk_weight, v_weight, x_f8_t, weight_f8_t, scaled_weight_scale,
                              qkv_scale, x_scale, grad_scale, qk, factor1, factor2)
        ctx.input_shape = x.shape
        ctx.num_heads, ctx.head_dim = num_heads, head_dim
        ctx.paired, ctx.key_offset = paired, key_offset
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        """The kernel above, then the two grad GEMMs, the qkv_scale reduction and the bank splits."""
        (qk_weight, v_weight, x_f8_t, weight_f8_t, scaled_weight_scale, qkv_scale,
         x_scale, grad_scale, qk, factor1, factor2) = ctx.saved_tensors
        num_heads, head_dim = ctx.num_heads, ctx.head_dim
        grad_v = grad_v.reshape(-1, num_heads, head_dim)
        tokens, heads2, qk_dim = qk.shape
        qkv_dim = 2 * num_heads * qk_dim + num_heads * head_dim
        assert heads2 == 2 * num_heads and grad_v.shape == (tokens, num_heads, head_dim)
        assert not (ctx.paired and ctx.key_offset)
        grad_f8 = torch.empty((tokens, qkv_dim), device=qk.device, dtype=torch.float8_e4m3fn)
        grad_f8_t = torch.empty((qkv_dim, tokens), device=qk.device, dtype=torch.float8_e4m3fn)
        block_t = 32   # feature rows run BLOCK_T bytes; 32 keeps every transposed store on a
                       # full 32 B sector, halving the write transactions on a 75 MB/call output.
        _qkv_norm_rope_pack_fp8_backward_kernel[(num_heads, triton.cdiv(tokens, block_t))](
            grad_q, grad_k, grad_v, qk, factor1, factor2, grad_f8, grad_f8_t, grad_scale,
            tokens=tokens, num_heads=num_heads, qk_dim=qk_dim, rotary_dim=qk_dim // 2,
            head_dim=head_dim, qkv_dim=qkv_dim,
            stride_qkt=qk.stride(0), stride_qkh=qk.stride(1), factor_stride_t=factor1.stride(0),
            stride_grad_q_t=grad_q.stride(0), stride_grad_q_h=grad_q.stride(1),
            stride_grad_k_t=grad_k.stride(0), stride_grad_k_h=grad_k.stride(1),
            stride_grad_v_t=grad_v.stride(0), stride_grad_v_h=grad_v.stride(1),
            PAIRED=ctx.paired, KEY_OFFSET=ctx.key_offset, BLOCK_T=block_t,
            BLOCK_QK=triton.next_power_of_2(qk_dim), BLOCK_V=triton.next_power_of_2(head_dim),
            num_warps=4 if qk_dim <= 64 else 8)
        mm8 = PackedFP8QKVPreFunction.mm8
        grad_weight = mm8(grad_f8_t, x_f8_t.T, grad_scale, x_scale, False)
        grad_input = mm8(grad_f8, weight_f8_t.T, grad_scale, scaled_weight_scale, False)
        # .to(qkv_scale.dtype): qkv_scale is fp32 in the `scalars` bank while the packed weights
        # are bf16, so autograd's outgoing-grad dtype check would reject the raw sum.
        grad_qkv_scale = (grad_weight * torch.cat((qk_weight, v_weight))).sum().to(qkv_scale.dtype)
        grad_qk_weight, grad_v_weight = grad_weight.split((2 * num_heads * qk_dim,
                                                           num_heads * head_dim))
        # 16 slots: one per forward argument; only x, the two weights and qkv_scale carry grads.
        return (grad_input.view(ctx.input_shape), grad_qk_weight * qkv_scale,
                grad_v_weight * qkv_scale, None, None, None, grad_qkv_scale, None, None, None,
                None, None, None, None, None, None)


PackedFP8QKVPre = PackedFP8QKVPreFunction.apply

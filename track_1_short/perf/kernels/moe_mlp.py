"""Mixture of experts: the relu^2 MLP of E experts with every GEMM in FP8 -- the grouped twin of mlp.py's fused
linear_relu_square, with the same fp8 formats and scales.

Layout. moe_dispatch sorts the token rows by expert and pads every expert's segment to a multiple of
BLOCK_M = 128 rows, so each 128-row tile belongs to one expert: tile_expert[i]. Tiles past the last
segment run on all-zero rows (expert clamped to E-1: no branch in the persistent loops, < E*128 rows of
waste) and their output rows are never read. The segment ends are `offs`. Padding rows are zero in the
forward and get zero gradient.

Kernels, all Triton except the optional Gluon "up" GEMM (moe_g3.py), one launch each (the point of this file: no ATen
glue between them):
  dispatch       2 kernels: per-block expert counts, then every row's padded slot (a stable counting
                 sort: no argsort), the padding slots, the group ends and tile_expert
  gather x       the token rows gathered into the padded layout, written straight to e4m3, row-major and
                 transposed, from one read
  up    post = relu(x @ W1[e].T)^2      the record's fused kernel plus a per-tile expert that offsets the
                                          stacked [E*h, D] weight rows; post in both layouts + its amax
  down  y    = post @ W2[e]             grouped over M; the output is gated per token, out = y * s
  gather g       the incoming gradient: gathered, times the token's gate s, e5m2 in both layouts, and the
                 gate's gradient <grad, y> per row, from one read of grad and y
  dW2[e]     = post_t[:, seg] @ g[seg]  grouped over K; an empty expert gets zeros from the kernel itself
  dpre       = 2 (g @ W2[e].T) relu(pre)  the up kernel's backward epilogue: e5m2, both layouts + amax
  dW1[e]     = dpre_t[:, seg] @ x[seg]  grouped over K, as dW2
  dx         = dpre @ W1[e]             grouped over M
Routing is top-1: each token goes to the expert of largest router probability (RoutedFP8MoE asserts topk == 1).
Formats and scales as the record: x e4m3 static 2^-4, W1/W2 e4m3 one scale per expert, post e4m3 delayed,
g e5m2 static 2^-6, dpre e5m2 delayed (one per layer). Every kernel reads its scales through pointers.
fp8 accumulation is Hopper's (no fp32 promotion), as in the record's own Triton kernels: promoting every BLOCK_K would
change dW by only ~0.7e-3 relative (against 6e-2 from the fp8 operands themselves) and costs time.
"""
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
# G3 (Gluon, warp-specialized) for the grouped "up" GEMM: checked bit for bit against the Triton grouped_lrs_kernel at
# start-up (prime_grouped_stage_cache(g3_k=...)); measured ~1.2-1.3x faster.  If moe_g3 or Gluon are missing, or the check
# fails, the Triton kernel runs and MOE_G3_FALLBACK_REASON says why.
try:
    from track_1_short.perf.kernels.moe_g3 import (
        g3_eligible as _g3_eligible,
        grouped_lrs_g3 as _grouped_lrs_g3,
        prime_g3 as _prime_g3,
    )
    MOE_G3_FALLBACK_REASON = None
except Exception as _g3_err:  # Gluon/triton version drift
    _grouped_lrs_g3 = None
    MOE_G3_FALLBACK_REASON = f"moe_g3 is not importable ({type(_g3_err).__name__}: {str(_g3_err)[:200]})"
# MOE_G3_ACTIVE (MOE_G3 is the same flag, the name the traced code reads): the "up" GEMMs dispatch to the Gluon kernels.
# True from import if moe_g3 imported; prime_grouped_stage_cache(..., g3_k=...) switches both off if its check fails.
MOE_G3 = MOE_G3_ACTIVE = _grouped_lrs_g3 is not None
E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
FP8 = (E4M3, E5M2)
# The gradients (g, dpre) are e5m2 and the fp8 accumulators are never promoted to fp32 mid-GEMM: the record's choices.  The
# e4m3-gradient and fp32-promotion variants that used to sit behind these three flags are removed; the flags stay as False
# constants for callers that assert on them.
GRAD_E4M3 = DW_PROMOTE = DX_PROMOTE = False


@triton.jit
def grouped_lrs_kernel(a_desc, b_desc, tile_expert_ptr, dequant_ptr, dq2_ptr, M, N, K,
                       f8_desc, t_desc, aux_desc, post_scale_ptr, amax_ptr, dpre_scale_ptr,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       NUM_SMS: tl.constexpr, FORWARD: tl.constexpr):
    """Forward: post = relu(a @ b[e].T * dq[e])^2 -> e4m3 (row and transposed), amax.
    Backward: dpre = 2 * (a @ b[e].T * dq[e]) * sqrt(post) -> e5m2 (row and transposed), amax.
    b is the experts' [E*N, K] weights stacked by rows; tile row-block pid_m uses expert tile_expert[pid_m].
    dq[e] = dequant[e] * dq2[0]: the expert's weight scale times the activation's (x or g)."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    dq2 = tl.load(dq2_ptr)
    if FORWARD:
        inv_post_scale = 1.0 / tl.load(post_scale_ptr)
        post_scale = 1.0
        inv_dpre_scale = 1.0
    else:
        inv_post_scale = 1.0
        post_scale = tl.load(post_scale_ptr)
        inv_dpre_scale = 1.0 / tl.load(dpre_scale_ptr)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        expert = tl.load(tile_expert_ptr + pid_m)
        dq = tl.load(dequant_ptr + expert) * dq2   # loaded before the K loop: its latency overlaps the GEMM
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_wn = expert * N + offs_bn          # this expert's rows in the stacked weights
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_wn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        accumulator *= dq

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        if FORWARD:
            c0 = acc0.to(tl.bfloat16)               # pre, bf16-rounded as the record
            c0 = tl.maximum(c0, 0)
            c0 = c0 * c0
            c1 = acc1.to(tl.bfloat16)
            c1 = tl.maximum(c1, 0)
            c1 = c1 * c1
            c0f = c0.to(tl.float32)
            c1f = c1.to(tl.float32)
            q0 = tl.minimum(c0f * inv_post_scale, 448.0).to(tl.float8e4nv)
            q1 = tl.minimum(c1f * inv_post_scale, 448.0).to(tl.float8e4nv)
            tile_max = tl.max(tl.max(tl.maximum(c0f, c1f), axis=1), axis=0)
        else:
            aux_tile = aux_desc.load([offs_am, offs_bn])     # e4m3 post, re-split like the accumulator
            aux_tile = tl.reshape(aux_tile, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            aux_tile = tl.permute(aux_tile, (0, 2, 1))
            a0_raw, a1_raw = tl.split(aux_tile)
            c0f = (2.0 * acc0 * tl.sqrt(a0_raw.to(tl.float32) * post_scale)).to(tl.bfloat16).to(tl.float32)
            c1f = (2.0 * acc1 * tl.sqrt(a1_raw.to(tl.float32) * post_scale)).to(tl.bfloat16).to(tl.float32)
            q0 = tl.maximum(tl.minimum(c0f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
            q1 = tl.maximum(tl.minimum(c1f * inv_dpre_scale, 57344.0), -57344.0).to(tl.float8e5)
            tile_max = tl.max(tl.max(tl.maximum(tl.abs(c0f), tl.abs(c1f)), axis=1), axis=0)
        f8_desc.store([offs_am, offs_bn], q0)
        f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
        t_desc.store([offs_bn, offs_am], tl.trans(q0))
        t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
        tl.atomic_max(amax_ptr, tile_max)


# num_stages per variant, resolved eagerly by prime_grouped_stage_cache() before torch.compile (the
# retry below cannot run under compile), exactly like mlp.py's stage cache.
_glrs_stage_cache = {}


def _launch_with_stage_retry(key, default_stages, launch):
    """Launch with the cached num_stages; eager calls step it down on a shared-memory overflow and cache it."""
    num_stages = _glrs_stage_cache.get(key, default_stages)
    if torch.compiler.is_compiling():
        assert key in _glrs_stage_cache, "call prime_grouped_stage_cache() before torch.compile"
        launch(num_stages)
        return
    while True:
        try:
            launch(num_stages)
            break
        except OutOfResources:
            if num_stages <= 1:
                raise
            num_stages -= 1
    _glrs_stage_cache[key] = num_stages


def _check_scalars(*ts):
    for t in ts:
        assert t.dtype == torch.float32 and t.numel() == 1, (t.dtype, t.shape)


def _grouped_lrs_impl(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, *, forward, post_scale,
                               amax, aux=None, dpre_scale=None):
    """Forward (a = x e4m3): (post e4m3 [M, N], its transpose). Backward (a = g e5m2, aux = the e4m3 post):
    (dpre e5m2 [M, N], its transpose). See the kernel. With MOE_G3 on, the Gluon kernel of moe_g3.py runs where its
    shapes fit (outputs and amax checked bit-identical to the Triton kernel at start-up); the Triton kernel otherwise."""
    if MOE_G3 and _g3_eligible(a_f8, b_f8, n_per_expert, forward, aux):    # it asserts the same contract itself
        return _grouped_lrs_g3(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, forward=forward,
                               post_scale=post_scale, amax=amax, aux=aux, dpre_scale=dpre_scale)
    return _grouped_lrs_triton(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, forward=forward,
                               post_scale=post_scale, amax=amax, aux=aux, dpre_scale=dpre_scale)


def _grouped_lrs_triton(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, *, forward, post_scale,
                        amax, aux=None, dpre_scale=None):
    """The Triton path of _grouped_lrs_impl."""
    M, K = a_f8.shape
    EN, K2 = b_f8.shape
    N = n_per_expert
    assert K == K2 and EN % N == 0, (a_f8.shape, b_f8.shape, N)
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, "rows must be 128-aligned and the expert width a multiple of 256"
    assert a_f8.dtype == (E4M3 if forward else E5M2) and b_f8.dtype == E4M3
    assert a_f8.is_contiguous() and b_f8.is_contiguous() and K % 16 == 0
    assert forward == (aux is None) and (forward or dpre_scale is not None)
    if not forward:
        assert aux.dtype == E4M3 and aux.shape == (M, N) and aux.is_contiguous(), "aux must be the e4m3 post [M, N]"
    assert tile_expert.dtype == torch.int32 and tile_expert.numel() == M // BLOCK_M
    assert dequant.dtype == torch.float32 and dequant.numel() == EN // N
    _check_scalars(dq2, post_scale, amax, *(() if forward else (dpre_scale,)))
    out_dtype = E4M3 if forward else E5M2
    f8 = torch.empty((M, N), device=a_f8.device, dtype=out_dtype)
    f8_t = torch.empty((N, M), device=a_f8.device, dtype=out_dtype)
    a_desc = TensorDescriptor.from_tensor(a_f8, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b_f8, [BLOCK_N, BLOCK_K])
    f8_desc = TensorDescriptor.from_tensor(f8, [BLOCK_M, BLOCK_N // 2])
    t_desc = TensorDescriptor.from_tensor(f8_t, [BLOCK_N // 2, BLOCK_M])
    aux_desc = f8_desc if forward else TensorDescriptor.from_tensor(aux, [BLOCK_M, BLOCK_N])
    num_sms = torch.cuda.get_device_properties(a_f8.device).multi_processor_count
    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    def launch(stages):
        grouped_lrs_kernel[grid](
            a_desc, b_desc, tile_expert, dequant, dq2, M, N, K,
            f8_desc, t_desc, aux_desc, post_scale, amax, post_scale if forward else dpre_scale,
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K, NUM_SMS=num_sms, FORWARD=forward,
            num_stages=stages, num_warps=8)

    _launch_with_stage_retry(("lrs", forward), 3, launch)
    return f8, f8_t


# -----------------------------------------------------------------------------
# Plain GEMMs grouped over M (down projection, dx): c = a @ b[e].T * dq[e], bf16 out.

@triton.jit
def grouped_mm_kernel(a_desc, b_desc, c_desc, tile_expert_ptr, dequant_ptr, dq2_ptr, M, N, K,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      NUM_SMS: tl.constexpr):
    """c[rows] = a[rows] @ b[e].T * dequant[e] * dq2[0] for the 128-row tiles of expert e = tile_expert[tile];
    b is the experts' [E*N, K] K-major weights stacked by rows."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    dq2 = tl.load(dq2_ptr)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        expert = tl.load(tile_expert_ptr + pid_m)
        dq = tl.load(dequant_ptr + expert) * dq2
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_wn = expert * N + offs_bn
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_wn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        accumulator *= dq
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([offs_am, offs_bn], acc0.to(tl.bfloat16))
        c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], acc1.to(tl.bfloat16))


MM_BLOCK_N = 256


def _grouped_mm_impl(a_f8, b_rows, tile_expert, dequant, dq2, n_per_expert):
    """[M, N] bf16: row tile i of a_f8 [M, K] (e4m3 or e5m2) times expert tile_expert[i]'s e4m3 b_rows block
    [N, K], scaled by dequant[e] * dq2. Rows past the last segment are computed from zero rows, never read."""
    M, K = a_f8.shape
    EN, K2 = b_rows.shape
    N = n_per_expert
    assert K == K2 and EN % N == 0 and M % BLOCK_M == 0 and N % MM_BLOCK_N == 0 and K % BLOCK_K == 0, (a_f8.shape, b_rows.shape)
    assert a_f8.dtype in FP8 and b_rows.dtype == E4M3 and a_f8.is_contiguous() and b_rows.is_contiguous()
    assert tile_expert.dtype == torch.int32 and tile_expert.numel() == M // BLOCK_M
    assert dequant.dtype == torch.float32 and dequant.numel() == EN // N
    _check_scalars(dq2)
    out = torch.empty((M, N), device=a_f8.device, dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a_f8, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b_rows, [MM_BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(out, [BLOCK_M, MM_BLOCK_N // 2])
    num_sms = torch.cuda.get_device_properties(a_f8.device).multi_processor_count
    grid = (min(num_sms, (M // BLOCK_M) * (N // MM_BLOCK_N)),)

    def launch(stages):
        grouped_mm_kernel[grid](a_desc, b_desc, c_desc, tile_expert, dequant, dq2, M, N, K,
                                BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=MM_BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
                                NUM_SMS=num_sms, num_stages=stages, num_warps=8)

    _launch_with_stage_retry(("mm",), 4, launch)
    return out


# -----------------------------------------------------------------------------
# Weight gradients: dW[e] = A[:, seg_e] @ B[:, seg_e].T, grouped over K (the token axis).

@triton.jit
def grouped_dw_kernel(a_desc, b_desc, c_desc, offs_ptr, scale_a_ptr, scale_b_ptr, acc_flag_ptr, H, N,
                      E: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr, NUM_SMS: tl.constexpr):
    """c[e*H + h, n] = sum over expert e's 128-aligned row segment of a[h, k] b[n, k], times both scales.
    a [H, M] and b [N, M] are the transposed fp8 activations (K-major, as fp8 wgmma wants both operands);
    an expert with an empty segment runs no K iteration and stores zeros. c += instead of c = when the device
    flag acc_flag[0] is nonzero (a gradient summed over steps)."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(H, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_expert = num_pid_m * num_pid_n
    scale = tl.load(scale_a_ptr) * tl.load(scale_b_ptr)
    add_old = tl.load(acc_flag_ptr) != 0
    for tile_id in tl.range(start_pid, E * tiles_per_expert, NUM_SMS):
        expert = tile_id // tiles_per_expert
        r = tile_id % tiles_per_expert
        pid_m = r // num_pid_n
        pid_n = r % num_pid_n
        seg_end = tl.load(offs_ptr + expert)
        seg_start = tl.load(offs_ptr + expert - 1, mask=expert > 0, other=0)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(seg_start, seg_end, BLOCK_SIZE_K):
            a = a_desc.load([offs_am, k])
            b = b_desc.load([offs_bn, k])
            accumulator = tl.dot(a, b.T, accumulator)
        accumulator *= scale
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        offs_cm = expert * H + offs_am
        if add_old:
            acc0 += c_desc.load([offs_cm, offs_bn]).to(tl.float32)
            acc1 += c_desc.load([offs_cm, offs_bn + BLOCK_SIZE_N // 2]).to(tl.float32)
        c_desc.store([offs_cm, offs_bn], acc0.to(tl.bfloat16))
        c_desc.store([offs_cm, offs_bn + BLOCK_SIZE_N // 2], acc1.to(tl.bfloat16))


DW_BLOCK_M, DW_BLOCK_N = 128, 256


def _grouped_weight_grad_impl(a_t, b_t, offs, scale_a, scale_b, n_experts, out, acc_flag):
    """[E, H, N] bf16: expert e's a_t[:, seg_e] @ b_t[:, seg_e].T * scale_a * scale_b (zeros if seg_e is empty), written
    into `out` ([E*H, N] bf16 contiguous); when acc_flag[0] != 0 (int32 [1] on the device) it is added to out's contents.
    a_t [H, M], b_t [N, M] fp8 (e4m3 or e5m2) contiguous; offs [E] int32, the 128-aligned segment ends."""
    H, M = a_t.shape
    N, M2 = b_t.shape
    E = n_experts
    assert M == M2 and M % BLOCK_K == 0 and H % DW_BLOCK_M == 0 and N % DW_BLOCK_N == 0, (a_t.shape, b_t.shape)
    assert a_t.dtype in FP8 and b_t.dtype in FP8 and a_t.is_contiguous() and b_t.is_contiguous()
    assert offs.dtype == torch.int32 and offs.numel() == E
    assert BLOCK_M % BLOCK_K == 0, "segments are BLOCK_M-aligned: a K step must not cross into the next expert"
    _check_scalars(scale_a, scale_b)
    assert out.shape == (E * H, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert acc_flag.dtype == torch.int32 and acc_flag.numel() == 1
    a_desc = TensorDescriptor.from_tensor(a_t, [DW_BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b_t, [DW_BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(out, [DW_BLOCK_M, DW_BLOCK_N // 2])
    num_sms = torch.cuda.get_device_properties(a_t.device).multi_processor_count
    grid = (min(num_sms, E * (H // DW_BLOCK_M) * (N // DW_BLOCK_N)),)

    def launch(stages):
        grouped_dw_kernel[grid](a_desc, b_desc, c_desc, offs, scale_a, scale_b, acc_flag, H, N,
                                E=E, BLOCK_SIZE_M=DW_BLOCK_M, BLOCK_SIZE_N=DW_BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
                                NUM_SMS=num_sms, num_stages=stages, num_warps=8)

    _launch_with_stage_retry(("dw", E), 4, launch)
    return out.view(E, H, N)


@torch.library.custom_op("nanogpt_moe::dw_into_bank", mutates_args={"bank_grad"})
def dw_into_bank(a_t: torch.Tensor, b_t: torch.Tensor, offs: torch.Tensor, scale_a: torch.Tensor,
                 scale_b: torch.Tensor, bank_grad: torch.Tensor, slot: int, acc_flag: torch.Tensor) -> None:
    """grouped_weight_grad written into bank_grad[slot:slot + E] ([L, H, N] bf16, the expert bank's gradient):
    the backward hands autograd no weight gradient, so nothing stacks the layers' dW into the bank's. When
    acc_flag[0] != 0 the dW is added to what is there (a gradient summed over a slow bank's period)."""
    E = offs.numel()
    L, H, N = bank_grad.shape
    assert 0 <= slot and slot + E <= L and bank_grad.is_contiguous()
    _grouped_weight_grad_impl(a_t, b_t, offs, scale_a, scale_b, E, out=bank_grad[slot:slot + E].view(E * H, N),
                              acc_flag=acc_flag)


@dw_into_bank.register_fake
def _(a_t, b_t, offs, scale_a, scale_b, bank_grad, slot, acc_flag) -> None:
    return None


def prime_grouped_stage_cache(n_experts, topk=1, g3_k=None):
    """Resolve num_stages for every variant on one-tile inputs. Call once, before torch.compile. `topk` must be 1 (the
    only routing the kernels support). `g3_k`: the model width D (the up GEMM's K) -- the Gluon kernels compile per K, so
    priming them here keeps the JIT out of the first training step; their output is checked bit by bit against the Triton
    kernel, and MOE_G3 / MOE_G3_ACTIVE fall back to False if anything differs (the reason goes to MOE_G3_FALLBACK_REASON
    and is printed on rank 0).  With g3_k=None the Gluon kernels are neither primed nor checked."""
    global MOE_G3, MOE_G3_ACTIVE, MOE_G3_FALLBACK_REASON
    assert topk == 1, "the MoE kernels route top-1 only"
    dev = torch.device("cuda")
    if MOE_G3 and g3_k is not None:
        reason = _prime_g3(_grouped_lrs_triton, K=g3_k)
        if reason is not None:
            MOE_G3 = MOE_G3_ACTIVE = False
            MOE_G3_FALLBACK_REASON = reason
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[moe_mlp] G3 (Gluon) disabled, using the Triton kernel: {reason}", flush=True)
    z = lambda dt, *shape: torch.zeros(*shape, device=dev, dtype=dt)
    one = torch.ones(1, device=dev, dtype=torch.float32)
    amax = torch.zeros(1, device=dev, dtype=torch.float32)
    tile_expert = torch.zeros(1, device=dev, dtype=torch.int32)
    grouped_linear_relu_square(z(E4M3, BLOCK_M, BLOCK_K), z(E4M3, BLOCK_N, BLOCK_K), tile_expert, one, one, BLOCK_N,
                               forward=True, post_scale=one, amax=amax)
    grouped_linear_relu_square(z(E5M2, BLOCK_M, BLOCK_K), z(E4M3, BLOCK_N, BLOCK_K), tile_expert, one, one, BLOCK_N,
                               forward=False, post_scale=one, amax=amax, aux=z(E4M3, BLOCK_M, BLOCK_N), dpre_scale=one)
    offs = torch.full((n_experts,), BLOCK_K, device=dev, dtype=torch.int32)
    _grouped_weight_grad_impl(z(E4M3, DW_BLOCK_M, BLOCK_K), z(E5M2, DW_BLOCK_N, BLOCK_K), offs, one, one, n_experts,
                              out=torch.zeros(n_experts * DW_BLOCK_M, DW_BLOCK_N, device=dev, dtype=torch.bfloat16),
                              acc_flag=torch.ones(1, device=dev, dtype=torch.int32))
    grouped_mm(z(E4M3, BLOCK_M, BLOCK_K), z(E4M3, MM_BLOCK_N, BLOCK_K), tile_expert, one, one, MM_BLOCK_N)
    grouped_mm(z(E5M2, BLOCK_M, BLOCK_K), z(E4M3, MM_BLOCK_N, BLOCK_K), tile_expert, one, one, MM_BLOCK_N)
    # the kernels with no stage knob, once eagerly: a compile failure surfaces here, not inside torch.compile
    rows = torch.zeros(3, 256, device=dev, dtype=torch.bfloat16)
    dest, src, *_ = moe_dispatch(torch.zeros(3, device=dev, dtype=torch.int64), n_experts)
    y_pad = torch.zeros(src.numel(), 256, device=dev, dtype=torch.bfloat16)
    gather_quant_dual(rows, src, one)
    grad_gather_quant(rows, y_pad, src, torch.ones(3, device=dev), one)
    dx_combine(y_pad, dest, torch.zeros(3, n_experts, device=dev),
               torch.zeros(n_experts, 256, device=dev, dtype=torch.bfloat16))
    torch.cuda.synchronize(dev)


# -----------------------------------------------------------------------------
# Dispatch: token rows -> expert segments padded to BLOCK_M, all on the device (no host sync), as a
# stable counting sort in two launches.

DISPATCH_BLOCK = 1024


def padded_rows(n_rows: int, n_experts: int) -> int:
    """Static row count of the padded layout: sum_e ceil(c_e / 128) <= ceil(n / 128) + E - 1 tiles."""
    return (n_rows + BLOCK_M - 1) // BLOCK_M * BLOCK_M + (n_experts - 1) * BLOCK_M


@triton.jit
def _dispatch_count_kernel(expert_ptr, block_counts_ptr, n, E_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """block_counts[b, e]: rows of block b routed to expert e."""
    b = tl.program_id(0)
    rows = b * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(expert_ptr + rows, mask=rows < n, other=-1).to(tl.int32)
    onehot = (e[:, None] == tl.arange(0, E_PAD)[None, :]).to(tl.int32)
    tl.store(block_counts_ptr + b * E_PAD + tl.arange(0, E_PAD), tl.sum(onehot, axis=0))


@triton.jit
def _dispatch_place_kernel(expert_ptr, block_counts_ptr, dest_ptr, src_ptr, offs_ptr, counts_ptr, tile_expert_ptr,
                           n, m_pad, num_blocks, E: tl.constexpr, E_PAD: tl.constexpr, NB_PAD: tl.constexpr,
                           BLOCK: tl.constexpr, TILE: tl.constexpr):
    """Program b: the slots of rows [b*BLOCK, (b+1)*BLOCK) (dest, and src at those slots), the padding slots
    among padded rows [b*BLOCK, (b+1)*BLOCK) (src = n), and those rows' tiles' experts. Every program
    rebuilds the per-expert totals and its block's start from block_counts (a few KB)."""
    b = tl.program_id(0)
    cols = tl.arange(0, E_PAD)
    blk = tl.arange(0, NB_PAD)
    bc = tl.load(block_counts_ptr + blk[:, None] * E_PAD + cols[None, :], mask=blk[:, None] < num_blocks, other=0)
    counts = tl.sum(bc, axis=0)                                     # [E_PAD] rows per expert
    before = tl.sum(tl.where(blk[:, None] < b, bc, 0), axis=0)      # rows of each expert in earlier blocks
    padded = (counts + TILE - 1) // TILE * TILE
    ends = tl.cumsum(padded, axis=0)                                # the group ends, offs
    starts = ends - padded
    ends_real = tl.where(cols < E, ends, 2147483647)                # pad columns are never a segment

    # token rows: slot = segment start + rows of the expert before this block + rank inside the block
    rows = b * BLOCK + tl.arange(0, BLOCK)
    valid = rows < n
    e = tl.load(expert_ptr + rows, mask=valid, other=-1).to(tl.int32)
    onehot = (e[:, None] == cols[None, :]).to(tl.int32)
    rank = tl.cumsum(onehot, axis=0) - onehot                       # same-expert rows earlier in the block
    dest = tl.sum(onehot * (rank + (starts + before)[None, :]), axis=1)
    tl.store(dest_ptr + rows, dest, mask=valid)
    tl.store(src_ptr + dest, rows, mask=valid)

    # padded rows no token lands on (a segment's tail, or past the last segment): the zero row n
    p = b * BLOCK + tl.arange(0, BLOCK)
    seg = tl.sum((ends_real[None, :] <= p[:, None]).to(tl.int32), axis=1)   # E past the last segment
    in_seg = seg[:, None] == cols[None, :]
    seg_start = tl.sum(tl.where(in_seg, starts[None, :], 0), axis=1)
    seg_count = tl.sum(tl.where(in_seg, counts[None, :], 0), axis=1)
    is_pad = (seg >= E) | (p - seg_start >= seg_count)
    tl.store(src_ptr + p, tl.zeros_like(p) + n, mask=(p < m_pad) & is_pad)

    # this chunk's tiles: the expert of their first row, clamped past the end
    t = b * (BLOCK // TILE) + tl.arange(0, BLOCK // TILE)
    t_seg = tl.sum((ends_real[None, :] <= (t * TILE)[:, None]).to(tl.int32), axis=1)
    tl.store(tile_expert_ptr + t, tl.minimum(t_seg, E - 1), mask=t * TILE < m_pad)

    if b == 0:
        tl.store(offs_ptr + cols, ends, mask=cols < E)
        tl.store(counts_ptr + cols, counts, mask=cols < E)


def _moe_dispatch_impl(row_expert: torch.Tensor, n_experts: int):
    """For each row its padded-layout row (dest), for each padded row its source row or n (src: n is the
    zero row), the group ends (offs), each 128-row tile's expert (tile_expert; past the end clamped to
    E-1) and the rows per expert (counts); all int32. Stable: rows keep their order inside an expert's
    segment, exactly as a stable argsort by expert."""
    assert row_expert.ndim == 1 and not row_expert.is_floating_point() and row_expert.is_contiguous()
    n = row_expert.size(0)
    E = n_experts
    m_pad = padded_rows(n, E)
    dev = row_expert.device
    E_PAD = max(2, triton.next_power_of_2(E))
    # The place kernel scans a [block, E_PAD] int32 one-hot through shared memory: keep it <= 64 KB
    # (1024 rows up to E_PAD = 16; E = 40 at 1024 rows asked 256 KB, over the H100's 227 KB).
    block = min(DISPATCH_BLOCK, max(BLOCK_M, 16384 // E_PAD))
    num_blocks = triton.cdiv(n, block)
    NB_PAD = max(2, triton.next_power_of_2(num_blocks))
    assert NB_PAD <= 512, "dispatch: too many row blocks for one program's view of block_counts"
    i32 = torch.int32
    block_counts = torch.empty((num_blocks, E_PAD), device=dev, dtype=i32)
    dest = torch.empty(n, device=dev, dtype=i32)
    src = torch.empty(m_pad, device=dev, dtype=i32)
    offs = torch.empty(E, device=dev, dtype=i32)
    counts = torch.empty(E, device=dev, dtype=i32)
    tile_expert = torch.empty(m_pad // BLOCK_M, device=dev, dtype=i32)
    _dispatch_count_kernel[(num_blocks,)](row_expert, block_counts, n, E_PAD=E_PAD, BLOCK=block, num_warps=4)
    _dispatch_place_kernel[(triton.cdiv(m_pad, block),)](
        row_expert, block_counts, dest, src, offs, counts, tile_expert, n, m_pad, num_blocks,
        E=E, E_PAD=E_PAD, NB_PAD=NB_PAD, BLOCK=block, TILE=BLOCK_M, num_warps=8)
    return dest, src, offs, tile_expert, counts


# -----------------------------------------------------------------------------
# Gathers into the padded layout, quantized in both layouts from one read.

GQ_BLOCK_M, GQ_BLOCK_D = 128, 128


@triton.jit
def _gather_quant_dual_kernel(x_ptr, src_ptr, f8_ptr, t_ptr, scale_ptr, n, M, D: tl.constexpr,
                              BM: tl.constexpr, BD: tl.constexpr):
    """f8[p] = e4m3(x[src[p]] / scale), zero where src[p] == n, and t = f8.T."""
    pm = tl.program_id(0)
    pd = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)                               # padded rows: M % BM == 0
    cols = pd * BD + tl.arange(0, BD)                               # D % BD == 0
    s = tl.load(src_ptr + rows)
    v = tl.load(x_ptr + s.to(tl.int64)[:, None] * D + cols[None, :], mask=(s < n)[:, None], other=0.0)
    q = tl.maximum(tl.minimum(v.to(tl.float32) * (1.0 / tl.load(scale_ptr)), 448.0), -448.0).to(tl.float8e4nv)
    tl.store(f8_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], q)
    tl.store(t_ptr + cols.to(tl.int64)[:, None] * M + rows[None, :], tl.trans(q))


def _gather_quant_dual_impl(x, src, scale):
    """(row [M, D], transposed [D, M]) e4m3 of x[src] / scale (src[p] == n = x.shape[0]: a zero row), from one read.
    x [n, D] holds the token rows; src indexes them in the padded layout."""
    n, D = x.shape
    M = src.numel()
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and src.dtype == torch.int32
    assert M % GQ_BLOCK_M == 0 and D % GQ_BLOCK_D == 0, (M, D)
    _check_scalars(scale)
    f8 = torch.empty((M, D), device=x.device, dtype=E4M3)
    f8_t = torch.empty((D, M), device=x.device, dtype=E4M3)
    _gather_quant_dual_kernel[(M // GQ_BLOCK_M, D // GQ_BLOCK_D)](
        x, src, f8, f8_t, scale, n, M, D=D, BM=GQ_BLOCK_M, BD=GQ_BLOCK_D, num_warps=8)
    return f8, f8_t


GG_BLOCK_M, GG_BLOCK_D = 64, 128


@triton.jit
def _grad_gather_quant_kernel(g_ptr, y_ptr, src_ptr, s_ptr, f8_ptr, t_ptr, ds_ptr, scale_ptr, n, M,
                              D: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
    """For padded rows p of this program, t = src[p] (n: a zero row):
         f8[p] = e5m2(grad[t] * s[t] / scale) and its transpose   (the gradient of y, the ungated output)
         ds[p] = sum_d grad[t, d] * y_pad[p, d]                   (the gradient of the gate s[t])
    One program covers whole rows (the D loop), so ds needs no atomics and is deterministic."""
    pm = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM)
    s_row = tl.load(src_ptr + rows)
    valid = s_row < n
    gate = tl.load(s_ptr + s_row, mask=valid, other=0.0)
    inv = 1.0 / tl.load(scale_ptr)
    g_base = g_ptr + s_row.to(tl.int64)[:, None] * D
    y_base = y_ptr + rows.to(tl.int64)[:, None] * D
    ds = tl.zeros((BM,), dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        g = tl.load(g_base + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        y = tl.load(y_base + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        ds += tl.sum(g * y, axis=1)
        v = g * gate[:, None]
        q = tl.maximum(tl.minimum(v * inv, 57344.0), -57344.0).to(tl.float8e5)
        tl.store(f8_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], q)
        tl.store(t_ptr + cols.to(tl.int64)[:, None] * M + rows[None, :], tl.trans(q))
    tl.store(ds_ptr + rows, ds)


def _grad_gather_quant_impl(grad, y_pad, src, gate, scale):
    """(g row [M, D], g transposed [D, M]) e5m2 of grad[src] * gate[src] / scale, and the gate's gradient per padded
    row, ds [M] fp32 (read it through dest). grad [n, D] bf16 in token order, gate [n] fp32, y_pad [M, D] bf16."""
    n, D = grad.shape
    M = src.numel()
    assert grad.dtype == torch.bfloat16 and grad.is_contiguous() and y_pad.shape == (M, D) and y_pad.is_contiguous()
    assert gate.dtype == torch.float32 and gate.shape == (n,) and src.dtype == torch.int32
    assert M % GG_BLOCK_M == 0 and D % GG_BLOCK_D == 0, (M, D)
    _check_scalars(scale)
    f8 = torch.empty((M, D), device=grad.device, dtype=E5M2)
    f8_t = torch.empty((D, M), device=grad.device, dtype=E5M2)
    ds = torch.empty(M, device=grad.device, dtype=torch.float32)
    _grad_gather_quant_kernel[(M // GG_BLOCK_M,)](grad, y_pad, src, gate, f8, f8_t, ds, scale, n, M, D=D,
                                                  BM=GG_BLOCK_M, BD=GG_BLOCK_D, num_warps=4)
    return f8, f8_t, ds


# -----------------------------------------------------------------------------
# dx of the routed layer: the experts' dx back in token order plus the router's dlogits @ W_router.

DX_BLOCK_M, DX_BLOCK_D = 64, 128


@triton.jit
def _dx_combine_kernel(dxp_ptr, dest_ptr, dlog_ptr, r_ptr, out_ptr, n, D: tl.constexpr, E: tl.constexpr,
                       BM: tl.constexpr, BD: tl.constexpr):
    """out[t] = dx_pad[dest[t]] + sum_e dlogits[t, e] * R[e], in fp32, one bf16 rounding. n tokens."""
    pm = tl.program_id(0)
    pd = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pd * BD + tl.arange(0, BD)
    valid = rows < n
    d = tl.load(dest_ptr + rows, mask=valid, other=0)
    acc = tl.load(dxp_ptr + d.to(tl.int64)[:, None] * D + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    for e in tl.static_range(E):
        w = tl.load(dlog_ptr + rows * E + e, mask=valid, other=0.0)
        r = tl.load(r_ptr + e * D + cols).to(tl.float32)
        acc += w[:, None] * r[None, :]
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], acc.to(tl.bfloat16), mask=valid[:, None])


def _dx_combine_impl(dx_pad, dest, dlogits, router_w):
    """[n, D] bf16: dx_pad[dest[t]] + dlogits @ router_w (dlogits [n, E] fp32, router_w [E, D] bf16, dest [n] int32)."""
    n, E = dlogits.shape
    D = dx_pad.size(1)
    assert dx_pad.is_contiguous() and dlogits.is_contiguous() and dlogits.dtype == torch.float32
    assert router_w.shape == (E, D) and router_w.is_contiguous() and dest.dtype == torch.int32 and D % DX_BLOCK_D == 0
    assert dest.shape == (n,), (dest.shape, n)
    out = torch.empty((n, D), device=dx_pad.device, dtype=torch.bfloat16)
    _dx_combine_kernel[(triton.cdiv(n, DX_BLOCK_M), D // DX_BLOCK_D)](
        dx_pad, dest, dlogits, router_w, out, n, D=D, E=E, BM=DX_BLOCK_M, BD=DX_BLOCK_D, num_warps=4)
    return out


# -----------------------------------------------------------------------------
# The autograd op: the experts' core, then the routed layer around it (router inside).

def _experts_forward(rows, src, tile_expert, up_row, down_col, w1_scales, w2_scales, x_scale, post_scale, post_amax):
    """y_pad [M, D] bf16 (ungated, padded layout) and what the backward needs: post_f8, post_t, x_t. rows [T, D]
    are the token rows, src indexes them in the padded layout."""
    D = rows.size(1)
    E, H, _ = down_col.shape
    assert up_row.shape == (E * H, D)
    x_f8, x_t = gather_quant_dual(rows, src, x_scale)
    post_f8, post_t = grouped_linear_relu_square(x_f8, up_row, tile_expert, w1_scales, x_scale, H,
                                                 forward=True, post_scale=post_scale, amax=post_amax)
    # down_col (E, h, D) column-major: its storage is W2[e].T [D, h], stacked
    y_pad = grouped_mm(post_f8, down_col.transpose(1, 2).reshape(E * D, H), tile_expert, w2_scales, post_scale, D)
    # Padded rows at or past offs[-1] of y_pad are computed from zero rows and never read.
    return y_pad, post_f8, post_t, x_t


def _experts_backward(ctx, grad, y_pad, post_f8, post_t, x_t, gate, src, offs, tile_expert, bank_grad):
    """(dx_pad [M, D] bf16, ds_pad [M] fp32 the gate's gradient per padded row).  dW1 / dW2 go straight into bank_grad's
    slots ctx.slot_up / ctx.slot_down (added to what is there when acc_flag[0] != 0). grad is [T, D], gate [T]."""
    E, H = ctx.E, ctx.H
    D = grad.size(1)
    g_f8, g_t, ds_pad = grad_gather_quant(grad.contiguous(), y_pad, src, gate, ctx.g_scale)
    torch.ops.nanogpt_moe.dw_into_bank(post_t, g_t, offs, ctx.post_scale, ctx.g_scale, bank_grad, ctx.slot_down,
                                       ctx.acc_flag)
    dpre_f8, dpre_t = grouped_linear_relu_square(g_f8, ctx.down_row, tile_expert, ctx.w2_scales, ctx.g_scale, H,
                                                 forward=False, aux=post_f8, post_scale=ctx.post_scale,
                                                 amax=ctx.dpre_amax, dpre_scale=ctx.dpre_scale)
    torch.ops.nanogpt_moe.dw_into_bank(dpre_t, x_t, offs, ctx.dpre_scale, ctx.x_scale, bank_grad, ctx.slot_up,
                                       ctx.acc_flag)
    dx_pad = grouped_mm(dpre_f8, ctx.up_col.transpose(1, 2).reshape(E * D, H), tile_expert, ctx.w1_scales,
                        ctx.dpre_scale, D)
    return dx_pad, ds_pad


def _keep_for_backward(ctx, w_fc, up_col, down_row, w1_scales, w2_scales, x_scale, g_scale, post_scale,
                       dpre_scale, dpre_amax, layer):
    """Persistent caches and scales (refreshed in place between steps) by name on ctx, as the record's
    FusedLinearReLUSquareFunction; intermediates go through save_for_backward. `layer`: the MoE layer's index
    k in the bank, whose c_fc / c_proj slots are [2kE, (2k+1)E) / [(2k+1)E, (2k+2)E) (for bank_grad)."""
    ctx.w1_scales, ctx.w2_scales, ctx.x_scale, ctx.g_scale = w1_scales, w2_scales, x_scale, g_scale
    ctx.post_scale, ctx.dpre_scale, ctx.dpre_amax = post_scale, dpre_scale, dpre_amax
    ctx.up_col, ctx.down_row = up_col, down_row
    E = w_fc.shape[0]
    ctx.E, ctx.H = E, w_fc.shape[1]
    ctx.slot_up, ctx.slot_down = 2 * layer * E, (2 * layer + 1) * E


class RoutedFP8MoE(torch.autograd.Function):
    """The whole routed layer in one op: logits = rows @ router_w.T (bf16, as F.linear), probs = softmax(logits.float()),
    (gate, expert) = probs.max(-1) (top-1), out[t] = gate[t] * post * MLP_{expert[t]}(rows[t]), the MLP being
    relu(rows @ W1[e].T)^2 @ W2[e] with every GEMM in fp8.

    Owning the router lets the backward fuse what autograd would do in separate passes over [n, D]: the gate's gradient
    comes out of the g gather, and dx = dx_pad[dest] + dlogits @ W_r is one kernel (dx_combine) instead of a gather, a
    copy and an accumulating GEMM.

    rows [T, D] bf16 (the normed token rows); router_w [E, D] bf16; post: 0-d, the folded post-lambda (or a constant 1);
    w_fc, w_proj [E, h, D] bf16 (only for their gradient's shape and dtype); up_row / down_row [E*h, D] e4m3 row-major and
    up_col / down_col [E, h, D] e4m3 column-major (whose storage is W[e].T stacked: the B of the down projection and of
    dx); w1_scales / w2_scales [E]; x_scale, g_scale: the static e4m3 input and e5m2 gradient scales; g_amax: unused
    (kept so the argument list does not change); post_scale / dpre_scale [1] (delayed) and post_amax / dpre_amax [1]
    (this step's, atomic-maxed); bank_grad: the expert bank's persistent gradient [L, h, D]: the experts' dW1 / dW2 are
    written into this layer's slots (see _keep_for_backward), so autograd gets None for w_fc / w_proj; layer: the MoE
    layer's index in the bank; acc_flag: int32 [1], nonzero = add to bank_grad instead of overwriting (a gradient
    summed over a slow bank's period); topk must be 1.
    Returns (out [T, D] bf16, probs [T, E] fp32 for the load-balancing loss, counts [E] int32 assignments per expert,
    not differentiable).
    """
    @staticmethod
    def forward(ctx, rows, router_w, post, w_fc, w_proj, up_row, up_col, down_row, down_col, w1_scales, w2_scales,
                x_scale, g_scale, g_amax, post_scale, post_amax, dpre_scale, dpre_amax, bank_grad, layer, acc_flag,
                topk=1):
        E = w_fc.shape[0]
        assert topk == 1, "the MoE kernels route top-1 only"
        assert bank_grad is not None, "the experts' dW is written straight into the bank's gradient"
        assert router_w.shape == (E, rows.size(1)) and post.numel() == 1
        probs = torch.softmax(torch.nn.functional.linear(rows, router_w).float(), dim=-1)
        gate, expert = probs.max(dim=-1)
        topi = expert[:, None]
        dest, src, offs, tile_expert, counts = moe_dispatch(expert, E)
        post_f = post.float().reshape(())
        scale = gate * post_f
        y_pad, post_f8, post_t, x_t = _experts_forward(rows, src, tile_expert, up_row, down_col, w1_scales,
                                                       w2_scales, x_scale, post_scale, post_amax)
        ctx.save_for_backward(y_pad, post_f8, post_t, x_t, scale, src, dest, offs, tile_expert, rows, router_w, probs,
                              topi, gate, post_f, bank_grad)
        _keep_for_backward(ctx, w_fc, up_col, down_row, w1_scales, w2_scales, x_scale, g_scale, post_scale,
                           dpre_scale, dpre_amax, layer)
        ctx.post_dtype, ctx.post_shape = post.dtype, post.shape
        ctx.acc_flag = acc_flag
        ctx.mark_non_differentiable(counts)
        out = y_pad.index_select(0, dest) * scale[:, None].to(torch.bfloat16)
        return out, probs, counts

    @staticmethod
    def backward(ctx, grad, dprobs, dcounts):
        st = ctx.saved_tensors
        y_pad, post_f8, post_t, x_t, scale, src, dest, offs, tile_expert = (
            st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8])
        rows, router_w, probs, topi, gate, post_f, bank_grad = st[9], st[10], st[11], st[12], st[13], st[14], st[15]
        dx_pad, ds_pad = _experts_backward(ctx, grad, y_pad, post_f8, post_t, x_t, scale, src, offs, tile_expert,
                                           bank_grad)
        ds = ds_pad.index_select(0, dest)                               # d out / d scale, per token
        dpost = (ds * gate).sum().to(ctx.post_dtype).reshape(ctx.post_shape) if ctx.needs_input_grad[2] else None
        # gate[t] = probs[t, topi[t]]: its gradient joins the load-balancing loss's at the chosen expert
        dp = torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp = dp.scatter_add(1, topi, (ds * post_f).view(1, -1).t())
        dlogits = probs * (dp - (probs * dp).sum(-1, keepdim=True))    # softmax backward, fp32
        dlogits_bf = dlogits.to(torch.bfloat16)                          # logits were bf16 (F.linear)
        dx = dx_combine(dx_pad, dest, dlogits_bf.float(), router_w)
        drouter = dlogits_bf.t() @ rows
        return (dx, drouter, dpost, None, None) + (None,) * 17


# -----------------------------------------------------------------------------
# Every Triton launch above runs inside a torch.library custom op, opaque to inductor, as mlp.py does with its
# _scaled_mm GEMMs. Measured: traced as user-defined Triton kernels, the descriptor-writing GEMMs gave a NaN dx in the
# compiled backward only (an output read before the kernel that writes it through its TensorDescriptor); as ops the
# compiled graph runs exactly the eager code.

Tensor = torch.Tensor


@torch.library.custom_op("nanogpt_moe::dispatch", mutates_args=())
def _dispatch_op(row_expert: Tensor, n_experts: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _moe_dispatch_impl(row_expert, n_experts)


@_dispatch_op.register_fake
def _(row_expert, n_experts):
    n = row_expert.shape[0]
    m_pad = padded_rows(n, n_experts)
    e = lambda size: row_expert.new_empty(size, dtype=torch.int32)
    return e((n,)), e((m_pad,)), e((n_experts,)), e((m_pad // BLOCK_M,)), e((n_experts,))


@torch.library.custom_op("nanogpt_moe::gather_x", mutates_args=())
def _gather_x_op(x: Tensor, src: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    return _gather_quant_dual_impl(x, src, scale)


@_gather_x_op.register_fake
def _(x, src, scale):
    M, D = src.shape[0], x.shape[1]
    return x.new_empty((M, D), dtype=E4M3), x.new_empty((D, M), dtype=E4M3)


@torch.library.custom_op("nanogpt_moe::gather_g", mutates_args=())
def _gather_g_op(grad: Tensor, y_pad: Tensor, src: Tensor, gate: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return _grad_gather_quant_impl(grad, y_pad, src, gate, scale)


@_gather_g_op.register_fake
def _(grad, y_pad, src, gate, scale):
    M, D = src.shape[0], grad.shape[1]
    return (grad.new_empty((M, D), dtype=E5M2), grad.new_empty((D, M), dtype=E5M2),
            grad.new_empty((M,), dtype=torch.float32))


@torch.library.custom_op("nanogpt_moe::lrs", mutates_args={"amax"})
def _lrs_op(a_f8: Tensor, b_f8: Tensor, tile_expert: Tensor, dequant: Tensor, dq2: Tensor, n_per_expert: int,
            forward: bool, post_scale: Tensor, amax: Tensor, aux: Optional[Tensor],
            dpre_scale: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    return _grouped_lrs_impl(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, forward=forward,
                             post_scale=post_scale, amax=amax, aux=aux, dpre_scale=dpre_scale)


@_lrs_op.register_fake
def _(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, forward, post_scale, amax, aux, dpre_scale):
    M, N = a_f8.shape[0], n_per_expert
    dt = E4M3 if forward else E5M2
    return a_f8.new_empty((M, N), dtype=dt), a_f8.new_empty((N, M), dtype=dt)


@torch.library.custom_op("nanogpt_moe::mm", mutates_args=())
def _mm_op(a_f8: Tensor, b_rows: Tensor, tile_expert: Tensor, dequant: Tensor, dq2: Tensor, n_per_expert: int) -> Tensor:
    return _grouped_mm_impl(a_f8, b_rows, tile_expert, dequant, dq2, n_per_expert)


@_mm_op.register_fake
def _(a_f8, b_rows, tile_expert, dequant, dq2, n_per_expert):
    return a_f8.new_empty((a_f8.shape[0], n_per_expert), dtype=torch.bfloat16)


@torch.library.custom_op("nanogpt_moe::dx_combine", mutates_args=())
def _dx_combine_op(dx_pad: Tensor, dest: Tensor, dlogits: Tensor, router_w: Tensor) -> Tensor:
    return _dx_combine_impl(dx_pad, dest, dlogits, router_w)


@_dx_combine_op.register_fake
def _(dx_pad, dest, dlogits, router_w):
    return dx_pad.new_empty((dlogits.shape[0], dx_pad.shape[1]), dtype=torch.bfloat16)


def moe_dispatch(row_expert, n_experts):
    """See _moe_dispatch_impl: (dest, src, offs, tile_expert, counts), int32."""
    return torch.ops.nanogpt_moe.dispatch(row_expert, n_experts)


def gather_quant_dual(x, src, scale):
    """See _gather_quant_dual_impl."""
    return torch.ops.nanogpt_moe.gather_x(x, src, scale)


def grad_gather_quant(grad, y_pad, src, gate, scale):
    """See _grad_gather_quant_impl."""
    return torch.ops.nanogpt_moe.gather_g(grad, y_pad, src, gate, scale)


def grouped_linear_relu_square(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, *, forward, post_scale,
                               amax, aux=None, dpre_scale=None):
    """See _grouped_lrs_impl."""
    return torch.ops.nanogpt_moe.lrs(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, forward, post_scale, amax,
                                     aux, dpre_scale)


def grouped_mm(a_f8, b_rows, tile_expert, dequant, dq2, n_per_expert):
    """See _grouped_mm_impl."""
    return torch.ops.nanogpt_moe.mm(a_f8, b_rows, tile_expert, dequant, dq2, n_per_expert)


def dx_combine(dx_pad, dest, dlogits, router_w):
    """See _dx_combine_impl."""
    return torch.ops.nanogpt_moe.dx_combine(dx_pad, dest, dlogits, router_w)


# -----------------------------------------------------------------------------
# Batched byte-transpose: dst (B, N, M) = src (B, M, N).transpose(1, 2), one launch for the whole stack.
# The experts' column-major e4m3 cache is built from the gathered row-major one with it (GPT.build_moe_cols):
# 80 matrices of 1536x768, which 80 transpose_copy launches (or a strided copy_) would move at a fraction of the
# bandwidth.

@triton.jit
def _transpose_copy_batched_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_b, dst_stride_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    src = src_ptr + pid_b * src_stride_b
    dst = dst_ptr + pid_b * dst_stride_b

    # Coalesced read of a (BLOCK_M, BLOCK_N) tile of src[b] (M, N), row-major
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tile = tl.load(src + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0)

    # Coalesced write to dst[b] (N, M), row-major: dst[b, n, m] = src[b, m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(dst + offs_n[:, None] * M + offs_m[None, :], tl.trans(tile), mask=mask_T)


def transpose_copy_batched(src: torch.Tensor, dst: torch.Tensor):
    """Batched byte transpose: dst[b] = src[b].T, src (B, M, N) and dst (B, N, M), both contiguous uint8.

    128x128 tiles, so each side issues full 128-byte segments (as transpose_copy's fp8 tile); the grid's
    third axis is the batch. A pure copy, hence bit-exact against src.transpose(1, 2).contiguous().
    """
    assert src.ndim == 3 and dst.ndim == 3
    B, M, N = src.shape
    assert dst.shape == (B, N, M), f"Expected dst shape ({B}, {N}, {M}), got {tuple(dst.shape)}"
    assert src.dtype == torch.uint8 and dst.dtype == torch.uint8, "pass uint8 views of fp8 caches"
    assert src.is_contiguous() and dst.is_contiguous()
    BLOCK_M = BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)
    _transpose_copy_batched_kernel[grid](
        src, dst,
        M, N,
        M * N, M * N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )

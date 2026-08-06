# fused_kernels.py — the full-FP8 MLP + fused-CE kernel package.
#
# Owns the "nanogpt" custom-op namespace — NEVER import two copies in the same
# process. Feature tags used in the comments below name this module's mechanisms;
# each is baked on/off by the trainer's configuration constants:
#   KX_WG2   dW2 = post_f8^T @ g_f8 via DUAL-LAYOUT EPILOGUE EMISSION (PR#344
#            register-tile tl.trans technique). No standalone transpose passes
#            anywhere — even bandwidth-optimal separate transpose passes
#            measured +10.7s/run. The transposed operands are written
#            by kernels that already hold the tiles in registers.
#   KX_C1D   dx = dpre_f8 @ W1_f8_col (row-major dpre emitted by the backward
#            kernel epilogue; col-major W1 cache refreshed post-step).
#   KX_C1E   dW1 = dpre_f8^T @ x_f8^T.T (transposed dpre from the backward
#            epilogue; transposed x from the forward dual-layout quantize).
#   KX_PDROP drop the bf16 `pre` store entirely (PR#322): backward reconstructs
#            relu(pre) = sqrt(post) from the fp8 `post` it reads as aux anyway.
#   KX_GE4   e4m3 gradients (g: dynamic per-tensor scale; dpre: delayed
#            per-layer scale) instead of e5m2 at static KX_GS.
#   KX_A2P   plumbing upgrades from PR#342: fused quantize+transpose weight
#            cache kernel with lagged (one-step-stale) scales, and per-SM
#            partial-amax slots instead of tl.atomic_max in GEMM epilogues.
#   KX_S3D   FP8 attention qkv DGRAD (dx = g @ W_qkv, K=2304 — C1D's shape
#            class): fused aten e5m2 cast of g +
#            the existing nanogpt::dx_f8 op.
#   KX_S3W   FP8 attention dW_qkv (g^T @ x, K=T — the splitK family)
#            via the existing nanogpt::wg1_f8 op. Transposed operands land in
#            trainer-side PREALLOCATED buffers (never
#            torch.empty a [2304,T] per step inside an op). 1=primary
#            (fp8->fp8 transpose_copy of the fused cast, or GE4 quantize),
#            2=route B (dual-layout quantize of bf16 g; DCE-independent).
#   KX_S3O   o-site dW (unused; dx_o K=768 is excluded and not implemented).
#   KX_S3_GE4  e4m3 wgrad g at a delayed per-layer scale (default ON: dW_qkv's
#            v-rows feed the vo weight bank; halves elementwise noise).
#
# Numerical rules respected throughout:
#   * use_fast_accum=False on EVERY gradient GEMM (NaN otherwise).
#   * explicit clamp before every fp8 cast (fp8 saturation).
#   * delayed per-layer scaling via the C1A machinery pattern.
#   * smem: emit variants start at num_stages=3 and fall
#     back automatically on triton OutOfResources.

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    from triton.runtime.errors import OutOfResources as _TritonOOR
except Exception:  # triton version drift — keep the fallback path importable
    class _TritonOOR(Exception):
        pass

E4M3_MAX_F = 448.0

# KX_LHBW: emit the lm_head/CE wgrad in bf16 instead of fp32. Halves the 154MB
# grad_w write AND the fp32 reduce_scatter bytes on Adam steps (the grad buffer takes
# grad_w's dtype at first backward). Same rounding class as the accepted MLP bf16
# wgrad path. Default 0 = stock fp32 bit-path.
_LHBW16 = True

# CE backward dead-saves (byte-identical
# when the knob is off). Backward only reads grad_input/x_f8/w_f8/mtp_weights — the
# saved logits/lse/x/targets/lm_head_weight are dead across the backward window.
import os as _r2_os
_R2CE = True  # CE backward saves only what backward reads (baked)

# Behavior flags. The trainer decides what buffers to pass; these flags decide
# what the kernels do with them. Read once at import (matching kernel-module's _C1C
# pattern); tests may override via set_flags() BEFORE any launches.
FLAGS = {
    "C1C":   False,   # legacy dW2 diagnostic (MEASURED DEAD: +10.7s, kept for provenance)
    "WG2":   True,   # FP8 dW2 via epilogue dual-layout
    "C1D":   True,   # FP8 dx
    "C1E":   True,   # FP8 dW1
    "PDROP": True, # drop pre; bwd reconstructs sqrt(post_f8)
    "GE4":   False,   # e4m3 grads w/ real scales
    "A2P":   True,   # per-SM amax + lagged weight quantize
    "S3D":   False,   # FP8 attention qkv dgrad
    "S3W":   0,     # FP8 attention dW_qkv (2 = route B)
    "S3O":   False,   # o-site dW (unused)
    "S3GE4": True,  # e4m3 wgrad g, delayed scale
    # Diagnostic knob: route the S3D dgrad-g cast through the triton
    # quantize kernel instead of the fused aten pointwise (removes the
    # inductor-fusion surface entirely; costs one extra bf16 read of g).
    "S3CAST_K": False,
}



# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99
# (unchanged from kernel-module)

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load A blocks for C[m,n] = A[m,:] @ A[n,:].T
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (128, 128, 64)
        num_stages, num_warps = (4, 8)
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (64, 128, 128)
        num_stages, num_warps = (4, 8)

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# -----------------------------------------------------------------------------
# Triton kernel for X.T @ X (tall matrices) — unchanged from kernel-module

@triton.jit
def XTX_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Compute C = A.T @ A where A is (M, K) and C is (K, K).
    """
    pid = tl.program_id(axis=0)
    batch_idx, k_idx, n_idx = _pid_to_block(
        pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XTX(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A.T @ A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"
    assert out.size(-1) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (128, 128, 64)
        num_stages, num_warps = (4, 8)
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (64, 128, 128)
        num_stages, num_warps = (4, 8)

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)
    XTX_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = (128, 128, 64)
    num_stages, num_warps = (4, 8)

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# -----------------------------------------------------------------------------
# Fused MLP kernel: relu(x @ W1.T)^2, by @andrewbriand, @jrauvola.
# Extension: dual-layout fp8 epilogue emission on BOTH passes.
#
# FORWARD extras (all constexpr-gated, dead-code-eliminated when off):
#   EMIT_F8      store post/post_scale as e4m3 row-major [M, N]   (C1A, as kernel-module)
#   EMIT_T       ALSO store the same fp8 tile transposed to [N, M] via a
#                register-tile tl.trans — PR#344's technique. Write-only cost.
#   STORE_PRE    store the bf16 pre tensor (off under KX_PDROP)
#   STORE_POST_BF store the bf16 post tensor (dead once dW2 is FP8)
# BACKWARD extras:
#   RECON_SQRT   aux is the fp8 `post` (not bf16 `pre`): reconstruct
#                relu(pre) = sqrt(dequant(post)). The backward reads aux anyway,
#                so this is free — and the fp8 aux read HALVES that traffic.
#   EMIT_DPRE    store dpre/dpre_scale as fp8 row-major [M, N]     (for dx, C1D)
#   EMIT_DPRE_T  ALSO store it transposed to [N, M]                (for dW1, C1E)
#   STORE_C_BF   store bf16 dpre (off only when C1D && C1E: nothing reads it)
#   GRAD_E4M3    dpre fp8 format: e4m3 (GE4) vs e5m2 (default)
# Both passes:
#   PARTIAL_AMAX amax goes to a per-SM slot (buffer is a [NUM_SMS] row) via an
#                UNCONTENDED per-slot tl.atomic_max, instead of every tile of a
#                full-occupancy GEMM hammering one global location (A2P, PR#342
#                item 3). NOTE: #342/#344 accumulate a loop-carried scalar
#                across the tl.range(flatten=True) persistent loop and store it
#                once at kernel end; flatten's pipelining does not guarantee
#                loop-carried scalars (their delayed-scale+headroom design
#                would silently absorb a wrong amax — our exact-equality gate
#                caught it). Per-slot atomics are exact by construction and
#                keep the contention-removal win.

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 dequant_scale_ptr,
                                 M, N, K,
                                 aux_f8_desc, post_scale_ptr, post_amax_ptr,
                                 aux_t_desc,
                                 c_f8_desc, c_t_desc, dpre_scale_ptr, dpre_amax_ptr,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 USE_FP8: tl.constexpr,
                                 EMIT_F8: tl.constexpr,
                                 EMIT_T: tl.constexpr,
                                 STORE_PRE: tl.constexpr,
                                 STORE_POST_BF: tl.constexpr,
                                 RECON_SQRT: tl.constexpr,
                                 EMIT_DPRE: tl.constexpr,
                                 EMIT_DPRE_T: tl.constexpr,
                                 STORE_C_BF: tl.constexpr,
                                 GRAD_E4M3: tl.constexpr,
                                 PARTIAL_AMAX: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        if USE_FP8:
            accumulator *= tl.load(dequant_scale_ptr)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        if FORWARD:
            # ---- half 0 ----
            c0 = acc0.to(dtype)                      # pre (bf16-rounded, as kernel-module)
            if STORE_PRE:
                c_desc.store([offs_am_c, offs_bn_c], c0)
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            if STORE_POST_BF:
                aux_desc.store([offs_am_c, offs_bn_c], c0_post)
            if EMIT_F8:
                inv_ps = 1.0 / tl.load(post_scale_ptr)
                # explicit clamp before every fp8 cast (post >= 0 by construction)
                q0 = tl.minimum(c0_post.to(tl.float32) * inv_ps, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am_c, offs_bn_c], q0)
                if EMIT_T:
                    aux_t_desc.store([offs_bn_c, offs_am_c], tl.trans(q0))
            # ---- half 1 ----
            c1 = acc1.to(dtype)
            if STORE_PRE:
                c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            if STORE_POST_BF:
                aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)
            if EMIT_F8:
                inv_ps = 1.0 / tl.load(post_scale_ptr)
                q1 = tl.minimum(c1_post.to(tl.float32) * inv_ps, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], q1)
                if EMIT_T:
                    aux_t_desc.store([offs_bn_c + BLOCK_SIZE_N // 2, offs_am_c], tl.trans(q1))
                tile_max = tl.maximum(
                    tl.max(tl.max(c0_post.to(tl.float32), axis=1), axis=0),
                    tl.max(tl.max(c1_post.to(tl.float32), axis=1), axis=0))
                if PARTIAL_AMAX:
                    # per-SM slot: uncontended (this pid owns the slot), exact.
                    tl.atomic_max(post_amax_ptr + start_pid, tile_max)
                else:
                    tl.atomic_max(post_amax_ptr, tile_max)
        else:
            # ---- backward: dpre = 2 * (g @ W2) * relu(pre) ----
            # half 0
            if RECON_SQRT:
                ps = tl.load(post_scale_ptr)
                aux0 = aux_desc.load([offs_am_c, offs_bn_c]).to(tl.float32)
                # aux is the e4m3 post (>= 0); relu(pre) = sqrt(post)
                c0 = (2.0 * acc0 * tl.sqrt(aux0 * ps)).to(dtype)
            else:
                c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
                c0 = acc0.to(dtype)
                c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)
            if STORE_C_BF:
                c_desc.store([offs_am_c, offs_bn_c], c0)
            # half 1
            if RECON_SQRT:
                ps = tl.load(post_scale_ptr)
                aux1 = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2]).to(tl.float32)
                c1 = (2.0 * acc1 * tl.sqrt(aux1 * ps)).to(dtype)
            else:
                c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
                c1 = acc1.to(dtype)
                c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)
            if STORE_C_BF:
                c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            # fp8 dpre emission (quantize from the bf16-rounded value so the fp8
            # path sees exactly what the bf16 path would have shipped — #344's
            # "match the old bf16 materialization" trick)
            if EMIT_DPRE or EMIT_DPRE_T:
                inv_ds = 1.0 / tl.load(dpre_scale_ptr)
                c0f = c0.to(tl.float32)
                c1f = c1.to(tl.float32)
                if GRAD_E4M3:
                    q0 = tl.maximum(tl.minimum(c0f * inv_ds, 448.0), -448.0).to(tl.float8e4nv)
                    q1 = tl.maximum(tl.minimum(c1f * inv_ds, 448.0), -448.0).to(tl.float8e4nv)
                else:
                    q0 = tl.maximum(tl.minimum(c0f * inv_ds, 57344.0), -57344.0).to(tl.float8e5)
                    q1 = tl.maximum(tl.minimum(c1f * inv_ds, 57344.0), -57344.0).to(tl.float8e5)
                if EMIT_DPRE:
                    c_f8_desc.store([offs_am_c, offs_bn_c], q0)
                    c_f8_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], q1)
                if EMIT_DPRE_T:
                    c_t_desc.store([offs_bn_c, offs_am_c], tl.trans(q0))
                    c_t_desc.store([offs_bn_c + BLOCK_SIZE_N // 2, offs_am_c], tl.trans(q1))
                tile_max = tl.maximum(
                    tl.max(tl.max(tl.abs(c0f), axis=1), axis=0),
                    tl.max(tl.max(tl.abs(c1f), axis=1), axis=0))
                if PARTIAL_AMAX:
                    # per-SM slot: uncontended (this pid owns the slot), exact.
                    tl.atomic_max(dpre_amax_ptr + start_pid, tile_max)
                else:
                    tl.atomic_max(dpre_amax_ptr, tile_max)


# COMPILE-SAFETY INVARIANT (fixv3, 2026-07-27): linear_relu_square executes
# INSIDE torch.compile's autograd-Function HOP subgraphs (fullgraph=True), where
# dynamo rejects any Python-state mutation from an outer scope with
# "HigherOrderOperator: Mutating a variable not in the current scope
# (SideEffects)". Therefore NO code reachable from linear_relu_square may:
# write a module global (the old lazy `_get_dummy_f32`), write a module-level
# dict (the old unconditional `_lrs_stage_cache[key] = ...`), or mutate a
# closure cell (the old `_tile()` nonlocal). Unused pointer args now get a
# per-call torch.empty(1) (dynamo-pure, no kernel launch); the stage cache is
# read-only under compile and written only on the eager path.

# num_stages cache per constexpr-variant: emit variants start at the C1A
# precedent (3) and step down automatically if a variant exceeds H100 smem.
# Written ONLY by eager calls (tests, prime_stage_cache); under torch.compile
# it is a read-only lookup -- call prime_stage_cache() before compiling so
# every reachable variant is resolved (defaults-variant initial values are the
# kernel-module-proven ones, so an unprimed cache is still safe with all new flags off).
_lrs_stage_cache = {}

def linear_relu_square(a, b, aux=None, a_f8=None, b_f8=None, dequant_scale_ptr=None,
                       emit_f8=False, post_scale=None, post_amax=None,
                       emit_t=False, store_pre=True, store_post_bf=True,
                       emit_dpre=False, emit_dpre_t=False, store_c_bf=True,
                       dpre_scale=None, dpre_amax=None, grad_e4m3=False,
                       partial_amax_mode=False):
    """kernel-module's fused MLP GEMM, extended with dual-layout fp8 epilogue emission.

    Returns (pre, post, post_f8, post_t) on the forward pass and
    (dpre, dpre_f8, dpre_t) on the backward pass; entries not requested are None.
    """
    M, K = a.shape
    N, Kb = b.shape
    dtype = a.dtype
    device = a.device
    use_fp8 = b_f8 is not None

    FORWARD = aux is None
    recon_sqrt = (not FORWARD) and aux.dtype == torch.float8_e4m3fn

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128 if use_fp8 else 64
    num_warps = 8

    if FORWARD:
        assert not (emit_t and not emit_f8), "EMIT_T rides on the EMIT_F8 quantize"
        # smem guard: post_t emission is only ever needed once dW2 is FP8, at
        # which point the bf16 post is dead — never pay for both.
        assert not (emit_t and store_post_bf), "post_t emit implies bf16 post is dead"
        c = torch.empty((M, N), device=device, dtype=dtype) if store_pre else None
        aux_out = torch.empty((M, N), device=device, dtype=dtype) if store_post_bf else None
        aux_f8 = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn) if emit_f8 else None
        aux_t = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn) if emit_t else None
        c_f8 = c_t = None
    else:
        c = torch.empty((M, N), device=device, dtype=dtype) if store_c_bf else None
        aux_out = aux
        aux_f8 = aux_t = None
        grad_fmt = torch.float8_e4m3fn if grad_e4m3 else torch.float8_e5m2
        c_f8 = torch.empty((M, N), device=device, dtype=grad_fmt) if emit_dpre else None
        c_t = torch.empty((N, M), device=device, dtype=grad_fmt) if emit_dpre_t else None
        assert store_c_bf or emit_dpre or emit_dpre_t

    # A separate minimal dummy tile so unused TMA descriptors never alias a live
    # output buffer (PR#322's descriptor-aliasing precaution). Straight-line
    # allocation (compile-safety invariant above: the old lazily-mutated
    # `_tile()` closure cell is exactly the construct the HOP tracer rejects).
    dummy_tile = None
    if c is None or aux_out is None:
        dummy_tile = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
                                 device=device, dtype=dtype)

    a_kernel = a_f8 if use_fp8 else a
    a_desc = TensorDescriptor.from_tensor(a_kernel, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_kernel = b_f8 if use_fp8 else b
    b_desc = TensorDescriptor.from_tensor(b_kernel, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c if c is not None else dummy_tile,
                                          [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux_out if aux_out is not None else dummy_tile,
                                            [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_f8_desc = (TensorDescriptor.from_tensor(aux_f8, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
                   if aux_f8 is not None else aux_desc)
    aux_t_desc = (TensorDescriptor.from_tensor(aux_t, [BLOCK_SIZE_N // 2, BLOCK_SIZE_M])
                  if aux_t is not None else aux_desc)
    c_f8_desc = (TensorDescriptor.from_tensor(c_f8, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
                 if c_f8 is not None else c_desc)
    c_t_desc = (TensorDescriptor.from_tensor(c_t, [BLOCK_SIZE_N // 2, BLOCK_SIZE_M])
                if c_t is not None else c_desc)

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    # The unified Triton signature requires a pointer for every scalar arg, but
    # the kernel never loads the ones whose constexpr paths are off. Per-call
    # torch.empty(1) instead of the old lazily-created module global: a
    # STORE_GLOBAL in here traces inside the autograd-Function HOP and dynamo
    # rejects it (compile-safety invariant above). empty(1) is dynamo-pure and
    # launch-free (allocation only).
    if (not use_fp8) or post_scale is None or post_amax is None \
            or dpre_scale is None or dpre_amax is None:
        _unused_ptr = torch.empty(1, dtype=torch.float32, device=device)

    if use_fp8:
        assert dequant_scale_ptr is not None
    else:
        # bf16 kernels never load the dequant pointer.
        dequant_scale_ptr = _unused_ptr

    if (FORWARD and emit_f8) or recon_sqrt:
        assert post_scale is not None
    if post_scale is None:
        post_scale = _unused_ptr
    if post_amax is None:
        post_amax = _unused_ptr
    if emit_dpre or emit_dpre_t:
        assert dpre_scale is not None and dpre_amax is not None
    if dpre_scale is None:
        dpre_scale = _unused_ptr
    if dpre_amax is None:
        dpre_amax = _unused_ptr

    if FORWARD:
        initial_stages = 3 if (emit_f8 or emit_t) else 4
    else:
        initial_stages = 3

    key = (FORWARD, use_fp8, emit_f8, emit_t, store_pre, store_post_bf,
           recon_sqrt, emit_dpre, emit_dpre_t, store_c_bf, grad_e4m3,
           partial_amax_mode)
    num_stages = _lrs_stage_cache.get(key, initial_stages)

    def _launch(_ns):
        linear_relu_square_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc,
            dequant_scale_ptr,
            M, N, K,
            aux_f8_desc, post_scale, post_amax,
            aux_t_desc,
            c_f8_desc, c_t_desc, dpre_scale, dpre_amax,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=1,
            NUM_SMS=NUM_SMS,
            FORWARD=FORWARD,
            USE_FP8=use_fp8,
            EMIT_F8=aux_f8 is not None,
            EMIT_T=aux_t is not None,
            STORE_PRE=FORWARD and store_pre,
            STORE_POST_BF=FORWARD and store_post_bf,
            RECON_SQRT=recon_sqrt,
            EMIT_DPRE=c_f8 is not None,
            EMIT_DPRE_T=c_t is not None,
            STORE_C_BF=(not FORWARD) and store_c_bf,
            GRAD_E4M3=grad_e4m3,
            PARTIAL_AMAX=partial_amax_mode,
            num_stages=_ns,
            num_warps=num_warps,
        )

    if torch.compiler.is_compiling():
        # Dynamo trace (constant-folded True). Two reasons this branch is a bare
        # cache-read + single launch: (1) writing _lrs_stage_cache here is a
        # module-dict mutation inside the autograd-Function HOP -> hard dynamo
        # error (SideEffects) -- the v3 crash; (2) the OOR retry could never
        # work here anyway: the launch is deferred into the compiled artifact,
        # so no exception can reach this try at trace time. The cache must be
        # primed eagerly (prime_stage_cache()) before compiling; unprimed keys
        # fall back to initial_stages == the kernel-module-proven defaults-variant values.
        _launch(num_stages)
    else:
        while True:
            try:
                _launch(num_stages)
                break
            except _TritonOOR:
                # smem overflow on this emit variant: step the pipeline down
                # (the C1A variant stepped 4->3 the same way).
                if num_stages <= 1:
                    raise
                num_stages -= 1
        _lrs_stage_cache[key] = num_stages

    if FORWARD:
        return c, aux_out, aux_f8, aux_t
    else:
        return c, c_f8, c_t


def prime_stage_cache(device=None):
    """Eagerly resolve num_stages for every linear_relu_square variant the
    current FLAGS can reach, by launching each once on one-tile inputs.

    Why this exists: under torch.compile the OutOfResources retry loop cannot
    run (the launch is deferred into the compiled artifact) and the cache may
    not be written (module-state mutation inside the autograd-Function HOP is
    a dynamo hard error). H100 smem pressure depends only on the constexpr
    variant (block sizes, pipeline stages, output-buffer set), NOT on M/N/K,
    so the tiny-shape resolution here is exact for training shapes. The Triton
    binaries compiled here land in the normal Triton cache and are reused by
    the compiled run, so the cost is one-time and untimed (call before the
    clock starts). Idempotent; call once per process after CUDA init and
    BEFORE torch.compile traces the model.
    """
    if not torch.cuda.is_available():
        return
    dev = torch.device(device) if device is not None else torch.device("cuda")
    M, N, K = (128, 256, 128)  # one (BLOCK_M, BLOCK_N) tile, one fp8 K-block
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    a = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.zeros(N, K, device=dev, dtype=torch.bfloat16)
    a_e4 = torch.zeros(M, K, device=dev, dtype=torch.float8_e4m3fn)
    b_e4 = torch.zeros(N, K, device=dev, dtype=torch.float8_e4m3fn)
    g_fmt = torch.float8_e4m3fn if FLAGS["GE4"] else torch.float8_e5m2
    g_f8 = torch.zeros(M, K, device=dev, dtype=g_fmt)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    slots = nsm if FLAGS["A2P"] else 1
    amax_f = torch.zeros(slots, dtype=torch.float32, device=dev)
    amax_b = torch.zeros(slots, dtype=torch.float32, device=dev)

    wg2, pdrop = FLAGS["WG2"], FLAGS["PDROP"]
    c1d, c1e = FLAGS["C1D"], FLAGS["C1E"]
    store_post_bf = not (wg2 or FLAGS["C1C"])

    # fwd, bf16 (eval path / DISABLE_FP8)
    linear_relu_square(a, b)
    # fwd, fp8 without the C1A emit set (KX_C1A=0 trainers)
    linear_relu_square(a, b, a_f8=a_e4, b_f8=b_e4, dequant_scale_ptr=one)
    # fwd, fp8 + C1A emit set -- mirrors FusedLinearReLUSquareFunction.forward
    linear_relu_square(a, b, a_f8=a_e4, b_f8=b_e4, dequant_scale_ptr=one,
                       emit_f8=True, post_scale=one, post_amax=amax_f,
                       emit_t=wg2, store_pre=not pdrop,
                       store_post_bf=store_post_bf,
                       partial_amax_mode=FLAGS["A2P"] and not pdrop)
    # bwd, bf16 aux, no C1B
    aux_bf = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    linear_relu_square(a, b, aux=aux_bf, post_scale=one)
    if pdrop:
        # bwd, sqrt-recon aux without C1B (C1A on, C1B off corner)
        aux_e4 = torch.zeros(M, N, device=dev, dtype=torch.float8_e4m3fn)
        linear_relu_square(a, b, aux=aux_e4, post_scale=one)
    # bwd, fp8 C1B (+ PDROP recon aux, + C1D/C1E dpre emits) -- mirrors .backward
    aux_bwd = (torch.zeros(M, N, device=dev, dtype=torch.float8_e4m3fn)
               if pdrop else aux_bf)
    kw = dict(aux=aux_bwd, a_f8=g_f8, b_f8=b_e4, dequant_scale_ptr=one,
              post_scale=one, grad_e4m3=FLAGS["GE4"],
              partial_amax_mode=FLAGS["A2P"] and not pdrop)
    if c1d or c1e:
        kw.update(emit_dpre=c1d, emit_dpre_t=c1e,
                  store_c_bf=not (c1d and c1e),
                  dpre_scale=one, dpre_amax=amax_b)
    linear_relu_square(a, b, **kw)
    torch.cuda.synchronize(dev)


# -----------------------------------------------------------------------------
# Dual-layout activation/grad quantize (PR#344 technique + explicit clamp).
# ONE read of the bf16 source, TWO fp8 writes (row-major + transposed) via a
# register-tile tl.trans. This is the only sanctioned way to obtain a
# transposed operand — standalone transpose passes measured +10.7s/run.

@triton.jit
def _quantize_dual_layout_kernel(
    src_ptr, row_ptr, t_ptr, scale_ptr, amax_ptr,
    M, N,
    src_stride_m, src_stride_n,
    t_stride_n,
    E4M3: tl.constexpr,
    EMIT_ROW: tl.constexpr,
    EMIT_T: tl.constexpr,
    EMIT_AMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    ).to(tl.float32)
    if EMIT_AMAX:
        # S3 delayed-scale support: single-slot atomic amax of the LOADED
        # values (the shipping non-A2P pattern; exact by construction).
        tl.atomic_max(amax_ptr, tl.max(tl.max(tl.abs(x), axis=1), axis=0))
    inv_s = 1.0 / tl.load(scale_ptr)
    if E4M3:
        q = tl.maximum(tl.minimum(x * inv_s, 448.0), -448.0).to(tl.float8e4nv)
    else:
        q = tl.maximum(tl.minimum(x * inv_s, 57344.0), -57344.0).to(tl.float8e5)
    if EMIT_ROW:
        tl.store(row_ptr + offs_m[:, None] * N + offs_n[None, :], q, mask=mask)
    if EMIT_T:
        mask_t = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(t_ptr + offs_n[:, None] * t_stride_n + offs_m[None, :],
                 tl.trans(q), mask=mask_t)


def quantize_dual_layout(src: torch.Tensor, scale: torch.Tensor,
                         fmt: torch.dtype = torch.float8_e5m2,
                         out_t: torch.Tensor | None = None,
                         emit_row: bool = True, emit_t: bool = True,
                         amax: torch.Tensor | None = None):
    """Quantize src [M, N] (bf16) to fp8: row-major [M, N] and/or transposed
    [N, M], reading src once. `scale` is a 0-D fp32 tensor. Original call form
    (src, scale, fmt) is unchanged bitwise (same arithmetic chain).

    S3 extensions (attention-backward fp8):
      out_t    PREALLOCATED transposed destination (a [N, M_max][:, :M] slice
               is fine: rows may be strided, last dim must be contiguous).
               Never allocate a [2304,T] temporary per step (+10.7s/run class).
      emit_row/emit_t  constexpr-gated stores (EMIT_ROW off = quantize_transposed).
      amax     optional 0-D fp32 slot: in-kernel atomic amax of the bf16 source
               (rides the read the kernel does anyway — delayed-scale refresh
               reads it post-step; no standalone amax reduction).
    """
    assert src.ndim == 2
    assert emit_row or emit_t
    M, N = src.shape
    row = torch.empty((M, N), device=src.device, dtype=fmt) if emit_row else None
    if emit_t:
        if out_t is None:
            t = torch.empty((N, M), device=src.device, dtype=fmt)
        else:
            assert out_t.shape == (N, M) and out_t.stride(1) == 1 and out_t.dtype == fmt
            t = out_t
    else:
        t = None
    BLOCK_M, BLOCK_N = (64, 128)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _quantize_dual_layout_kernel[grid](
        src,
        row if row is not None else src,
        t if t is not None else src,
        scale,
        amax if amax is not None else scale,
        M, N,
        src.stride(0), src.stride(1),
        t.stride(0) if t is not None else M,
        E4M3=fmt == torch.float8_e4m3fn,
        EMIT_ROW=row is not None,
        EMIT_T=t is not None,
        EMIT_AMAX=amax is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return row, t


def quantize_transposed(src: torch.Tensor, scale: torch.Tensor,
                        dst_t: torch.Tensor,
                        fmt: torch.dtype = torch.float8_e4m3fn,
                        amax: torch.Tensor | None = None):
    """One bf16 read -> ONE transposed fp8 write into a preallocated dst_t
    [N, M] (S3 §4.4). Optional in-kernel amax for delayed scales."""
    return quantize_dual_layout(src, scale, fmt=fmt, out_t=dst_t,
                                emit_row=False, emit_t=True, amax=amax)[1]


# -----------------------------------------------------------------------------
# Fused weight-cache quantize with lagged scales (PR#342 P1+P3, generalized to
# emit BOTH layouts). Replaces the aten amax pass + bf16 temp + non-coalesced
# copy_ in quantize_mlp_fp8: one coalesced read of the bank, fp8 row and/or
# transposed writes, per-tile amax emitted for the NEXT step's scale.

@triton.jit
def _quantize_weights_dual_kernel(
    w_ptr, row_ptr, col_ptr, scale_ptr, partial_amax_ptr,
    w_stride_l, w_stride_h, w_stride_d,
    H: tl.constexpr, D: tl.constexpr,
    num_tiles_d: tl.constexpr, num_tiles: tl.constexpr,
    EMIT_ROW: tl.constexpr, EMIT_COL: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    tile_h = tile // num_tiles_d
    tile_d = tile % num_tiles_d
    offs_h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)
    v = tl.load(
        w_ptr + layer * w_stride_l + offs_h[:, None] * w_stride_h
        + offs_d[None, :] * w_stride_d,
        mask=mask, other=0.0,
    ).to(tl.float32)
    s = tl.load(scale_ptr + layer)
    q = tl.maximum(tl.minimum(v / s, 448.0), -448.0).to(tl.float8e4nv)
    if EMIT_ROW:
        tl.store(row_ptr + layer * H * D + offs_h[:, None] * D + offs_d[None, :],
                 q, mask=mask)
    if EMIT_COL:
        mask_t = (offs_d[:, None] < D) & (offs_h[None, :] < H)
        tl.store(col_ptr + layer * H * D + offs_d[:, None] * H + offs_h[None, :],
                 tl.trans(q), mask=mask_t)
    tl.store(partial_amax_ptr + layer * num_tiles + tile,
             tl.max(tl.max(tl.abs(v), axis=1), axis=0))


def quantize_weights_dual_ntiles(H: int, D: int, block: int = 64) -> int:
    """Partial-amax slot count the trainer must allocate per layer."""
    return triton.cdiv(H, block) * triton.cdiv(D, block)


def quantize_mlp_weights_dual(bank: torch.Tensor, scales: torch.Tensor,
                              partial_amax: torch.Tensor,
                              row: torch.Tensor | None = None,
                              col_t: torch.Tensor | None = None,
                              headroom: float = 1.12,
                              update_scales: bool = True):
    """Quantize a weight bank [L, H, D] (bf16) into fp8 caches with a LAGGED
    per-layer scale.

    row    : [L, H, D] e4m3 contiguous (row-major cache), or None.
    col_t  : [L, D, H] e4m3 contiguous — pass the .transpose(1,2) view of a
             col-major cache built with the kernel-module idiom
             `zeros_like(bank).transpose(1,2).contiguous().transpose(1,2)`.
    scales : [L] fp32. When update_scales, refreshed FIRST from partial_amax
             (which holds LAST call's per-tile amaxes) with `headroom`; weights
             move <<1%/step under Muon so a one-step-stale scale + clamp is
             loss-neutral (PR#342 measured; bootstrap the first ~16 calls with
             exact aten scales and update_scales=False).
    partial_amax : [L, quantize_weights_dual_ntiles(H, D)] fp32.
    """
    L, H, D = bank.shape
    BLOCK_H = BLOCK_D = 64
    num_tiles_d = triton.cdiv(D, BLOCK_D)
    num_tiles = triton.cdiv(H, BLOCK_H) * num_tiles_d
    assert partial_amax.shape == (L, num_tiles), \
        f"partial_amax must be ({L}, {num_tiles}), got {tuple(partial_amax.shape)}"
    assert row is not None or col_t is not None
    if row is not None:
        assert row.shape == (L, H, D) and row.is_contiguous()
    if col_t is not None:
        assert col_t.shape == (L, D, H) and col_t.is_contiguous()
    if update_scales:
        torch.clamp(partial_amax.amax(dim=1) * (headroom / 448.0),
                    min=1e-12, out=scales)
    _quantize_weights_dual_kernel[(L, num_tiles)](
        bank,
        row if row is not None else bank,
        col_t if col_t is not None else bank,
        scales, partial_amax,
        bank.stride(0), bank.stride(1), bank.stride(2),
        H=H, D=D, num_tiles_d=num_tiles_d, num_tiles=num_tiles,
        EMIT_ROW=row is not None, EMIT_COL=col_t is not None,
        BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )


# -----------------------------------------------------------------------------
# FP8 attention projection ops (unchanged from kernel-module)

@torch.library.custom_op("nanogpt::af8s", mutates_args=())
def af8s_op(x: torch.Tensor, w_f8: torch.Tensor, w_bf: torch.Tensor,
            ws_t: torch.Tensor, x_s: float) -> torch.Tensor:
    """FP8 attention projection fwd, static scales: y = x @ w.T (lambda applied
    by the caller on the output, keeping its autograd path intact). w_f8 cached
    per step (no per-forward quantize); ws_t = per-layer weight scale (0-D fp32
    tensor, refreshed with the cache); x_s static (RMS-normed inputs).
    Backward bf16 via w_bf."""
    x_f8 = (x * (1.0 / x_s)).to(torch.float8_e4m3fn)
    return torch._scaled_mm(
        x_f8, w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=x.new_tensor(x_s, dtype=torch.float32),
        scale_b=ws_t,
        use_fast_accum=True,
    )

@af8s_op.register_fake
def _(x, w_f8, w_bf, ws_t, x_s):
    return x.new_empty((x.shape[0], w_f8.shape[0]), dtype=torch.bfloat16)

def _af8s_backward(ctx, grad_out):
    x, w_bf = ctx.saved_tensors
    g = grad_out.contiguous()
    grad_x = g @ w_bf.type_as(g)
    grad_w = g.T @ x.type_as(g)
    return grad_x, None, grad_w, None, None

def _af8s_setup(ctx, inputs, output):
    x, w_f8, w_bf, ws_t, x_s = inputs
    ctx.save_for_backward(x, w_bf)

af8s_op.register_autograd(_af8s_backward, setup_context=_af8s_setup)


@torch.library.custom_op("nanogpt::af8sd", mutates_args=())
def af8sd_op(x: torch.Tensor, w_f8: torch.Tensor, w_bf: torch.Tensor,
             ws_t: torch.Tensor) -> torch.Tensor:
    """FP8 attention projection fwd, DYNAMIC per-call input scale + static cached
    weight scale (o-site variant of af8s). The o-proj input y is unbounded (no
    norm between SDPA and o-proj) and float8_e4m3fn has no inf, so a static input
    scale turns outlier tokens into NaN; one amax reduction per call bounds it.
    Backward bf16 via w_bf."""
    xs = x.abs().amax().clamp_min(1e-6).float() / 448.0
    x_f8 = (x / xs).to(torch.float8_e4m3fn)
    return torch._scaled_mm(
        x_f8, w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=xs,
        scale_b=ws_t,
        use_fast_accum=True,
    )

@af8sd_op.register_fake
def _(x, w_f8, w_bf, ws_t):
    return x.new_empty((x.shape[0], w_f8.shape[0]), dtype=torch.bfloat16)

def _af8sd_backward(ctx, grad_out):
    x, w_bf = ctx.saved_tensors
    g = grad_out.contiguous()
    grad_x = g @ w_bf.type_as(g)
    grad_w = g.T @ x.type_as(g)
    return grad_x, None, grad_w, None

def _af8sd_setup(ctx, inputs, output):
    x, w_f8, w_bf, ws_t = inputs
    ctx.save_for_backward(x, w_bf)

af8sd_op.register_autograd(_af8sd_backward, setup_context=_af8sd_setup)


@torch.library.custom_op("nanogpt::af8", mutates_args=())
def af8_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP8 forward for attention projections: y = x @ w.T with dynamic per-tensor
    e4m3 scales (inputs are RMS-normed => stable range). Backward stays bf16."""
    xs = x.abs().amax().clamp_min(1e-6).float() / 448.0
    ws = w.abs().amax().clamp_min(1e-6).float() / 448.0
    x_f8 = (x / xs).to(torch.float8_e4m3fn)
    w_f8 = (w / ws).to(torch.float8_e4m3fn)
    return torch._scaled_mm(
        x_f8, w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=xs, scale_b=ws,
        use_fast_accum=True,
    )

@af8_op.register_fake
def _(x, w):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)

def _af8_backward(ctx, grad_out):
    x, w = ctx.saved_tensors
    g = grad_out.contiguous()
    grad_x = g @ w.to(g.dtype)
    grad_w = g.T @ x.to(g.dtype)
    return grad_x, grad_w

def _af8_setup(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)

af8_op.register_autograd(_af8_backward, setup_context=_af8_setup)

# -----------------------------------------------------------------------------
# FP8 MLP scaled_mm wrappers (opaque to inductor — house mm_t pattern).
# fp32 accumulation (use_fast_accum=False) on EVERY gradient GEMM: NaN risk otherwise.

@torch.library.custom_op("nanogpt::wg_f8", mutates_args=())
def wg_f8_op(post_f8: torch.Tensor, g_f8: torch.Tensor, post_scale: torch.Tensor,
             grad_s: float) -> torch.Tensor:
    """LEGACY C1C diagnostic (do NOT ship): dW2 via standalone .contiguous()
    transposes. The tiled-transpose variant of this measured +10.7s
    per run — layout must come from a producing kernel's epilogue instead
    (KX_WG2). Kept for diagnostics only."""
    return torch._scaled_mm(
        post_f8.T.contiguous(),
        g_f8.T.contiguous().T,
        out_dtype=torch.bfloat16,
        scale_a=post_scale,
        scale_b=post_f8.new_tensor(grad_s, dtype=torch.float32),
        use_fast_accum=False,
    )

@wg_f8_op.register_fake
def _(post_f8, g_f8, post_scale, grad_s):
    return post_f8.new_empty((post_f8.shape[1], g_f8.shape[1]), dtype=torch.bfloat16)

@torch.library.custom_op("nanogpt::dp_f8", mutates_args=())
def dp_f8_op(post_f8: torch.Tensor, w2_f8: torch.Tensor, post_scale: torch.Tensor,
             w2_scale: torch.Tensor) -> torch.Tensor:
    """FP8 down-projection forward. post_f8 is emitted by the fused MLP kernel
    epilogue (delayed per-layer scaling); this op is just the scaled matmul,
    wrapped so inductor treats it as opaque (same pattern as nanogpt::mm_t)."""
    return torch._scaled_mm(
        post_f8, w2_f8,
        out_dtype=torch.bfloat16,
        scale_a=post_scale,
        scale_b=w2_scale,
        use_fast_accum=True,
    )

@dp_f8_op.register_fake
def _(post_f8, w2_f8, post_scale, w2_scale):
    return post_f8.new_empty((post_f8.shape[0], w2_f8.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("nanogpt::wg2_f8", mutates_args=())
def wg2_f8_op(post_t_f8: torch.Tensor, g_t_f8: torch.Tensor,
              post_scale: torch.Tensor, g_scale: torch.Tensor) -> torch.Tensor:
    """KX_WG2: dW2 = post^T @ g, both operands ALREADY in wgrad layout:
    post_t_f8 [3072, T] e4m3 row-major (forward-epilogue emit, K contiguous),
    g_t_f8    [768, T]  fp8 row-major (dual-layout grad quantize); its .T is a
    zero-copy column-major [T, 768] view — the TN pair Hopper FP8 WGMMA needs.
    Output [3072, 768] bf16, matching mlp_bank grad dtype (NOT fp32:
    that would double reduce_scatter volume)."""
    return torch._scaled_mm(
        post_t_f8, g_t_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=post_scale,
        scale_b=g_scale,
        use_fast_accum=False,
    )

@wg2_f8_op.register_fake
def _(post_t_f8, g_t_f8, post_scale, g_scale):
    return post_t_f8.new_empty((post_t_f8.shape[0], g_t_f8.shape[0]),
                               dtype=torch.bfloat16)


@torch.library.custom_op("nanogpt::wg1_f8", mutates_args=())
def wg1_f8_op(dpre_t_f8: torch.Tensor, x_t_f8: torch.Tensor,
              dpre_scale: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """KX_C1E: dW1 = dpre^T @ x. dpre_t_f8 [3072, T] (backward-epilogue emit,
    e5m2 or e4m3 under GE4), x_t_f8 [768, T] e4m3 (forward dual-layout quantize
    of normed — the same quantization the up-projection already commits to)."""
    return torch._scaled_mm(
        dpre_t_f8, x_t_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=dpre_scale,
        scale_b=x_scale,
        use_fast_accum=False,
    )

@wg1_f8_op.register_fake
def _(dpre_t_f8, x_t_f8, dpre_scale, x_scale):
    return dpre_t_f8.new_empty((dpre_t_f8.shape[0], x_t_f8.shape[0]),
                               dtype=torch.bfloat16)


@torch.library.custom_op("nanogpt::dx_f8", mutates_args=())
def dx_f8_op(dpre_f8: torch.Tensor, w1_f8_col: torch.Tensor,
             dpre_scale: torch.Tensor, w1_scale: torch.Tensor) -> torch.Tensor:
    """KX_C1D: dx = dpre @ W1. dpre_f8 [T, 3072] row-major (backward-epilogue
    emit), w1_f8_col [3072, 768] column-major cache (refreshed post-step, off
    the critical path; shares the up-projection's per-layer scale)."""
    return torch._scaled_mm(
        dpre_f8, w1_f8_col,
        out_dtype=torch.bfloat16,
        scale_a=dpre_scale,
        scale_b=w1_scale,
        use_fast_accum=False,
    )

@dx_f8_op.register_fake
def _(dpre_f8, w1_f8_col, dpre_scale, w1_scale):
    return dpre_f8.new_empty((dpre_f8.shape[0], w1_f8_col.shape[1]),
                             dtype=torch.bfloat16)


# -----------------------------------------------------------------------------
# S3: attention projections — bf16 FORWARD, fp8 BACKWARD GEMMs only.
# ZERO new custom ops: dx_qkv is C1D's
# nanogpt::dx_f8 verbatim (K=2304, the shape class that shipped; the
# K=768 FORWARD stays bf16 here on purpose); dW_qkv is C1E's
# nanogpt::wg1_f8 verbatim (K=T, the splitK family).

class AttnProjBwdFP8(torch.autograd.Function):
    """bf16 forward y = x @ w_eff.T (identical dispatch to the F.linear it
    replaces — nothing in the forward graph changes); fp8 backward GEMMs.
    w_eff is the bf16 lambda*W tensor: returning its grad keeps the existing
    pointwise chain to sa_lambdas and the banks bit-identical.

    Args (all trailing optional; the trainer passes what its KX_S3* need):
      x        [T, in] bf16       w_eff [out, in] bf16 (= lambda * bank rows)
      w_col_f8 [out, in] e4m3 COL-major cache of lambda*W (post-step refresh,
               lambda folded: sign in values, scale positive)   <- KX_S3D
      w_cs     0-D f32 per-layer scale of that cache
      gs_t     0-D f32 static e5m2 scale for the dgrad g row (KX_S3_GS)
      g_s_inv  python float 1/KX_S3_GS (compile-time constant for the fused
               cast — never x.new_tensor per call, the C2S H2D defect)
      x_t_f8   [in, T] fp8 activation operand in wgrad layout, PRE-FILLED by
               the trainer's forward-side quantize_transposed <- KX_S3W
      x_s      0-D f32 its scale (static 2^-4 for attn_in_normed — clip-free
               by the sqrt(768) proof; delayed per-layer amax for o's y)
      g_scale  0-D f32 wgrad-g scale (== gs_t when e5m2; delayed per-layer
               buffer element under KX_S3_GE4)
      g_amax   0-D f32 amax slot for the delayed refresh (GE4 only)
      gt_buf   [out, T] fp8 PREALLOCATED transposed-g destination (a
               [out, T_max][:, :T] slice)   <- KX_S3W
      ge4      bool: wgrad g in e4m3 at the delayed scale (default ON in the
               trainer: dW_qkv's v-rows feed the vo weight bank)
      route_b  bool (KX_S3W==2): DCE-independent fallback — one dual-layout
               quantize of bf16 g feeds BOTH backward GEMMs.
    """

    @staticmethod
    def forward(ctx, x, w_eff, w_col_f8=None, w_cs=None, gs_t=None, g_s_inv=1.0,
                x_t_f8=None, x_s=None, g_scale=None, g_amax=None, gt_buf=None,
                ge4=False, route_b=False):
        assert not (route_b and gt_buf is None), "route B is an KX_S3W mode"
        y = x @ w_eff.T                      # bf16, traced, NOT wrapped
        ctx.save_for_backward(x, w_eff)
        # Non-differentiable operands (caches/buffers/scales) are ctx-stashed —
        # the shipping FusedLinearReLUSquareFunction pattern for graph INPUTS
        # (the defaults-regression lesson only bans stashing INTERMEDIATES).
        ctx.w_col_f8 = w_col_f8
        ctx.w_cs = w_cs
        ctx.gs_t = gs_t
        ctx.g_s_inv = g_s_inv
        ctx.x_t_f8 = x_t_f8
        ctx.x_s = x_s
        ctx.g_scale = g_scale
        ctx.g_amax = g_amax
        ctx.gt_buf = gt_buf
        ctx.ge4 = ge4
        ctx.route_b = route_b
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w_eff = ctx.saved_tensors
        g = grad_out.contiguous()
        s3d = ctx.w_col_f8 is not None
        s3w = ctx.gt_buf is not None

        g_f8 = None
        row_scale = ctx.gs_t
        if s3d and not ctx.route_b:
                # (1) row-major e5m2 dgrad operand: PURE ATEN POINTWISE —
                # inductor fuses it into the kernel producing g (FA3-bwd ->
                # rotary/qk-norm/cat chain). Explicit clamp before the cast
                # (aten fp32->fp8 is non-saturating). Deploy check: kernel census
                # must show NO standalone cast kernel.
            g_f8 = (g * ctx.g_s_inv).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)

        # ---- dW = g^T @ x  (K=T) -------------------------------------------
        if s3w:
            if ctx.route_b:
                # route B (KX_S3W=2): no DCE dependency, no fused cast — one
                # dual-layout quantize of bf16 g feeds both GEMMs.
                fmt = torch.float8_e4m3fn if ctx.ge4 else torch.float8_e5m2
                g_f8, _ = quantize_dual_layout(g, ctx.g_scale, fmt=fmt,
                                               out_t=ctx.gt_buf, amax=ctx.g_amax)
            elif ctx.ge4 or g_f8 is None:
                # e4m3 wgrad operand at the delayed per-layer scale (or the
                # o-site / S3W-without-S3D case): quantize-transpose the bf16
                # g; the amax for next step's scale rides the read in-kernel.
                fmt = torch.float8_e4m3fn if ctx.ge4 else torch.float8_e5m2
                quantize_transposed(g, ctx.g_scale, ctx.gt_buf, fmt=fmt,
                                    amax=ctx.g_amax)
            else:
                # e5m2 twin of the dx operand: fp8->fp8 transpose into the
                # preallocated buffer. With S3D on, this removes the last bf16
                # reader of g and inductor DCEs the 226 MB/layer bf16 store
                # (S3 §3.3; verify offline with gate G2 before any cluster run).
                transpose_copy(g_f8, ctx.gt_buf)
            dW = torch.ops.nanogpt.wg1_f8(ctx.gt_buf, ctx.x_t_f8,
                                          ctx.g_scale, ctx.x_s)
        else:
            dW = g.T @ x                     # bf16, unchanged (keeps g alive)

        # ---- dx = g @ w_eff  (K=2304) --------------------------------------
        if s3d:
            if ctx.route_b:
                row_scale = ctx.g_scale
            dx = torch.ops.nanogpt.dx_f8(g_f8, ctx.w_col_f8, row_scale, ctx.w_cs)
        else:
            dx = g @ w_eff                   # bf16 (o-site: K=768 fp8 measured slower)

        return (dx, dW) + (None,) * 11


class FusedLinearReLUSquareFunction(torch.autograd.Function):
    """kernel-module's fused MLP autograd function extended to the full-FP8 MLP package.

    Positional arg layout (superset of kernel-module's 13; all trailing args optional —
    the trainer passes exactly what its configuration flags require):
       0 x        [T, 768] bf16 (normed residual)
       1 W1       [3072, 768] bf16      2 W2  [3072, 768] bf16
       3 W1_f8    e4m3 row cache        4 dequant_scale (x_s*w1_s, 0-D f32)
       5 x_f8     [T, 768] e4m3
       6 W2_f8    e4m3 col-major        7 w2_scale (0-D)           <- KX_C1A
       8 post_scale (0-D, delayed)      9 post_amax ([1] / [NUM_SMS] slot)
      10 w2_f8_row e4m3 row-major      11 grad_s (float, static)   <- KX_C1B
      12 dq_bwd   (0-D: w2_s*grad_s; under KX_GE4: w2_s alone)
      13 x_f8_t   [768, T] e4m3        14 x_scale (0-D, amax/448)  <- KX_C1E
      15 w1_f8_col e4m3 col-major      16 w1_scale (0-D)           <- KX_C1D
      17 g_scale  (0-D f32 = KX_GS; superseded by dynamic scale under KX_GE4)
      18 dpre_scale (0-D, delayed)     19 dpre_amax (slot)         <- KX_C1D/E
    """

    @staticmethod
    def forward(ctx, x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None,
                W2_f8=None, w2_scale=None, post_scale=None, post_amax=None,
                w2_f8_row=None, grad_s=None, dq_bwd=None,
                x_f8_t=None, x_scale=None, w1_f8_col=None, w1_scale=None,
                g_scale=None, dpre_scale=None, dpre_amax=None):
        x_flat = x.view((-1, x.shape[-1]))
        emit = W2_f8 is not None
        wg2 = FLAGS["WG2"] and emit and w2_f8_row is not None
        pdrop = FLAGS["PDROP"] and emit
        # bf16 post's only consumer is a bf16 dW2 (the down-proj fwd reads
        # post_f8 under C1A): once dW2 is FP8 the store is dead.
        # C1C's fp8 dW2 additionally needs the C1B grad cast (w2_f8_row).
        store_post_bf = not (emit and (wg2 or (FLAGS["C1C"] and w2_f8_row is not None)))
        if W1_f8 is not None:
            assert x_f8 is not None and dequant_scale is not None
            x_f8v = x_f8.view((-1, x_f8.shape[-1]))
            pre, post, post_f8, post_t = linear_relu_square(
                x_flat, W1, a_f8=x_f8v, b_f8=W1_f8,
                dequant_scale_ptr=dequant_scale,
                emit_f8=emit, post_scale=post_scale, post_amax=post_amax,
                emit_t=wg2, store_pre=not pdrop, store_post_bf=store_post_bf,
                # PDROP forces the shipping single-slot atomic amax path
                # (measured: A2P+PDROP -> mid-run NaN; each alone
                # clean. The combo was the only fwd kernel variant never
                # compiled by any gate test: EMIT_F8=1, STORE_PRE=0,
                # STORE_POST_BF=1, PARTIAL_AMAX=1. Scale algebra audited
                # clean, so the hazard class is variant-specific codegen;
                # keep PDROP off the PARTIAL_AMAX variants until root-caused
                # on GPU. Atomic into slot 0 of the [NUM_SMS] row is fully
                # compatible with the trainer's .amax(dim=1) refresh — no
                # trainer change. Costs only P2 (~0.05-0.15 ms) under PDROP;
                # P1 (the measured 0.9 ms/step weight kernel) is unaffected.)
                partial_amax_mode=FLAGS["A2P"] and emit and not pdrop)
        else:
            pre, post, post_f8, post_t = linear_relu_square(
                x_flat, W1, emit_f8=emit,
                post_scale=post_scale, post_amax=post_amax,
                partial_amax_mode=FLAGS["A2P"] and emit and not pdrop)
        if emit:
            x3 = torch.ops.nanogpt.dp_f8(post_f8, W2_f8, post_scale, w2_scale)
        else:
            x3 = post @ W2
        # Saved-tensor contract: IDENTICAL to kernel-module at defaults —
        # save_for_backward(x, W1, W2, pre, post) — with pre/post dropped only
        # under the variant flags that delete those tensors. Defaults-regression
        # discipline (measured +3.3mb without it): plain ctx-attr
        # stashing of graph INTERMEDIATES (pre/post) was one of only two
        # structural deltas from kernel-module inside the compiled autograd-Function HOP;
        # extras that kernel-module itself ctx-stashes (post_f8, post_scale, ...) stay
        # ctx-stashed, matching kernel-module's layout exactly. gate_tier0.py asserts
        # bitwise kernel-module-equivalence eager AND compiled before any full-run screen.
        saved = [x, W1, W2]
        if pre is not None:
            saved.append(pre)
        if post is not None:
            saved.append(post)
        ctx.save_for_backward(*saved)
        ctx.saved_layout = (pre is not None, post is not None)
        ctx.post_f8 = post_f8 if emit else None
        ctx.post_t = post_t
        ctx.post_scale = post_scale
        ctx.w2_f8_row = w2_f8_row
        ctx.grad_s = grad_s
        ctx.dq_bwd = dq_bwd
        ctx.x_f8_t = x_f8_t
        ctx.x_scale = x_scale
        ctx.w1_f8_col = w1_f8_col
        ctx.w1_scale = w1_scale
        ctx.g_scale = g_scale
        ctx.dpre_scale = dpre_scale
        ctx.dpre_amax = dpre_amax
        ctx.pdrop = pdrop
        ctx.wg2 = wg2
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Dynamo-robust unpack (47c_t0gate crash post-mortem): no star-unpack
        # of saved_tensors, no bool-as-index — constant-bool branches with
        # literal integer indices only. ctx.saved_layout is trace-time-constant
        # Python metadata (same proven pattern as ctx.pdrop/ctx.wg2).
        st = ctx.saved_tensors
        x, W1, W2 = st[0], st[1], st[2]
        has_pre, has_post = ctx.saved_layout
        if has_pre and has_post:      # defaults / bf16 / C1A-without-fp8-dW2
            pre, post = st[3], st[4]
        elif has_pre:                 # WG2 (bf16 post dead), no PDROP
            pre, post = st[3], None
        elif has_post:                # PDROP without WG2 (pre dead)
            pre, post = None, st[3]
        else:                         # PDROP + WG2 (both dead)
            pre = post = None
        g_flat = grad_output.view((-1, grad_output.shape[-1]))
        c1b = ctx.w2_f8_row is not None
        wg2 = ctx.wg2
        c1e = FLAGS["C1E"] and c1b and ctx.x_f8_t is not None and ctx.dpre_scale is not None
        c1d = FLAGS["C1D"] and c1b and ctx.w1_f8_col is not None and ctx.dpre_scale is not None
        ge4 = FLAGS["GE4"]

        # ---- 1. incoming-grad quantize (one pass; dual layout only for WG2) --
        g_f8 = g_f8_t = None
        dq = ctx.dq_bwd
        g_scale_t = ctx.g_scale
        if c1b:
            if ge4:
                # e4m3 grads need a real scale: dynamic per-tensor amax (g is
                # only [T, 768] — the reduce is cheap). dq_bwd carries w2_scale
                # alone under GE4; multiply the grad scale back in here.
                g_scale_t = (g_flat.detach().abs().amax().float()
                             .clamp_min(1e-12) / E4M3_MAX_F).reshape(())
                dq = ctx.dq_bwd * g_scale_t
                fmt = torch.float8_e4m3fn
            else:
                fmt = torch.float8_e5m2
            if wg2:
                if g_scale_t is None:
                    g_scale_t = g_flat.new_tensor(ctx.grad_s, dtype=torch.float32)
                g_f8, g_f8_t = quantize_dual_layout(g_flat, g_scale_t, fmt=fmt)
            elif ge4:
                g_f8 = torch.clamp(g_flat.float() / g_scale_t,
                                   -E4M3_MAX_F, E4M3_MAX_F).to(fmt)
            else:
                # kernel-module's C1B cast, unchanged (static e5m2 scale)
                g_f8 = (g_flat / ctx.grad_s).to(torch.float8_e5m2)

        # ---- 2. dW2 --------------------------------------------------------
        if wg2:
            dW2 = torch.ops.nanogpt.wg2_f8(ctx.post_t, g_f8_t,
                                           ctx.post_scale, g_scale_t)
        else:
            dW2 = post.T @ grad_output  # kernel-module-exact default expression (unflattened)

        # ---- 3. dpre (+ fp8 emits from the same kernel's epilogue) ----------
        # Under PDROP the aux the backward reads IS the fp8 post (it has to
        # read *some* aux; the fp8 one is half the bytes of the bf16 pre).
        aux_src = ctx.post_f8 if ctx.pdrop else pre
        if c1b:
            dpre, dpre_f8, dpre_t = linear_relu_square(
                g_flat, W2, aux=aux_src, a_f8=g_f8, b_f8=ctx.w2_f8_row,
                dequant_scale_ptr=dq, post_scale=ctx.post_scale,
                emit_dpre=c1d, emit_dpre_t=c1e,
                store_c_bf=not (c1d and c1e),
                dpre_scale=ctx.dpre_scale, dpre_amax=ctx.dpre_amax,
                grad_e4m3=ge4,
                partial_amax_mode=FLAGS["A2P"] and not ctx.pdrop)
        else:
            dpre, dpre_f8, dpre_t = linear_relu_square(
                g_flat, W2, aux=aux_src, post_scale=ctx.post_scale)

        # ---- 4. dW1 ----------------------------------------------------------
        if c1e:
            dW1 = torch.ops.nanogpt.wg1_f8(dpre_t, ctx.x_f8_t,
                                           ctx.dpre_scale, ctx.x_scale)
        else:
            dW1 = dpre.T @ x  # kernel-module-exact default expression (unflattened)

        # ---- 5. dx -----------------------------------------------------------
        if c1d:
            dx = torch.ops.nanogpt.dx_f8(dpre_f8, ctx.w1_f8_col,
                                         ctx.dpre_scale, ctx.w1_scale)
        else:
            dx = dpre @ W1

        return (dx.view(x.shape), dW1, dW2) + (None,) * 18


# -----------------------------------------------------------------------------
# Tiled transpose copy kernel: dst (N, M) = src (M, N).T
# Uses coalesced reads from src and coalesced writes to dst via tl.trans().
# NOTE: even THIS bandwidth-optimal standalone pass
# measured +10.7 s/run when put on the backward critical path for the MLP
# wgrads. It stays here for the lm_head path (where the record itself already
# pays this bill) — do not add new critical-path call sites; emit layouts from
# producing kernels instead.

@triton.jit
def _transpose_copy_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced write to dst (N, M): dst[n, m] = src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile), mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced reads AND writes,
    achieving near memory-bandwidth-limited performance.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = (64, 128)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Tiled transpose-add kernel: dst (M, N) += src (N, M).T  (unchanged from kernel-module)

@triton.jit
def _transpose_add_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T where src is (M, N) and dst is (N, M)."""
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = (32, 32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_add_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Fused softcapped cross-entropy (unchanged from kernel-module, incl. C7 lm_head FP8 reuse)

CE_KERNEL_BLOCK_SIZE = 256
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """
#include <cuda_bf16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2(float x) {
    uint16_t packed;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
        : "=h"(packed)
        : "f"(x), "f"(0.0f)
    );
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (packed & (0xFF << 8)) >> 8;
    return result;
}

struct __align__(16) __nv_bfloat168 {
    __nv_bfloat16 data[8];
    __device__ __nv_bfloat16& operator[](int i) { return data[i]; }
    __device__ const __nv_bfloat16& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

extern "C"
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    const int64_t* __restrict__ prefix_targets,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param,
    double grad_scale_param,
    double prefix_weight_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __nv_bfloat16 smem[];

  static_assert(VEC_WIDTH == 8);

  const __nv_bfloat16 *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  float thread_max = -CUDART_INF_F;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 result = *(__nv_bfloat168*)(&block_logit_ptr[idx]);
      __nv_bfloat168 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = __bfloat162float(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2bfloat16(tmp);
        tmp = A * tmp;
        thread_max = max(tmp, thread_max);
      }
      *(__nv_bfloat168*)(&smem[idx]) = result_sigmoid;
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_maxs[NUM_WARPS];
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

  if (threadIdx.x % 32 == 0) {
    block_maxs[warp_id] = thread_max;
  }

  __syncthreads();

  float block_max = -CUDART_INF_F;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_max = fmaxf(block_max, block_maxs[i]);
  }

  float thread_sum = 0.0f;
  #pragma unroll 2
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_bfloat168 l;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      l = *(__nv_bfloat168*)(&smem[idx]);
    }
    #pragma unroll
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = A * __bfloat162float(l[k]);
      tmp = __expf(tmp - block_max);
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_sum += tmp;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = block_max + __logf(block_sum);

  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __bfloat162float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    // KX_PTP: auxiliary prefix-token CE (PR#337 port); ptgt=-1 or pw=0 => exact no-op
    {
      int64_t ptgt = prefix_targets[blockIdx.x];
      if (ptgt >= 0 && ptgt < VOCAB_SIZE) {
        float z_p = A * __bfloat162float(smem[ptgt]);
        total_loss += (float)prefix_weight_param * (lse - z_p);
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }
  // KX_PTP: this row's softmax carries the prefix CE too (pw=0 or ptgt<0 => +0.0f, bit-identical)
  int64_t ptgt_row = prefix_targets[blockIdx.x];
  if (ptgt_row >= 0) {
    S_w += (float)prefix_weight_param;
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 sigmoid_us = *(__nv_bfloat168*)(&smem[idx]);
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j++) {
        float sigmoid_u = __bfloat162float(sigmoid_us[j]);
        float z = A * sigmoid_u;
        float p = __expf(z - lse);

        float term1 = S_w * p;
        float term2 = 0.0f;

        float grad_z = term1 - term2;
        float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        auto result_tmp = f32_to_fp8_e5m2(grad_x);
        result[j] = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      }
      *(__nv_fp8_e5m28*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = result;
    }
  }

  __syncthreads();

  // KX_PTP: thread n_predict fixes up the prefix-target entry; every fixup thread's
  // term2 includes the prefix contribution so duplicate-entry writes stay identical.
  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && ((float)prefix_weight_param > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];

    float sigmoid_u = __bfloat162float(smem[target]);
    float z = A * sigmoid_u;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }
    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += (float)prefix_weight_param;
    }

    float grad_z = term1 - term2;
    float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
    auto result_tmp = f32_to_fp8_e5m2(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""

ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
    "ce_fwd_bwd_kernel",
    compute_capability="90",
    cuda_include_dirs=["/usr/local/cuda/include/"],
    nvcc_options=["-lineinfo", "--use_fast_math"],
)
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)

_PTP_DUMMY = None  # KX_PTP: all(-1) prefix-target dummy for the default path — allocated ONCE via ptp_init() at startup; never inside compiled code
_PTP_W_RUNTIME = None  # KX_PTP: live prefix weight, read EAGERLY inside the opaque custom op so the ramp never enters a compiled graph (a python-float weight in the graph forces a full recompile at every stage flip — ~12s each on-clock on fresh caches). Trainer sets this; graph passes a constant sentinel.

def ptp_init(device, max_rows: int = 131072):
    """Eager allocation of the default-path prefix dummy (dynamo must never see a
    lazy alloc inside the compiled forward). Called from trainer module scope."""
    global _PTP_DUMMY
    if _PTP_DUMMY is None or _PTP_DUMMY.device != device or _PTP_DUMMY.numel() < max_rows:
        _PTP_DUMMY = torch.full((max_rows,), -1, dtype=torch.int64, device=device)

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_targets: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
    grad_scale: float,
    prefix_weight: float,
) -> None:
    if _PTP_W_RUNTIME is not None and prefix_weight > 0.0:
        prefix_weight = _PTP_W_RUNTIME
    grid = (n_rows, 1, 1)
    ce_fwd_bwd_kernel(
        grid,
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, prefix_targets, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s, grad_scale, prefix_weight),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5, w_f8_in=None, prefix_targets=None, prefix_weight=0.0, w_f8_row=None):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        if w_f8_in is not None:
            # FP8 copy refreshed post-step by the trainer (C7): quantize cost moves
            # off the serial forward path into the comm-overlapped optimizer phase.
            w_f8 = w_f8_in
        else:
            w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        # KX_PTP: default path slices the startup-allocated dummy (no alloc, no branch
        # on tensor state — dynamo-safe); ptp_init() guarantees capacity 131072 rows
        if prefix_targets is None:
            assert n_rows <= _PTP_DUMMY.numel(), "ptp_init max_rows too small for this batch"
            prefix_targets = _PTP_DUMMY[:n_rows]
            prefix_weight = 0.0
        else:
            prefix_targets = prefix_targets.contiguous()
        ce_fwd_bwd(logits, targets, mtp_weights, prefix_targets, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale, float(prefix_weight))

        # KX_LMF8T: w_f8_in is the col-major cache (zero-copy forward view above);
        # backward needs the row-major layout (w_f8.T must be a col-major view for
        # _scaled_mm's mat2) — the trainer passes it as w_f8_row.
        w_f8_bwd = w_f8_row if w_f8_row is not None else w_f8
        if _R2CE:
            ctx.save_for_backward(mtp_weights, x_f8, w_f8_bwd, grad_input)
        else:
            ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8_bwd, grad_input)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        # grad-count parity: the trainer omits w_f8_row entirely when LMF8T is off, so
        # backward must return exactly as many grads as forward received args.
        ctx._has_w15 = w_f8_row is not None
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        if _R2CE:
            mtp_weights, x_f8, w_f8, grad_input = ctx.saved_tensors
            n_rows, n_cols = grad_input.shape
        else:
            logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input = ctx.saved_tensors
            n_rows, n_cols = logits.shape
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_predict = mtp_weights.shape[0]

        grad_output = grad_output.contiguous()

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)  # (50304, n_rows) row-major

        grad_w = torch._scaled_mm(
            x_f8_T,            # (768, n_rows) row-major
            grad_input_T.T,    # (n_rows, 50304) column-major view
            out_dtype=(torch.bfloat16 if _LHBW16 else torch.float32),
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        # one slot per forward arg: x, targets, mtp_weights, lm_head_weight,
        # x_s, w_s, grad_s, grad_scale, A, B, C, w_f8_in, prefix_targets, prefix_weight[, w_f8_row]
        if ctx._has_w15:
            return grad_x, None, None, grad_w, None, None, None, None, None, None, None, None, None, None, None
        return grad_x, None, None, grad_w, None, None, None, None, None, None, None, None, None, None

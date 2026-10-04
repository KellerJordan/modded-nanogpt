# triton_kernels.py -- the full-FP8 MLP + fused-CE kernels.  Owns the "nanogpt" custom-op namespace: NEVER import two
# copies in one process.  Dual-layout rule: every transposed fp8 operand is written by the kernel that already holds
# the tile in registers, never by a standalone transpose pass (even a bandwidth-optimal one measured +10.7 s/run).
# use_fast_accum=False on EVERY gradient GEMM (NaN otherwise), and an explicit clamp before every fp8 cast.


import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    from triton.runtime.errors import OutOfResources as _TritonOOR
except Exception:  # triton version drift
    class _TritonOOR(Exception):
        pass


# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

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
    # Load A[m, k] -> shape (BM, BK)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # Load A[n, k] -> shape (BN, BK). Transpose to get (BK, BN) for accumulation.
    # Loading (BN, BK) is coalesced because stride_c is 1 (contiguous dim is k).
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
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 8
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

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
# Triton kernel for X.T @ X (tall matrices)
# Computes C = A.T @ A where A is (M, K) and output C is (K, K)

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
    This is the transpose variant of XXT for tall matrices.
    
    The output matrix C is symmetric, so we compute upper triangle and mirror.
    We iterate over blocks of M (the reduction dimension after transpose).
    """
    pid = tl.program_id(axis=0)
    # Note: Output is (K, K), so we use K for the output grid
    batch_idx, k_idx, n_idx = _pid_to_block(
        pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed (symmetry optimization)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # For A.T @ A:
    # - A.T has shape (K, M), so A.T[k, m] = A[m, k]
    # - We load blocks from columns k_idx and n_idx of A (which are rows of A.T)
    # - We reduce over M (the shared dimension)
    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K  # Output row indices (columns of A)
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K  # Output col indices (columns of A)
    offs_m = tl.arange(0, BLOCK_SIZE_K)  # Reduction dimension (rows of A)

    # Pointers for loading A[:, k_idx:k_idx+BLOCK] (transposed view is A.T[k_idx:, :])
    # at_ptrs loads A.T block: A.T[offs_k, offs_m] = A[offs_m, offs_k]
    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # a_ptrs loads A block for the other factor: A.T[offs_m, offs_n].T = A[offs_m, offs_n]
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of M (the reduction dimension)
    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        # Load A.T[offs_k, offs_m] = A[offs_m, offs_k] -> shape (BLOCK_K, BLOCK_M)
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # Load A[offs_m, offs_n] -> shape (BLOCK_K, BLOCK_N)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # C[k, n] = sum_m A.T[k, m] * A[m, n] = sum_m A[m, k] * A[m, n]
        # at.T @ a: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal (symmetry)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XTX(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A.T @ A
    
    For tall matrices (M > K), this is more efficient than transposing
    and using XXT because the intermediate products are smaller (K x K vs M x M).
    
    Args:
        A: Input tensor of shape (M, K) or (batch, M, K)
        out: Output tensor of shape (K, K) or (batch, K, K)
    
    Returns:
        out: The same output tensor, filled with A.T @ A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"
    assert out.size(-1) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 8
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

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
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
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
    
    # Coalesced loads similar to XXT_kernel
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

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

    # Hardcoded config based on H100 autotuning (M=768)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    num_stages, num_warps = 4, 8

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
# Triton kernel for MLP: relu(x @ W1.T)^2, by @andrewbriand, @jrauvola
# Extension: dual-layout fp8 epilogue emission on BOTH passes.  The constexpr EMIT_*/STORE_*/RECON_SQRT flags select
# the epilogue, dead-code-eliminated when off; each transposed store is a register-tile tl.trans, so it costs writes
# only.  EMIT_T is the fp8 complement of STORE_POST_BF, EMIT_DPRE of the bf16 dpre store.

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 dequant_scale_ptr,
                                 M, N, K,
                                 aux_f8_desc, post_scale_ptr, post_amax_ptr, aux_t_desc,
                                 c_f8_desc, c_t_desc, dpre_scale_ptr, dpre_amax_ptr, aux_w_desc,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 USE_FP8: tl.constexpr,
                                 EMIT_F8: tl.constexpr, EMIT_T: tl.constexpr, STORE_PRE: tl.constexpr,
                                 STORE_POST_BF: tl.constexpr, RECON_SQRT: tl.constexpr, EMIT_DPRE: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # Epilogue scalars load once per CTA, ABOVE the tile loop: no desc.store target ever aliases them.
    if USE_FP8:
        dq_h = tl.load(dequant_scale_ptr)
    else:
        dq_h = 1.0
    if EMIT_F8:
        inv_ps_h = 1.0 / tl.load(post_scale_ptr)
    else:
        inv_ps_h = 1.0
    if RECON_SQRT:
        ps_h = tl.load(post_scale_ptr)
    else:
        ps_h = 1.0
    if EMIT_DPRE:
        inv_ds_h = 1.0 / tl.load(dpre_scale_ptr)
    else:
        inv_ds_h = 1.0

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
            accumulator *= dq_h

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        if FORWARD:
            c0 = acc0.to(dtype)                      # pre (bf16-rounded)
            if STORE_PRE:
                c_desc.store([offs_am, offs_bn], c0)
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            if STORE_POST_BF:
                aux_desc.store([offs_am, offs_bn], c0_post)
            if EMIT_F8:
                q0 = tl.minimum(c0_post.to(tl.float32) * inv_ps_h, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am, offs_bn], q0)
                if EMIT_T:
                    aux_t_desc.store([offs_bn, offs_am], tl.trans(q0))
            c1 = acc1.to(dtype)
            if STORE_PRE:
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            if STORE_POST_BF:
                aux_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1_post)
            if EMIT_F8:
                q1 = tl.minimum(c1_post.to(tl.float32) * inv_ps_h, 448.0).to(tl.float8e4nv)
                aux_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
                if EMIT_T:
                    aux_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
                # atomic_max, not a loop-carried scalar (tl.range(flatten=True) does not guarantee those); max is
                # order-free, so one reduce(maximum(c0, c1)) equals maxing the two reductions.
                tile_max = tl.max(tl.max(tl.maximum(c0_post.to(tl.float32), c1_post.to(tl.float32)), axis=1), axis=0)
                tl.atomic_max(post_amax_ptr, tile_max)
        else:
            # aux holds `post`; relu(pre) = sqrt(post). dpre = 2 * (grad @ W2) * relu(pre).
            if RECON_SQRT:
                # ONE [BM, BN] TMA box, re-split by the SAME reshape -> permute -> split as the accumulator.
                _qe = aux_w_desc.load([offs_am, offs_bn])
                _qe = tl.reshape(_qe, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                _qe = tl.permute(_qe, (0, 2, 1))
                a0_raw, a1_raw = tl.split(_qe)
                c0 = (2.0 * acc0 * tl.sqrt(a0_raw.to(tl.float32) * ps_h)).to(dtype)
            else:
                c0_pre = aux_desc.load([offs_am, offs_bn])
                c0 = acc0.to(dtype)
                c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)
            if not EMIT_DPRE:
                c_desc.store([offs_am, offs_bn], c0)
            if RECON_SQRT:
                c1 = (2.0 * acc1 * tl.sqrt(a1_raw.to(tl.float32) * ps_h)).to(dtype)
            else:
                c1_pre = aux_desc.load([offs_am, offs_bn + BLOCK_SIZE_N // 2])
                c1 = acc1.to(dtype)
                c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)
            if not EMIT_DPRE:
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            if EMIT_DPRE:
                c0f = c0.to(tl.float32)
                c1f = c1.to(tl.float32)
                q0 = tl.maximum(tl.minimum(c0f * inv_ds_h, 57344.0), -57344.0).to(tl.float8e5)
                q1 = tl.maximum(tl.minimum(c1f * inv_ds_h, 57344.0), -57344.0).to(tl.float8e5)
                c_f8_desc.store([offs_am, offs_bn], q0)
                c_f8_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], q1)
                c_t_desc.store([offs_bn, offs_am], tl.trans(q0))
                c_t_desc.store([offs_bn + BLOCK_SIZE_N // 2, offs_am], tl.trans(q1))
                tile_max = tl.max(tl.max(tl.maximum(tl.abs(c0f), tl.abs(c1f)), axis=1), axis=0)
                tl.atomic_max(dpre_amax_ptr, tile_max)


# -----------------------------------------------------------------------------
# fp8 glue between the GEMMs.  The cost here is PASSES over the [T, 768] residual stream, not arithmetic: the row-major
# half is ATen that inductor folds into the producer's epilogue, the transposed half one coalesced fp8->fp8 pass
# (75.5 MB not 151.0 MB, -43.6 us).
# -----------------------------------------------------------------------------

@torch.library.custom_op("nanogpt::glue_fp8_t", mutates_args=())
def glue_fp8_t_op(src: torch.Tensor) -> torch.Tensor:
    """dst [N, M] = src [M, N].T for a row-major 1-byte tensor.  Opaque to inductor so the transpose stays one
    coalesced pass, not a strided pointwise; CUDA-graph safe; byte-exact (transpose_copy sees a uint8 view)."""
    assert src.ndim == 2 and src.stride(1) == 1 and src.element_size() == 1
    dst = torch.empty((src.shape[1], src.shape[0]), device=src.device, dtype=src.dtype)
    transpose_copy(src.view(torch.uint8), dst.view(torch.uint8))
    return dst

@glue_fp8_t_op.register_fake
def _(src):
    return src.new_empty((src.shape[1], src.shape[0]), dtype=src.dtype)

def quantize_dual_layout_fused(src: torch.Tensor, scale: torch.Tensor, fmt: torch.dtype):
    """Dual-layout fp8 quantize of src [M, N] (0-D fp32 `scale`): (row [M, N], t [N, M]) from ONE read.  The row-major
    half is ATen, so inductor fuses it into src's producer and rounds THAT kernel's fp32 registers, not the stored
    bf16 -- one rounding, not two: never bit-identical to a standalone quantize, never farther."""
    assert src.ndim == 2
    lim = 448.0 if fmt == torch.float8_e4m3fn else 57344.0
    row = torch.clamp(src.to(torch.float32) * (1.0 / scale), -lim, lim).to(fmt)
    return row, torch.ops.nanogpt.glue_fp8_t(row)


@triton.jit
def _quantize_weights_dual_kernel(w_ptr, row_ptr, col_ptr, scale_ptr, partial_amax_ptr, w_stride_l, w_stride_h,
                                  w_stride_d, H: tl.constexpr, D: tl.constexpr, num_tiles_d: tl.constexpr,
                                  num_tiles: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    tile_h = tile // num_tiles_d
    tile_d = tile % num_tiles_d
    offs_h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)
    v = tl.load(w_ptr + layer * w_stride_l + offs_h[:, None] * w_stride_h + offs_d[None, :] * w_stride_d, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + layer)
    q = tl.maximum(tl.minimum(v / s, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row_ptr + layer * H * D + offs_h[:, None] * D + offs_d[None, :], q, mask=mask)
    mask_t = (offs_d[:, None] < D) & (offs_h[None, :] < H)
    tl.store(col_ptr + layer * H * D + offs_d[:, None] * H + offs_h[None, :], tl.trans(q), mask=mask_t)
    tl.store(partial_amax_ptr + layer * num_tiles + tile, tl.max(tl.max(tl.abs(v), axis=1), axis=0))

def quantize_weights_dual_ntiles(H: int, D: int) -> int:
    """Partial-amax slots per layer: one per tile of _quantize_weights_dual_kernel's 64x64 grid."""
    return triton.cdiv(H, 64) * triton.cdiv(D, 64)

def quantize_mlp_weights_dual(bank: torch.Tensor, scales: torch.Tensor, partial_amax: torch.Tensor,
                              row: torch.Tensor, col_t: torch.Tensor, update_scales: bool = True):
    """Quantize a weight bank [L, H, D] (bf16) into BOTH fp8 caches from ONE read: `row` [L, H, D] e4m3 contiguous,
    `col_t` [L, D, H] e4m3, the .transpose(1,2) view of zeros_like(bank).transpose(1,2).contiguous().  The per-layer
    `scales` [L] LAG by a step -- update_scales refreshes them FIRST from the LAST call's per-tile amaxes, taking the
    reduction off the critical path; weights move <<1%/step under ANVIL, so the lag is loss-neutral."""
    L, H, D = bank.shape
    BLOCK_H = BLOCK_D = 64
    num_tiles_d = triton.cdiv(D, BLOCK_D)
    num_tiles = triton.cdiv(H, BLOCK_H) * num_tiles_d
    assert partial_amax.shape == (L, num_tiles)
    if update_scales:
        # 12% headroom, wider than FP8_POST_HEADROOM (1.03) because this amax is a step behind.
        torch.clamp(partial_amax.amax(dim=1) * (1.12 / 448.0), min=1e-12, out=scales)
    _quantize_weights_dual_kernel[(L, num_tiles)](
        bank, row, col_t, scales, partial_amax, bank.stride(0), bank.stride(1), bank.stride(2), H=H, D=D,
        num_tiles_d=num_tiles_d, num_tiles=num_tiles, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)


# COMPILE-SAFETY INVARIANT: linear_relu_square executes inside torch.compile's autograd-Function HOP subgraphs,
# where dynamo rejects any Python-state mutation from an outer scope ("Mutating a variable not in the current scope
# (SideEffects)").  So nothing reachable from it may write a module global, a module-level dict or a closure cell:
# unused pointer args take a per-call torch.empty(1), and the num_stages cache below is READ-ONLY under compile.
# It holds num_stages per constexpr variant -- emit variants start at 3 and step down if one exceeds H100 smem --
# and is written ONLY by eager calls, i.e. prime_stage_cache().
_lrs_stage_cache = {}

def linear_relu_square(a, b, aux=None, a_f8=None, b_f8=None, dequant_scale_ptr=None,
                       emit_f8=False, post_scale=None, post_amax=None,
                       emit_t=False, store_pre=True, store_post_bf=True,
                       emit_dpre=False, dpre_scale=None, dpre_amax=None):
    """Fused MLP GEMM with dual-layout fp8 epilogue emission.  Returns (pre, post, post_f8, post_t) forward and
    (dpre, dpre_f8, dpre_t) backward, None where not requested; emit_dpre asks for BOTH fp8 dpre layouts, exactly
    when the bf16 dpre is dead, so the backward emits fp8 or bf16 and never both."""
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    use_fp8 = b_f8 is not None
    FORWARD = aux is None
    recon_sqrt = (not FORWARD) and aux.dtype == torch.float8_e4m3fn
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # One tile shape for BOTH passes: the backward measured 0.3824 ms on 128x256, 0.4102 ms transposed.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128 if use_fp8 else 64
    num_warps = 8
    # EMIT_T rides on EMIT_F8's quantize, and post_t only exists once dW2 is FP8 -- when the bf16 post is dead.
    assert emit_f8 or not emit_t
    assert not (emit_t and store_post_bf)
    _e4, _e5 = torch.float8_e4m3fn, torch.float8_e5m2
    _dpre8 = emit_dpre and not FORWARD
    c, aux_out, aux_f8, aux_t, c_f8, c_t = [
        torch.empty((_r, _n), device=a.device, dtype=_d) if _w else None
        for _w, _r, _n, _d in (
            (store_pre if FORWARD else not emit_dpre, M, N, dtype),   # c       pre | bf16 dpre
            (FORWARD and store_post_bf,               M, N, dtype),   # aux     bf16 post
            (FORWARD and emit_f8,                     M, N, _e4),     # aux_f8  e4m3 post, row
            (FORWARD and emit_t,                      N, M, _e4),     # aux_t   e4m3 post, .T
            (_dpre8,                                  M, N, _e5),     # c_f8    e5m2 dpre, row
            (_dpre8,                                  N, M, _e5))]    # c_t     e5m2 dpre, .T
    if not FORWARD:
        aux_out = aux
    # A dummy tile keeps an unused TMA descriptor off a live output buffer; an epilogue that is off aliases a live
    # descriptor instead, and the wide [BM, BN] box exists exactly when RECON_SQRT reads it.  Straight-line, all
    # trace-time bools: a lazily-mutated cell is what the HOP rejects, and dynamo will not trace `is_` on descriptors.
    dummy_tile = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), device=a.device, dtype=dtype) if c is None or aux_out is None else None
    _box, _tbox = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], [BLOCK_SIZE_N // 2, BLOCK_SIZE_M]
    a_kernel = a_f8 if use_fp8 else a
    a_desc = TensorDescriptor.from_tensor(a_kernel, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_kernel = b_f8 if use_fp8 else b
    b_desc = TensorDescriptor.from_tensor(b_kernel, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c if c is not None else dummy_tile, _box)
    aux_desc = TensorDescriptor.from_tensor(aux_out if aux_out is not None else dummy_tile, _box)
    aux_f8_desc = TensorDescriptor.from_tensor(aux_f8, _box) if aux_f8 is not None else aux_desc
    aux_t_desc = TensorDescriptor.from_tensor(aux_t, _tbox) if aux_t is not None else aux_desc
    c_f8_desc = TensorDescriptor.from_tensor(c_f8, _box) if c_f8 is not None else c_desc
    c_t_desc = TensorDescriptor.from_tensor(c_t, _tbox) if c_t is not None else c_desc
    aux_w_desc = (TensorDescriptor.from_tensor(aux_out, [BLOCK_SIZE_M, BLOCK_SIZE_N]) if recon_sqrt else aux_desc)
    # The Triton signature always wants a pointer even where the kernel never loads it, so an off scalar arg takes
    # one scratch element -- per-call, never a module global.  An emit arm without its scale would read uninitialized
    # memory, hence the asserts.
    assert use_fp8 == (dequant_scale_ptr is not None)
    assert post_scale is not None or not ((FORWARD and emit_f8) or recon_sqrt)
    assert not emit_dpre or (dpre_scale is not None and dpre_amax is not None)
    _unused_ptr = torch.empty(1, dtype=torch.float32, device=a.device)
    dequant_scale_ptr, post_scale, post_amax, dpre_scale, dpre_amax = [
        _unused_ptr if _p is None else _p
        for _p in (dequant_scale_ptr, post_scale, post_amax, dpre_scale, dpre_amax)]

    key = (FORWARD, use_fp8, emit_f8, emit_t, store_pre, store_post_bf, recon_sqrt, emit_dpre,
           BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps)
    num_stages = _lrs_stage_cache.get(key, 4 if (FORWARD and not (emit_f8 or emit_t)) else 3)
    _alt_grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)), )

    def _launch(_ns):
        linear_relu_square_kernel[_alt_grid](
            a_desc, b_desc, c_desc, aux_desc, dequant_scale_ptr, M, N, K,
            aux_f8_desc, post_scale, post_amax, aux_t_desc,
            c_f8_desc, c_t_desc, dpre_scale, dpre_amax, aux_w_desc,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_SMS=NUM_SMS,
            FORWARD=FORWARD, USE_FP8=use_fp8,
            EMIT_F8=aux_f8 is not None, EMIT_T=aux_t is not None,
            STORE_PRE=FORWARD and store_pre, STORE_POST_BF=FORWARD and store_post_bf,
            RECON_SQRT=recon_sqrt, EMIT_DPRE=c_f8 is not None,
            num_stages=_ns, num_warps=num_warps,
        )

    if torch.compiler.is_compiling():
        _launch(num_stages)     # deferred into the artifact: the OOR retry cannot fire, unprimed keys take the default
    else:
        while True:
            try:
                _launch(num_stages)
                break
            except _TritonOOR:      # smem overflow on this emit variant: step the pipeline down
                if num_stages <= 1:
                    raise
                num_stages -= 1
        _lrs_stage_cache[key] = num_stages
    return (c, aux_out, aux_f8, aux_t) if FORWARD else (c, c_f8, c_t)

def prime_stage_cache():
    """Resolve num_stages for every linear_relu_square variant the trainer reaches by launching each once on
    one-tile inputs -- smem pressure depends only on the constexpr variant, NOT on M/N/K.  Call once, BEFORE
    torch.compile traces: under compile the OOR retry cannot fire."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    M, N, K = (128, 256, 128)  # one (BLOCK_M, BLOCK_N) tile, one fp8 K-block
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    _z = lambda fmt, *shape: torch.zeros(*shape, device=dev, dtype=fmt)
    a, b = _z(torch.bfloat16, M, K), _z(torch.bfloat16, N, K)
    a_e4, b_e4 = _z(torch.float8_e4m3fn, M, K), _z(torch.float8_e4m3fn, N, K)
    g_f8, aux_e4 = _z(torch.float8_e5m2, M, K), _z(torch.float8_e4m3fn, M, N)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    amax = torch.zeros(nsm, dtype=torch.float32, device=dev)
    # NOT mechanical (recon_sqrt comes from aux.dtype), so each row names the caller it mirrors.  ReLUSqrdMLP takes
    # fp8 weights iff fp8 activations and eval has no backward, so these are the three reachable rows.
    for _kw in ({},                                                       # eval-path fwd, bf16
                {"a_f8": a_e4, "b_f8": b_e4, "dequant_scale_ptr": one, "emit_f8": True,
                 "emit_t": True, "post_scale": one, "post_amax": amax,
                 "store_pre": False, "store_post_bf": False},             # .forward
                {"aux": aux_e4, "a_f8": g_f8, "b_f8": b_e4, "dequant_scale_ptr": one,
                 "post_scale": one, "emit_dpre": True, "dpre_scale": one,
                 "dpre_amax": amax}):                                     # .backward
        linear_relu_square(a, b, **_kw)
    torch.cuda.synchronize(dev)


# FP8 MLP scaled_mm wrappers: ONE emitter, four registered names.  Each stays its own op so it is opaque to inductor and
# appears by name in the traced graph; they differ only in which operand needs transposing and in use_fast_accum -- False
# on EVERY gradient GEMM (NaN otherwise).  Operands arrive in the layout _scaled_mm wants: dp takes the row-major post
# against the col-major W2 cache; wg2 and wg1 take the two epilogue emits, whose .T is a zero-copy column-major view --
# the TN pair Hopper FP8 WGMMA needs; dx takes the row-major dpre against w1_f8_col.  Output is bf16 to match the bank
# grad dtype; fp32 would double the reduce_scatter volume.

def _f8_mm_op(name, transpose_b, fast_accum):
    def _fn(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(a, b.T if transpose_b else b, out_dtype=torch.bfloat16, scale_a=a_scale,
                                scale_b=b_scale, use_fast_accum=fast_accum)
    _fn.__name__ = name + "_op"
    _op = torch.library.custom_op("nanogpt::" + name, _fn, mutates_args=())
    _op.register_fake(lambda a, b, a_scale, b_scale:
                      a.new_empty((a.shape[0], b.shape[0] if transpose_b else b.shape[1]), dtype=torch.bfloat16))
    return _op

dp_f8_op = _f8_mm_op("dp_f8", False, True)     # x3  = post_f8 @ W2_f8       (down projection)
wg2_f8_op = _f8_mm_op("wg2_f8", True, False)   # dW2 = post_t_f8 @ g_t_f8.T
wg1_f8_op = _f8_mm_op("wg1_f8", True, False)   # dW1 = dpre_t_f8 @ x_t_f8.T
dx_f8_op = _f8_mm_op("dx_f8", False, False)    # dx  = dpre_f8 @ w1_f8_col

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    """Fused MLP autograd function extended to the full-FP8 MLP package. x is the [T, 768] bf16 normed residual and
    W1/W2 the [H, 768] bf16 bank slices; every trailing arg is an fp8 cache or scale, None when not supplied, and
    together they select the fp8 arms. Layouts: x_f8 is e4m3 [T, 768] and x_f8_t its [768, T] transpose; W2_f8
    (col-major) and w2_f8_row are the same weights in the two layouts _scaled_mm needs; dequant_scale is x_s*w1_s,
    dq_bwd w2_s*grad_s, and post_scale/dpre_scale are delayed, fed by the amax slots post_amax/dpre_amax."""
    @staticmethod
    def forward(ctx, x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None, W2_f8=None, w2_scale=None,
                post_scale=None, post_amax=None, w2_f8_row=None, dq_bwd=None, x_f8_t=None, x_scale=None,
                w1_f8_col=None, w1_scale=None, g_scale=None, dpre_scale=None, dpre_amax=None, p_fold=None):
        # `emit` is ONE decision for the whole fp8 MLP (fp8 down projection, transposed post emit, fp8 dW2); under it
        # both bf16 stores are dead.  The saved-tensor layout and the backward's flags assume all-or-nothing pairs.
        assert (W2_f8 is None) == (w2_f8_row is None)
        assert (x_f8_t is None) == (w1_f8_col is None)
        emit = W2_f8 is not None
        x_flat = x.view((-1, x.shape[-1]))
        if W1_f8 is not None:
            assert x_f8 is not None and dequant_scale is not None
            pre, post, post_f8, post_t = linear_relu_square(
                x_flat, W1, a_f8=x_f8.view((-1, x_f8.shape[-1])), b_f8=W1_f8, emit_f8=emit, emit_t=emit,
                dequant_scale_ptr=dequant_scale, post_scale=post_scale, post_amax=post_amax,
                store_pre=not emit, store_post_bf=not emit)
        else:
            pre, post, post_f8, post_t = linear_relu_square(x_flat, W1, emit_f8=emit, post_scale=post_scale, post_amax=post_amax)
        x3 = (torch.ops.nanogpt.dp_f8(post_f8, W2_f8, post_scale, w2_scale) if emit else post @ W2)
        # Graph INTERMEDIATES must go through save_for_backward: ctx-attr stashing inside the compiled HOP pins them
        # past their last use (+3.3 MB).  `p_fold` is the residual site's post-lambda, folded by the caller into the
        # down-projection dequant scales, so it rides along only for its own gradient.  Both tails are trace-time
        # constant, so the concatenation resolves before the HOP sees the call.
        ctx.save_for_backward(*((x, W1, W2) + ((pre, post) if pre is not None else ()) + ((p_fold,) if p_fold is not None else ())))
        # Persistent caches and scales, kept by NAME so the backward reads them unindexed.
        ctx.saved_bf16, ctx.fold_p = pre is not None, p_fold is not None
        ctx.post_f8, ctx.post_t, ctx.post_scale = (post_f8 if emit else None), post_t, post_scale
        ctx.w2_f8_row, ctx.dq_bwd, ctx.x_f8_t, ctx.x_scale = w2_f8_row, dq_bwd, x_f8_t, x_scale
        ctx.w1_f8_col, ctx.w1_scale, ctx.g_scale = w1_f8_col, w1_scale, g_scale
        ctx.dpre_scale, ctx.dpre_amax = dpre_scale, dpre_amax
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Dynamo-robust unpack: no star-unpack of saved_tensors, no bool-as-index; ctx.saved_bf16 is metadata.
        st = ctx.saved_tensors
        x, W1, W2 = st[0], st[1], st[2]
        if ctx.saved_bf16:            # bf16 MLP
            pre, post = st[3], st[4]
            p_fold = st[5] if ctx.fold_p else None
        else:                         # fp8 path: the fp8 post is the aux
            pre = post = None
            p_fold = st[3] if ctx.fold_p else None
        g_flat = grad_output.view((-1, grad_output.shape[-1]))
        # The forward's pair asserts make "w2_f8_row present" exactly its `emit`, and the epilogue emits both fp8
        # dpre layouts or neither, so dW1 and dx pair up.
        fp8_gemm = ctx.w2_f8_row is not None
        fp8_emit = fp8_gemm and ctx.x_f8_t is not None and ctx.w1_f8_col is not None and ctx.dpre_scale is not None

        # ---- 1. incoming-grad quantize (one pass, dual layout) and 2. dW2 = post^T @ g ----
        if fp8_gemm:
            g_f8, g_f8_t = quantize_dual_layout_fused(g_flat, ctx.g_scale, fmt=torch.float8_e5m2)
            dW2 = torch.ops.nanogpt.wg2_f8(ctx.post_t, g_f8_t, ctx.post_scale, ctx.g_scale)
        else:
            g_f8 = None
            dW2 = post.T @ grad_output  # default eager expression (unflattened)

        # ---- 2b. MLPFOLD: p's fold moved into weight space.  Forward computed out = post @ (p * W2), so dW2 above
        # is d/d(p*W2) = post^T g, dL/dW2 = p * dW2 and dL/dp = <dW2, W2> -- a [mlp_hdim, 768] dot, NO division by p,
        # replacing the residual site's two [T, 768] passes (13.0 MB against 151.0 MB).  grad_p stays None unfolded,
        # which is also autograd's trailing slot for it.
        grad_p = None
        if ctx.fold_p:
            grad_p = (dW2.float() * W2.float()).sum().to(p_fold.dtype)
            dW2 = dW2 * p_fold

        # ---- 3. dpre (+ fp8 emits from the same epilogue), then dW1 and dx.  The aux read IS the fp8 post where
        # there is one: half the bf16 pre's bytes.
        aux_src = pre if ctx.post_f8 is None else ctx.post_f8
        if fp8_gemm:
            dpre, dpre_f8, dpre_t = linear_relu_square(
                g_flat, W2, aux=aux_src, a_f8=g_f8, b_f8=ctx.w2_f8_row,
                dequant_scale_ptr=ctx.dq_bwd, post_scale=ctx.post_scale,
                emit_dpre=fp8_emit, dpre_scale=ctx.dpre_scale, dpre_amax=ctx.dpre_amax)
        else:
            dpre, dpre_f8, dpre_t = linear_relu_square(g_flat, W2, aux=aux_src, post_scale=ctx.post_scale)
        if fp8_emit:
            dW1 = torch.ops.nanogpt.wg1_f8(dpre_t, ctx.x_f8_t, ctx.dpre_scale, ctx.x_scale)
            dx = torch.ops.nanogpt.dx_f8(dpre_f8, ctx.w1_f8_col, ctx.dpre_scale, ctx.w1_scale)
        else:
            dW1 = dpre.T @ x  # default eager expressions (unflattened)
            dx = dpre @ W1

        # One return slot per forward input (19, + p_fold).
        return (dx.view(x.shape), dW1, dW2) + (None,) * 16 + (grad_p,)


# -----------------------------------------------------------------------------
# Tiled transpose copy kernel: dst (N, M) = src (M, N).T
# Uses coalesced reads from src and coalesced writes to dst via tl.trans().
# Replaces PyTorch's elementwise copy_ which uses a naive 75k-block kernel
# with non-coalesced writes, saturating all SMs and blocking NCCL.

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

    BLOCK_M, BLOCK_N = 64, 128
    if src.element_size() == 1 and dst.stride(1) == 1:
        # BLOCK_M is the length in ELEMENTS of each contiguous destination run, so at
        # 64x128 a 1-byte dtype would store half a 128-byte L2 line. Pure copy: BIT-EXACT.
        BLOCK_M = 128
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
# Tiled transpose-add kernel: dst (M, N) += src (N, M).T
# Same tiling strategy as transpose_copy but with a fused read-add-write.
# Replaces PyTorch's .add_(src.T) which uses the same 75k-block elementwise
# kernel with non-coalesced reads from the transposed operand.

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

    # Coalesced read from src (M, N)
    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced read-add-write on dst (N, M): dst[n, m] += src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced access on both src and dst,
    replacing PyTorch's .add_(src.T) which has non-coalesced reads from the
    transposed operand.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    # 64x128 / 8 warps rather than 32x32 / 4: both streams then issue 128-byte
    # segments (src offs_n-fastest at BLOCK_N, dst offs_m-fastest at BLOCK_M).
    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_add_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


# ---- fused softcapped cross-entropy (reuses the lm_head FP8 weight cache) ----
# ONE CUDA kernel does the forward AND backward in a single pass over the vocabulary, so the [T, 50304] logits are never re-read: the lm_head GEMM writes
# them as e4m3 CODE BYTES, decoded by a ~4-op hardware conversion (an arithmetic decode costs 0.643 ms/step more at the bandwidth-bound 2.109 TB/s). A
# row's loss adds a prefix-token CE at prefix_weight to the MTP look-ahead, a no-op at weight 0; its sigmoid cache is fp16 as bf16 rounds 1-sigma to 0.

CE_KERNEL_BLOCK_SIZE = 256
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

struct __align__(16) __half8 {
    __half data[8];
    __device__ __half& operator[](int i) { return data[i]; }
    __device__ const __half& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2_fast(float x) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(x));
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (uint8_t)(packed & 0xFFu);
    return result;
}

__device__ __forceinline__ uint16_t f32x2_to_fp8_e5m2x2(float lo, float hi) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(hi), "f"(lo));
    return packed;
}

__device__ __forceinline__ unsigned int fp8_e5m2x2_pair_to_word(uint16_t a, uint16_t b) {
    unsigned int w;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(w) : "r"((unsigned int)a), "r"((unsigned int)b));
    return w;
}

struct __align__(8) __nv_uchar8 {
    unsigned char data[8];
    __device__ unsigned char& operator[](int i) { return data[i]; }
    __device__ const unsigned char& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ float ce_e4m3_logit_to_f32(unsigned char b) {
    unsigned int h2;
    unsigned short in = (unsigned short)b;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    unsigned short lo = (unsigned short)(h2 & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(lo));
    return f;
}

#define CE_IN_T unsigned char
#define CE_IN8_T __nv_uchar8
#define CE_LOGIT_TO_F32(v) ce_e4m3_logit_to_f32(v)

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

//__device__ float sigmoid(float x) {
//  return 1.0f / (1.0f + __expf(-x));
//}
__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

extern "C"
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const CE_IN_T* __restrict__ logits,
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

  extern __shared__ __half smem[];

  static_assert(VEC_WIDTH == 8);

  const CE_IN_T *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  // FIXED-MAX lse. z = A*sigmoid((l+B)/C) with A=23 is bounded to (0, A], so exp(z - A) is in (exp(-A), 1] and the block sum in [VOCAB*exp(-A),
  // VOCAB] -- 5e-6 .. 5e4 at VOCAB=50304, nowhere near fp32's range. The online block max exists only to bound that exponent, so the constant A
  // stands in for it, the exp-sum rides the SAME smem pass, and the __syncthreads below is the only one before smem is read cross-thread.

  float thread_sum = 0.0f;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      CE_IN8_T result = *(CE_IN8_T*)(&block_logit_ptr[idx]);
      __half8 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2half(tmp);
      }
      *(__half8*)(&smem[idx]) = result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = A * __half2float(result_sigmoid[k]);
        thread_sum += __expf(tmp - A);
      }
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_sums[NUM_WARPS];

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

  float lse = A + __logf(block_sum);

  // Prefix token prediction target for this position (T' = longest-prefix token of
  // the immediate next-token target T). prefix_targets[i] < 0 => no valid prefix, ignored.
  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __half2float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    // Same CE logic as MTP, but the target is the prefix token T' at this position.
    {
      int64_t ptgt = prefix_targets[blockIdx.x];
      if (ptgt >= 0 && ptgt < VOCAB_SIZE) {
        float z_p = A * __half2float(smem[ptgt]);
        total_loss += (float)prefix_weight_param * (lse - z_p);
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  // Total weight over active predictions at this position (used in the softmax-normalizer
  // gradient term). Include the prefix prediction only when it has a valid target.
  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }
  int64_t ptgt_row = prefix_targets[blockIdx.x];
  if (ptgt_row >= 0) {
    S_w += (float)prefix_weight_param;
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __half8 sigmoid_us = *(__half8*)(&smem[idx]);
      uint16_t pk[VEC_WIDTH / 2];
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j += 2) {
        float g[2];
        #pragma unroll
        for (int t = 0; t < 2; t++) {
          float sigmoid_u = __half2float(sigmoid_us[j + t]);
          float z = A * sigmoid_u;
          float p = __expf(z - lse);

          float term1 = S_w * p;
          float term2 = 0.0f;

          float grad_z = term1 - term2;
          g[t] = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        }
        pk[j >> 1] = f32x2_to_fp8_e5m2x2(g[0], g[1]);
      }
      unsigned long long packed8 =
          ((unsigned long long)fp8_e5m2x2_pair_to_word(pk[2], pk[3]) << 32)
          | (unsigned long long)fp8_e5m2x2_pair_to_word(pk[0], pk[1]);
      *(unsigned long long*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = packed8;
    }
  }

  __syncthreads();

  // Sparse correction for target columns. Threads [0, n_predict) handle the future MTP
  // targets; thread n_predict handles the prefix target. term2 for a column sums the
  // weights of every prediction (MTP + prefix) whose target lands on that column, so
  // duplicate columns across threads write identical values (idempotent, race-free).
  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && ((float)prefix_weight_param > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];

    float sigmoid_u = __half2float(smem[target]);
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
    auto result_tmp = f32_to_fp8_e5m2_fast(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""

# nvrtc needs the CUDA headers; CUDA_HOME is what torch's own extension builder consults.
CUDA_INCLUDE_DIRS = [os.path.join(os.environ.get("CUDA_HOME") or "/usr/local/cuda", "include")]

ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
    "ce_fwd_bwd_kernel",
    compute_capability="90",
    cuda_include_dirs=CUDA_INCLUDE_DIRS,
    nvcc_options=["-lineinfo", "--use_fast_math"],
)
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)

# Live prefix-CE weight, read EAGERLY inside the opaque custom op: a python float
# reaching the graph forces a full recompile at every ramp flip (~12 s on-clock), so
# the graph passes a constant sentinel.
_PTP_W_RUNTIME = None

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


# ---- cached fp32 scale scalars ----------------------------------------------
# torch._scaled_mm wants per-tensor scales as 0-d fp32 CUDA tensors, and the CE's are the lm_head's FIXED
# x_s / w_s / grad_s: 19.04 us per rebuild against 6.46 us cached, and the cached tensor is bit-identical --
# _scaled_mm never writes its scales.  Inert under torch.compile: a dict gaining a key guards a ~12 s recompile.
_CE_SCALAR_CACHE = {}

def _ce_scalar(ref: torch.Tensor, v: float) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return ref.new_tensor(v, dtype=torch.float32)
    key = (float(v), ref.device)
    t = _CE_SCALAR_CACHE.get(key)
    if t is None:
        _CE_SCALAR_CACHE[key] = t = ref.new_tensor(v, dtype=torch.float32)
    return t

def _ce_backward_gemms(grad_input, w_t, x_f8, x_s, w_s, grad_s):
    """The CE backward tail both CE Functions share: the three scale scalars,
    dx = grad @ W, and the bf16 wgrad via two fp8 transposes (dW = x^T @ grad)."""
    n_rows, n_cols = grad_input.shape
    x_scale = _ce_scalar(grad_input, x_s)
    w_scale = _ce_scalar(grad_input, w_s)
    grad_scale = _ce_scalar(grad_input, grad_s)

    grad_x = torch._scaled_mm(
        grad_input, w_t.T,
        out_dtype=torch.bfloat16,
        scale_a=grad_scale,
        scale_b=w_scale,
        use_fast_accum=False,
    )

    x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
    transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

    grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
    transpose_copy(grad_input, grad_input_T)  # (n_cols, n_rows) row-major

    grad_w = torch._scaled_mm(
        x_f8_T, grad_input_T.T,    # (768, n_rows) row-major @ (n_rows, n_cols) column-major view
        out_dtype=torch.bfloat16,  # bf16 wgrad: halves the grad_w write + reduce_scatter bytes
        scale_a=x_scale,
        scale_b=grad_scale,
        use_fast_accum=False,
    )
    return grad_x, grad_w

def _ce_logit_gemm(x, x_f8, w_col, x_s, w_s):
    """The lm_head GEMM, emitting the RAW logit as an e4m3 code byte; w_col must be column-major.
    torch._scaled_mm IGNORES scale_result for an fp8 out_dtype, so scale_a/scale_b descale the
    accumulator and the stored byte is the raw e4m3 RNE logit."""
    return torch._scaled_mm(
        x_f8, w_col,
        out_dtype=torch.float8_e4m3fn,
        scale_a=_ce_scalar(x, x_s),
        scale_b=_ce_scalar(x, w_s),
        use_fast_accum=True,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    """Full-vocabulary softcapped CE: the fp8 lm_head GEMM into e4m3 logit code bytes, the fused fwd+bwd kernel, then the two fp8 gradient GEMMs --
    the e5m2 grad_input the kernel leaves behind IS the backward's input, and no logit tensor survives the forward. Both GEMM halves are staticmethods
    because SampledSoftcappedCrossEntropy runs the same two. w_f8_in (col-major) and w_f8_row are the trainer's post-step fp8 lm_head copies from the
    comm-overlapped optimizer phase; lm_head_weight only carries the grad. A, B, C are spelled again in train_gpt.py's eval arm and in CE_KERNEL_SOURCE."""
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, w_f8_in, prefix_targets, prefix_weight, w_f8_row):
        # softcap z = A*sigmoid((l+B)/C); train_gpt.py's eval arm and CE_KERNEL_SOURCE
        # hold the same constants -- the three copies must move together.
        A, B, C = 23.0, 5.0, 7.5

        assert w_f8_in is not None and w_f8_row is not None, "the fp8 lm_head caches are required, col-major and row-major together"
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        # The trainer's post-step fp8 lm_head copies in the two layouts, quantized in
        # the comm-overlapped optimizer phase; lm_head_weight only carries the grad.
        logits = _ce_logit_gemm(x, x_f8, w_f8_in.T.contiguous().T, x_s, w_s)

        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        prefix_targets = prefix_targets.contiguous()
        ce_fwd_bwd(logits, targets, mtp_weights, prefix_targets, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale, float(prefix_weight))

        # The backward's mat2 needs w_f8.T column-major, so it takes the ROW-major cache.
        ctx.save_for_backward(x_f8, w_f8_row, grad_input)
        ctx.params = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8, grad_input = ctx.saved_tensors
        x_s, w_s, grad_s = ctx.params
        grad_x, grad_w = _ce_backward_gemms(grad_input, w_f8, x_f8, x_s, w_s, grad_s)
        # One return slot per forward arg; grad_w lands on lm_head_weight (slot 4).
        return grad_x, None, None, grad_w, None, None, None, None, None, None, None, None

# SNS -- Shared-Negative Sampled softmax for the EARLY stages, where the full vocabulary's 2.5-5 GB of logit traffic per
# step buys resolution not yet worth paying for.  Until the cutover the three GEMMs run at width P << 50304 against a
# per-step SHARED candidate set C, with the CE CUDA kernel reused UNCHANGED at VOCAB_SIZE = P.
_SNS_KERNELS: dict = {}   # P -> ce_fwd_bwd kernel compiled at VOCAB_SIZE = P
_SNS_WC: dict = {}        # P -> fp8 (P, 768)  gathered lm_head rows, row-major
_SNS_WCT: dict = {}       # P -> fp8 (768, P)  the same rows transposed (the backward's mat2)
_SNS_POSD = None          # int32 [vocab]  C's device inverse, -1 off-set
_SNS_ARD = None           # int32 [P_max]  0..P_max-1, the scatter payload
# The per-step index buffers, in upload order, each with the row count a step slices out of it: C is the candidate class
# ids (ascending), TPOS and PPOS the position IN C of each token's target and prefix target, -1 for an absent prefix.
_SNS_BUFS = (("C", "P"), ("TPOS", "T"), ("PPOS", "T"))
_SNS_IDX: dict = {}       # name -> its int64 device buffer, ONE identity for the process's life: dynamo never sees a new tensor
# RACE CONTRACT: SampledSoftcappedCrossEntropy keeps no defensive copy of _SNS_WCT[P], legal only because exactly one
# micro-batch runs per step, so its backward is enqueued before the next sns_gather().  The H2D half: sns_upload_async.

def sns_init(device, p_values, vocab_size: int, max_rows: int, model_dim: int):
    """One CE kernel per distinct candidate count P, plus every persistent SNS buffer. Called ONCE at module scope BEFORE torch.compile: nvrtc is untimed."""
    global _SNS_POSD, _SNS_ARD
    for p in p_values:
        # The 8-wide smem / grad_input stores need P % (BLOCK_SIZE * 8) == 0; the weight pair is per P because a sliced (768, P) view is not _scaled_mm's mat2.
        assert 0 < p <= vocab_size and p % (CE_KERNEL_BLOCK_SIZE * 8) == 0, f"X_SNS candidate count {p} is not a legal CE vocabulary"
        _src = f"\nconstexpr int VOCAB_SIZE = {p};\nconstexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n" + CE_KERNEL_SOURCE
        _SNS_KERNELS[p] = torch.cuda._compile_kernel(_src, "ce_fwd_bwd_kernel", compute_capability="90",
                                                     cuda_include_dirs=CUDA_INCLUDE_DIRS, nvcc_options=["-lineinfo", "--use_fast_math"])
        _SNS_KERNELS[p].set_shared_memory_config(p * 2)
        _SNS_WC[p], _SNS_WCT[p] = (torch.zeros(_s, dtype=torch.float8_e4m3fn, device=device) for _s in ((p, model_dim), (model_dim, p)))
    for _nm, _k in _SNS_BUFS:   # PPOS starts at the kernel's no-op, so a step that skips its upload reads no prefix
        _SNS_IDX[_nm] = torch.full(({"P": max(p_values), "T": max_rows}[_k],), -1 if _nm == "PPOS" else 0, dtype=torch.int64, device=device)
    _SNS_POSD = torch.empty(vocab_size, dtype=torch.int32, device=device)   # C's inverse: rebuilt in sns_gather, never uploaded
    _SNS_ARD = torch.arange(max(p_values), dtype=torch.int32, device=device)

def sns_gather(w_f8_cm, sns_p: int):
    """EAGER (never traced): refill _SNS_WC / _SNS_WCT from this step's candidate rows of the col-major lm_head fp8 cache, and rebuild C's device
    inverse. The cache's strides are (1, 768), so its transpose IS the contiguous matrix the gather wants; the uint8 reinterpret moves identical bytes."""
    cand, wc, wct = _SNS_IDX["C"][:sns_p], _SNS_WC[sns_p], _SNS_WCT[sns_p]
    torch.index_select(w_f8_cm.T.view(torch.uint8), 0, cand, out=wc.view(torch.uint8))
    transpose_copy(wc.view(torch.uint8), wct.view(torch.uint8))
    _SNS_POSD.fill_(-1)
    _SNS_POSD.index_copy_(0, cand, _SNS_ARD[:sns_p])


# ---- fused SNS grad_w densify -----------------------------------------------
# The SNS backward's wgrad is a dense (768, P) slab of candidate POSITIONS; the optimizer wants a contiguous
# (768, vocab) grad with non-candidate columns zeroed.  Rather than `zeros(); index_copy_(1, cand, ...)` -- a
# 77 MB zero pass plus dim-1 SCATTERED 2-byte stores -- this kernel writes DENSELY and gathers:
# out[r, c] = src[r, pos[c]] or 0.  No arithmetic touches the payload, so stored bits are loaded bits: BIT-EXACT.

@triton.jit
def _sns_densify_kernel(SRC, POS, DST, R, V, src_stride_r, dst_stride_r,
                        BLOCK_R: tl.constexpr, BLOCK_V: tl.constexpr):
    offs_v = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    m_v, m_r = offs_v < V, offs_r < R
    pos = tl.load(POS + offs_v, mask=m_v, other=-1)
    keep = pos >= 0
    p = tl.where(keep, pos, 0).to(tl.int64)          # in-range dummy for the masked lanes
    # C is ascending, so a vocab tile's kept positions are CONSECUTIVE: one short contiguous run per src row.
    src = tl.load(SRC + offs_r[:, None].to(tl.int64) * src_stride_r + p[None, :],
                  mask=m_r[:, None] & keep[None, :], other=0.0)
    tl.store(DST + offs_r[:, None].to(tl.int64) * dst_stride_r + offs_v[None, :].to(tl.int64),
             src, mask=m_r[:, None] & m_v[None, :])

@torch.library.custom_op("nanogpt::sns_densify", mutates_args={"dst"})
def sns_densify(src: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor) -> None:
    _sns_densify_kernel[(triton.cdiv(dst.shape[1], 256), triton.cdiv(dst.shape[0], 32))](
        src, pos, dst, *dst.shape, src.stride(0), dst.stride(0), BLOCK_R=32, BLOCK_V=256, num_warps=4, num_stages=3)

@torch.library.custom_op("nanogpt::sns_ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def sns_ce_fwd_bwd(logits: torch.Tensor, targets: torch.Tensor, mtp_weights: torch.Tensor, prefix_targets: torch.Tensor,
                   losses: torch.Tensor, grad_input: torch.Tensor, n_rows: int, n_predict: int, A: float, B: float,
                   C: float, grad_s: float, grad_scale: float, prefix_weight: float, sns_p: int) -> None:
    if _PTP_W_RUNTIME is not None and prefix_weight > 0.0:
        prefix_weight = _PTP_W_RUNTIME
    _SNS_KERNELS[sns_p]((n_rows, 1, 1), (CE_KERNEL_BLOCK_SIZE, 1, 1), (logits, targets, mtp_weights, prefix_targets, losses,
                        grad_input, n_rows, n_predict, A, B, C, grad_s, grad_scale, prefix_weight), shared_mem=sns_p * 2)

class SampledSoftcappedCrossEntropy(torch.autograd.Function):
    """Sampled-candidate variant of FusedSoftcappedCrossEntropy: the same arguments plus a trailing `sns_p`, this stage's padded candidate
    count (a python int, so a traced constant). The three GEMMs run at width P against the row-gather sns_gather leaves in _SNS_WC/_SNS_WCT.
    `targets` / `prefix_targets` are taken for signature parity and shape checking only; the kernel is fed the host-built POSITION vectors."""
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, w_f8_cm, prefix_targets, prefix_weight, sns_p):
        A, B, C = 23.0, 5.0, 7.5      # softcap, as in FusedSoftcappedCrossEntropy.forward
        n_rows = x.shape[0]
        assert targets.shape[0] == n_rows and prefix_targets.shape[0] == n_rows
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x, x_f8, _SNS_WC[sns_p].T, x_s, w_s)   # .T is column-major
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, sns_p), dtype=torch.float8_e5m2, device=logits.device)
        sns_ce_fwd_bwd(logits.contiguous(), _SNS_IDX["TPOS"][:n_rows], mtp_weights.contiguous(), _SNS_IDX["PPOS"][:n_rows], losses,
                       grad_input, n_rows, mtp_weights.shape[0], A, B, C, grad_s, grad_scale, float(prefix_weight), sns_p)
        ctx.save_for_backward(x_f8, _SNS_WCT[sns_p], grad_input)
        ctx.params = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, wc_t, grad_input = ctx.saved_tensors
        grad_x, grad_w_c = _ce_backward_gemms(grad_input, wc_t, x_f8, *ctx.params)
        # Dense full-extent grad, sized from _SNS_POSD -- it IS the vocab-length map densify gathers through. The tied
        # transpose_add writes into it in place and the reduce_scatter needs a contiguous payload; freshly allocated every
        # backward (AccumulateGrad may adopt it as .grad) and never zeroed, since the gather writes every element.
        grad_w = torch.empty((x_f8.shape[1], _SNS_POSD.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        torch.ops.nanogpt.sns_densify(grad_w_c.contiguous(), _SNS_POSD, grad_w)
        return grad_x, None, None, grad_w, None, None, None, None, None, None, None, None

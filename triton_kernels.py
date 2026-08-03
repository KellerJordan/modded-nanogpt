import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 post_fp8_desc, partial_amax_ptr,
                                 dequant_scale_ptr, activation_scale_ptr,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 USE_FP8: tl.constexpr,
                                 EMIT_FP8: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    partial_amax = 0.0
    if EMIT_FP8:
        inverse_activation_scale = 1.0 / tl.load(activation_scale_ptr)

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

        c0 = acc0.to(dtype)
        if FORWARD:
            # Store ONLY post = relu(pre)^2 (drop the redundant `pre` materialization).
            # Backward reconstructs relu(pre) = sqrt(post) in-kernel, so the full
            # (M, N) pre tensor never round-trips HBM.
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            c_desc.store([offs_am_c, offs_bn_c], c0_post)
            if EMIT_FP8:
                c0_fp8 = tl.minimum(c0_post * inverse_activation_scale, 448.0)
                post_fp8_desc.store(
                    [offs_am_c, offs_bn_c], c0_fp8.to(tl.float8e4nv)
                )
                partial_amax = tl.maximum(
                    partial_amax,
                    tl.max(tl.max(c0_post.to(tl.float32), axis=1), axis=0),
                )
        else:
            # aux holds `post`; relu(pre) = sqrt(post). dpre = 2 * (grad @ W2) * relu(pre).
            c0_post = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.sqrt(c0_post.to(tl.float32))
            c_desc.store([offs_am_c, offs_bn_c], c0.to(dtype))

        c1 = acc1.to(dtype)
        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)
            if EMIT_FP8:
                c1_fp8 = tl.minimum(c1_post * inverse_activation_scale, 448.0)
                post_fp8_desc.store(
                    [offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2],
                    c1_fp8.to(tl.float8e4nv),
                )
                partial_amax = tl.maximum(
                    partial_amax,
                    tl.max(tl.max(c1_post.to(tl.float32), axis=1), axis=0),
                )
        else:
            c1_post = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.sqrt(c1_post.to(tl.float32))
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1.to(dtype))

    if EMIT_FP8:
        tl.store(partial_amax_ptr + start_pid, partial_amax)


@triton.jit
def reduce_mlp_activation_scales_kernel(
    partial_amax_ptr,
    scale_ptr,
    partial_stride,
    partial_count: tl.constexpr,
    HEADROOM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        partial_amax_ptr + layer * partial_stride + offsets,
        mask=offsets < partial_count,
        other=0.0,
    )
    amax = tl.max(values, axis=0)
    tl.store(scale_ptr + layer, tl.maximum(amax, 1.0e-12) * (HEADROOM / 448.0))


@triton.jit
def quantize_transpose_mlp_down_weights_kernel(
    weight_ptr,
    output_ptr,
    row_output_ptr,
    scale_ptr,
    weight_layer_stride,
    hidden_dim: tl.constexpr,
    model_dim: tl.constexpr,
    num_tiles_d: tl.constexpr,
    EMIT_ROW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    layer = tl.program_id(0)
    tile = tl.program_id(1)
    tile_h = tile // num_tiles_d
    tile_d = tile % num_tiles_d
    offsets_h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offsets_h[:, None] < hidden_dim) & (offsets_d[None, :] < model_dim)
    input_offsets = (
        layer * weight_layer_stride
        + offsets_h[:, None] * model_dim
        + offsets_d[None, :]
    )
    values = tl.load(weight_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + layer)
    quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
    output_offsets = (
        layer * model_dim * hidden_dim
        + offsets_d[:, None] * hidden_dim
        + offsets_h[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        tl.trans(quantized).to(tl.float8e4nv),
        mask=tl.trans(mask),
    )
    if EMIT_ROW:
        # Row-major (hidden, model) copy for the FP8 backward's dpre GEMM, which
        # needs a contiguous last dim. Free here: the tile is already in registers.
        # NB: must use the output's own contiguous layer stride, NOT input_offsets --
        # `weights` is a non-contiguous bank slice whose layer stride is 2x this one.
        row_offsets = (
            layer * hidden_dim * model_dim
            + offsets_h[:, None] * model_dim
            + offsets_d[None, :]
        )
        tl.store(row_output_ptr + row_offsets, quantized.to(tl.float8e4nv), mask=mask)


_dummy_f32 = None  # lazily initialized 1-element tensor for unused pointer args

def _get_dummy_f32(device):
    global _dummy_f32
    if _dummy_f32 is None or _dummy_f32.device != device:
        _dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    return _dummy_f32

def linear_relu_square(
    a,
    b,
    aux=None,
    a_f8=None,
    b_f8=None,
    dequant_scale_ptr=None,
    activation_scale=None,
    partial_amax=None,
):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    use_fp8 = b_f8 is not None
    emit_fp8 = activation_scale is not None

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128 if use_fp8 else 64

    FORWARD = False
    if aux is None:
        FORWARD = True
        # Forward stores only `post` (into `c`); aux_desc is never accessed on the
        # forward path. Use a SEPARATE minimal [BM, BN//2] dummy (NOT `c`) so the two
        # TMA descriptors don't alias the same buffer (aliasing can perturb the
        # forward kernel's memory analysis / pipelining).
        aux = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), device=a.device, dtype=dtype)

    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_kernel = a_f8 if use_fp8 else a
    a_desc = TensorDescriptor.from_tensor(a_kernel, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_kernel = b_f8 if use_fp8 else b
    b_desc = TensorDescriptor.from_tensor(b_kernel, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    if emit_fp8:
        assert FORWARD and use_fp8 and partial_amax is not None
        assert partial_amax.numel() >= NUM_SMS
        post_fp8 = torch.empty((M, N), device=a.device, dtype=torch.float8_e4m3fn)
        post_fp8_desc = TensorDescriptor.from_tensor(
            post_fp8, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2]
        )
        num_stages = 3
    else:
        post_fp8 = None
        post_fp8_desc = aux_desc
        activation_scale = _get_dummy_f32(a.device)
        partial_amax = _get_dummy_f32(a.device)

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    if use_fp8:
        assert dequant_scale_ptr is not None
    else:
        # The unified Triton signature requires a pointer, but bf16 kernels never load it.
        dequant_scale_ptr = _get_dummy_f32(a.device)

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        post_fp8_desc, partial_amax,
        dequant_scale_ptr, activation_scale,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        USE_FP8=use_fp8,
        EMIT_FP8=emit_fp8,
        num_stages=num_stages,
        num_warps=num_warps
    )

    # On the forward path `c` now holds `post`; no separate `pre` tensor is produced.
    if emit_fp8:
        return c, post_fp8
    return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        W1,
        W2,
        W1_f8=None,
        dequant_scale=None,
        x_f8=None,
        W2_f8=None,
        W2_scale=None,
        activation_scale=None,
        partial_amax=None,
    ):
        # Forward stores only `post = relu(x @ W1.T)^2`; `pre` is never materialized.
        x_flat = x.view((-1, x.shape[-1]))
        if W1_f8 is not None:
            assert x_f8 is not None and dequant_scale is not None
            x_f8 = x_f8.view((-1, x_f8.shape[-1]))
            if W2_f8 is not None:
                # Also emit an FP8 copy of `post` (plus per-SM partial amax) so the down
                # projection can run through _scaled_mm.
                assert W2_scale is not None and activation_scale is not None
                assert partial_amax is not None
                post, post_f8 = linear_relu_square(
                    x_flat,
                    W1,
                    a_f8=x_f8,
                    b_f8=W1_f8,
                    dequant_scale_ptr=dequant_scale,
                    activation_scale=activation_scale,
                    partial_amax=partial_amax,
                )
            else:
                post = linear_relu_square(x_flat, W1, a_f8=x_f8, b_f8=W1_f8, dequant_scale_ptr=dequant_scale)
        else:
            post = linear_relu_square(x_flat, W1)
        if W2_f8 is not None:
            x3 = torch._scaled_mm(
                post_f8,
                W2_f8,
                out_dtype=torch.bfloat16,
                scale_a=activation_scale,
                scale_b=W2_scale,
                use_fast_accum=True,
            )
        else:
            x3 = post @ W2
        # Backward stays BF16: it consumes `post` (and W2), not their FP8 copies.
        ctx.save_for_backward(x, W1, W2, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        # dpre kernel reconstructs relu(pre) = sqrt(post) from `post` (passed as aux),
        # avoiding the redundant `pre` HBM read/write entirely.
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=post)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2, None, None, None, None, None, None, None


def reduce_mlp_activation_scales(partial_amax, scales, headroom=1.25):
    num_layers, partial_count = partial_amax.shape
    assert scales.numel() >= num_layers
    block_size = triton.next_power_of_2(partial_count)
    reduce_mlp_activation_scales_kernel[(num_layers,)](
        partial_amax,
        scales,
        partial_amax.stride(0),
        partial_count=partial_count,
        HEADROOM=headroom,
        BLOCK_SIZE=block_size,
        num_stages=1,
        num_warps=4,
    )


def quantize_transpose_mlp_down_weights(
    weights,
    output_storage,
    scales,
    row_output=None,
):
    """Quantize the MLP down weights with an exact-current scale.

    Always emits the transposed (model, hidden) storage used by the forward
    _scaled_mm. Pass `row_output` to additionally emit the row-major
    (hidden, model) copy that the FP8 backward's dpre GEMM requires; both come
    from a single read of the weights.
    """
    num_layers, hidden_dim, model_dim = weights.shape
    assert output_storage.shape == (num_layers, model_dim, hidden_dim)
    emit_row = row_output is not None
    if emit_row:
        assert row_output.shape == (num_layers, hidden_dim, model_dim)
        assert row_output.is_contiguous()
    block_h = 64
    block_d = 64
    num_tiles_d = triton.cdiv(model_dim, block_d)
    num_tiles = triton.cdiv(hidden_dim, block_h) * num_tiles_d
    quantize_transpose_mlp_down_weights_kernel[(num_layers, num_tiles)](
        weights,
        output_storage,
        row_output if emit_row else weights,  # unused pointer when EMIT_ROW is False
        scales,
        weights.stride(0),
        hidden_dim=hidden_dim,
        model_dim=model_dim,
        num_tiles_d=num_tiles_d,
        EMIT_ROW=emit_row,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_stages=1,
        num_warps=4,
    )


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

    Uses a 32x32 tiled Triton kernel with coalesced access on both src and dst,
    replacing PyTorch's .add_(src.T) which has non-coalesced reads from the
    transposed operand.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 32, 32
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

//__device__ float sigmoid(float x) {
//  return 1.0f / (1.0f + __expf(-x));
//}
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
    const float* __restrict__ prefix_weight_ptr,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param,
    double grad_scale_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;
  float prefix_weight = prefix_weight_ptr[0];

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

  // Prefix token prediction target for this position (T' = longest-prefix token of
  // the immediate next-token target T). prefix_targets[i] < 0 => no valid prefix, ignored.
  int64_t prefix_target = prefix_targets[blockIdx.x];
  bool prefix_valid = (prefix_target >= 0 && prefix_target < VOCAB_SIZE);

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
    // Same CE logic as MTP, but the target is the prefix token T' at this position.
    if (prefix_valid) {
      float z_target = A * __bfloat162float(smem[prefix_target]);
      total_loss += prefix_weight * (lse - z_target);
    }
    losses[blockIdx.x] = total_loss;
  }

  // Total weight over active predictions at this position (used in the softmax-normalizer
  // gradient term). Include the prefix prediction only when it has a valid target.
  float S_w = prefix_valid ? prefix_weight : 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
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

  // Sparse correction for target columns. Threads [0, n_predict) handle the future MTP
  // targets; thread n_predict handles the prefix target. term2 for a column sums the
  // weights of every prediction (MTP + prefix) whose target lands on that column, so
  // duplicate columns across threads write identical values (idempotent, race-free).
  if (threadIdx.x <= n_predict) {
    int i = threadIdx.x;
    int64_t target;
    bool valid;
    if (i < n_predict) {
      int64_t target_idx = blockIdx.x + i;
      valid = (target_idx < batch_size);
      target = valid ? targets[target_idx] : -1;
      valid = valid && (target >= 0 && target < VOCAB_SIZE);
    } else {
      target = prefix_target;
      valid = prefix_valid;
    }

    if (valid) {
      float sigmoid_u = __bfloat162float(smem[target]);
      float z = A * sigmoid_u;
      float p = __expf(z - lse);

      float term1 = S_w * p;
      float term2 = 0.0f;

      for (int k = 0; k < n_predict; k++) {
        int64_t target_idx = blockIdx.x + k;
        if (target_idx < batch_size && targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
      if (prefix_valid && prefix_target == target) {
        term2 += prefix_weight;
      }

      float grad_z = term1 - term2;
      float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
      auto result_tmp = f32_to_fp8_e5m2(grad_x);
      auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
    }
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

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_targets: torch.Tensor,
    prefix_weight: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
    grad_scale: float,
) -> None:
    grid = (n_rows, 1, 1)
    ce_fwd_bwd_kernel(
        grid,
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s, grad_scale),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, prefix_targets, prefix_weight, lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
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

        if prefix_targets is None:
            prefix_targets = torch.full((n_rows,), -1, dtype=torch.int64, device=logits.device)
        prefix_targets = prefix_targets.contiguous()

        if prefix_weight is None:
            prefix_weight = torch.zeros(1, dtype=torch.float32, device=logits.device)
        prefix_weight = prefix_weight.reshape(1).to(torch.float32).contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        ce_fwd_bwd(logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale)

        ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input = ctx.saved_tensors
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = logits.shape
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
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, None, None, None, None, grad_w, None, None, None

# -----------------------------------------------------------------------------
# FP8 MLP kernels and quantization helpers

@triton.jit
def _fp8_relu_square_forward_kernel(
    a_desc, b_desc, post_desc, post_t_desc,
    dequant_scale_ptr, post_scale_ptr, partial_amax_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    local_amax = 0.0

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_n, offs_k])
            acc = tl.dot(a, b.T, acc)
        acc *= tl.load(dequant_scale_ptr)

        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        post_scale = tl.load(post_scale_ptr)

        pre0 = acc0.to(tl.bfloat16)
        post0 = tl.maximum(pre0, 0.0)
        post0 *= post0
        post0_f32 = post0.to(tl.float32)
        q0 = (post0_f32 / post_scale).to(tl.float8e4nv)
        post_desc.store([offs_m, offs_n], q0)
        post_t_desc.store([offs_n, offs_m], tl.trans(q0))
        local_amax = tl.maximum(local_amax, tl.max(tl.max(post0_f32, axis=1), axis=0))

        pre1 = acc1.to(tl.bfloat16)
        post1 = tl.maximum(pre1, 0.0)
        post1 *= post1
        post1_f32 = post1.to(tl.float32)
        q1 = (post1_f32 / post_scale).to(tl.float8e4nv)
        n1 = offs_n + BLOCK_N // 2
        post_desc.store([offs_m, n1], q1)
        post_t_desc.store([n1, offs_m], tl.trans(q1))
        local_amax = tl.maximum(local_amax, tl.max(tl.max(post1_f32, axis=1), axis=0))

    tl.store(partial_amax_ptr + start_pid, local_amax)


@triton.jit
def _fp8_relu_square_backward_kernel(
    a_desc, b_desc, post_desc, dpre_desc, dpre_t_desc,
    dequant_scale_ptr, dpre_scale_ptr, post_scale_ptr, partial_amax_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    local_amax = 0.0

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_n, offs_k])
            acc = tl.dot(a, b.T, acc)
        acc *= tl.load(dequant_scale_ptr)

        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        dpre_scale = tl.load(dpre_scale_ptr)

        post0 = post_desc.load([offs_m, offs_n]).to(tl.float32) * tl.load(post_scale_ptr)
        dpre0 = 2.0 * acc0.to(tl.bfloat16) * tl.sqrt(post0).to(tl.bfloat16)
        dpre0_f32 = dpre0.to(tl.float32)
        q0 = (dpre0_f32 / dpre_scale).to(tl.float8e4nv)
        dpre_desc.store([offs_m, offs_n], q0)
        dpre_t_desc.store([offs_n, offs_m], tl.trans(q0))
        local_amax = tl.maximum(local_amax, tl.max(tl.max(tl.abs(dpre0_f32), axis=1), axis=0))

        n1 = offs_n + BLOCK_N // 2
        post1 = post_desc.load([offs_m, n1]).to(tl.float32) * tl.load(post_scale_ptr)
        dpre1 = 2.0 * acc1.to(tl.bfloat16) * tl.sqrt(post1).to(tl.bfloat16)
        dpre1_f32 = dpre1.to(tl.float32)
        q1 = (dpre1_f32 / dpre_scale).to(tl.float8e4nv)
        dpre_desc.store([offs_m, n1], q1)
        dpre_t_desc.store([n1, offs_m], tl.trans(q1))
        local_amax = tl.maximum(local_amax, tl.max(tl.max(tl.abs(dpre1_f32), axis=1), axis=0))

    tl.store(partial_amax_ptr + start_pid, local_amax)


@triton.jit
def _quantize_dual_layout_kernel(
    src, row, transposed, scale_ptr, partial_amax_ptr,
    M, N,
    src_stride_m, src_stride_n,
    NUM_PID_N: tl.constexpr,
    EMIT_AMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(
        src + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask,
        other=0.0,
    )
    scaled = x.to(tl.float32) / tl.load(scale_ptr)
    # E4M3FN overflow becomes NaN rather than infinity. Delayed scales can miss
    # a transient range spike, so saturate explicitly instead of poisoning all
    # downstream GEMMs from a single outlier.
    q = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row + offs_m[:, None] * N + offs_n[None, :], q, mask=mask)
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        transposed + offs_n[:, None] * M + offs_m[None, :],
        tl.trans(q),
        mask=transposed_mask,
    )
    if EMIT_AMAX:
        tile_amax = tl.max(tl.max(tl.abs(x.to(tl.float32)), axis=1), axis=0)
        tl.store(partial_amax_ptr + pid_m * NUM_PID_N + pid_n, tile_amax)


@triton.jit
def _quantize_dual_layout_batched_kernel(
    src, row, transposed, scale_ptr,
    M, N,
    src_stride_b, src_stride_m, src_stride_n,
    row_stride_b, transposed_stride_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(
        src + batch * src_stride_b + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask,
        other=0.0,
    )
    q = (x.to(tl.float32) / tl.load(scale_ptr + batch)).to(tl.float8e4nv)
    tl.store(
        row + batch * row_stride_b + offs_m[:, None] * N + offs_n[None, :],
        q,
        mask=mask,
    )
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        transposed + batch * transposed_stride_b + offs_n[:, None] * M + offs_m[None, :],
        tl.trans(q),
        mask=transposed_mask,
    )


@triton.jit
def _batched_matrix_amax_partial_kernel(
    src, partial_amax,
    MATRIX_ELEMENTS: tl.constexpr,
    SRC_STRIDE_B: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Read each matrix once and emit a small set of exact amax partials."""
    batch = tl.program_id(0)
    part = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    local_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in tl.range(
        part * BLOCK_SIZE, MATRIX_ELEMENTS, NUM_PARTS * BLOCK_SIZE,
    ):
        indices = start + offsets
        values = tl.load(
            src + batch * SRC_STRIDE_B + indices,
            mask=indices < MATRIX_ELEMENTS,
            other=0.0,
        ).to(tl.float32)
        local_amax = tl.maximum(local_amax, tl.abs(values))
    tl.store(
        partial_amax + batch * NUM_PARTS + part,
        tl.max(local_amax, axis=0),
    )


@triton.jit
def _packed_batched_matrix_amax_partial_kernel(
    first, second, partial_amax,
    FIRST_ELEMENTS: tl.constexpr,
    SECOND_ELEMENTS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Amax over a logical row-pack without materializing the concatenation."""
    batch = tl.program_id(0)
    part = tl.program_id(1)
    total_elements = FIRST_ELEMENTS + SECOND_ELEMENTS
    offsets = tl.arange(0, BLOCK_SIZE)
    local_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in tl.range(
        part * BLOCK_SIZE, total_elements, NUM_PARTS * BLOCK_SIZE,
    ):
        indices = start + offsets
        valid = indices < total_elements
        in_first = indices < FIRST_ELEMENTS
        first_values = tl.load(
            first + batch * FIRST_STRIDE_B + indices,
            mask=valid & in_first,
            other=0.0,
        )
        second_indices = tl.maximum(indices - FIRST_ELEMENTS, 0)
        second_values = tl.load(
            second + batch * SECOND_STRIDE_B + second_indices,
            mask=valid & ~in_first,
            other=0.0,
        )
        values = first_values.to(tl.float32) + second_values.to(tl.float32)
        local_amax = tl.maximum(local_amax, tl.abs(values))
    tl.store(
        partial_amax + batch * NUM_PARTS + part,
        tl.max(local_amax, axis=0),
    )


@triton.jit
def _reduce_mlp_weight_scales_kernel(
    partial_amax, up_scales, down_scales,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_PARTS
    up = tl.load(
        partial_amax + (2 * layer) * NUM_PARTS + offsets,
        mask=mask, other=0.0,
    )
    down = tl.load(
        partial_amax + (2 * layer + 1) * NUM_PARTS + offsets,
        mask=mask, other=0.0,
    )
    tl.store(
        up_scales + layer,
        tl.maximum(tl.max(up, axis=0), 1.0e-12) / 448.0,
    )
    tl.store(
        down_scales + layer,
        tl.maximum(tl.max(down, axis=0), 1.0e-12) / 448.0,
    )


@triton.jit
def _quantize_dual_layout_packed_batched_kernel(
    first, second, row, transposed, scale_ptr,
    M, N,
    FIRST_ROWS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    ROW_STRIDE_B: tl.constexpr,
    TRANSPOSED_STRIDE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Quantize a logical row-pack directly from its two source allocations."""
    batch = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_first = offs_m[:, None] < FIRST_ROWS
    first_offsets = offs_m[:, None] * N + offs_n[None, :]
    second_rows = tl.maximum(offs_m - FIRST_ROWS, 0)
    second_offsets = second_rows[:, None] * N + offs_n[None, :]
    source_ptrs = tl.where(
        in_first,
        first + batch * FIRST_STRIDE_B + first_offsets,
        second + batch * SECOND_STRIDE_B + second_offsets,
    )
    values = tl.load(
        source_ptrs,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    scaled = values / tl.load(scale_ptr + batch)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(
        row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :],
        quantized,
        mask=valid,
    )
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        transposed + batch * TRANSPOSED_STRIDE_B
        + offs_n[:, None] * M + offs_m[None, :],
        tl.trans(quantized),
        mask=transposed_mask,
    )


def quantize_dual_layout(
    src: torch.Tensor,
    scale: torch.Tensor,
    partial_amax: torch.Tensor | None = None,
    row: torch.Tensor | None = None,
    transposed: torch.Tensor | None = None,
):
    M, N = src.shape
    if row is None:
        assert transposed is None
        row = torch.empty((M, N), device=src.device, dtype=torch.float8_e4m3fn)
        transposed = torch.empty((N, M), device=src.device, dtype=torch.float8_e4m3fn)
    else:
        assert row.shape == (M, N) and row.dtype == torch.float8_e4m3fn
        assert transposed is not None
        assert transposed.shape == (N, M) and transposed.dtype == torch.float8_e4m3fn
    block_m, block_n = 64, 128
    num_pid_m, num_pid_n = triton.cdiv(M, block_m), triton.cdiv(N, block_n)
    emit_amax = partial_amax is not None
    if emit_amax:
        assert src.is_contiguous() and partial_amax.numel() >= num_pid_m * num_pid_n
    else:
        partial_amax = _get_dummy_f32(src.device)
    _quantize_dual_layout_kernel[(num_pid_m, num_pid_n)](
        src, row, transposed, scale, partial_amax,
        M, N, src.stride(0), src.stride(1),
        NUM_PID_N=num_pid_n, EMIT_AMAX=emit_amax,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_stages=2, num_warps=8,
    )
    return row, transposed


def quantize_dual_layout_batched(
    src: torch.Tensor,
    scales: torch.Tensor,
    row: torch.Tensor | None = None,
    transposed: torch.Tensor | None = None,
):
    B, M, N = src.shape
    assert scales.shape == (B,)
    if row is None:
        row = torch.empty((B, M, N), device=src.device, dtype=torch.float8_e4m3fn)
    else:
        assert row.shape == (B, M, N) and row.dtype == torch.float8_e4m3fn
    if transposed is None:
        transposed = torch.empty((B, N, M), device=src.device, dtype=torch.float8_e4m3fn)
    else:
        assert transposed.shape == (B, N, M)
        assert transposed.dtype == torch.float8_e4m3fn
    assert row.is_contiguous() and transposed.is_contiguous()
    block_m, block_n = 64, 128
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), B)
    _quantize_dual_layout_batched_kernel[grid](
        src, row, transposed, scales,
        M, N,
        src.stride(0), src.stride(1), src.stride(2),
        row.stride(0), transposed.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_stages=2, num_warps=8,
    )
    return row, transposed


def update_mlp_weight_scales(weights, partial_amax, up_scales, down_scales):
    """Compute exact per-projection scales without a full-size `abs` temporary."""
    B, projections, M, N = weights.shape
    assert projections == 2 and weights.is_contiguous()
    num_parts = partial_amax.shape[-1]
    assert partial_amax.shape == (B, projections, num_parts)
    assert up_scales.shape == down_scales.shape == (B,)
    flat = weights.view(B * projections, M, N)
    block_size = 2048
    _batched_matrix_amax_partial_kernel[(B * projections, num_parts)](
        flat, partial_amax,
        MATRIX_ELEMENTS=M * N,
        SRC_STRIDE_B=flat.stride(0),
        NUM_PARTS=num_parts,
        BLOCK_SIZE=block_size,
        num_stages=1,
        num_warps=8,
    )
    _reduce_mlp_weight_scales_kernel[(B,)](
        partial_amax, up_scales, down_scales,
        NUM_PARTS=num_parts,
        BLOCK_SIZE=triton.next_power_of_2(num_parts),
        num_stages=1,
        num_warps=4,
    )


def quantize_dual_layout_packed_batched(
    first, second, scales, partial_amax, row, transposed,
):
    """Exact-scale dual-layout quantization of a logical row concatenation."""
    B, first_rows, N = first.shape
    B2, second_rows, N2 = second.shape
    assert (B2, N2) == (B, N)
    assert first.device == second.device and first.dtype == second.dtype
    assert first.stride(1) == second.stride(1) == N
    assert first.stride(2) == second.stride(2) == 1
    total_rows = first_rows + second_rows
    assert scales.shape == (B,)
    num_parts = partial_amax.shape[-1]
    assert partial_amax.shape == (B, num_parts)
    assert row.shape == (B, total_rows, N) and row.dtype == torch.float8_e4m3fn
    assert transposed.shape == (B, N, total_rows)
    assert transposed.dtype == torch.float8_e4m3fn
    assert row.is_contiguous() and transposed.is_contiguous()

    block_size = 2048
    _packed_batched_matrix_amax_partial_kernel[(B, num_parts)](
        first, second, partial_amax,
        FIRST_ELEMENTS=first_rows * N,
        SECOND_ELEMENTS=second_rows * N,
        FIRST_STRIDE_B=first.stride(0),
        SECOND_STRIDE_B=second.stride(0),
        NUM_PARTS=num_parts,
        BLOCK_SIZE=block_size,
        num_stages=1,
        num_warps=8,
    )
    reduce_mlp_activation_scales(partial_amax, scales, headroom=1.0)

    block_m, block_n = 64, 128
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(N, block_n), B)
    _quantize_dual_layout_packed_batched_kernel[grid](
        first, second, row, transposed, scales,
        total_rows, N,
        FIRST_ROWS=first_rows,
        FIRST_STRIDE_B=first.stride(0),
        SECOND_STRIDE_B=second.stride(0),
        ROW_STRIDE_B=row.stride(0),
        TRANSPOSED_STRIDE_B=transposed.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_stages=2,
        num_warps=8,
    )
    return row, transposed


def fp8_relu_square_forward(
    x_f8: torch.Tensor,
    w_f8: torch.Tensor,
    dequant_scale: torch.Tensor,
    post_scale: torch.Tensor,
    partial_amax: torch.Tensor | None = None,
):
    M, K = x_f8.shape
    N, Kw = w_f8.shape
    assert K == Kw
    post = torch.empty((M, N), device=x_f8.device, dtype=torch.float8_e4m3fn)
    post_t = torch.empty((N, M), device=x_f8.device, dtype=torch.float8_e4m3fn)
    num_sms = torch.cuda.get_device_properties(x_f8.device).multi_processor_count
    if partial_amax is None:
        partial_amax = torch.empty(num_sms, device=x_f8.device, dtype=torch.float32)
    block_m, block_n, block_k = 128, 256, 128
    grid = (min(num_sms, triton.cdiv(M, block_m) * triton.cdiv(N, block_n)),)
    _fp8_relu_square_forward_kernel[grid](
        TensorDescriptor.from_tensor(x_f8, [block_m, block_k]),
        TensorDescriptor.from_tensor(w_f8, [block_n, block_k]),
        TensorDescriptor.from_tensor(post, [block_m, block_n // 2]),
        TensorDescriptor.from_tensor(post_t, [block_n // 2, block_m]),
        dequant_scale, post_scale, partial_amax,
        M, N, K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, NUM_SMS=num_sms,
        num_stages=3, num_warps=8,
    )
    return post, post_t, partial_amax


def fp8_relu_square_backward(
    grad_f8: torch.Tensor,
    w_f8: torch.Tensor,
    dequant_scale: torch.Tensor,
    post: torch.Tensor,
    dpre_scale: torch.Tensor,
    post_scale: torch.Tensor,
    partial_amax: torch.Tensor | None = None,
):
    M, K = grad_f8.shape
    N, Kw = w_f8.shape
    assert K == Kw and post.shape == (M, N)
    dpre = torch.empty((M, N), device=grad_f8.device, dtype=torch.float8_e4m3fn)
    dpre_t = torch.empty((N, M), device=grad_f8.device, dtype=torch.float8_e4m3fn)
    num_sms = torch.cuda.get_device_properties(grad_f8.device).multi_processor_count
    if partial_amax is None:
        partial_amax = torch.empty(num_sms, device=grad_f8.device, dtype=torch.float32)
    block_m, block_n, block_k = 128, 256, 128
    grid = (min(num_sms, triton.cdiv(M, block_m) * triton.cdiv(N, block_n)),)
    _fp8_relu_square_backward_kernel[grid](
        TensorDescriptor.from_tensor(grad_f8, [block_m, block_k]),
        TensorDescriptor.from_tensor(w_f8, [block_n, block_k]),
        TensorDescriptor.from_tensor(post, [block_m, block_n // 2]),
        TensorDescriptor.from_tensor(dpre, [block_m, block_n // 2]),
        TensorDescriptor.from_tensor(dpre_t, [block_n // 2, block_m]),
        dequant_scale, dpre_scale, post_scale, partial_amax,
        M, N, K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, NUM_SMS=num_sms,
        num_stages=3, num_warps=8,
    )
    return dpre, dpre_t, partial_amax


def _scaled_mm(a, b, a_scale, b_scale, *, fast_accum=False):
    return torch._scaled_mm(
        a,
        b,
        out_dtype=torch.bfloat16,
        scale_a=a_scale,
        scale_b=b_scale,
        use_fast_accum=fast_accum,
    )


class FusedFP8MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        W1,
        W2,
        W1_f8,
        W1_f8_t,
        W2_f8,
        W2_f8_t,
        W1_scale,
        W2_scale,
        x_scale,
        post_scale,
        dpre_scale,
        post_partial_amax,
        dpre_partial_amax,
        x_f8,
        x_f8_t,
    ):
        x_flat = x.view((-1, x.shape[-1]))
        post_f8, post_f8_t, _ = fp8_relu_square_forward(
            x_f8.view_as(x_flat),
            W1_f8,
            x_scale * W1_scale,
            post_scale,
            post_partial_amax,
        )
        out = _scaled_mm(post_f8, W2_f8_t.T, post_scale, W2_scale, fast_accum=True)
        ctx.save_for_backward(
            W1_f8_t,
            W2_f8,
            post_f8,
            post_f8_t,
            x_f8_t,
            W1_scale,
            W2_scale,
            x_scale,
            post_scale,
            dpre_scale,
            dpre_partial_amax,
        )
        ctx.input_shape = x.shape
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        (
            W1_f8_t,
            W2_f8,
            post_f8,
            post_f8_t,
            x_f8_t,
            W1_scale,
            W2_scale,
            x_scale,
            post_scale,
            dpre_scale,
            dpre_partial_amax,
        ) = ctx.saved_tensors
        grad = grad_output.view((-1, grad_output.shape[-1])).contiguous()
        # Output gradients are substantially more volatile than RMS-normalized
        # MLP inputs, so keep their scale exact-current rather than clipping to a
        # one-step-lagged range.
        grad_scale = (grad.detach().abs().amax().float().clamp_min(1e-12) / 448.0).view(1)
        grad_f8, grad_f8_t = quantize_dual_layout(grad, grad_scale)
        dW2 = _scaled_mm(
            post_f8_t, grad_f8_t.T, post_scale, grad_scale,
            fast_accum=False,
        )
        dpre_f8, dpre_f8_t, _ = fp8_relu_square_backward(
            grad_f8,
            W2_f8,
            grad_scale * W2_scale,
            post_f8,
            dpre_scale,
            post_scale,
            dpre_partial_amax,
        )
        dW1 = _scaled_mm(
            dpre_f8_t, x_f8_t.T, dpre_scale, x_scale,
            fast_accum=False,
        )
        dx = _scaled_mm(
            dpre_f8, W1_f8_t.T, dpre_scale, W1_scale,
            fast_accum=False,
        )
        return (
            dx.view(ctx.input_shape),
            dW1,
            dW2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

# -----------------------------------------------------------------------------
# Reduced-dimensional QK and packed QKV kernels

@triton.jit
def _qk_norm_rope_pad_forward_kernel(
    qk, factor1, factor2, out_q, out_k,
    tokens: tl.constexpr,
    rows: tl.constexpr,
    num_heads: tl.constexpr,
    heads2: tl.constexpr,
    qk_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    padded_dim: tl.constexpr,
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
    x_ptrs = (
        qk + token[:, None] * stride_qkt + input_head[:, None] * stride_qkh
        + offs_d[None, :]
    )
    x = tl.load(
        x_ptrs,
        mask=mask, other=0.0,
    )
    # RoPE swaps adjacent lanes.  The full Q/K tile is already resident, so
    # form the swapped view in registers instead of reading it from HBM again.
    x_flip = tl.reshape(
        tl.flip(tl.reshape(x, (BLOCK_M, BLOCK_D // 2, 2)), dim=2),
        (BLOCK_M, BLOCK_D),
    )
    x = x.to(tl.float32)
    x_flip = x_flip.to(tl.float32)
    rstd = tl.rsqrt(
        tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7
    )
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
    f1 = tl.load(
        factor1 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_d[None, :]
        if PAIRED else factor1 + token[:, None] * factor_stride_t + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    f2 = tl.load(
        factor2 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_d[None, :]
        if PAIRED else factor2 + token[:, None] * factor_stride_t + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    normalized = x * rstd[:, None]
    normalized_flip = x_flip * rstd[:, None]
    y = f1 * normalized + f2 * normalized_flip

    if KEY_OFFSET:
        shift_row = (input_head >= num_heads) & (token > 0)
        previous_token = tl.maximum(token - 1, 0)
        x_previous = tl.load(
            qk + previous_token[:, None] * stride_qkt + input_head[:, None] * stride_qkh
            + offs_d[None, :],
            mask=mask & shift_row[:, None], other=0.0,
        ).to(tl.float32)
        previous_rstd = tl.rsqrt(
            tl.sum(x_previous * x_previous, axis=1) / qk_dim
            + 1.1920928955078125e-7
        )
        shift = shift_row[:, None] & (offs_d[None, :] >= rotary_dim)
        y = tl.where(shift, x_previous * previous_rstd[:, None], y)

    output_ptrs = (
        output_token[:, None] * stride_out_t
        + output_head[:, None] * padded_dim + offs_d[None, :]
    )
    tl.store(
        out_q + output_ptrs,
        y,
        mask=mask & (input_head[:, None] < num_heads),
    )
    tl.store(
        out_k + output_ptrs,
        y,
        mask=mask & (input_head[:, None] >= num_heads),
    )
    padding = padded_dim - qk_dim
    padding_ptrs = output_ptrs + qk_dim
    padding_mask = (offs_m[:, None] < rows) & (offs_d[None, :] < padding)
    tl.store(
        out_q + padding_ptrs, 0.0,
        mask=padding_mask & (input_head[:, None] < num_heads),
    )
    tl.store(
        out_k + padding_ptrs, 0.0,
        mask=padding_mask & (input_head[:, None] >= num_heads),
    )


@triton.jit
def _qk_norm_rope_pad_backward_kernel(
    grad_q, grad_k, qk, factor1, factor2, grad_qk,
    tokens: tl.constexpr,
    rows: tl.constexpr,
    num_heads: tl.constexpr,
    heads2: tl.constexpr,
    qk_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    padded_dim: tl.constexpr,
    stride_qkt: tl.constexpr,
    stride_qkh: tl.constexpr,
    factor_stride_t: tl.constexpr,
    stride_grad_t: tl.constexpr,
    PAIRED: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < rows) & (offs_d[None, :] < qk_dim)
    offs_flip = offs_d ^ 1

    token = offs_m // heads2
    input_head = offs_m % heads2
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

    output_ptrs = (
        output_token[:, None] * stride_grad_t
        + output_head[:, None] * padded_dim + offs_d[None, :]
    )
    is_query = input_head[:, None] < num_heads

    x = tl.load(
        qk + token[:, None] * stride_qkt + input_head[:, None] * stride_qkh
        + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    grad = tl.where(
        is_query,
        tl.load(grad_q + output_ptrs, mask=mask & is_query, other=0.0),
        tl.load(grad_k + output_ptrs, mask=mask & ~is_query, other=0.0),
    )
    grad_flip = tl.reshape(
        tl.flip(tl.reshape(grad, (BLOCK_M, BLOCK_D // 2, 2)), dim=2),
        (BLOCK_M, BLOCK_D),
    )
    grad = grad.to(tl.float32)
    grad_flip = grad_flip.to(tl.float32)
    f1 = tl.load(
        factor1 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_d[None, :]
        if PAIRED else factor1 + token[:, None] * factor_stride_t + offs_d[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    f2_flip = tl.load(
        factor2 + token[:, None] * factor_stride_t + factor_offset[:, None] + offs_flip[None, :]
        if PAIRED else factor2 + token[:, None] * factor_stride_t + offs_flip[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)

    rstd = tl.rsqrt(
        tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7
    )
    normalized = x * rstd[:, None]
    grad_normalized = f1 * grad + f2_flip * grad_flip
    if KEY_OFFSET:
        next_token = tl.minimum(token + 1, tokens - 1)
        grad_next = tl.load(
            grad_k + next_token[:, None] * stride_grad_t
            + logical_head[:, None] * padded_dim + offs_d[None, :],
            mask=mask & ~is_query & (token[:, None] < tokens - 1), other=0.0,
        ).to(tl.float32)
        stationary_grad = tl.where(
            token[:, None] == 0, grad + grad_next, grad_next
        )
        stationary = (
            (input_head[:, None] >= num_heads)
            & (offs_d[None, :] >= rotary_dim)
        )
        grad_normalized = tl.where(stationary, stationary_grad, grad_normalized)
    correction = tl.sum(grad_normalized * normalized, axis=1) / qk_dim
    dx = rstd[:, None] * (
        grad_normalized - normalized * correction[:, None]
    )
    tl.store(
        grad_qk + offs_m[:, None] * qk_dim + offs_d[None, :],
        dx,
        mask=mask,
    )


def qk_norm_rope_pad_forward(
    qk, factor1, factor2, num_heads, padded_dim,
    paired=False, key_offset=False, block_m=None, num_warps=2,
):
    tokens, heads2, qk_dim = qk.shape
    rotary_dim = qk_dim // 2
    factor_dim = qk_dim * (2 if paired else 1)
    assert heads2 == 2 * num_heads
    assert factor1.shape == factor2.shape == (tokens, factor_dim)
    assert not (paired and key_offset)
    assert qk_dim <= padded_dim
    # Shifted keys require an extra token-row load and favor smaller row tiles.
    if block_m is None:
        block_m = 4 if key_offset else 8
    output_tokens = tokens * (2 if paired else 1)
    output_heads = num_heads // 2 if paired else num_heads
    # Keep Q and K in independent allocations. Returning views into one shared
    # allocation makes functionalization clone both views before this mutating
    # Triton call and copy them back afterwards to preserve alias semantics.
    output_shape = (output_tokens, output_heads, padded_dim)
    out_q = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    out_k = torch.empty(output_shape, device=qk.device, dtype=qk.dtype)
    rows = tokens * heads2
    block_d = triton.next_power_of_2(qk_dim)
    _qk_norm_rope_pad_forward_kernel[(triton.cdiv(rows, block_m),)](
        qk, factor1, factor2, out_q, out_k,
        tokens=tokens, rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=rotary_dim, padded_dim=padded_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_out_t=out_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_M=block_m, BLOCK_D=block_d, num_warps=num_warps,
    )
    return out_q, out_k


def qk_norm_rope_pad_backward(
    grad_q, grad_k, qk, factor1, factor2, num_heads, paired=False, key_offset=False,
    block_m=8, num_warps=4,
):
    tokens, heads2, qk_dim = qk.shape
    rotary_dim = qk_dim // 2
    padded_dim = grad_q.shape[-1]
    grad_qk = torch.empty_like(qk)
    rows = tokens * heads2
    block_d = triton.next_power_of_2(qk_dim)
    _qk_norm_rope_pad_backward_kernel[(triton.cdiv(rows, block_m),)](
        grad_q, grad_k, qk, factor1, factor2, grad_qk,
        tokens=tokens, rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=rotary_dim, padded_dim=padded_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_grad_t=grad_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_M=block_m, BLOCK_D=block_d, num_warps=num_warps,
    )
    return grad_qk


class QKNormRoPEPadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, qk, factor1, factor2, num_heads, padded_dim, paired, key_offset,
    ):
        out_q, out_k = qk_norm_rope_pad_forward(
            qk, factor1, factor2, num_heads, padded_dim, paired, key_offset,
        )
        ctx.save_for_backward(qk, factor1, factor2)
        ctx.num_heads = num_heads
        ctx.paired = paired
        ctx.key_offset = key_offset
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        qk, factor1, factor2 = ctx.saved_tensors
        grad_qk = qk_norm_rope_pad_backward(
            grad_q, grad_k, qk, factor1, factor2,
            ctx.num_heads, ctx.paired, ctx.key_offset,
        )
        return grad_qk, None, None, None, None, None, None


QKNormRoPEPad = QKNormRoPEPadFunction.apply


@triton.jit
def _qkv_norm_rope_pack_fp8_backward_kernel(
    grad_q, grad_k, grad_v, qk, factor1, factor2,
    grad_row, grad_transposed, grad_scale_ptr,
    tokens: tl.constexpr,
    num_heads: tl.constexpr,
    qk_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    head_dim: tl.constexpr,
    padded_dim: tl.constexpr,
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
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    logical_head = tl.program_id(1)
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

    factor_ptrs = (
        token[:, None] * factor_stride_t + factor_offset + offs_d[None, :]
        if PAIRED else token[:, None] * factor_stride_t + offs_d[None, :]
    )
    factor_flip_ptrs = (
        token[:, None] * factor_stride_t + factor_offset + offs_flip[None, :]
        if PAIRED else token[:, None] * factor_stride_t + offs_flip[None, :]
    )
    f1 = tl.load(factor1 + factor_ptrs, mask=qk_mask, other=0.0).to(tl.float32)
    f2_flip = tl.load(
        factor2 + factor_flip_ptrs, mask=qk_mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(grad_scale_ptr)

    for qk_kind in tl.static_range(2):
        input_head = logical_head + qk_kind * num_heads
        x = tl.load(
            qk + token[:, None] * stride_qkt + input_head * stride_qkh
            + offs_d[None, :],
            mask=qk_mask, other=0.0,
        ).to(tl.float32)
        if qk_kind == 0:
            grad_ptr = (
                grad_q + output_token[:, None] * stride_grad_q_t
                + output_head * stride_grad_q_h + offs_d[None, :]
            )
        else:
            grad_ptr = (
                grad_k + output_token[:, None] * stride_grad_k_t
                + output_head * stride_grad_k_h + offs_d[None, :]
            )
        grad = tl.load(grad_ptr, mask=qk_mask, other=0.0)
        grad_flip = tl.reshape(
            tl.flip(
                tl.reshape(grad, (BLOCK_T, BLOCK_QK // 2, 2)), dim=2
            ),
            (BLOCK_T, BLOCK_QK),
        )
        grad = grad.to(tl.float32)
        grad_flip = grad_flip.to(tl.float32)

        rstd = tl.rsqrt(
            tl.sum(x * x, axis=1) / qk_dim + 1.1920928955078125e-7
        )
        normalized = x * rstd[:, None]
        grad_normalized = f1 * grad + f2_flip * grad_flip
        if KEY_OFFSET and qk_kind == 1:
            next_token = tl.minimum(token + 1, tokens - 1)
            grad_next = tl.load(
                grad_k + next_token[:, None] * stride_grad_k_t
                + output_head * stride_grad_k_h + offs_d[None, :],
                mask=qk_mask & (token[:, None] < tokens - 1), other=0.0,
            ).to(tl.float32)
            stationary_grad = tl.where(
                token[:, None] == 0, grad + grad_next, grad_next
            )
            grad_normalized = tl.where(
                offs_d[None, :] >= rotary_dim,
                stationary_grad,
                grad_normalized,
            )
        correction = tl.sum(grad_normalized * normalized, axis=1) / qk_dim
        dx = rstd[:, None] * (
            grad_normalized - normalized * correction[:, None]
        )
        # Match the old BF16 QK-gradient materialization before FP8 conversion.
        q = (dx.to(tl.bfloat16).to(tl.float32) / scale).to(tl.float8e4nv)
        feature = input_head * qk_dim + offs_d
        tl.store(
            grad_row + token[:, None] * qkv_dim + feature[None, :],
            q,
            mask=qk_mask,
        )
        tl.store(
            grad_transposed + feature[:, None] * tokens + token[None, :],
            tl.trans(q),
            mask=(offs_d[:, None] < qk_dim) & token_mask[None, :],
        )

    offs_v = tl.arange(0, BLOCK_V)
    v_mask = token_mask[:, None] & (offs_v[None, :] < head_dim)
    grad_v_value = tl.load(
        grad_v + token[:, None] * stride_grad_v_t
        + logical_head * stride_grad_v_h + offs_v[None, :],
        mask=v_mask, other=0.0,
    ).to(tl.float32)
    qv = (grad_v_value / scale).to(tl.float8e4nv)
    v_feature = 2 * num_heads * qk_dim + logical_head * head_dim + offs_v
    tl.store(
        grad_row + token[:, None] * qkv_dim + v_feature[None, :],
        qv,
        mask=v_mask,
    )
    tl.store(
        grad_transposed + v_feature[:, None] * tokens + token[None, :],
        tl.trans(qv),
        mask=(offs_v[:, None] < head_dim) & token_mask[None, :],
    )


def qkv_norm_rope_pack_fp8_backward(
    grad_q, grad_k, grad_v, qk, factor1, factor2,
    grad_scale, num_heads, head_dim, paired=False, key_offset=False,
):
    tokens, heads2, qk_dim = qk.shape
    rotary_dim = qk_dim // 2
    assert heads2 == 2 * num_heads
    assert grad_v.shape == (tokens, num_heads, head_dim)
    assert not (paired and key_offset)
    padded_dim = grad_q.shape[-1]
    qkv_dim = 2 * num_heads * qk_dim + num_heads * head_dim
    grad_row = torch.empty(
        (tokens, qkv_dim), device=qk.device, dtype=torch.float8_e4m3fn
    )
    grad_transposed = torch.empty(
        (qkv_dim, tokens), device=qk.device, dtype=torch.float8_e4m3fn
    )
    block_t = 16 if qk_dim <= 64 else 8
    num_token_blocks = triton.cdiv(tokens, block_t)
    _qkv_norm_rope_pack_fp8_backward_kernel[
        (num_token_blocks, num_heads)
    ](
        grad_q, grad_k, grad_v, qk, factor1, factor2,
        grad_row, grad_transposed, grad_scale,
        tokens=tokens, num_heads=num_heads, qk_dim=qk_dim, rotary_dim=rotary_dim,
        head_dim=head_dim, padded_dim=padded_dim, qkv_dim=qkv_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0),
        stride_grad_q_t=grad_q.stride(0), stride_grad_q_h=grad_q.stride(1),
        stride_grad_k_t=grad_k.stride(0), stride_grad_k_h=grad_k.stride(1),
        stride_grad_v_t=grad_v.stride(0), stride_grad_v_h=grad_v.stride(1),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_T=block_t,
        BLOCK_QK=triton.next_power_of_2(qk_dim),
        BLOCK_V=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return grad_row, grad_transposed


class PackedFP8QKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, qk_weight, v_weight, weight_f8, weight_f8_t,
        weight_scale, qkv_scale, x_scale, grad_scale,
        factor1, factor2, num_heads, paired, key_offset,
    ):
        qk_dim = qk_weight.shape[0] // (2 * num_heads)
        head_dim = v_weight.shape[0] // num_heads
        attn_qk_dim = qk_dim if qk_dim <= 64 else head_dim
        qk_features = 2 * num_heads * qk_dim
        x_flat = x.reshape(-1, x.shape[-1])
        x_f8, x_f8_t = quantize_dual_layout(x_flat, x_scale)
        scaled_weight_scale = weight_scale * qkv_scale
        if paired:
            # A packed QKV result leaves V as a strided suffix of every token
            # row.  Paired attention must then materialize V to merge the token
            # and head axes.  Two column-sliced GEMMs read X once more but make
            # V dense at birth, avoiding the larger BF16 read/write repack.
            qk = torch._scaled_mm(
                x_f8, weight_f8[:qk_features].T,
                out_dtype=torch.bfloat16,
                scale_a=x_scale,
                scale_b=scaled_weight_scale,
                use_fast_accum=True,
            ).view(-1, 2 * num_heads, qk_dim)
            v = torch._scaled_mm(
                x_f8, weight_f8[qk_features:].T,
                out_dtype=torch.bfloat16,
                scale_a=x_scale,
                scale_b=scaled_weight_scale,
                use_fast_accum=True,
            ).view(-1, num_heads, head_dim)
        else:
            qkv = torch._scaled_mm(
                x_f8, weight_f8.T,
                out_dtype=torch.bfloat16,
                scale_a=x_scale,
                scale_b=scaled_weight_scale,
                use_fast_accum=True,
            )
            qk = qkv[:, :qk_features].view(-1, 2 * num_heads, qk_dim)
            v = qkv[:, qk_features:].view(-1, num_heads, head_dim)
        q, k = qk_norm_rope_pad_forward(
            qk, factor1, factor2, num_heads, attn_qk_dim,
            paired, key_offset,
        )
        ctx.save_for_backward(
            qk_weight, v_weight,
            x_f8_t, weight_f8_t, scaled_weight_scale, qkv_scale,
            x_scale, grad_scale,
            qk, factor1, factor2,
        )
        ctx.input_shape = x.shape
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.paired = paired
        ctx.key_offset = key_offset
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        (
            qk_weight, v_weight,
            x_f8_t, weight_f8_t, scaled_weight_scale, qkv_scale,
            x_scale, grad_scale,
            qk, factor1, factor2,
        ) = ctx.saved_tensors
        grad_v = grad_v.reshape(-1, ctx.num_heads, ctx.head_dim)
        grad_f8, grad_f8_t = qkv_norm_rope_pack_fp8_backward(
            grad_q, grad_k, grad_v, qk, factor1, factor2,
            grad_scale, ctx.num_heads, ctx.head_dim,
            ctx.paired, ctx.key_offset,
        )
        grad_weight = torch._scaled_mm(
            grad_f8_t, x_f8_t.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=x_scale,
            use_fast_accum=False,
        )
        grad_input = torch._scaled_mm(
            grad_f8, weight_f8_t.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=scaled_weight_scale,
            use_fast_accum=False,
        )
        grad_qkv_scale = (
            grad_weight * torch.cat((qk_weight, v_weight))
        ).sum()
        qk_features = 2 * ctx.num_heads * qk.shape[-1]
        grad_qk_weight, grad_v_weight = grad_weight.split(
            (qk_features, ctx.num_heads * ctx.head_dim)
        )
        grad_qk_weight = grad_qk_weight * qkv_scale
        grad_v_weight = grad_v_weight * qkv_scale
        return (
            grad_input.view(ctx.input_shape), grad_qk_weight, grad_v_weight,
            None, None, None, grad_qkv_scale, None, None, None, None,
            None, None, None,
        )


PackedFP8QKV = PackedFP8QKVFunction.apply

"""Hand-fused kernels for the ANVIL optimizer tail. Every kernel here is a drop-in for a chain of inductor / ATen
kernels and is written to reproduce that chain's arithmetic bit for bit (same fp32 operation order, same fma
contractions); this was verified by byte-comparing the fused path against the stock path in offline tests (the
comparison is not repeated at start-up), not argued."""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from track_1_short.perf.kernels import polar_express as _pe


@triton.jit
def _pre_kernel(G, V, X, MOM, BF, BW, n, BLOCK: tl.constexpr):
    """velocity rails + Nesterov blend + bf16 cast in ONE pass (stock: three inductor kernels and a 52 MB scratch
    copy of the fp32 rails for the mlp bank).  V is [2, n] fp32 updated IN PLACE, G [n] bf16, X [n] bf16 out."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    v0 = tl.load(V + offs, mask=m, other=0.0)
    v1 = tl.load(V + n + offs, mask=m, other=0.0)
    bf = tl.load(BF)
    mom = tl.load(MOM)
    bw = tl.load(BW)
    # fast rail: aten lerp(v0, g, 1 - bf) as inductor lowers it
    w0 = 1.0 - bf
    ge0 = tl.abs(w0) >= 0.5
    w0 = tl.where(ge0, w0 - 1.0, w0)
    v0n = tl.math.fma(w0, g - v0, tl.where(ge0, g, v0))
    # slow rail: weight 1 - 0.98 < 0.5
    v1n = tl.math.fma(0.019999999552965164, g - v1, v1)
    tl.store(V + offs, v0n, mask=m)
    tl.store(V + n + offs, v1n, mask=m)
    # blend + Nesterov lookahead: g.lerp_(m_eff, mom)
    wm = 1.0 - bw
    m_eff = tl.math.fma(bw, v0n, wm * v1n)
    ge1 = tl.abs(mom) >= 0.5
    wl = tl.where(ge1, mom - 1.0, mom)
    xo = tl.math.fma(wl, m_eff - g, tl.where(ge1, m_eff, g))
    tl.store(X + offs, xo.to(tl.bfloat16), mask=m)


def pre_fused(grad, velocity, out, mom_t, bf_t, bw_t, block=2048, num_warps=8):
    assert grad.is_contiguous() and velocity.is_contiguous() and out.is_contiguous()
    n = grad.numel()
    assert velocity.numel() == 2 * n and out.numel() == n
    _pre_kernel[(triton.cdiv(n, block),)](grad, velocity, out, mom_t, bf_t, bw_t, n, BLOCK=block, num_warps=num_warps)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Gram-family kernels under finer tiles. Same K sequence per output element => same bytes (verified by byte-comparing
# against the stock tiles in offline tests).

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) per (kernel, output edge); picked by offline tuning
_GRAM_CFG = {
    ("XTX", 768): (64, 128, 128, 4, 3),
    ("XXT", 768): (64, 128, 128, 4, 3),
    ("XXT", 256): (64, 64, 128, 4, 3),
    ("baAA", 768): (64, 128, 128, 4, 3),
    ("baAA", 256): (64, 64, 64, 4, 4),
}


def gram_tall(X, out):
    """out = X^T X (symmetric), X [B, M, K]."""
    B, M, K = X.shape
    bm, bn, bk, nw, ns = _GRAM_CFG[("XTX", K)]
    _pe.XTX_kernel[(B * triton.cdiv(K, bm) * triton.cdiv(K, bn),)](
        A_ptr=X, C_ptr=out, M=M, K=K, a_stride_b=X.stride(0), a_stride_r=X.stride(1), a_stride_c=X.stride(2),
        c_stride_b=out.stride(0), c_stride_r=out.stride(1), c_stride_c=out.stride(2),
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8, LOWER_UPPER=1, num_stages=ns, num_warps=nw)
    return out


@triton.jit
def _xxt_copy_kernel(
    A_ptr, C_ptr, COPY_ptr,
    M, K, num_gram,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, LOWER_UPPER: tl.constexpr,
):
    """polar_express.XXT_kernel, line for line, for pid < num_gram; the CTAs past that range copy A into COPY
    (same layout), one [BLOCK_SIZE_M, K] strip each. That replaces the separate memcpy the defensive-copy rule of
    baddbmm costs, on SMs the Gram leaves idle, with the Gram CTAs' code path untouched (a store of the dot
    operand from inside the Gram loop changes the compiler's MMA choice and with it the bytes)."""
    pid = tl.program_id(axis=0)
    if pid >= num_gram:
        cpid = pid - num_gram
        n_rb = tl.cdiv(M, BLOCK_SIZE_M)
        cb = cpid // n_rb
        rb = cpid % n_rb
        rows = rb * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rmask = rows < M
        for k0 in range(0, K, BLOCK_SIZE_K):
            cols = k0 + tl.arange(0, BLOCK_SIZE_K)
            cm = rmask[:, None] & (cols[None, :] < K)
            offs = cb * a_stride_b + rows[:, None] * a_stride_r + cols[None, :] * a_stride_c
            tl.store(COPY_ptr + offs, tl.load(A_ptr + offs, mask=cm, other=0.0), mask=cm)
        return
    batch_idx, m_idx, n_idx = _pe._pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
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
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def gram_wide(X, out, copy=None):
    """out = X X^T (symmetric), X [B, M, K]; with `copy`, also copy = X (bitwise)."""
    B, M, K = X.shape
    bm, bn, bk, nw, ns = _GRAM_CFG[("XXT", M)]
    n_gram = B * triton.cdiv(M, bm) * triton.cdiv(M, bn)
    grid = (n_gram,)
    common = dict(M=M, K=K, a_stride_b=X.stride(0), a_stride_r=X.stride(1), a_stride_c=X.stride(2),
                  c_stride_b=out.stride(0), c_stride_r=out.stride(1), c_stride_c=out.stride(2),
                  BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8, LOWER_UPPER=1, num_stages=ns, num_warps=nw)
    if copy is None:
        _pe.XXT_kernel[grid](A_ptr=X, C_ptr=out, **common)
    else:
        assert copy.is_contiguous() and X.is_contiguous() and copy.shape == X.shape
        assert X.stride(0) == copy.stride(0) and X.stride(1) == copy.stride(1) and X.stride(2) == copy.stride(2)
        _xxt_copy_kernel[(n_gram + B * triton.cdiv(M, bm),)](A_ptr=X, C_ptr=out, COPY_ptr=copy, num_gram=n_gram, **common)
    return out


def poly_gram(A, alpha, beta, out):
    """out = beta * A + alpha * A @ A (A symmetric [B, M, M])."""
    B, M, _ = A.shape
    bm, bn, bk, nw, ns = _GRAM_CFG[("baAA", M)]
    _pe.ba_plus_cAA_kernel[(B * triton.cdiv(M, bm) * triton.cdiv(M, bn),)](
        A_ptr=A, C_ptr=out, M=M, a_stride_b=A.stride(0), a_stride_r=A.stride(1), a_stride_c=A.stride(2),
        c_stride_b=out.stride(0), c_stride_r=out.stride(1), c_stride_c=out.stride(2), alpha=alpha, beta=beta,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8, LOWER_UPPER=1, num_stages=ns, num_warps=nw)
    return out


@torch.compile(dynamic=False, fullgraph=True)
def gram_norm(X, A):
    """The stock normalization of the cascade, verbatim: unit-ish Frobenius scale from the Gram trace."""
    tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
    d = tr.sqrt() * 1.05 + 1e-6
    X = (X.float() / d).bfloat16()
    A = (A.float() / d.square()).bfloat16()
    return X, A


@torch.compile(dynamic=False, fullgraph=True)
def gram_norm_dup(X, A):
    """gram_norm that also returns a second copy of the normalized X (one more store in the same kernel)."""
    tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
    d = tr.sqrt() * 1.05 + 1e-6
    X = (X.float() / d).bfloat16()
    A = (A.float() / d.square()).bfloat16()
    return X, A, X.clone()


# ---- tall step: X' = bf16( bf16(X @ B) + a * X ), B symmetric. Exactly bmm(X, B) followed by add_(X, alpha=a) ----------
@triton.jit
def _gemm_xb_kernel(xa_desc, b_desc, xe_desc, c_desc, alpha, M, N, K,
                    BATCH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    per_batch = num_pid_m * num_pid_n
    num_tiles = BATCH * per_batch
    k_tiles = tl.cdiv(K, BK)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        bi = tile_id // per_batch
        t = tile_id % per_batch
        pid_m = t // num_pid_n
        pid_n = t % num_pid_n
        row = bi * M + pid_m * BM          # row in the flattened [Bt*M, N] view of X and C
        brow = bi * N + pid_n * BN         # row in the flattened [Bt*N, N] view of B
        col = pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = xa_desc.load([row, ki * BK])
            b = b_desc.load([brow, ki * BK])
            acc = tl.dot(a, b.T, acc)
        c = acc.to(tl.bfloat16)
        x = xe_desc.load([row, col]).to(tl.float32)
        out = tl.math.fma(x, alpha, c.to(tl.float32))
        c_desc.store([row, col], out.to(tl.bfloat16))


def gemm_xb_add(X, B, out, alpha, BM=128, BN=128, BK=64, num_warps=4, num_stages=4):
    Bt, M, N = X.shape
    assert B.shape == (Bt, N, N) and out.shape == X.shape and X.is_contiguous() and B.is_contiguous() and out.is_contiguous()
    X2, B2, O2 = X.view(Bt * M, N), B.view(Bt * N, N), out.view(Bt * M, N)
    xa = TensorDescriptor.from_tensor(X2, [BM, BK])
    bd = TensorDescriptor.from_tensor(B2, [BN, BK])
    xe = TensorDescriptor.from_tensor(X2, [BM, BN])
    cd = TensorDescriptor.from_tensor(O2, [BM, BN])
    nsm = torch.cuda.get_device_properties(X.device).multi_processor_count
    tiles = Bt * triton.cdiv(M, BM) * triton.cdiv(N, BN)
    _gemm_xb_kernel[(min(nsm, tiles),)](xa, bd, xe, cd, alpha, M, N, N, BATCH=Bt, BM=BM, BN=BN, BK=BK, NUM_SMS=nsm,
                                        num_warps=num_warps, num_stages=num_stages)
    return out


def polar(X, split_baddbmm, maps):
    """The stock cascade from the Gram on. Same GEMM numerics everywhere: tall -> fused bmm+add (verified bitwise equal to
    the pair in offline tests); wide -> the stock cuBLAS baddbmm, minus its defensive-copy kernel (the Gram kernel writes that copy)."""
    is_tall = X.size(-2) > X.size(-1)
    n = min(X.size(-1), X.size(-2))
    A = torch.empty((*X.shape[:-2], n, n), device=X.device, dtype=X.dtype)
    if is_tall:
        gram_tall(X, A)
        X, A = gram_norm(X, A)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        for k, (a, b, c) in enumerate(maps):
            if k > 0:
                gram_tall(X, A)
            poly_gram(A, c, b, B)
            if split_baddbmm:
                gemm_xb_add(X, B, C, a)
            else:
                torch.baddbmm(X, X, B, beta=a, out=C)
            X, C = C, X
    else:
        gram_wide(X, A)
        X, A, C = gram_norm_dup(X, A)
        B = torch.empty_like(A)
        for k, (a, b, c) in enumerate(maps):
            if k > 0:
                gram_wide(X, A, copy=C)
            poly_gram(A, c, b, B)
            if split_baddbmm:
                torch.bmm(B, X, out=C)
                C.add_(X, alpha=a)
            else:
                torch.baddbmm(C, B, X, beta=a, out=C)
            X, C = C, X
    return X

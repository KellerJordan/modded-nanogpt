"""Tiled transpose copy / transpose-add: the tied embed <-> lm_head sync, and the fp8 transposes of
the MLP, the loss and the sampled-softmax row slabs.

What it replaces: `dst.copy_(src.T)` and `dst.add_(src.T)`, which PyTorch runs as an elementwise
kernel of ~75k blocks with non-coalesced accesses on the transposed side, saturating every SM (and so
blocking the NCCL kernels that overlap it).
Why it is faster: each program moves one 2-D tile, reading it coalesced from src and writing it
coalesced to dst through a register-tile tl.trans, so both sides issue full 128-byte segments and
the kernel runs near memory bandwidth.
Invariants: a pure copy (or one add per element), so bit-exact against the PyTorch expression; fp8
callers pass uint8 views, which move identical bytes.

Provenance: the master track; the fp8 and transpose-add tile shapes from record #360.
"""
import torch
import triton
import triton.language as tl

# Fp8 sources with at least this many elements take transpose_copy's big-tile path (the CE grad_input
# is >= 168 M elements).
_TC_BIG_NUMEL = 1 << 27

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
    num_warps = 8
    if src.element_size() == 1 and dst.stride(1) == 1:
        # BLOCK_M is the length in elements of each contiguous destination run, so for fp8 a
        # 64-wide tile would store half a 128-byte L2 line. A pure copy, so still bit-exact.
        BLOCK_M = 128
        if (src.stride(1) == 1 and src.numel() >= _TC_BIG_NUMEL
                and src.stride(0) % 16 == 0 and dst.stride(0) % 16 == 0):  # 16 B-aligned rows, else keep the 128x128 tile
            # The multi-hundred-MB CE grad_input transposes (49152 x 14336 ... 40960 x 50304, 1.4-4.1 GB of traffic
            # per call) are HBM-bound and cold: a pure D2D copy of the same bytes runs 3.04 TB/s, the 128x128 tile
            # 2.84. 64x256 with 16 warps (256 B source runs, 64 B destination runs) measured 2.92 / 2.88 TB/s.
            # Below this size (the [T, 768] glue transposes) every tile shape is within noise.
            BLOCK_M, BLOCK_N, num_warps = 64, 256, 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
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

    # 64x128 / 8 warps rather than 32x32 / 4 (record #360): both streams then issue 128-byte
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


# -----------------------------------------------------------------------------
# Tied-gradient fold: lm_acc = bf16(bf16(lm_acc + lm_g) + bf16(emb_acc + emb_g).T), bit-exact with the
# two accumulator adds followed by transpose_add. emb_acc is not updated.

@triton.jit
def _tied_grad_fold_kernel(
    emb_acc_ptr, emb_g_ptr, lm_acc_ptr, lm_g_ptr,
    M, N,
    emb_acc_s0, emb_acc_s1, emb_g_s0, emb_g_s1,
    lm_acc_s0, lm_acc_s1, lm_g_s0, lm_g_s1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # embed side (M, N), coalesced
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ea = tl.load(emb_acc_ptr + offs_m[:, None] * emb_acc_s0 + offs_n[None, :] * emb_acc_s1, mask=mask, other=0.0)
    eg = tl.load(emb_g_ptr + offs_m[:, None] * emb_g_s0 + offs_n[None, :] * emb_g_s1, mask=mask, other=0.0)
    e = (ea.to(tl.float32) + eg.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")

    # lm_head side (N, M), coalesced
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    lm_ptrs = lm_acc_ptr + offs_n[:, None] * lm_acc_s0 + offs_m[None, :] * lm_acc_s1
    la = tl.load(lm_ptrs, mask=mask_T, other=0.0)
    lg = tl.load(lm_g_ptr + offs_n[:, None] * lm_g_s0 + offs_m[None, :] * lm_g_s1, mask=mask_T, other=0.0)
    lsum = (la.to(tl.float32) + lg.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    out = (lsum.to(tl.float32) + tl.trans(e).to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(lm_ptrs, out, mask=mask_T)


def tied_grad_fold(emb_acc: torch.Tensor, emb_g: torch.Tensor, lm_acc: torch.Tensor, lm_g: torch.Tensor):
    assert emb_acc.ndim == 2 and emb_acc.shape == emb_g.shape
    M, N = emb_acc.shape
    assert lm_acc.shape == (N, M) and lm_g.shape == (N, M), f"lm_head grads {lm_acc.shape}/{lm_g.shape} vs embed ({M}, {N})"
    for t in (emb_acc, emb_g, lm_acc, lm_g):
        assert t.dtype == torch.bfloat16, f"tied_grad_fold is bf16-only (got {t.dtype})"
    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tied_grad_fold_kernel[grid](
        emb_acc, emb_g, lm_acc, lm_g,
        M, N,
        emb_acc.stride(0), emb_acc.stride(1), emb_g.stride(0), emb_g.stride(1),
        lm_acc.stride(0), lm_acc.stride(1), lm_g.stride(0), lm_g.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )

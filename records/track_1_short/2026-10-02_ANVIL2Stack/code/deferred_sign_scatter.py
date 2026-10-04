"""Optional fused raw-gradient sign application and compact scatter; not integrated."""
import torch
import triton
import triton.language as tl


@triton.jit
def _signed_scatter_add_kernel(SRC, IDX, DST, TAB, RAW_INPUTS, NORM_MAP, T,
                               D: tl.constexpr, RMASK: tl.constexpr,
                               BLOCK: tl.constexpr, NCHUNK: tl.constexpr):
    pid = tl.program_id(0)
    t = tl.where(pid < T, pid, pid - T).to(tl.int64)
    cur = tl.load(RAW_INPUTS + t)
    pm1 = tl.load(RAW_INPUTS + (t - 1), t >= 1, other=0)
    pm2 = tl.load(RAW_INPUTS + (t - 2), t >= 2, other=0)
    if NORM_MAP is not None:
        cur = tl.load(NORM_MAP + cur.to(tl.int64))
        # Mask the mapped loads too: a missing neighbor is not norm_map[0].
        pm1 = tl.load(NORM_MAP + pm1.to(tl.int64), t >= 1, other=0)
        pm2 = tl.load(NORM_MAP + pm2.to(tl.int64), t >= 2, other=0)
    si = tl.where(t >= 1, ((30011 * pm1) ^ (48271 * cur)) & RMASK, 0)
    ss = tl.where(t >= 2, ((26801 * pm2) ^ (39779 * pm1) ^ (58699 * cur)) & RMASK, 0)
    srow = tl.where(pid < T, si, ss).to(tl.int64)
    drow = tl.load(IDX + pid)
    o64 = tl.arange(0, BLOCK).to(tl.int64)
    sbase = t.to(tl.int64) * D
    tbase = srow.to(tl.int64) * D
    dbase = drow.to(tl.int64) * D
    for k in tl.static_range(NCHUNK):
        koff = k * BLOCK
        g = tl.load(SRC + (sbase + koff + o64)).to(tl.float32)
        sign = tl.load(TAB + (tbase + koff + o64)).to(tl.float32)
        # Preserve the materialized sink's BF16 rounding before scatter widening.
        val = (g * sign).to(tl.bfloat16).to(tl.float32)
        tl.atomic_add(DST + (dbase + koff + o64), val, sem="relaxed")


def _launch_spec(src, idx, dst, tab, raw_inputs, norm_map):
    """Metadata-only validation, also usable with real CPU/meta tensors in tests."""
    tensors = (src, idx, dst, tab, raw_inputs)
    if norm_map is not None:
        tensors += (norm_map,)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all inputs must be tensors (norm_map may be None)")
    if not all(x.is_contiguous() for x in tensors):
        raise ValueError("all inputs must be contiguous")
    if not all(x.device == src.device for x in tensors):
        raise ValueError("all inputs must be on the same device")
    if src.ndim != 2 or src.dtype != torch.bfloat16:
        raise ValueError("src must be BF16 [T, D]")
    t, d = src.shape
    if d <= 0:
        raise ValueError("D must be positive")
    if dst.ndim != 2 or dst.shape[1] != d or dst.dtype not in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        raise ValueError("dst must be FP16/BF16/FP32 [N, D]")
    if t and dst.shape[0] == 0:
        raise ValueError("a nonempty src needs destination rows")
    if idx.ndim != 1 or idx.numel() != 2 * t or idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("idx must be int32/int64 [2T]")
    if tab.ndim != 2 or tab.shape[1] != d or tab.dtype != torch.bfloat16:
        raise ValueError("tab must be BF16 [R, D]")
    r = tab.shape[0]
    if r == 0 or r & (r - 1):
        raise ValueError("R must be a positive power of two")
    if raw_inputs.ndim != 1 or raw_inputs.numel() != t or raw_inputs.dtype != torch.int32:
        raise ValueError("raw_inputs must be int32 [T]")
    if norm_map is not None and (norm_map.ndim != 1 or norm_map.dtype != torch.int32):
        raise ValueError("norm_map must be an int32 vector")
    if norm_map is not None and t and norm_map.numel() == 0:
        raise ValueError("a nonempty src needs a nonempty norm_map")
    block = 1024
    while block > 1 and d % block:
        block //= 2
    if t and block < 32:
        raise ValueError(f"row width {d} has no usable tile")
    return t, d, r - 1, block


def signed_scatter_add(src, idx, dst, tab, raw_inputs, norm_map=None):
    """Add signed BF16-rounded src rows to dst in place and return dst.

    CUDA-only, with idx[:T] for bigrams and idx[T:] for trigrams. Each idx
    must be in [0, N); with norm_map, raw IDs must be in its index range.
    Bounds are caller preconditions, not synchronizing host checks. Inputs
    must not overlap dst and must remain unchanged until the launch finishes.
    Repeated destinations within/across channels are legal; atomic landing
    order, hence floating-point reduction bits, is unspecified. This wrapper
    supplies no autograd/custom-op registration or token-lifetime management.
    """
    t, d, rmask, block = _launch_spec(src, idx, dst, tab, raw_inputs, norm_map)
    if not src.is_cuda:
        raise ValueError("signed_scatter_add requires CUDA tensors")
    if t:
        _signed_scatter_add_kernel[(2 * t,)](
            src, idx, dst, tab, raw_inputs, norm_map, t,
            D=d, RMASK=rmask, BLOCK=block, NCHUNK=d // block,
            num_warps=max(1, min(8, block // 64)), num_stages=1,
        )
    return dst

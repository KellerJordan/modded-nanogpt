"""Fused bf16 -> fp32 lerp for the tail accumulators (record #360).

`dst.lerp_(src.float(), rate)` makes ATen materialize an fp32 copy of `src` first; this kernel
widens in registers instead, one pass over 10 bytes/element rather than two over 18.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lerp_upcast_kernel(SRC, DST, n_elements, rate, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    s = tl.load(DST + offs, mask=m, other=0.0)
    e = tl.load(SRC + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(DST + offs, tl.math.fma(rate, e - s, s), mask=m)


def lerp_upcast_(dst: torch.Tensor, src: torch.Tensor, rate: float) -> None:
    """dst (fp32) <- lerp(dst, src.float(), rate), in place."""
    assert src.is_contiguous() and dst.is_contiguous() and src.numel() == dst.numel()
    assert dst.dtype == torch.float32
    BLOCK = 4096
    _lerp_upcast_kernel[(triton.cdiv(dst.numel(), BLOCK),)](
        src, dst, dst.numel(), float(rate), BLOCK=BLOCK, num_warps=8, num_stages=2,
    )

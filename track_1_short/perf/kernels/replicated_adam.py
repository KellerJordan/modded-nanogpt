"""One Triton launch for the Adam step of every replicated parameter (perf/replicated_adam.py).

Each element looks up its parameter ("segment") and reads that segment's row of the scalar table:
beta1, 1-beta1, beta2, 1-beta2, eps, step_size, eff_wd, active. The arithmetic is AnvilAndAdam's
_adam_update_step statement for statement, in fp32 (bf16 upcast exactly): ATen's add_ / addcmul_
contract to FFMA under nvcc's default -fmad=true, which the tl.math.fma spellings mirror; tl.sqrt and
`/` are IEEE (sqrt.rn / div.rn) like ATen's; the bf16 store rounds to nearest even.

Provenance: record #360 (ANVIL2), fuse_tiny_kernels.py `_afe_fused_adam` ("FUSE").
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024
# Columns of the scalar table, one row per segment.
SCALAR_COLUMNS = ("beta1", "one_minus_beta1", "beta2", "one_minus_beta2", "eps", "step_size", "eff_wd", "active")
assert len(SCALAR_COLUMNS) == 8  # the kernel's row stride


@triton.jit
def _fused_adam_kernel(P, G, EA, ES, SEG, SC, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    base = tl.load(SEG + offs, mask=m, other=0) * 8
    b1 = tl.load(SC + base + 0, mask=m, other=0.0)
    a1 = tl.load(SC + base + 1, mask=m, other=0.0)
    b2 = tl.load(SC + base + 2, mask=m, other=0.0)
    a2 = tl.load(SC + base + 3, mask=m, other=0.0)
    eps = tl.load(SC + base + 4, mask=m, other=0.0)
    step_size = tl.load(SC + base + 5, mask=m, other=0.0)
    wd = tl.load(SC + base + 6, mask=m, other=0.0)
    active = tl.load(SC + base + 7, mask=m, other=0.0)
    # An inactive segment (no gradient this step) is skipped whole, moments included.
    m = m & (active != 0.0)
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    ea = tl.load(EA + offs, mask=m, other=0.0)
    es = tl.load(ES + offs, mask=m, other=0.0)
    p = tl.load(P + offs, mask=m, other=0.0).to(tl.float32)
    ea = tl.math.fma(a1, g, ea * b1)                 # exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
    es = tl.math.fma(a2 * g, g, es * b2)             # exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    u = ea / (tl.sqrt(es) + eps) * step_size
    # Cautious weight decay: update.addcmul_(p, (update * p) > 0, value=eff_wd)
    u = tl.math.fma(wd * p, tl.where(u * p > 0.0, 1.0, 0.0), u)
    tl.store(EA + offs, ea, mask=m)
    tl.store(ES + offs, es, mask=m)
    tl.store(P + offs, (p - u).to(P.dtype.element_ty), mask=m)


def fused_adam_(param: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
                segment: torch.Tensor, scalars: torch.Tensor) -> None:
    """In place over flat buffers: param/grad (same dtype), fp32 moments, int32 segment per element,
    scalars [segments, len(SCALAR_COLUMNS)] fp32 on the device."""
    n = param.numel()
    _fused_adam_kernel[(triton.cdiv(n, BLOCK),)](param, grad, exp_avg, exp_avg_sq, segment, scalars, n,
                                                 BLOCK=BLOCK, num_warps=4)

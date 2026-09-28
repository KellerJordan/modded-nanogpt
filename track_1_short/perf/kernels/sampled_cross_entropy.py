"""The sampled-softmax training loss (see sampled_softmax.py): the fused CE kernel at width P.

What it replaces: FusedSoftcappedCrossEntropy over all 50304 classes, on the steps where the loss
normalizes over a candidate set C of P classes. Why it is faster: the logit GEMM, the CE kernel and both
gradient GEMMs run at width P instead of 50304. The CE CUDA kernel is reused unchanged, compiled once per
P with VOCAB_SIZE = P, and fed each token's target position in C instead of its class id.

Invariants: the GEMMs read the (P, D) / (D, P) row slabs SampledSoftmax gathered from the fp8 lm_head
cache for this step, and the Function saves the (D, P) slab without copying it -- legal only under the
one-microbatch-per-step race contract in perf/sampled_softmax_overlap.py.

Provenance: record #360 (ANVIL2).
"""
import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE,
    SOFTCAP_A,
    SOFTCAP_B,
    SOFTCAP_C,
    _ce_backward_gemms,
    _ce_logit_gemm,
    compile_ce_kernel,
)
from track_1_short.sampled_softmax import ALL_CANDIDATE_COUNTS

# The kernel's 8-wide shared-memory and grad stores need P % (BLOCK_SIZE * 8) == 0.
for _p in ALL_CANDIDATE_COUNTS:
    assert _p % (CE_KERNEL_BLOCK_SIZE * 8) == 0, f"candidate count {_p} is not a legal CE width"
# P -> the CE kernel compiled at VOCAB_SIZE = P. Built once at import (nvrtc, off the clock), never mutated.
SAMPLED_CE_KERNELS = {p: compile_ce_kernel(p) for p in ALL_CANDIDATE_COUNTS}


@torch.library.custom_op("nanogpt::sampled_ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def sampled_ce_fwd_bwd(
    logits: torch.Tensor,
    target_pos: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_pos: torch.Tensor,
    prefix_weight: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
) -> None:
    """ce_fwd_bwd with the kernel of width P = logits.shape[1]."""
    P = logits.shape[1]
    SAMPLED_CE_KERNELS[P](
        (n_rows, 1, 1),
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, target_pos, mtp_weights, prefix_pos, prefix_weight, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s),
        shared_mem=P * 2,
    )


# The backward's wgrad is a dense (D, P) slab over candidate positions; the optimizer wants the
# (D, V) lm_head gradient with non-candidate columns zero. Instead of zeros() + index_copy_ along dim 1
# (a 77 MB zero pass plus scattered 2-byte stores), this writes densely and gathers:
# out[r, c] = src[r, pos[c]] if pos[c] >= 0 else 0. A pure copy, so bit-exact.

@triton.jit
def _densify_kernel(SRC, POS, DST, R, V, src_stride_r, dst_stride_r,
                    BLOCK_R: tl.constexpr, BLOCK_V: tl.constexpr):
    offs_v = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    m_v, m_r = offs_v < V, offs_r < R
    pos = tl.load(POS + offs_v, mask=m_v, other=-1)
    keep = pos >= 0
    p = tl.where(keep, pos, 0).to(tl.int64)  # in-range dummy for the masked lanes
    # C is ascending, so a vocab tile's kept positions are consecutive: one short contiguous run per row.
    src = tl.load(SRC + offs_r[:, None].to(tl.int64) * src_stride_r + p[None, :],
                  mask=m_r[:, None] & keep[None, :], other=0.0)
    tl.store(DST + offs_r[:, None].to(tl.int64) * dst_stride_r + offs_v[None, :].to(tl.int64),
             src, mask=m_r[:, None] & m_v[None, :])

@torch.library.custom_op("nanogpt::sampled_densify", mutates_args={"dst"})
def sampled_densify(src: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor) -> None:
    _densify_kernel[(triton.cdiv(dst.shape[1], 256), triton.cdiv(dst.shape[0], 32))](
        src, pos, dst, *dst.shape, src.stride(0), dst.stride(0), BLOCK_R=32, BLOCK_V=256, num_warps=4, num_stages=3)


class SampledSoftcappedCrossEntropy(torch.autograd.Function):
    """FusedSoftcappedCrossEntropy over a candidate set of P classes.

    `rows` (P, D) / `rows_t` (D, P) are the candidates' fp8 lm_head rows; target_pos / prefix_pos are each
    token's target and prefix target as positions in C; vocab_pos maps a class to its position (-1 if
    not a candidate) and sizes the dense lm_head gradient. `lm_head_weight` only carries that gradient.
    """
    @staticmethod
    def forward(ctx, x, mtp_weights, target_pos, prefix_pos, prefix_weight, lm_head_weight,
                rows, rows_t, vocab_pos, x_s, w_s, grad_s):
        n_rows = x.shape[0]
        P = rows.shape[0]
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, rows.T, x_s, w_s)  # rows.T is column-major
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, P), dtype=torch.float8_e5m2, device=logits.device)
        sampled_ce_fwd_bwd(logits.contiguous(), target_pos, mtp_weights.contiguous(), prefix_pos,
                           prefix_weight.reshape(1).to(torch.float32).contiguous(), losses, grad_input,
                           n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
        ctx.save_for_backward(x_f8, rows_t, grad_input, vocab_pos)
        ctx.scales = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, rows_t, grad_input, vocab_pos = ctx.saved_tensors
        grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)
        # Freshly allocated (AccumulateGrad may adopt it as .grad) and never zeroed: densify writes every element.
        grad_w = torch.empty((x_f8.shape[1], vocab_pos.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w)
        # One slot per forward input; grad_w lands on lm_head_weight.
        return grad_x, None, None, None, None, grad_w, None, None, None, None, None, None

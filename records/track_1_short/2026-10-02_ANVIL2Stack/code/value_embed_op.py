"""Value-embedding gather with a custom scatter-add backward (design notes in the record README).

Forward: the native four-plane advanced-index gather, kept visible to Dynamo.  Only the backward is opaque: it
scatter-adds the four per-plane bf16 adjoints into the dense [4*V, D] fp16 gradient (VE_GRAD_DTYPE) in ONE
pass, accumulating in fp32 and letting tl.atomic_add round to fp16 at sem="relaxed", so it reproduces within
the CRN band, not bitwise -- colliding atomics land in unspecified order and fp16 addition is non-associative.
One program per destination (plane, token) row, one launch per plane, so the loads vectorise and the fp16
atomics pack (f16x2); REJECTED, measured: a flat element-per-thread map over the [4, t, D] adjoint space does
neither, since at D = 768 a 1024-lane block spans 1.33 rows.  PLANES/VOCAB/WIDTH must agree with the trainer's
constants, and `_into` accumulates into the trainer's persistent fp16 cycle buffer -- the table takes an Adam
step only every few steps and no dense gradient is allocated in between, so the buffer IS the gradient when
the step comes.  The plain `..._selected_load` serves the eval forward, which runs under torch.no_grad().
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PLANES, VOCAB, WIDTH, VE_GRAD_DTYPE = 4, 50304, 768, torch.float16

# Row-packed tiling for a 768-wide row; a segment is the destination stride too, so one SEG serves both sides.
_VESC_SEG, _VESC_NSEG, _VESC_WARPS = 256, 3, 4
assert _VESC_SEG * _VESC_NSEG == WIDTH
assert PLANES * VOCAB * WIDTH < 2 ** 31, "[value-embed] row scatter addresses in int32"

@triton.jit
def _vesc_row_scatter_kernel(src, token_ids, out, V: tl.constexpr, D_DST: tl.constexpr,
                             SEG: tl.constexpr, NSEG: tl.constexpr):
    """One program per (plane, token) row of ONE plane's adjoint.  `out` already points at this
    plane's [V, D_DST] slab, so `token` alone indexes the destination row and `dbase` is a uniform
    scalar.  Every offset is int32: the largest is (PLANES*V - 1) * D_DST = 154.4 M."""
    pid = tl.program_id(0)
    token = tl.load(token_ids + pid, eviction_policy="evict_last")
    token = tl.where(token < 0, token + V, token)
    tl.device_assert((0 <= token) & (token < V), "index out of bounds: 0 <= token < 50304")
    o = tl.arange(0, SEG)
    sbase = pid * (SEG * NSEG)
    dbase = token * D_DST
    for k in tl.static_range(NSEG):
        value = tl.load(src + (sbase + k * SEG + o), eviction_policy="evict_first").to(tl.float32)
        tl.atomic_add(out + (dbase + k * SEG + o), value, sem="relaxed")

def _check_sink(sink: torch.Tensor, token_ids: torch.Tensor) -> None:
    """Contract of the persistent accumulation buffer: fp16 [4*V, D], same device."""
    assert sink.shape == (PLANES * VOCAB, WIDTH)
    assert sink.dtype == VE_GRAD_DTYPE and sink.is_contiguous()
    assert sink.device == token_ids.device

@torch.library.custom_op(
    "nanogpt_slg::selected_load_backward_into",
    mutates_args={"output"},
)
def selected_load_backward_into_op(token_ids: torch.Tensor, grad0: torch.Tensor,
                                   grad1: torch.Tensor, grad2: torch.Tensor,
                                   grad3: torch.Tensor, output: torch.Tensor) -> None:
    _check_sink(output, token_ids)
    t = token_ids.numel()
    if t == 0:
        return
    for p, grad in enumerate((grad0, grad1, grad2, grad3)):
        assert grad.shape == (t, WIDTH), \
            f"[value-embed] plane {p} adjoint {tuple(grad.shape)} != {(t, WIDTH)}"
        assert grad.dtype == torch.bfloat16 and grad.is_contiguous()
        assert grad.device == token_ids.device
        _vesc_row_scatter_kernel[(t,)](
            grad, token_ids, output[p * VOCAB:(p + 1) * VOCAB],
            V=VOCAB, D_DST=WIDTH, SEG=_VESC_SEG, NSEG=_VESC_NSEG,
            num_warps=_VESC_WARPS, num_stages=1,
        )

@selected_load_backward_into_op.register_fake
def _selected_load_backward_into_fake(token_ids, grad0, grad1, grad2, grad3, output) -> None:
    assert token_ids.ndim == 1 and grad0.shape == (token_ids.numel(), WIDTH)
    assert output.shape == (PLANES * VOCAB, WIDTH)

class _ValueEmbeddingSelectedLoadInto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor):
        _check_sink(output, token_ids)
        ctx.save_for_backward(token_ids, output)
        return value_embedding_planes_selected_load(weight, token_ids)

    @staticmethod
    def backward(ctx, grad0, grad1, grad2, grad3):
        token_ids, output = ctx.saved_tensors
        selected_load_backward_into_op(token_ids, grad0, grad1, grad2, grad3, output)
        return None, None, None

def value_embedding_planes_selected_load(
    weight: torch.Tensor, token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """THE four-plane gather, taken by the eval forward (under torch.no_grad(), so no backward) and
    by _ValueEmbeddingSelectedLoadInto.forward alike.  Contract: 1-D int32 cuda ids, bf16 [4*V, D]."""
    assert token_ids.ndim == 1 and token_ids.dtype == torch.int32
    assert token_ids.is_contiguous() and token_ids.device.type == "cuda"
    assert weight.shape == (PLANES * VOCAB, WIDTH)
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert weight.device == token_ids.device
    gathered = weight.view(PLANES, VOCAB, WIDTH)[:, token_ids]
    return gathered[0], gathered[1], gathered[2], gathered[3]

def value_embedding_planes_selected_load_into(
    weight: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _ValueEmbeddingSelectedLoadInto.apply(weight, token_ids, output)

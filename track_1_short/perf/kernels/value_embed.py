"""The value-embedding lookup with a backward that accumulates into one persistent fp16 buffer.

What it replaces: autograd's backward of the plain gather in GPT.forward,

    planes = value_embeds.view(P, V, D)[:, token_ids]          # P planes of [T, D]

which allocates a dense [P * V, D] bf16 gradient (309 MB at P = 4, V = 50304, D = 768), zero-fills it,
index-puts the planes' gradients into it, and then adds it into value_embeds.grad -- every step, although
the step touches a few thousand token rows.

Why it is faster: the forward stays the native gather (visible to inductor); only the backward is an
opaque op. It scatter-adds the P per-plane bf16 gradients straight into the caller's persistent fp16
[P * V, D] buffer, one launch per plane and one program per (plane, token) row, so the loads vectorise
and the fp16 atomics pack two lanes (f16x2). No gradient-sized tensor is allocated, and the buffer
accumulates across steps: at value_embeds' update event it already IS the cycle's gradient
(perf/value_embed_pull.py compacts its touched rows and zeroes them).

Invariants: the backward only adds at rows p * V + t for this step's tokens t, so the buffer is nonzero
only at the rows of the current cycle's tokens. Accumulation is in fp32 per add, rounded to fp16 by the
atomic: colliding atomics land in unspecified order, so it is reproducible within fp16 rounding, not
bitwise. The eval forward (under torch.no_grad) passes no buffer and takes the plain gather.

Provenance: record #360 (ANVIL2): value_embed_op.py (`value_embedding_planes_selected_load[_into]`,
`selected_load_backward_into`, the V-lite buffer `_velite_buf`).
"""
import torch
import triton
import triton.language as tl

GRAD_DTYPE = torch.float16
# A 768-wide row is three 256-lane segments, on both the source and the destination side (record #360).
SEGMENT = 256
NUM_WARPS = 4


@triton.jit
def _row_scatter_add_kernel(src, token_ids, out, V: tl.constexpr, D: tl.constexpr,
                            SEG: tl.constexpr, NSEG: tl.constexpr):
    """One program per token row of ONE plane's gradient; `out` points at that plane's [V, D] slab.
    int32 offsets: the largest is V * D, within one plane's slab (the caller asserts the table < 2**31)."""
    pid = tl.program_id(0)
    token = tl.load(token_ids + pid, eviction_policy="evict_last")
    tl.device_assert((0 <= token) & (token < V), "token id out of range")
    o = tl.arange(0, SEG)
    src_base = pid * D
    dst_base = token * D
    for k in tl.static_range(NSEG):
        value = tl.load(src + (src_base + k * SEG + o), eviction_policy="evict_first").to(tl.float32)
        tl.atomic_add(out + (dst_base + k * SEG + o), value, sem="relaxed")


@torch.library.custom_op("nanogpt::value_embed_grad_accumulate", mutates_args={"grad_accum"})
def value_embed_grad_accumulate(token_ids: torch.Tensor, plane_grads: list[torch.Tensor],
                                grad_accum: torch.Tensor) -> None:
    """grad_accum[p * V + token_ids[i]] += plane_grads[p][i], for every plane p and token i."""
    num_planes, t = len(plane_grads), token_ids.numel()
    vocab, d = grad_accum.shape[0] // num_planes, grad_accum.shape[1]
    assert grad_accum.dtype == GRAD_DTYPE and grad_accum.is_contiguous()
    assert token_ids.dtype == torch.int32 and token_ids.is_contiguous()
    assert d % SEGMENT == 0 and grad_accum.numel() < 2 ** 31
    if t == 0:
        return
    for p, grad in enumerate(plane_grads):
        assert grad.shape == (t, d) and grad.dtype == torch.bfloat16
        _row_scatter_add_kernel[(t,)](grad.contiguous(), token_ids, grad_accum[p * vocab:(p + 1) * vocab],
                                      V=vocab, D=d, SEG=SEGMENT, NSEG=d // SEGMENT,
                                      num_warps=NUM_WARPS, num_stages=1)


@value_embed_grad_accumulate.register_fake
def _(token_ids, plane_grads, grad_accum) -> None:
    return None


class _ValueEmbedLookup(torch.autograd.Function):
    """The gather's gradient goes into `grad_accum`, never to value_embeds.grad."""
    @staticmethod
    def forward(ctx, weight, token_ids, grad_accum, num_planes):
        ctx.save_for_backward(token_ids, grad_accum)
        return value_embed_gather(weight, token_ids, num_planes)

    @staticmethod
    def backward(ctx, *plane_grads):
        token_ids, grad_accum = ctx.saved_tensors
        torch.ops.nanogpt.value_embed_grad_accumulate(token_ids, list(plane_grads), grad_accum)
        return None, None, None, None


def value_embed_gather(weight: torch.Tensor, token_ids: torch.Tensor, num_planes: int) -> tuple[torch.Tensor, ...]:
    """The plain gather: [P * V, D] at [T] int32 token ids -> P planes of [T, D]."""
    return tuple(weight.view(num_planes, -1, weight.shape[1])[:, token_ids].unbind(0))


def value_embed_lookup(weight: torch.Tensor, token_ids: torch.Tensor, num_planes: int,
                       grad_accum: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    """P planes of [T, D]. In training `grad_accum` ([P * V, D] fp16) receives their gradient; None in eval."""
    if grad_accum is None:
        return value_embed_gather(weight, token_ids, num_planes)
    return _ValueEmbedLookup.apply(weight, token_ids, grad_accum, num_planes)

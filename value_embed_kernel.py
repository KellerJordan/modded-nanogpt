"""SLG standalone exact-topology value-embedding scatter kernel.

Uses the shipping XBLOCK=1024, W4/S1 launch and issues the same per-thread
relaxed BF16 atomic contribution as the generated scatter it replaces; it
loads only the active one of the five plane-adjoint pointers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


VOCAB = 50304
WIDTH = 768
PLANES = 5
XBLOCK = 1024
NUM_WARPS = 4
NUM_STAGES = 1


@triton.jit
def _veg_selected_load_kernel(
    token_ids,
    grad4,
    grad3,
    grad2,
    grad1,
    grad0,
    output,
    T: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Keep the exact shipping thread-to-contribution mapping.
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x1 = (xindex // D) % T
    plane_span = T * D
    x2 = xindex // plane_span
    x3 = xindex % plane_span
    x0 = xindex % D

    token = tl.load(token_ids + x1, eviction_policy="evict_last")
    selected_ptr = tl.where(x2 == 4, grad4 + x3, grad0 + x3)
    selected_ptr = tl.where(x2 == 3, grad3 + x3, selected_ptr)
    selected_ptr = tl.where(x2 == 2, grad2 + x3, selected_ptr)
    selected_ptr = tl.where(x2 == 1, grad1 + x3, selected_ptr)
    value = tl.load(selected_ptr, eviction_policy="evict_last").to(tl.float32)

    token = tl.where(token < 0, token + V, token)
    tl.device_assert(
        (0 <= token) & (token < V),
        "index out of bounds: 0 <= token < 50304",
    )
    tl.atomic_add(output + x0 + D * token + V * D * x2, value, sem="relaxed")


def _validate(
    token_ids: torch.Tensor,
    grads: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> int:
    assert token_ids.ndim == 1 and token_ids.dtype == torch.int32
    assert token_ids.is_contiguous() and token_ids.device.type == "cuda"
    assert len(grads) == PLANES
    t = token_ids.numel()
    for grad in grads:
        assert grad.shape == (t, WIDTH)
        assert grad.dtype == torch.bfloat16 and grad.is_contiguous()
        assert grad.device == token_ids.device
    assert output.shape == (PLANES, VOCAB, WIDTH)
    assert output.dtype == torch.bfloat16 and output.is_contiguous()
    assert output.device == token_ids.device
    assert (t * WIDTH) % XBLOCK == 0
    return t


def _launch(kernel, token_ids, grads, output) -> None:
    t = _validate(token_ids, grads, output)
    grid = (PLANES * t * WIDTH // XBLOCK,)
    kernel[grid](
        token_ids,
        grads[4],
        grads[3],
        grads[2],
        grads[1],
        grads[0],
        output,
        T=t,
        V=VOCAB,
        D=WIDTH,
        XBLOCK=XBLOCK,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def launch_selected_load(token_ids, grads, output) -> None:
    _launch(_veg_selected_load_kernel, token_ids, grads, output)

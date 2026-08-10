"""SLG standalone exact-topology value-embedding scatter kernels.

Both bodies use the shipping XBLOCK=1024, W4/S1 launch and issue the same
per-thread relaxed BF16 atomic contribution.  The candidate changes only how
the active one of five plane-adjoint pointers is loaded.
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
def _veg_shipping_five_load_kernel(
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
    # Mechanically preserve the generated `_19` index decomposition.
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x1 = (xindex // D) % T
    plane_span = T * D
    x2 = xindex // plane_span
    x3 = xindex % plane_span
    x0 = xindex % D

    token = tl.load(token_ids + x1, eviction_policy="evict_last")
    tmp9 = tl.load(grad4 + x3, eviction_policy="evict_last").to(tl.float32)
    tmp14 = tl.load(grad3 + x3, eviction_policy="evict_last").to(tl.float32)
    tmp19 = tl.load(grad2 + x3, eviction_policy="evict_last").to(tl.float32)
    tmp24 = tl.load(grad1 + x3, eviction_policy="evict_last").to(tl.float32)
    tmp29 = tl.load(grad0 + x3, eviction_policy="evict_last").to(tl.float32)

    token = tl.where(token < 0, token + V, token)
    tl.device_assert(
        (0 <= token) & (token < V),
        "index out of bounds: 0 <= tmp4 < 50304",
    )
    value = tl.where(x2 == 4, tmp9, 0.0)
    value = value + tl.where(x2 == 3, tmp14, 0.0)
    value = value + tl.where(x2 == 2, tmp19, 0.0)
    value = value + tl.where(x2 == 1, tmp24, 0.0)
    value = value + tl.where(x2 == 0, tmp29, 0.0)
    tl.atomic_add(output + x0 + D * token + V * D * x2, value, sem="relaxed")


@triton.jit
def _veg_candidate_selected_load_kernel(
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
        "index out of bounds: 0 <= tmp4 < 50304",
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


def launch_shipping(token_ids, grads, output) -> None:
    _launch(_veg_shipping_five_load_kernel, token_ids, grads, output)


def launch_candidate(token_ids, grads, output) -> None:
    _launch(_veg_candidate_selected_load_kernel, token_ids, grads, output)


def compile_and_audit_ir(token_ids, grads, output) -> dict:
    """Compile without launching and prove five-to-one adjoint loads in TTIR."""
    t = _validate(token_ids, grads, output)
    grid = (PLANES * t * WIDTH // XBLOCK,)
    common = dict(
        grid=grid,
        T=t,
        V=VOCAB,
        D=WIDTH,
        XBLOCK=XBLOCK,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    args = (
        token_ids,
        grads[4],
        grads[3],
        grads[2],
        grads[1],
        grads[0],
        output,
    )
    shipping = _veg_shipping_five_load_kernel.warmup(*args, **common)
    candidate = _veg_candidate_selected_load_kernel.warmup(*args, **common)
    shipping_ttir = shipping.asm["ttir"]
    candidate_ttir = candidate.asm["ttir"]
    shipping_loads = shipping_ttir.count("tt.load")
    candidate_loads = candidate_ttir.count("tt.load")
    assert shipping_loads == 6, shipping_loads
    assert candidate_loads == 2, candidate_loads
    return {
        "shipping_hash": shipping.hash,
        "candidate_hash": candidate.hash,
        "shipping_total_global_load_ops": shipping_loads,
        "candidate_total_global_load_ops": candidate_loads,
        "shipping_adjoint_load_ops": shipping_loads - 1,
        "candidate_adjoint_load_ops": candidate_loads - 1,
        "shipping_num_warps": shipping.metadata.num_warps,
        "candidate_num_warps": candidate.metadata.num_warps,
        "shipping_num_stages": shipping.metadata.num_stages,
        "candidate_num_stages": candidate.metadata.num_stages,
        "shipping_shared_bytes": shipping.metadata.shared,
        "candidate_shared_bytes": candidate.metadata.shared,
    }

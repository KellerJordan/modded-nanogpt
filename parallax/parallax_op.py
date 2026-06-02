"""Parallax kernels registered as torch.library custom ops.

Wraps the upstream ``parallax.triton.parallax_fwd`` / ``parallax_bwd`` Triton
kernels in a ``torch.library.custom_op`` pair with a registered autograd
connection, so that ``torch.compile`` treats the kernel as an opaque op
(traceable, no graph break) instead of bailing out at the autograd.Function.

Drop-in for the upstream ``parallax_func``: same ``(B, H_q, L_q, D)`` /
``(B, H_kv, L_kv, D)`` shape convention, same ``qk_scale`` and
``window_size_left`` kwargs. Heads are folded into the batch dimension
before the kernel call (matching the upstream wrapper).

Usage from a sibling script in this directory::

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parallax_op import parallax_func
    # ... model.compile(dynamic=False) now compiles around the kernel
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Make the upstream Parallax repo importable. Use PARALLAX_PATH (the same
# env var the existing variant scripts use).
_PARALLAX_PATH = os.environ.get('PARALLAX_PATH', '')
if _PARALLAX_PATH and _PARALLAX_PATH not in sys.path:
    sys.path.insert(0, _PARALLAX_PATH)

from parallax.triton.parallax_fwd import parallax_fwd as _parallax_fwd  # noqa: E402
from parallax.triton.parallax_bwd import parallax_bwd as _parallax_bwd  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Forward op (registered once; subsequent imports are no-ops).
# Inputs are the folded 3-D layout (B*H, L, D); the public wrapper below
# does the (B, H, L, D) → (B*H, L, D) reshape so callers see the same
# interface as the upstream parallax_func.
# ────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op("parallax::fwd", mutates_args=())
def _fwd_op(q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            qk_scale: float, n_rep: int, window_size_left: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _parallax_fwd(q, r, k, v, qk_scale, n_rep, window_size_left)


@_fwd_op.register_fake
def _(q, r, k, v, qk_scale, n_rep, window_size_left):
    B, L, D = q.shape
    return (
        torch.empty((B, L, D), device=q.device, dtype=q.dtype),       # o
        torch.empty((B, L, D), device=q.device, dtype=q.dtype),       # barv
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),  # d1
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),  # bart
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),  # m
    )


@torch.library.custom_op("parallax::bwd", mutates_args=())
def _bwd_op(q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            o: torch.Tensor, barv: torch.Tensor,
            d1: torch.Tensor, bart: torch.Tensor, m: torch.Tensor,
            grad_o: torch.Tensor,
            qk_scale: float, n_rep: int, window_size_left: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o,
                         qk_scale, n_rep, window_size_left)


@_bwd_op.register_fake
def _(q, r, k, v, o, barv, d1, bart, m, grad_o,
      qk_scale, n_rep, window_size_left):
    return (
        torch.empty_like(q),  # grad_q
        torch.empty_like(r),  # grad_r
        torch.empty_like(k),  # grad_k
        torch.empty_like(v),  # grad_v
    )


def _setup_context(ctx, inputs, output):
    q, r, k, v, qk_scale, n_rep, window_size_left = inputs
    o, barv, d1, bart, m = output
    ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
    ctx.qk_scale = qk_scale
    ctx.n_rep = n_rep
    ctx.window_size_left = window_size_left


def _backward(ctx, grad_o, grad_barv, grad_d1, grad_bart, grad_m):
    # Only ``o`` is consumed downstream; grad_barv/d1/bart/m arrive as zero
    # or None from autograd and are ignored.
    q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
    grad_q, grad_r, grad_k, grad_v = _bwd_op(
        q, r, k, v, o, barv, d1, bart, m, grad_o.contiguous(),
        ctx.qk_scale, ctx.n_rep, ctx.window_size_left,
    )
    return grad_q, grad_r, grad_k, grad_v, None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup_context)


# ────────────────────────────────────────────────────────────────────────────
# Public wrapper — same signature as parallax.triton.parallax_func.
# ────────────────────────────────────────────────────────────────────────────

def parallax_func(q: torch.Tensor, r: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor,
                  qk_scale: float | None = None,
                  window_size_left: int = -1) -> torch.Tensor:
    """Compile-safe causal Parallax forward, with autograd.

    Args:
        q, r: ``(B, H_q, L_q, D)`` bf16/fp16.
        k, v: ``(B, H_kv, L_kv, D)`` same dtype as q, with H_q % H_kv == 0.
        qk_scale: defaults to ``1 / sqrt(D)``.
        window_size_left: causal sliding-window length (FA2 convention).

    Returns:
        ``(B, H_q, L_q, D)`` tensor with the same dtype as q.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallax_func requires bf16 or fp16 inputs, got {q.dtype}")
    B, H_q, L_q, D = q.shape
    _, H_kv, L_kv, _ = k.shape
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv}"
    n_rep = H_q // H_kv
    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D)

    q3 = q.reshape(B * H_q, L_q, D)
    r3 = r.reshape(B * H_q, L_q, D)
    k3 = k.reshape(B * H_kv, L_kv, D)
    v3 = v.reshape(B * H_kv, L_kv, D)

    o, _barv, _d1, _bart, _m = _fwd_op(
        q3, r3, k3, v3,
        float(qk_scale), int(n_rep), int(window_size_left),
    )
    return o.reshape(B, H_q, L_q, D)


__all__ = ["parallax_func"]

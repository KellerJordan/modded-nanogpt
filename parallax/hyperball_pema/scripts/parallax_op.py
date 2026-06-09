"""Compile-safe wrapper around ``parallax.triton.parallax_func``.

The kernel is exposed via:
  * ``_fwd_op`` / ``_bwd_op`` — ``torch.library.custom_op``s with fake impls,
    so AOTAutograd's retrace sees opaque ops instead of the Triton kernel's
    ``.data_ptr()``.
  * ``_ParallaxFunction`` — ``torch.autograd.Function`` wiring autograd over
    the two custom_ops. Mirrors how native ATen autograd ops are structured.
  * ``parallax_func`` — public wrapper, ``@torch.compiler.allow_in_graph``ed
    so Dynamo emits a single graph node for the call.

Also forces AOTAutograd's min-cut partitioner to **save** rather than
**recompute** activations across the forward/backward boundary
(``recomputable_ops = OrderedSet()``). Without this, Inductor's joint-graph
lowering hits a stride/recompute interaction (pytorch/pytorch#159469-family)
that produces deterministic NaN gradients in the attention backward —
reproduced under MHA + GQA with multiple optimizers (SOAP-H, DynMuon).
Side effects: small memory bump for saved activations; affects all
``torch.compile`` invocations in the process, not just the parallax op.
Acceptable here because there's no user-level ``torch.utils.checkpoint``
in the codebase, so the partitioner's recompute heuristic wasn't yielding
much memory anyway.

Drop-in for the upstream ``parallax.triton.parallax_func.parallax_func``.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_PARALLAX_PATH = os.environ.get('PARALLAX_PATH', '')
if _PARALLAX_PATH and _PARALLAX_PATH not in sys.path:
    sys.path.insert(0, _PARALLAX_PATH)


# Force AOTAutograd's min-cut partitioner to save (not recompute) activations
# across the fwd/bwd boundary. See module docstring for why.
try:
    import torch._functorch.partitioners as _ptn

    _ORIG_GET_DEFAULT_OP_LIST = _ptn.get_default_op_list

    def _no_recompute_op_list():
        op_types = _ORIG_GET_DEFAULT_OP_LIST()
        try:
            op_types.recomputable_ops.clear()
        except (AttributeError, TypeError):
            from dataclasses import replace
            try:
                from torch.utils._ordered_set import OrderedSet
            except ImportError:
                OrderedSet = set
            op_types = replace(op_types, recomputable_ops=OrderedSet())
        return op_types

    _ptn.get_default_op_list = _no_recompute_op_list
except Exception as _e:  # noqa: BLE001
    print(f"[parallax_op] partitioner patch skipped: {type(_e).__name__}: {_e}",
          file=sys.stderr)

from parallax.triton.parallax_fwd import parallax_fwd as _parallax_fwd  # noqa: E402
from parallax.triton.parallax_bwd import parallax_bwd as _parallax_bwd  # noqa: E402


@torch.library.custom_op("parallax::fwd", mutates_args=())
def _fwd_op(q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            qk_scale: float, n_rep: int, window_size_left: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _parallax_fwd(q, r, k, v, qk_scale, n_rep, window_size_left)


@_fwd_op.register_fake
def _(q, r, k, v, qk_scale, n_rep, window_size_left):
    B, L, D = q.shape
    return (
        torch.empty((B, L, D), device=q.device, dtype=q.dtype),
        torch.empty((B, L, D), device=q.device, dtype=q.dtype),
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),
        torch.empty((B, L, 1), device=q.device, dtype=torch.float32),
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
        torch.empty_like(q), torch.empty_like(r),
        torch.empty_like(k), torch.empty_like(v),
    )


class _ParallaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, r, k, v, qk_scale, n_rep, window_size_left):
        o, barv, d1, bart, m = _fwd_op(q, r, k, v, qk_scale, n_rep, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.qk_scale = qk_scale
        ctx.n_rep = n_rep
        ctx.window_size_left = window_size_left
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        grad_q, grad_r, grad_k, grad_v = _bwd_op(
            q, r, k, v, o, barv, d1, bart, m, grad_o.contiguous(),
            ctx.qk_scale, ctx.n_rep, ctx.window_size_left,
        )
        return grad_q, grad_r, grad_k, grad_v, None, None, None


@torch.compiler.allow_in_graph
def parallax_func(q: torch.Tensor, r: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor,
                  qk_scale: float | None = None,
                  window_size_left: int = -1) -> torch.Tensor:
    """Causal Parallax with autograd, compile-safe.

    Args:
        q, r: ``(B, H_q, L_q, D)`` bf16/fp16.
        k, v: ``(B, H_kv, L_kv, D)`` same dtype as q, with H_q % H_kv == 0.
        qk_scale: defaults to ``1 / sqrt(D)``.
        window_size_left: causal sliding-window length (FA2 convention).
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallax_func requires bf16 or fp16 inputs, got {q.dtype}")
    B, H_q, L_q, D = q.shape
    _, H_kv, L_kv, _ = k.shape
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv}"
    n_rep = H_q // H_kv
    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D)

    o = _ParallaxFunction.apply(
        q.reshape(B * H_q, L_q, D),
        r.reshape(B * H_q, L_q, D),
        k.reshape(B * H_kv, L_kv, D),
        v.reshape(B * H_kv, L_kv, D),
        float(qk_scale), int(n_rep), int(window_size_left),
    )
    return o.reshape(B, H_q, L_q, D)


__all__ = ["parallax_func"]

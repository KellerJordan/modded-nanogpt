"""Residual-site fusions: the plain residual expressions, with a backward that reads each activation once.

What they replace: the residual updates `x = r * x + p * y`, where r (resid lambda) and p (post lambda)
are learned 0-D scalars. Their forward is one elementwise pass, but in the backward every scalar's
gradient is a reduction over a whole [T, 768] activation, which the stock autograd graph runs as its own
pass over (grad, x) next to the pass that forms the activation gradient.

Why they are faster (the three fusions of record #360):
  - scale / scale_add: autograd Functions whose backward forms the scalars' gradients as row sums,
    stacked into one reduction, from the same reads of the incoming gradient that form r * grad.
  - The attention post-lambda rides the attention output projection instead (AttnArgs.o_gain, see
    model/attention.py): the O weight is already multiplied by the 0-D sa_lambdas[1], so a second
    0-D gain costs nothing in the forward, and its gradient is a reduction over the 768 x 768 weight
    rather than over the T x 768 output. (The fp8 MLP does the same: its post-lambda is folded into the
    down-projection scale, perf/kernels/mlp.py.)
  - rms_norm_with_head: the value-embedding gate reads the first 6 channels of norm(x). Read as a
    slice of the normed tensor, the slice's gradient is padded back to full width and folded into
    the norm's backward; recomputed from the norm's shared per-row rstd, it stays 6 columns wide.
The gains are record #360's measurements; they were not re-measured separately here.

Invariants:
  - scale / scale_add take 0-D scalars only: a per-token coefficient (the MUDD layer's mu[k]) keeps plain
    autograd, since its gradient is not a whole-tensor reduction.
  - Forward values are the plain expressions' exactly. The scalar gradients accumulate in fp32, as the
    compiled plain graph does; in eager bf16 they differ from plain autograd, which rounds each
    product to bf16 before summing.
  - o_gain moves the post-lambda's rounding: y @ (s * p * W) rather than p * (y @ (s * W)). Equal in
    exact arithmetic, not bitwise.
  - rms_norm_with_head's normed tensor is F.rms_norm's (same fp32 upcast, same eps) for bf16 or fp32 input.
"""
import torch
from torch import Tensor

# F.rms_norm's default eps: its opmath (fp32) epsilon, for bf16 and fp32 inputs alike.
RMS_NORM_EPS = torch.finfo(torch.float32).eps


class _ScaleResidual(torch.autograd.Function):
    """r * x for a 0-D r."""
    @staticmethod
    def forward(ctx, r, x):
        ctx.save_for_backward(r, x)
        return r * x

    @staticmethod
    def backward(ctx, grad):
        r, x = ctx.saved_tensors
        grad_r = (grad.float() * x.float()).sum(dim=-1).reshape(-1).sum()
        return grad_r.to(r.dtype), r * grad


class _ScaleAddResidual(torch.autograd.Function):
    """r * x + p * y for 0-D r and p."""
    @staticmethod
    def forward(ctx, r, x, p, y):
        ctx.save_for_backward(r, x, p, y)
        return r * x + p * y

    @staticmethod
    def backward(ctx, grad):
        r, x, p, y = ctx.saved_tensors
        grad_f = grad.float()
        # Both scalars' row sums in one reduction.
        grad_rp = torch.stack([(grad_f * t.float()).sum(dim=-1).reshape(-1) for t in (x, y)]).sum(dim=-1)
        return grad_rp[0].to(r.dtype), r * grad, grad_rp[1].to(p.dtype), p * grad


def scale(r: Tensor, x: Tensor) -> Tensor:
    """r * x, for a 0-D scalar r."""
    assert r.ndim == 0
    return _ScaleResidual.apply(r, x)


def scale_add(r: Tensor, x: Tensor, p: Tensor, y: Tensor) -> Tensor:
    """r * x + p * y, for 0-D scalars r and p."""
    assert r.ndim == 0 and p.ndim == 0
    return _ScaleAddResidual.apply(r, x, p, y)


def rms_norm_with_head(x: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """(norm(x), norm(x)[..., :width]), the head recomputed from the shared per-row rstd."""
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    normed = (x_f * rstd).to(x.dtype)
    head = (x[..., :width].float() * rstd).to(x.dtype)
    return normed, head

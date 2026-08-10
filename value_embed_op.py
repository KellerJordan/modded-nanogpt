"""Opaque SLG gather-backward seam for the trainer.

The forward remains the native five-plane advanced-index expression so Dynamo
can retain the shipping forward graph.  Only the custom autograd backward is
opaque: it allocates the same dense BF16 value-embedding gradient and launches
the already-gated selected-adjoint-load kernel with the shipping grid,
thread-to-contribution mapping, and relaxed atomic stores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


PLANES = 5
VOCAB = 50304
WIDTH = 768
EXPECTED_KERNEL_SHA256 = (
    "cdfff4f283c3da3bcfa92df4b07bcc47b22171958a9f10f55b238e56e3085f43"
)

_KERNEL_PATH = Path(__file__).with_name("value_embed_kernel.py")
_kernel_sha256 = hashlib.sha256(_KERNEL_PATH.read_bytes()).hexdigest()
assert _kernel_sha256 == EXPECTED_KERNEL_SHA256, (
    f"SLG kernel source mismatch: got {_kernel_sha256}, "
    f"expected {EXPECTED_KERNEL_SHA256}"
)

from value_embed_kernel import launch_candidate  # noqa: E402


def _check_token_ids(token_ids: torch.Tensor) -> None:
    assert token_ids.ndim == 1
    assert token_ids.dtype == torch.int32
    assert token_ids.is_contiguous()
    assert token_ids.device.type == "cuda"


def _check_weight(weight: torch.Tensor, token_ids: torch.Tensor) -> None:
    assert weight.shape == (PLANES * VOCAB, WIDTH)
    assert weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    assert weight.device == token_ids.device


def _check_grad(grad: torch.Tensor, token_ids: torch.Tensor) -> None:
    assert grad.shape == (token_ids.numel(), WIDTH)
    assert grad.dtype == torch.bfloat16
    assert grad.is_contiguous()
    assert grad.device == token_ids.device


@torch.library.custom_op(
    "nanogpt_slg::selected_load_backward",
    mutates_args=(),
)
def selected_load_backward_op(
    token_ids: torch.Tensor,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    grad3: torch.Tensor,
    grad4: torch.Tensor,
) -> torch.Tensor:
    _check_token_ids(token_ids)
    grads = (grad0, grad1, grad2, grad3, grad4)
    for grad in grads:
        _check_grad(grad, token_ids)
    output = torch.zeros(
        (PLANES * VOCAB, WIDTH),
        device=token_ids.device,
        dtype=torch.bfloat16,
    )
    launch_candidate(token_ids, grads, output.view(PLANES, VOCAB, WIDTH))
    return output


@torch.library.custom_op(
    "nanogpt_slg::selected_load_backward_into",
    mutates_args={"output"},
)
def selected_load_backward_into_op(
    token_ids: torch.Tensor,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    grad3: torch.Tensor,
    grad4: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Accumulate directly into a persistent two-step BF16 cycle buffer."""
    _check_token_ids(token_ids)
    grads = (grad0, grad1, grad2, grad3, grad4)
    for grad in grads:
        _check_grad(grad, token_ids)
    _check_weight(output, token_ids)
    launch_candidate(token_ids, grads, output.view(PLANES, VOCAB, WIDTH))


@selected_load_backward_into_op.register_fake
def _selected_load_backward_into_fake(
    token_ids: torch.Tensor,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    grad3: torch.Tensor,
    grad4: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del grad1, grad2, grad3, grad4
    assert token_ids.ndim == 1 and grad0.shape == (token_ids.numel(), WIDTH)
    assert output.shape == (PLANES * VOCAB, WIDTH)


@selected_load_backward_op.register_fake
def _selected_load_backward_fake(
    token_ids: torch.Tensor,
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    grad3: torch.Tensor,
    grad4: torch.Tensor,
) -> torch.Tensor:
    del grad1, grad2, grad3, grad4
    assert token_ids.ndim == 1 and grad0.shape == (token_ids.numel(), WIDTH)
    return torch.empty(
        (PLANES * VOCAB, WIDTH),
        device=grad0.device,
        dtype=grad0.dtype,
    )


class _ValueEmbeddingSelectedLoad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, token_ids: torch.Tensor):
        _check_token_ids(token_ids)
        _check_weight(weight, token_ids)
        ctx.save_for_backward(token_ids)
        gathered = weight.view(PLANES, VOCAB, WIDTH)[:, token_ids]
        # Separate outputs preserve the five shipping adjoint producers.  This
        # avoids materializing a stacked [5,T,D] adjoint before the custom op.
        return gathered[0], gathered[1], gathered[2], gathered[3], gathered[4]

    @staticmethod
    def backward(ctx, grad0, grad1, grad2, grad3, grad4):
        (token_ids,) = ctx.saved_tensors
        grads = (grad0, grad1, grad2, grad3, grad4)
        assert all(grad is not None for grad in grads)
        for grad in grads:
            _check_grad(grad, token_ids)
        grad_weight = selected_load_backward_op(token_ids, *grads)
        return grad_weight, None


class _ValueEmbeddingSelectedLoadInto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor):
        _check_token_ids(token_ids)
        _check_weight(weight, token_ids)
        _check_weight(output, token_ids)
        ctx.save_for_backward(token_ids, output)
        gathered = weight.view(PLANES, VOCAB, WIDTH)[:, token_ids]
        return gathered[0], gathered[1], gathered[2], gathered[3], gathered[4]

    @staticmethod
    def backward(ctx, grad0, grad1, grad2, grad3, grad4):
        token_ids, output = ctx.saved_tensors
        grads = (grad0, grad1, grad2, grad3, grad4)
        assert all(grad is not None for grad in grads)
        for grad in grads:
            _check_grad(grad, token_ids)
        selected_load_backward_into_op(token_ids, *grads, output)
        return None, None, None


def value_embedding_planes_selected_load(
    weight: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _ValueEmbeddingSelectedLoad.apply(weight, token_ids)


def value_embedding_planes_selected_load_into(
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _ValueEmbeddingSelectedLoadInto.apply(weight, token_ids, output)

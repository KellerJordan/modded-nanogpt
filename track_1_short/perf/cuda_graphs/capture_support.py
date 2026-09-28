"""Helpers every CUDA-graph owner in this package shares: the pre-capture warm count and the
snapshot / restore / bitwise-compare used by the captures and their self-checks."""
import torch

# Eager iterations on the capture stream before recording, so cuBLAS handles and workspaces, inductor
# autotuning and Triton compiles all happen outside the capture (record #360 warms every capture 3 times).
WARM_ITERATIONS = 3


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of two tensors' values (NaN-safe, any dtype)."""
    as_bytes = lambda t: t.contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(as_bytes(a), as_bytes(b))


def snapshot(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: t.detach().clone() for name, t in tensors.items()}


@torch.no_grad()
def restore(tensors: dict[str, torch.Tensor], saved: dict[str, torch.Tensor]):
    """In place: the graphs bake these addresses."""
    for name, t in tensors.items():
        t.copy_(saved[name])

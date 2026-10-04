"""Deterministic first-occurrence slots for sparse CUDA gradient rows."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _reset_slots(GIDX, SLOTS, N, BLOCK: tl.constexpr):
    positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = positions < N
    rows = tl.load(GIDX + positions, mask=mask, other=0)
    # Duplicate stores are intentional: every writer stores the same sentinel.
    tl.store(SLOTS + rows, 2147483647, mask=mask)


@triton.jit(do_not_specialize=["N"])
def _select_slots(GIDX, SLOTS, N, BLOCK: tl.constexpr):
    positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = positions < N
    rows = tl.load(GIDX + positions, mask=mask, other=0)
    tl.atomic_min(SLOTS + rows, positions, mask=mask, sem="relaxed")


def prepare_grad_rows(gidx: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Return int32 first-occurrence positions and update only touched slots.

    Both arguments must be contiguous, one-dimensional int32 CUDA tensors on
    the same device, with nonoverlapping storage. The caller guarantees that
    every row in gidx is in [0, slots.numel()); values are not inspected here.
    Untouched slots retain their previous contents and must not be consumed as
    part of this batch. Slot positions are not dense unique-row ordinals.

    Reset, atomic minimum, and gather run in order on the device's current CUDA
    stream. Separate kernel launches ensure all resets finish before minima
    are computed; callers must synchronize any other stream accessing slots.
    """
    for name, tensor in (("gidx", gidx), ("slots", slots)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if gidx.device != slots.device:
        raise ValueError("gidx and slots must be on the same CUDA device")
    if gidx.device.type != "cuda":
        raise ValueError("gidx and slots must be CUDA tensors")

    n = gidx.numel()
    if n > 2147483647:
        raise ValueError("gidx length must fit in signed int32")
    if n == 0:
        return torch.empty_like(gidx)

    block = 256
    grid = (triton.cdiv(n, block),)
    with torch.cuda.device(gidx.device):
        _reset_slots[grid](gidx, slots, n, BLOCK=block, num_warps=4)
        _select_slots[grid](gidx, slots, n, BLOCK=block, num_warps=4)
        return torch.index_select(slots, 0, gidx)

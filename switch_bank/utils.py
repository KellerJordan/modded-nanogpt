import math
from functools import lru_cache
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F


def _sanitize(t: Tensor, *, value: float = 0.0) -> Tensor:
    if torch.isfinite(t).all():
        return t
    return torch.nan_to_num(t, nan=value, posinf=value, neginf=value)


def _safe_softmax(logits: Tensor, dim: int) -> Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0)
    probs = logits.softmax(dim=dim)
    probs = _sanitize(probs)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return probs / denom


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def rampdown_multiplier(progress: float, start_frac: float, end_frac: float) -> float:
    if end_frac < 0:
        return 1.0
    if progress >= end_frac:
        return 0.0
    if start_frac < 0 or start_frac >= end_frac:
        return 1.0
    if progress <= start_frac:
        return 1.0
    span = max(end_frac - start_frac, 1e-6)
    frac = 1.0 - (progress - start_frac) / span
    return min(max(frac, 0.0), 1.0)


def compute_train_micro_len(train_seq_len: int, grad_accum_steps: int, train_micro_seq_len: int | None) -> int:
    if train_micro_seq_len is not None:
        micro = train_micro_seq_len
    else:
        approx = max(train_seq_len // grad_accum_steps, 128)
        approx = (approx // 128) * 128
        if approx == 0:
            approx = 128
        micro = approx
    assert micro % 128 == 0, "train_micro_seq_len must be a multiple of 128 tokens (block size)"
    return micro


def summarize(values: Iterable[Tensor | float], reducer) -> dict:
    # Placeholder utility for potential future reductions; kept for parity with planned structure.
    return {}

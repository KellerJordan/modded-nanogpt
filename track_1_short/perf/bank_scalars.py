"""The ANVIL banks' per-step scalars in one device buffer, refreshed by one pinned H2D per optimizer step.

What it replaces: passing each bank's momentum, decay, rail beta / weight and per-matrix learning rates
to its update as Python floats (a new constant, i.e. a recompile, whenever one changes) or as a dozen
small host-to-device copies.

Why it is faster: every scalar every bank reads lives in ONE fp32 device buffer, and the optimizer
uploads the whole step's values with one non-blocking copy from a pinned host row, at the top of
step() and outside the captured graphs (perf/cuda_graphs/optimizer_graphs.py). A replay then reads the
step's values from memory instead of baking constants.

Invariants: the device buffer and the views handed to each AnvilBank are allocated once and only
written in place (the optimizer graphs bake their addresses); a pinned row is rewritten only after its
previous H2D retired (one event per row, and BANK_SCALAR_SLOTS rows, deeper than the host's run-ahead
of the device in optimizer steps).

Provenance: record #360 (ANVIL2).
"""
import numpy as np
import torch
from torch import Tensor

BANK_SCALAR_SLOTS = 16
# The 0-D per-bank scalars, in their order in a bank's part of the buffer.
BANK_SCALAR_FIELDS = ("momentum", "eff_wd", "fast_beta", "fast_weight")


class BankScalarStaging:
    """Layout: per bank the BANK_SCALAR_FIELDS, then (after all banks' fields) each bank's per-matrix
    learning rates, `chunk_sizes[i]` of them for bank i."""

    def __init__(self, chunk_sizes: list[int], device: torch.device):
        num_fields = len(BANK_SCALAR_FIELDS)
        self.chunk_sizes = chunk_sizes
        self.lr_offsets = (num_fields * len(chunk_sizes) + np.concatenate(([0], np.cumsum(chunk_sizes)))).tolist()
        size = self.lr_offsets[-1]
        self.device_buffer = torch.zeros(size, dtype=torch.float32, device=device)
        self.slots = [(torch.zeros(size, dtype=torch.float32, pin_memory=True), torch.cuda.Event())
                      for _ in range(BANK_SCALAR_SLOTS)]
        self.next_slot = 0

    def scalars(self, bank: int) -> dict[str, Tensor]:
        """Bank `bank`'s 0-D device scalars, by field name."""
        num_fields = len(BANK_SCALAR_FIELDS)
        views = self.device_buffer[bank * num_fields:(bank + 1) * num_fields].unbind(0)
        return dict(zip(BANK_SCALAR_FIELDS, views))

    def eff_lr(self, bank: int) -> Tensor:
        """Bank `bank`'s per-matrix learning rates, [chunk, 1, 1] on the device."""
        lo, n = self.lr_offsets[bank], self.chunk_sizes[bank]
        return self.device_buffer[lo:lo + n].view(n, 1, 1)

    def upload(self, fields: list[tuple[float, ...]], eff_lrs: list[list[float]]):
        """One step's values, per bank: its BANK_SCALAR_FIELDS values and its per-matrix learning rates.
        Each value rounds to fp32 once, at the store."""
        host, uploaded = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % BANK_SCALAR_SLOTS
        uploaded.synchronize()
        row = host.numpy()
        num_fields = len(BANK_SCALAR_FIELDS)
        for i, (values, lrs) in enumerate(zip(fields, eff_lrs, strict=True)):
            row[i * num_fields:(i + 1) * num_fields] = values
            row[self.lr_offsets[i]:self.lr_offsets[i + 1]] = np.array(lrs)
        self.device_buffer.copy_(host, non_blocking=True)
        uploaded.record()

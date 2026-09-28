"""Pinned host slots for the data loader's per-batch H2D copies.

What it replaces: `.to("cuda", non_blocking=True)` straight from the loader's pageable CPU tensors. From
pageable memory the copy is staged through a driver buffer and does not return until the host side is
done, so each batch costs host time on the critical path of whoever fetches it (the training step, or the
validation's on-clock batch reads).

Why it is faster: from page-locked memory the copy is a true async DMA, queued and forgotten. With the
training step replayed as CUDA graphs the host otherwise runs ahead of the device, so every host stall
that remains shows up directly in the step time.

Invariant: a slot is rewritten only after its previous H2D retired (one event per slot, recorded after
its copies). The device tensors are fresh allocations, so no reader of a batch ever sees a slot reused.
One staging object serves the training and validation loaders (created once in main(), off the clock).

Provenance: record #360 (ANVIL2): `_PinnedRing`, `_pinld_slot`, `_loader_pin_cap`, `_sgs_pinned_like`.
#360 guards its rings with the CUDA-graph run-ahead throttle; here each slot has its own event.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

# Slots in the ring (record #360's _PINLD_RING): deeper than the loader's lookahead, so a slot's copy has
# long retired when it comes round again.
PINNED_BATCH_SLOTS = 16


@dataclass(slots=True)
class PinnedSlot:
    inputs: Tensor        # int32 [max_tokens]
    targets: Tensor       # int64 [max_tokens]
    cum_seqlens: Tensor   # int32 [max_cu_seqlens_rows]
    ngram_ids: Tensor     # int32 [2 * max_tokens]
    uploaded: torch.cuda.Event


class PinnedBatchStaging:
    def __init__(self, max_tokens: int, max_cu_seqlens_rows: int, device: torch.device):
        pinned = lambda n, dtype: torch.empty(n, dtype=dtype, pin_memory=True)
        self.device = device
        self.slots = [PinnedSlot(pinned(max_tokens, torch.int32), pinned(max_tokens, torch.int64),
                                 pinned(max_cu_seqlens_rows, torch.int32), pinned(2 * max_tokens, torch.int32),
                                 torch.cuda.Event())
                      for _ in range(PINNED_BATCH_SLOTS)]
        self.next_slot = 0

    def upload(self, inputs: Tensor, targets: Tensor, cum_seqlens: Tensor, ngram_ids: Tensor) -> tuple[Tensor, ...]:
        """Device copies of one batch's host tensors, through the next pinned slot."""
        slot = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % PINNED_BATCH_SLOTS
        slot.uploaded.synchronize()
        out = []
        for host, buf in ((inputs, slot.inputs), (targets, slot.targets), (cum_seqlens, slot.cum_seqlens),
                          (ngram_ids, slot.ngram_ids)):
            n = host.numel()
            assert n <= buf.numel() and host.dtype == buf.dtype, f"batch tensor {host.dtype}[{n}] overflows its pinned slot"
            buf[:n].copy_(host)
            out.append(buf[:n].to(self.device, non_blocking=True))
        slot.uploaded.record()
        return tuple(out)

"""Deferred post-optimizer all-gathers: the bank gathers' WAIT moves from the end of the optimizer step
to the first real consumer, the next step's fp8 weight refresh.

What it replaces: waiting every all-gather at the end of AnvilAndAdam.step, then refreshing the fp8
weight caches, so the compute stream idles until the largest collectives of the step (the qk/vo/mlp bank
gathers) have finished.

Why it is faster: the gathers are launched as before but not waited; the compute stream goes on with
the step's tail (tail-average ticks, the next step's row-pull uploads, cache restores) while they run.
flush() then waits each gather right before ITS OWN consumer, and the fp8 refresh, moved here from the
end of the optimizer step, runs in two halves around the waits:

    wait qk_bank, vo_bank gathers  ->  refresh the attention fp8 caches (model.quantize_attn_fp8)
    complete value_embeds' replica land (perf/value_embed_pull.py; its row exchange was served in the
                                         event's optimizer step)
    wait mlp_bank gather            ->  refresh the MLP fp8 caches and, after an Adam step, the lm_head
                                        fp8 copies (model.quantize_mlp_fp8)

Ordering contract. Between the end of step s's optimizer and flush(), nothing may read a bank's full
replica or the peers' rows of the value_embeds replica. The readers, all after flush():
  - the fp8 refresh itself (flush, both halves)
  - the training forward and backward of step s+1 (bf16 bank slices, the value_embeds replica)
  - the sampled-softmax row gather of step s+1 (reads the lm_head fp8 copy the refresh writes)
  - validation: the forward, value_embeds' gather_replica, and at the end the tail-average ship, whose
    decontraction norms read every full bank and whose gathers write them
  - the checkpoint's state_dict
What does run in between reads neither a full bank nor the value_embeds replica's peer rows:
  - the tail-average ticks (tail_average.py): only this rank's own shard, which the in-place all-gather
    never writes
  - row_prefetch.forward_slots (train_step): the n-gram cache and table, want-list uploads and count
    exchanges; at the first step of a run (nothing in flight yet) also the initial synchronous fills
  - sampled_softmax.upload (train_step): host-built candidate ids and positions copied to their own
    device buffers; the lm_head rows are gathered by sampled_softmax.gather, after flush()
Collectives issued in between are fine: NCCL runs them after the gathers, in launch order.

Both refresh halves go through Fp8RefreshGraphs (perf/cuda_graphs/fp8_refresh_graphs.py), which
replays their captured graphs once the fp8 exact-scale bootstrap is over and runs them eagerly before.

The refresh changes no number: nothing writes a bank, lm_head or the fp8 scale state between the
optimizer step and flush (eval forwards run the bf16 path).

Provenance: record #360 (ANVIL2): `ags_flush`, `_AGS_NOW` / `_AGS_PENDING` / `_AGS_QPEND` ("AGTAIL"),
`_TG2.refresh_attn` / `refresh_mlp`.
"""
import torch

from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs
from track_1_short.perf.value_embed_pull import ValueEmbedPull

ATTN_BANK_LABELS = ("qk_bank", "vo_bank")
BANK_LABELS = (*ATTN_BANK_LABELS, "mlp_bank")  # work order
DEFERRED_LABELS = frozenset(BANK_LABELS)


class DeferredGathers:
    """The bank gathers and fp8 refresh one optimizer step leaves in flight. Created in main()."""

    def __init__(self, value_embeds: ValueEmbedPull, fp8_refresh: Fp8RefreshGraphs):
        self.value_embeds = value_embeds
        self.fp8_refresh = fp8_refresh
        self.gathers: dict[str, torch.futures.Future] = {}
        # The fp8 refresh the last optimizer step owes: its refresh_lm, or None when nothing is owed.
        self.refresh_lm: bool | None = None

    def hand_over(self, gathers: dict[str, torch.futures.Future], refresh_lm: bool):
        """After an optimizer step: its deferred gathers, and whether its refresh includes lm_head."""
        assert self.refresh_lm is None and not self.gathers, "an optimizer step ran before the previous flush"
        assert set(gathers) <= DEFERRED_LABELS
        self.gathers = gathers
        self.refresh_lm = refresh_lm

    def _wait(self, labels):
        for label in labels:
            future = self.gathers.pop(label, None)
            if future is not None:
                future.wait()

    def flush(self):
        """Before the next reader of the banks (see the contract above): wait each gather before its
        consumer and run the owed fp8 refresh. A no-op when nothing is in flight."""
        refresh_lm, self.refresh_lm = self.refresh_lm, None
        self._wait(ATTN_BANK_LABELS)
        if refresh_lm is not None:
            self.fp8_refresh.refresh_attn()
        self.value_embeds.complete_land()
        self._wait(BANK_LABELS)
        if refresh_lm is not None:
            self.fp8_refresh.refresh_mlp(refresh_lm)

    def flush_final(self):
        """Before the final validation: wait everything, and drop the owed fp8 refresh -- its only
        consumer is a training forward, and none follows."""
        self.refresh_lm = None
        self._wait(BANK_LABELS)
        self.value_embeds.complete_land()

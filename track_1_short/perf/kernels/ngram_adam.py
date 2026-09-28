"""The n-gram table's row-sparse Adam event (ngram_table.py) as Triton kernels: an owner-side row merge
without a sort, and a fused per-row update that replays the second-moment decay a row slept through.

What it replaces, in plain torch (entries = the owner's own cycle rows, then every row a peer asked for;
a row appears once per rank that read it):

    grad = zeros(num_distinct_rows, D); grad.index_add_(0, segment_of_entry, entry_grads)  # sort-based plan
    grad /= world
    v *= beta2                                               # EVERY row of the shard, every event
    v[rows] += (1 - beta2) * grad.square().mean(1)
    update = grad / (sqrt(v[rows]) + eps) * step_size
    update += where(update * p > 0, p * decay, 0)            # cautious weight decay
    p[rows] -= update

Why it is faster:
  1. No merge plan. claim_rows writes entry_of_row[row_e] = e for every entry e; entries of one row race
     and whichever write lands is THE entry of that row. The merge then scatter-adds every entry's
     gradient row straight into its row's claimed slot of an [entries, D] buffer (row_scatter.py with
     row_map), and the update runs one program per entry, of which only the claimed one commits. No sort,
     no segment ids, no distinct-row count on the host.
  2. No per-event pass over all V/world rows. With beta1 = 0 a row that got no gradient has an update of
     exactly 0 (the weight decay is gated on the update), so its weights never change and only its second
     moment owes the event's beta2. Each row records the event it is current to (last_event); a history
     keeps every event's beta2; and the next time a row is touched the kernel replays the missed decays in
     registers -- the same multiplies in the same order as the eager decay, so the same bits.
  3. One program per row fuses the gradient read, the row mean square, the moment update, the update, the
     weight decay and the write back; the per-event scalars are launch arguments, not device tensors.
  4. bring_rows_current (the gradient-free form) stamps the rows the next cycle reads up to the current
     event, so their replay at the next event is short.

Invariants: entries index rows local to this rank's shard (int32); the merge buffer is indexed by entry;
the 1/world average of the merged gradient is folded into grad_mul (and its square into sq_mul);
beta2_history[e] holds event e's beta2 for every event 1..current (rows start at event 0).

Provenance: record #360 (ANVIL2): bigram_kernels.py `_bgadam_rows_kernel_b0` / `bgadam_rows_launch`
(lazy replay, `_BGADAM_CTX` passes 1 and 2), `bgsp_merge_gradients`. #360 merges into a dense
[V/world, D] bf16 gradient buffer (16.2 GB) and elects the committer with atomic_max on last_event; the
claim map here costs 42 MB and the same number of launches (this trainer's peak memory has no 16 GB to spare).
"""
import torch
import triton
import triton.language as tl

CLAIM_BLOCK = 1024


# `num_entries` / `event` change every event: without do_not_specialize, Triton would compile a variant per
# value class (== 1, divisible by 16) the first time one appears, possibly on the clock.
@triton.jit(do_not_specialize=["num_entries"])
def _claim_rows_kernel(ROWS, ENTRY_OF_ROW, num_entries, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_entries
    rows = tl.load(ROWS + offs, mask=mask, other=0)
    tl.store(ENTRY_OF_ROW + rows, offs, mask=mask)


@triton.jit(do_not_specialize=["event"])
def _adam_rows_kernel(P, V, LAST_EVENT, BETA2_HISTORY, G, ROWS, ENTRY_OF_ROW, event,
                      beta2, eps, step_size, decay, grad_mul, sq_mul,
                      D: tl.constexpr, HAS_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    """One program per entry. HAS_GRAD: this event's update of the entry's row, if it is the row's claimed
    entry (G row = entry). Without: replay the row's second moment up to `event` (rows must not repeat)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    row = tl.load(ROWS + pid)
    row64 = row.to(tl.int64)
    last = tl.load(LAST_EVENT + row)
    if HAS_GRAD:
        commit = tl.load(ENTRY_OF_ROW + row) == pid
        # the decays of events last+1 .. event-1; this event's own is part of the update below
        missed = tl.where(commit, event - 1 - last, 0)
    else:
        missed = event - last
    v = tl.load(V + row64)
    for i in range(missed):
        v = v * tl.load(BETA2_HISTORY + (last + 1 + i))
    if HAS_GRAD:
        if commit:
            g = tl.load(G + (pid.to(tl.int64) * D + offs), mask=mask, other=0.0).to(tl.float32)
            p = tl.load(P + (row64 * D + offs), mask=mask, other=0.0).to(tl.float32)
            m = g * grad_mul
            # The row's mean square (masked lanes load 0 and add nothing).
            v = v * beta2 + (tl.sum(g * g, axis=0) / D) * sq_mul
            u = (m / (tl.sqrt(v) + eps)) * step_size
            u = u + tl.where((u * p) > 0, p * decay, 0.0)
            p = p - u
            tl.store(V + row64, v)
            tl.store(P + (row64 * D + offs), p.to(P.dtype.element_ty), mask=mask)
            tl.store(LAST_EVENT + row, event)
    else:
        tl.store(V + row64, v)
        tl.store(LAST_EVENT + row, event)


def claim_rows(rows: torch.Tensor, entry_of_row: torch.Tensor):
    """entry_of_row[rows[e]] = e for every entry e; for a repeated row, one of its entries (unspecified which)."""
    assert rows.dtype == torch.int32 and entry_of_row.dtype == torch.int32 and rows.is_contiguous()
    n = rows.numel()
    if n:
        _claim_rows_kernel[(triton.cdiv(n, CLAIM_BLOCK),)](rows, entry_of_row, n, BLOCK=CLAIM_BLOCK, num_warps=4)


def adam_rows_(shard: torch.Tensor, exp_avg_sq: torch.Tensor, last_event: torch.Tensor, beta2_history: torch.Tensor,
               grad: torch.Tensor, rows: torch.Tensor, entry_of_row: torch.Tensor, event: int, *, beta2: float,
               eps: float, step_size: float, decay: float, grad_mul: float, sq_mul: float):
    """Event `event`'s update of every row in `rows` (entries; rows may repeat, entry_of_row from claim_rows),
    reading the merged gradient of entry e at grad[e]. In place on shard / exp_avg_sq / last_event."""
    _check_state(shard, exp_avg_sq, last_event, beta2_history, rows, event)
    assert grad.shape == (rows.numel(), shard.shape[1]) and grad.is_contiguous()
    assert entry_of_row.dtype == torch.int32 and entry_of_row.shape == last_event.shape
    if rows.numel():
        _adam_rows_kernel[(rows.numel(),)](
            shard, exp_avg_sq, last_event, beta2_history, grad, rows, entry_of_row, event,
            float(beta2), float(eps), float(step_size), float(decay), float(grad_mul), float(sq_mul),
            D=shard.shape[1], HAS_GRAD=True, BLOCK=triton.next_power_of_2(shard.shape[1]), num_warps=4,
        )


def bring_rows_current(shard: torch.Tensor, exp_avg_sq: torch.Tensor, last_event: torch.Tensor,
                       beta2_history: torch.Tensor, rows: torch.Tensor, event: int):
    """Replay the second moment of `rows` (no repeats) up to event `event`. The weights need nothing."""
    _check_state(shard, exp_avg_sq, last_event, beta2_history, rows, event)
    if rows.numel():
        _adam_rows_kernel[(rows.numel(),)](
            shard, exp_avg_sq, last_event, beta2_history, shard, rows, last_event, event,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            D=shard.shape[1], HAS_GRAD=False, BLOCK=triton.next_power_of_2(shard.shape[1]), num_warps=4,
        )


def _check_state(shard, exp_avg_sq, last_event, beta2_history, rows, event):
    # beta1 = 0 on a bf16 table is what makes the weights replay-invariant (see the module docstring).
    assert shard.dtype == torch.bfloat16 and shard.is_contiguous() and shard.shape[0] < 2 ** 31
    assert exp_avg_sq.dtype == torch.float32 and exp_avg_sq.shape == shard.shape[:1]
    assert last_event.dtype == torch.int32 and last_event.shape == shard.shape[:1]
    assert beta2_history.dtype == torch.float32 and 0 < event < beta2_history.numel()
    assert rows.dtype == torch.int32 and rows.is_contiguous()

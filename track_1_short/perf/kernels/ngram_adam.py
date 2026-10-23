"""The n-gram table's row-sparse Adam event (ngram_table.py) as Triton kernels: an owner-side deterministic
row merge fused with the per-row update, and the second-moment replay a row slept through.

What the merge + update replace, in plain torch (own = this rank's rows of the cycle, recv = every row a
peer asked for, both as fp16 payloads; a row appears once per rank that read it, own first, then the peers
in rank order):

    for each distinct row r, its entries e_1..e_k in that order:
        g = 0 (bf16)
        for e in entries: g = bf16(g + bf16(payload[e]))   # a sequential bf16 chain, rounded after every add
    v[r] *= beta2 for the events r missed                     # replayed lazily, see below
    v[r] = v[r] * beta2 + g.square().mean() * sq_mul
    update = g * grad_mul / (sqrt(v[r]) + eps) * step_size
    update += where(update * p > 0, p * decay, 0)             # cautious weight decay
    p[r] -= update

Why it looks like this:
  1. The merge is deterministic. The entries are sorted by row (stable, so ties keep source order) and ONE
     program per distinct row sums its run in that order, rounding to bf16 after every addition -- what a
     sequential bf16 atomic chain from zero produces. With one or two contributors this equals the atomics'
     result in any order (a two-term bf16 sum is commutative); with three or more, atomics land in
     arbitrary order and the bits vary from run to run. Rounding each addend to bf16 first is what the
     peers' former fp16 -> bf16 wire cast did, now done by the receiver, so the wire carries fp16 views of
     the compact gradient with no send copy.
  2. No gradient buffer. The merge reads the contributions where they are (this rank's compact rows and
     the packed receive buffer), so there is no [entries, D] scratch, no atomics and no clearing, and the
     sum never leaves registers: it feeds the row mean square and the update directly.
  3. No per-event pass over all V/world rows. With beta1 = 0 a row that got no gradient has an update of
     exactly 0 (the weight decay is gated on the update), so its weights never change and only its second
     moment owes the event's beta2. Each row records the event it is current to (last_event); a history
     keeps every event's beta2; and the next time a row is touched the kernel replays the missed decays in
     registers -- the same multiplies in the same order as the eager decay, so the same bits.
  4. bring_rows_current (the gradient-free form) stamps the rows the next cycle reads up to the current
     event, so their replay at the next event is short.

Invariants: rows index this rank's shard (int32); the 1/world average of the merged gradient is folded into
grad_mul (and its square into sq_mul); beta2_history[e] holds event e's beta2 for every event 1..current
(rows start at event 0).

Provenance: record #360 (ANVIL2): bigram_kernels.py `_bgadam_rows_kernel_b0` / `bgadam_rows_launch` (lazy
replay). The deterministic merge is this trainer's: #360 scatters into a dense 16.2 GB [V/world, 768]
bf16 gradient buffer with atomics.
"""
import torch
import triton
import triton.language as tl


# `num_entries` / `num_own` / `event` change every event: without do_not_specialize, Triton would compile a
# variant per value class (== 1, divisible by 16) the first time one appears, possibly on the clock.
@triton.jit(do_not_specialize=["num_entries", "num_own", "event"])
def _merge_adam_kernel(P, V, LAST_EVENT, BETA2_HISTORY, OWN, RECV, SORTED_ROWS, ORDER,
                       num_entries, num_own, event, beta2, eps, step_size, decay, grad_mul, sq_mul,
                       D: tl.constexpr, BLOCK: tl.constexpr):
    """One program per entry; only the first entry of each run of equal rows does the work: it sums the
    run in source order and applies this event's update to the row."""
    pid = tl.program_id(0)
    row = tl.load(SORTED_ROWS + pid)
    prev = tl.load(SORTED_ROWS + tl.maximum(pid - 1, 0))
    if (pid == 0) | (prev != row):
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        o64 = offs.to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        j = pid
        go = pid >= 0
        while go:
            src = tl.load(ORDER + j)  # int64 index into [own | recv]
            c_own = tl.load(OWN + src * D + o64, mask=mask & (src < num_own), other=0.0).to(tl.float32)
            c_recv = tl.load(RECV + (src - num_own) * D + o64, mask=mask & (src >= num_own), other=0.0).to(tl.float32)
            c = (c_own + c_recv).to(tl.bfloat16).to(tl.float32)  # the addend as bf16; the other term is 0
            acc = (acc + c).to(tl.bfloat16).to(tl.float32)       # bf16 add, round to nearest even
            j += 1
            go = (j < num_entries) & (tl.load(SORTED_ROWS + tl.minimum(j, num_entries - 1)) == row)
        row64 = row.to(tl.int64)
        last = tl.load(LAST_EVENT + row)
        v = tl.load(V + row64)
        # the decays of events last+1 .. event-1; this event's own is part of the update below
        for i in range(event - 1 - last):
            v = v * tl.load(BETA2_HISTORY + (last + 1 + i))
        p = tl.load(P + (row64 * D + offs), mask=mask, other=0.0).to(tl.float32)
        m = acc * grad_mul
        # The row's mean square (masked lanes hold 0 and add nothing).
        v = v * beta2 + (tl.sum(acc * acc, axis=0) / D) * sq_mul
        u = (m / (tl.sqrt(v) + eps)) * step_size
        u = u + tl.where((u * p) > 0, p * decay, 0.0)
        p = p - u
        tl.store(V + row64, v)
        tl.store(P + (row64 * D + offs), p.to(P.dtype.element_ty), mask=mask)
        tl.store(LAST_EVENT + row, event)


@triton.jit(do_not_specialize=["event"])
def _replay_rows_kernel(V, LAST_EVENT, BETA2_HISTORY, ROWS, event):
    """One program per row (rows must not repeat): replay the row's second moment up to `event`."""
    pid = tl.program_id(0)
    row = tl.load(ROWS + pid)
    last = tl.load(LAST_EVENT + row)
    v = tl.load(V + row.to(tl.int64))
    for i in range(event - last):
        v = v * tl.load(BETA2_HISTORY + (last + 1 + i))
    tl.store(V + row.to(tl.int64), v)
    tl.store(LAST_EVENT + row, event)


def merge_adam_rows_(shard: torch.Tensor, exp_avg_sq: torch.Tensor, last_event: torch.Tensor,
                     beta2_history: torch.Tensor, own_vals: torch.Tensor, own_rows: torch.Tensor,
                     recv_vals: torch.Tensor, recv_rows: torch.Tensor, event: int, *, beta2: float, eps: float,
                     step_size: float, decay: float, grad_mul: float, sq_mul: float):
    """Event `event`'s update of every row a contribution names: own_vals[i] (fp16 [n_own, D]) is the
    gradient of shard row own_rows[i], recv_vals[i] that of row recv_rows[i]; rows are int32, local to the
    shard, unique within own_rows and within each peer's block of recv_rows. In place on shard / exp_avg_sq /
    last_event. The contributions of a row are summed in the order own, then the peers by rank."""
    _check_state(shard, exp_avg_sq, last_event, beta2_history, event)
    d = shard.shape[1]
    num_own, num_recv = own_rows.numel(), recv_rows.numel()
    assert own_rows.dtype == torch.int32 and recv_rows.dtype == torch.int32
    assert own_rows.is_contiguous() and recv_rows.is_contiguous()
    assert own_vals.shape == (num_own, d) and recv_vals.shape == (num_recv, d)
    assert own_vals.is_contiguous() and recv_vals.is_contiguous() and own_vals.dtype == recv_vals.dtype
    assert own_vals.dtype in (torch.float16, torch.bfloat16)
    n = num_own + num_recv
    if n:
        keys = torch.cat([own_rows, recv_rows]) if num_own else recv_rows
        sorted_rows, order = torch.sort(keys, stable=True)
        _merge_adam_kernel[(n,)](
            shard, exp_avg_sq, last_event, beta2_history, own_vals, recv_vals, sorted_rows, order,
            n, num_own, event, float(beta2), float(eps), float(step_size), float(decay), float(grad_mul),
            float(sq_mul), D=d, BLOCK=triton.next_power_of_2(d), num_warps=4,
        )


def bring_rows_current(shard: torch.Tensor, exp_avg_sq: torch.Tensor, last_event: torch.Tensor,
                       beta2_history: torch.Tensor, rows: torch.Tensor, event: int):
    """Replay the second moment of `rows` (no repeats) up to event `event`. The weights need nothing."""
    _check_state(shard, exp_avg_sq, last_event, beta2_history, event)
    assert rows.dtype == torch.int32 and rows.is_contiguous()
    if rows.numel():
        _replay_rows_kernel[(rows.numel(),)](exp_avg_sq, last_event, beta2_history, rows, event, num_warps=1)


def _check_state(shard, exp_avg_sq, last_event, beta2_history, event):
    # beta1 = 0 on a bf16 table is what makes the weights replay-invariant (see the module docstring).
    assert shard.dtype == torch.bfloat16 and shard.is_contiguous() and shard.shape[0] < 2 ** 31
    assert exp_avg_sq.dtype == torch.float32 and exp_avg_sq.shape == shard.shape[:1]
    assert last_event.dtype == torch.int32 and last_event.shape == shard.shape[:1]
    assert beta2_history.dtype == torch.float32 and 0 < event < beta2_history.numel()

"""The replicated Adam parameters as flat buffers: one all_reduce and one fused Adam launch per dtype.

What it replaces: per replicated parameter (the scalars, lambdas, gates and MUDD weights, 16 small
tensors), its own all_reduce and its own compiled Adam call with two 0-D scalar uploads -- about a hundred
tiny launches per Adam step, each far shorter than its launch overhead.

Why it is faster: the gradients are copied into one flat buffer per dtype (one foreach copy) and reduced
with one all_reduce each; one Triton kernel then runs the Adam step for every element, reading its
parameter's betas, step size and decay from a small scalar table (perf/kernels/replicated_adam.py). The
arithmetic is AnvilAndAdam._adam_update_step's, element for element.

Invariants:
  - After construction each parameter's .data and its exp_avg / exp_avg_sq are views into the flat
    buffers. Nothing may rebind them afterwards: model.load_state_dict and AnvilAndAdam.load_state_dict
    copy in place, and AnvilAndAdam.reset() leaves Adam state alone.
  - The gradient views tile each flat gradient buffer exactly, so the all_reduce moves only gradients.
  - Segment ids follow the optimizer's parameter order, identical on every rank.
  - The scalar table reaches the device through pinned slots; a slot is rewritten only after its
    previous upload retired (one event per slot).
  - A parameter without a gradient this step is skipped whole (its row is inactive), as the per-parameter
    path skips it; its stale slice of the flat gradient is reduced but never read.

Provenance: record #360 (ANVIL2): fuse_tiny_kernels.py `afe_flatten` ("FLAT") and `afe_run_fused`
("FUSE"), AnvilAndAdam `_far_flats` (the Phase 0 flat all_reduce).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from track_1_short.perf.kernels.replicated_adam import SCALAR_COLUMNS, fused_adam_

# Deeper than the host's run-ahead of the device in optimizer steps, so a slot is long retired when reused.
SCALAR_TABLE_SLOTS = 16


@dataclass
class FlatGroup:
    """The replicated parameters of one dtype, laid end to end in parameter order."""
    params: list[nn.Parameter]
    grad: Tensor        # the all_reduce buffer
    param: Tensor       # every parameter's .data is a view of this
    exp_avg: Tensor     # fp32
    exp_avg_sq: Tensor  # fp32
    segment: Tensor     # int32 per element: the parameter's row in the scalar table
    grad_views: dict[nn.Parameter, Tensor]


@torch.no_grad()
def flatten_replicated(params: list[nn.Parameter], param_states: dict, device: torch.device) -> list[FlatGroup]:
    """Copy each parameter and its Adam moments into flat buffers (one set per dtype) and re-point them
    at their slices. Bit-identical: bitwise copies, then views of the same shape and dtype."""
    segment_of = {p: i for i, p in enumerate(params)}
    groups = []
    for dtype in dict.fromkeys(p.dtype for p in params):  # first-seen order
        members = [p for p in params if p.dtype == dtype]
        n = sum(p.numel() for p in members)
        group = FlatGroup(
            params=members,
            grad=torch.zeros(n, dtype=dtype, device=device),
            param=torch.empty(n, dtype=dtype, device=device),
            exp_avg=torch.empty(n, dtype=torch.float32, device=device),
            exp_avg_sq=torch.empty(n, dtype=torch.float32, device=device),
            segment=torch.empty(n, dtype=torch.int32, device=device),
            grad_views={},
        )
        offset = 0
        for p in members:
            state = param_states[p]
            assert state["exp_avg"].dtype == torch.float32 and state["exp_avg"].shape == p.shape
            span = slice(offset, offset + p.numel())
            offset += p.numel()
            group.param[span].copy_(p.data.reshape(-1))
            group.exp_avg[span].copy_(state["exp_avg"].reshape(-1))
            group.exp_avg_sq[span].copy_(state["exp_avg_sq"].reshape(-1))
            group.segment[span] = segment_of[p]
            group.grad_views[p] = group.grad[span].view(p.shape)
            p.data = group.param[span].view(p.shape)
            state["exp_avg"] = group.exp_avg[span].view(p.shape)
            state["exp_avg_sq"] = group.exp_avg_sq[span].view(p.shape)
        assert offset == n
        groups.append(group)
    return groups


def adam_scalar_table(params: list[nn.Parameter], param_cfgs: dict, param_states: dict,
                      active: set[nn.Parameter]) -> np.ndarray:
    """Advance each active parameter's Adam step and return the kernel's scalar table, one row per
    parameter (SCALAR_COLUMNS). Same arithmetic as AnvilAndAdam._adam_update, in Python doubles
    rounded to fp32 once, at the store."""
    table = np.zeros((len(params), len(SCALAR_COLUMNS)), dtype=np.float32)
    for i, p in enumerate(params):
        if p not in active:
            continue
        cfg, state = param_cfgs[p], param_states[p]
        state["step"] += 1
        t = state["step"]
        beta1, beta2 = cfg.adam_betas
        lr = cfg.lr * cfg.lr_mul
        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        table[i] = (beta1, 1 - beta1, beta2, 1 - beta2, cfg.eps,
                    lr * (bias2 ** 0.5 / bias1), lr * lr * cfg.weight_decay * cfg.wd_mul, 1.0)
    return table


class ReplicatedAdam:
    """Owns the flat buffers of the replicated Adam parameters; AnvilAndAdam.step calls launch_reduce
    (Phase 0, Adam steps only) and update (at the first replicated label of its work order)."""

    def __init__(self, params: list[nn.Parameter], param_cfgs: dict, param_states: dict, device: torch.device):
        self.params = params
        self.param_cfgs = param_cfgs
        self.param_states = param_states
        self.groups = flatten_replicated(params, param_states, device)
        shape = (len(params), len(SCALAR_COLUMNS))
        self.table_slots = [(torch.zeros(shape, dtype=torch.float32, pin_memory=True), torch.cuda.Event())
                            for _ in range(SCALAR_TABLE_SLOTS)]
        self.next_slot = 0
        self.device_table = torch.zeros(shape, dtype=torch.float32, device=device)
        self.active: set[nn.Parameter] = set()
        self.reduces = []

    @torch.no_grad()
    def launch_reduce(self):
        """Copy this step's gradients into the flat buffers and start their all_reduces (async)."""
        assert not self.reduces
        self.active = {p for p in self.params if p.grad is not None}
        for group in self.groups:
            live = [p for p in group.params if p in self.active]
            if live:
                torch._foreach_copy_([group.grad_views[p] for p in live], [p.grad for p in live])
            self.reduces.append(dist.all_reduce(group.grad, op=dist.ReduceOp.AVG, async_op=True).get_future())

    @torch.no_grad()
    def update(self):
        """Wait for the reduces and run the Adam step of every replicated parameter at once."""
        for future in self.reduces:
            future.wait()
        self.reduces = []
        table = adam_scalar_table(self.params, self.param_cfgs, self.param_states, self.active)
        if not self.active:
            return
        host, uploaded = self.table_slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % SCALAR_TABLE_SLOTS
        uploaded.synchronize()
        host.copy_(torch.from_numpy(table))
        self.device_table.copy_(host, non_blocking=True)
        uploaded.record()
        for group in self.groups:
            fused_adam_(group.param, group.grad, group.exp_avg, group.exp_avg_sq, group.segment, self.device_table)

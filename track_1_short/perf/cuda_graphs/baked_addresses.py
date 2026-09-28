"""The address census across the post-warmup reset: every tensor a captured graph bakes must survive it in place.

A graph replays with the device addresses it was captured with. The warmup captures every graph and then
the run resets the model, optimizer and tables to their initial state; a reset that rebinds a tensor
(`p.data = ...`, `self.buf = torch.zeros(...)`, a fresh nn.Buffer) instead of writing it in place would
leave the graphs reading or writing freed memory -- silently. So main() takes this census before the
reset and asserts it is unchanged after, role by role (address and shape).

Roles covered: every parameter and buffer of the model (fp8 caches and scales, YaRN tables, n-gram row
cache, prefix table, ...), the step graphs' own inputs and the external tensors they bake
(StepGraphs.baked_tensors), and the optimizer banks' tensors (optimizer_graphs.bank_tensors) together
with the live optimizer state they alias (assert_banks_alias_live_state).

Provenance: record #360 (ANVIL2): `_baked_tensors`, `_cf_baked_ptrs`, `_cf_check_ptrs`.
"""
from torch import nn

from track_1_short.optim.anvil import AnvilAndAdam
from track_1_short.perf.cuda_graphs.optimizer_graphs import bank_tensors
from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs


def baked_addresses(model: nn.Module, optimizer: AnvilAndAdam, step_graphs: StepGraphs) -> dict[str, tuple[int, tuple]]:
    """role -> (data_ptr, shape) of every tensor a captured graph may read or write. `model` is the
    uncompiled GPT."""
    roles = [*((f"model.{n}", t) for n, t in model.named_parameters()),
             *((f"model.{n}", t) for n, t in model.named_buffers()),
             *step_graphs.baked_tensors(),
             *bank_tensors(optimizer)]
    roles = [(role, t) for role, t in roles if t.numel()]
    census = {role: (t.data_ptr(), tuple(t.shape)) for role, t in roles}
    assert len(census) == len(roles), "duplicate census roles"
    return census


def assert_addresses_unchanged(before: dict[str, tuple[int, tuple]], after: dict[str, tuple[int, tuple]]):
    """Every role of the census taken before the reset is at the same address, with the same shape, after it."""
    moved = {role: (before[role], after.get(role)) for role in before if after.get(role) != before[role]}
    assert not moved, f"rebound or reshaped across the reset (a graph baked them): {moved}"

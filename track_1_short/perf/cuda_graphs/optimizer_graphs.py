"""CUDA graphs of the optimizer tail: one graph per ANVIL bank update (qk_bank, vo_bank, mlp_bank).

What it replaces: running anvil_bank_update (optim/anvil.py) eagerly for each bank every step -- the
twin-rail momentum, six rounds of the whitening cascade (Gram, polynomial, update GEMMs per round), the
lane equalizer and the cautious-decay update: ~50 launches per bank, most of them short.

Why it is faster: each bank's update is the same work on the same buffers every step, so one replay
launches all of it. Record #360 measured ~0.9 s over the run for these graphs.

What is captured: anvil_bank_update(bank) for each bank, between the wait on its reduce-scatter and the
launch of its all-gather (AnvilAndAdam._anvil_update calls AnvilBankGraphs.update there: main() hands
it to the optimizer as its bank_update).

What must stay address-stable: every tensor of the AnvilBank (optim/anvil.py): the persistent
reduce-scatter destination, the velocity / lane-energy / mantissa state, the bank's parameter slice, and
the device scalar buffer that perf/bank_scalars.py refreshes with one H2D at the top of every step() --
outside the graphs, which is why the per-step learning rate, momentum, decay and rail weights reach a
replay as values in memory rather than baked constants. baked_addresses.py censuses them across the
post-warmup reset, and assert_banks_alias_live_state re-derives them from the live optimizer.

Capture protocol: captured once, between two warmup steps (capture_plan.py); each graph is then
self-checked bitwise against the eager body at its first use.

Provenance: record #360 (ANVIL2): AnvilAndAdam `_tgr_*` ("TAILGRAPH"). Deviation: no free-memory gate
(TAIL_GRAPH_FLOOR_MIB): #360's run refused to start when the gate skipped a capture, so here a capture
that does not fit simply fails.
"""
import dataclasses

import torch

from track_1_short.optim.anvil import AnvilAndAdam, AnvilBank, anvil_bank_update
from track_1_short.perf.cuda_graphs.capture_support import WARM_ITERATIONS, bits_equal, restore, snapshot


class AnvilBankGraphs:
    """Runs each bank's update for the optimizer: eagerly until captured, then as a replay of its
    graph. Created in main()."""

    def __init__(self, device: torch.device):
        self.stream = torch.cuda.Stream(device)
        # The three graphs always replay in capture order (work order), one after another: one pool.
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self.unchecked: set[str] = set()
        self.sealed = False

    def update(self, bank: AnvilBank):
        graph = self.graphs.get(bank.label)
        if graph is None:
            # Before the capture (the first warmup step) the body runs eagerly; after seal() never.
            assert not self.sealed, f"no optimizer graph for {bank.label}"
            anvil_bank_update(bank)
        elif bank.label in self.unchecked:
            self.unchecked.discard(bank.label)
            self._self_check(bank, graph)
        else:
            graph.replay()

    @torch.no_grad()
    def capture(self, banks: list[AnvilBank]):
        """Record every bank's update, in replay order. Runs between optimizer steps; leaves the state as found."""
        assert not self.graphs, "the optimizer graphs are captured once"
        for bank in banks:
            saved = snapshot(bank.mutated())
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(WARM_ITERATIONS):
                    anvil_bank_update(bank)
            torch.cuda.current_stream().wait_stream(self.stream)
            restore(bank.mutated(), saved)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # A capture records without executing, so the state stays as restored.
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode="thread_local"):
                anvil_bank_update(bank)
            torch.cuda.synchronize()
            self.graphs[bank.label] = graph
        self.unchecked = set(self.graphs)

    def _self_check(self, bank: AnvilBank, graph: torch.cuda.CUDAGraph):
        """At a graph's first use: the eager body and the replay, from the same state, must agree bit for
        bit. Catches a graph that baked a scalar's value or a stale address instead of reading it live."""
        before = snapshot(bank.mutated())
        anvil_bank_update(bank)
        eager = snapshot(bank.mutated())
        restore(bank.mutated(), before)
        graph.replay()  # the step's real update
        bad = [name for name, t in bank.mutated().items() if not bits_equal(t, eager[name])]
        assert not bad, f"optimizer graph {bank.label}: replay != eager on {bad}"

    def seal(self):
        assert self.graphs and not self.unchecked, f"optimizer graphs missing or unchecked: {sorted(self.unchecked)}"
        self.sealed = True


def assert_banks_alias_live_state(optimizer: AnvilAndAdam):
    """The tensors each bank (and so its graph) holds are still the optimizer's live state (run after the
    post-warmup reset, which must have restored everything in place)."""
    for label, bank in optimizer.banks.items():
        moved = [name for name, t in optimizer.live_bank_state(label).items()
                 if getattr(bank, name).data_ptr() != t.data_ptr()]
        assert not moved, f"optimizer bank {label}: {moved} rebound since capture"


def bank_tensors(optimizer: AnvilAndAdam):
    """(role, tensor) for every tensor the optimizer graphs bake."""
    for label, bank in optimizer.banks.items():
        for field in dataclasses.fields(bank):
            value = getattr(bank, field.name)
            if isinstance(value, torch.Tensor):
                yield f"bank.{label}.{field.name}", value

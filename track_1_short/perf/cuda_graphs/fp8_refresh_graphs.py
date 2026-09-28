"""CUDA graphs of the fp8 weight-cache refresh: one for the attention half, two for the MLP half.

What it replaces: model.quantize_attn_fp8() and model.quantize_mlp_fp8(refresh_lm) run eagerly after
every optimizer step (perf/deferred_gathers.py runs them before the next forward): per width group an
amax and a dual-layout quantize for attention; for the MLP two dual quantizes, the delayed-scale updates,
the products that fold the post-lambdas into the down-projection scales and, after an Adam step, the fp8
lm_head copies -- ~45 small launches.

Why it is faster: the refresh is the same work on the same buffers every step, so a replay launches it
as one graph (record #360: one replay replaces ~43.5 of the 45.5 launches).

What is captured: quantize_attn_fp8(); quantize_mlp_fp8(refresh_lm=False) and (refresh_lm=True), the
latter also writing the lm_head copies. Only the delayed-scale path is captured: the first
FP8_EXACT_SCALE_CALLS calls after init or the post-warmup reset set the weight scales exactly (a
different, host-branched body), so those run eagerly -- the timed run's first 15 refreshes.

What must stay address-stable: the banks, lm_head, post_lambdas (read), and every buffer the refresh
writes: the attention caches (attn_fp8.*), the _mlp_* caches and scales, the _lm_head_f8_* copies. All
are model buffers or parameters, allocated once and written in place (baked_addresses.py checks them).
The host counter model.mlp_quantize_calls, which a replay does not touch, is bumped here instead.

Capture protocol: captured once, between two warmup steps after the exact-scale bootstrap
(capture_plan.py); each of the three graphs is self-checked bitwise against the eager body at its first
use, which the plan schedules before the warmup ends.

Provenance: record #360 (ANVIL2): `_TailGraphs2` ("tgr2"). Deviation: no free-memory gate
(TAIL_GRAPH2_FLOOR_MIB), as for the optimizer graphs.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from track_1_short.model.gpt import FP8_EXACT_SCALE_CALLS
from track_1_short.perf.cuda_graphs.capture_support import WARM_ITERATIONS, bits_equal, restore, snapshot

# Buffer-name prefixes of what each half writes.
ATTN_OUTPUT_PREFIXES = ("attn_fp8.",)
MLP_OUTPUT_PREFIXES = ("_mlp_", "_lm_head_f8_")


@dataclass(slots=True)
class CapturedRefresh:
    name: str
    body: Callable[[], None]
    outputs: dict[str, torch.Tensor]   # what this half writes, checked by its self-check
    counter_bumps: int                 # model.mlp_quantize_calls increments the eager body makes
    graph: torch.cuda.CUDAGraph
    pool: object                       # private pool: the halves alternate by cadence, not capture order


class Fp8RefreshGraphs:
    """The fp8 refresh DeferredGathers.flush runs: eager until captured, and during the exact-scale bootstrap."""

    def __init__(self, model: nn.Module):
        self.model = model  # the uncompiled GPT: the host counter must be set on the module itself
        buffers = dict(model.named_buffers())
        self.attn_outputs = {n: t for n, t in buffers.items() if n.startswith(ATTN_OUTPUT_PREFIXES)}
        self.mlp_outputs = {n: t for n, t in buffers.items() if n.startswith(MLP_OUTPUT_PREFIXES)}
        assert self.attn_outputs and self.mlp_outputs
        self.captured: dict[str, CapturedRefresh] = {}  # "attn", "mlp", "mlp+lm_head"
        self.unchecked: set[str] = set()
        self.sealed = False

    def refresh_attn(self):
        self._refresh("attn", self.model.quantize_attn_fp8)

    def refresh_mlp(self, refresh_lm: bool):
        self._refresh("mlp+lm_head" if refresh_lm else "mlp", lambda: self.model.quantize_mlp_fp8(refresh_lm=refresh_lm))

    def _refresh(self, name: str, body: Callable[[], None]):
        """Both halves take the same predicate, so one step never mixes an eager half with a replayed one."""
        captured = self.captured.get(name)
        if captured is None or self.model.mlp_quantize_calls < FP8_EXACT_SCALE_CALLS:
            # After seal() every graph exists; eager here is only the bootstrap.
            assert captured is not None or not self.sealed, f"no fp8 refresh graph {name}"
            body()
        elif name in self.unchecked:
            self.unchecked.discard(name)
            self._self_check(captured)
        else:
            captured.graph.replay()
            self.model.mlp_quantize_calls += captured.counter_bumps

    @torch.no_grad()
    def capture(self):
        """Record the three variants. Runs between steps; the state (buffers and host counter) is left as found."""
        assert not self.captured, "the fp8 refresh graphs are captured once"
        # The captured body must be the delayed-scale one (see the header).
        assert self.model.mlp_quantize_calls >= FP8_EXACT_SCALE_CALLS, "fp8 refresh capture inside the bootstrap"
        stream = torch.cuda.Stream()
        m = self.model
        variants = (("attn", m.quantize_attn_fp8, self.attn_outputs, 0),
                    ("mlp", lambda: m.quantize_mlp_fp8(refresh_lm=False), self.mlp_outputs, 1),
                    ("mlp+lm_head", lambda: m.quantize_mlp_fp8(refresh_lm=True), self.mlp_outputs, 1))
        everything = {**self.attn_outputs, **self.mlp_outputs}
        saved, calls = snapshot(everything), m.mlp_quantize_calls
        for name, body, outputs, bumps in variants:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(WARM_ITERATIONS):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            restore(everything, saved)
            torch.cuda.synchronize()
            pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            # Recording runs the Python (the counter bump) but not the kernels.
            with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
                body()
            torch.cuda.synchronize()
            m.mlp_quantize_calls = calls
            self.captured[name] = CapturedRefresh(name, body, outputs, bumps, graph, pool)
        self.unchecked = set(self.captured)

    @torch.no_grad()
    def _self_check(self, captured: CapturedRefresh):
        """At a graph's first use: eager body and replay, from the same state, must agree bit for bit."""
        everything = {**self.attn_outputs, **self.mlp_outputs}
        before, calls = snapshot(everything), self.model.mlp_quantize_calls
        captured.body()
        eager = snapshot(captured.outputs)
        restore(everything, before)
        captured.graph.replay()  # the step's real refresh
        self.model.mlp_quantize_calls = calls + captured.counter_bumps
        bad = [n for n, t in captured.outputs.items() if not bits_equal(t, eager[n])]
        assert not bad, f"fp8 refresh graph {captured.name}: replay != eager on {bad}"

    def seal(self):
        assert set(self.captured) == {"attn", "mlp", "mlp+lm_head"}, f"fp8 refresh graphs missing: {sorted(self.captured)}"
        assert not self.unchecked, f"fp8 refresh graphs never self-checked: {sorted(self.unchecked)}"
        self.sealed = True

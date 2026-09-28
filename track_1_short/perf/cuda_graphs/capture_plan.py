"""Which warmup step captures which CUDA graph, decided before the warmup runs (bookkeeping only, no CUDA).

Every graph is captured during the untimed warmup and proven there by a self-check against the eager
code on the same buffers, so the timed run only replays:
  - step graphs (step_graphs.py), one pair per StepGraphKey: the first warmup visit of a key captures,
    the second runs its self-check. Every key the timed run reaches therefore needs two warmup visits,
    which plan_warmup adds (the first step showing each key, and the one after it).
  - optimizer tail graphs (optimizer_graphs.py): captured at the top of the second warmup step, self-checked
    in that step's optimizer.
  - fp8 refresh graphs (fp8_refresh_graphs.py): captured once the MLP's exact-scale bootstrap is over (the
    captured body is the delayed-scale path), after which each of the three graphs (attention; MLP
    without / with lm_head) self-checks at a later refresh. That needs later warmup steps of both
    parities: the refresh after an Adam (odd) step includes lm_head, the one after an even step does not.
The two tail captures run at the top of a warmup position, right after DeferredGathers.flush, so no
collective still reads a buffer their warm iterations touch.

The warmup visits its steps in ascending order, as the timed run does: the YaRN tables and attention
scales are rebuilt at each window change, so stepping backwards would leave them in a state the timed run
never has, and a step graph bakes the attention scale (StepGraphs re-checks it at every replay).

Provenance: record #360 (ANVIL2): `_cg_key_at`, `_cg_visits` / `_cg_thin`, `_CF_MAX`, the tgr / tgr2
capture positions. #360 visits a contiguous capture prefix 0..7 last; this plan captures inside the
ascending pass instead.
"""
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from track_1_short.config import BLOCK_SIZE, VIRTUAL_SEQ_CAP
from track_1_short.data import cu_seqlens_rows

# Untimed warmup steps allowed (record #360's _CF_MAX). The count is printed every run; a schedule
# edit that grows the warmup past this fails at startup instead of silently lengthening it.
WARMUP_STEP_BUDGET = 40


class StepGraphKey(NamedTuple):
    """Everything a captured training step freezes: the shapes, and the Python values the compiled
    model specializes on (a replay cannot re-check dynamo's guards, so these must match exactly)."""
    tokens: int            # tokens per rank (one microbatch per step)
    cu_seqlens_rows: int   # packed cu_seqlens table length
    mtp_targets: int       # multi-token-prediction weights
    candidates: int        # sampled-softmax candidate count P, 0 = full softmax
    ws_short: int          # attention windows, in tokens
    ws_long: int
    max_seq_len: int       # longest attention segment


def step_graph_key(schedule, candidate_counts: list[int], step: int, world_size: int) -> StepGraphKey:
    """The key of training step `step`, from the schedule alone (one microbatch per step)."""
    stage, _ = schedule.lookup(step)
    tokens = stage.batch_size // world_size
    return StepGraphKey(
        tokens=tokens,
        cu_seqlens_rows=cu_seqlens_rows(tokens),
        mtp_targets=len(stage.mtp_weights_start),
        candidates=candidate_counts[step],
        ws_short=stage.window_sizes[0] * BLOCK_SIZE,
        ws_long=stage.window_sizes[1] * BLOCK_SIZE,
        max_seq_len=min(stage.train_max_seq_len, VIRTUAL_SEQ_CAP),
    )


@dataclass(frozen=True)
class CapturePlan:
    steps: list[int]                # the warmup steps, ascending
    visits: dict[StepGraphKey, int]  # warmup visits per step-graph key the timed run reaches
    optimizer_capture_at: int       # position in `steps` whose top (after the flush) captures the optimizer graphs
    fp8_capture_at: int             # ... and the fp8 refresh graphs


def plan_warmup(base_steps: list[int], step_keys: list[StepGraphKey], is_adam_step,
                fp8_exact_scale_calls: int) -> CapturePlan:
    """Extend the compile warmup's steps so every graph gets its capture and its self-check.

    step_keys[s] is step s's key for every timed training step s. Assumes main()'s warmup: the MLP fp8
    caches are quantized once at build time and once in the flush at the top of every warmup position
    p >= 1 (the refresh the previous position's optimizer owes), so after that flush the MLP has been
    quantized p + 1 times.
    """
    first_step = {}
    for step, key in enumerate(step_keys):
        first_step.setdefault(key, step)
    steps = sorted(set(base_steps) | {s + offset for s in first_step.values() for offset in (0, 1)
                                      if s + offset < len(step_keys)})
    assert len(steps) <= WARMUP_STEP_BUDGET, f"{len(steps)} warmup steps > budget {WARMUP_STEP_BUDGET}: {steps}"
    visits = Counter(step_keys[s] for s in steps)
    # One visit captures, the next self-checks; a key with fewer would be captured on the clock or
    # replayed unchecked.
    thin = {key: visits[key] for key in first_step if visits[key] < 2}
    assert not thin, f"step-graph keys with < 2 warmup visits: {thin}"

    # The optimizer graphs are captured at the top of position 1 (position 0's eager step compiled the
    # body) and self-checked in position 1's optimizer.
    assert len(steps) >= 2
    # The fp8 capture at the top of position p needs the bootstrap over (p + 1 >= fp8_exact_scale_calls
    # quantizes) and, after it, refreshes of both refresh_lm values: those run at the tops of positions
    # p+1 .. n-1, owed by steps[p .. n-2].
    fp8_at = next((p for p in range(fp8_exact_scale_calls - 1, len(steps) - 1)
                   if {is_adam_step(s) for s in steps[p:-1]} == {False, True}), None)
    assert fp8_at is not None, f"no warmup position leaves both refresh parities for the fp8 self-checks: {steps}"
    return CapturePlan(steps=steps, visits=dict(visits), optimizer_capture_at=1, fp8_capture_at=fp8_at)

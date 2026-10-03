"""Tail averaging (record #360): the run ends on a blend of late-training weight averages
instead of the raw final iterate.

Four accumulators track the last few hundred steps. Each rank averages only the shard of each
weight it owns in the optimizer (1/8 of the weights), in fp32. At the end of training, before the
final validation, each accumulator is "shipped": written back into its weights, then every shipped
weight is all-gathered and rescaled to its pre-ship Frobenius norm ("decontraction": averaging
shrinks norms, and the model was trained at the un-averaged scale).

  accumulator         weights              ticks                          ship
  tail-ema            lm_head, embed       Adam steps, last 298           lerp(final, ema, 0.65)
  tail-avg            vo_bank, mlp_bank    every 4th step, last 250       replaced by the average
  value-embed-avg     value_embeds         its update events, last 250    replaced by the average
  bank-blend          qk/vo/mlp banks      every 4th step, last 298       lerp(current, ema, B)

bank-blend ships last, so on vo/mlp it blends into the tail average. Its B is 0.55, lowered by a
Richardson correction on vo/mlp: a seeded EMA lags the final iterate, and
(1-B)*A + B*X + w*(A - X) == lerp(A, X, B - w).
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from track_1_short.config import STEP_SCALE, scaled_steps
from track_1_short.ngram_table import NGRAM_ADAM_PERIOD4_START
from track_1_short.perf.kernels.lerp import lerp_upcast_

TAIL_EMA_WINDOW = scaled_steps(298)
TAIL_EMA_BLEND = 0.65

TAIL_AVG_WINDOW = scaled_steps(250)
TAIL_AVG_PERIOD = 4
TAIL_AVG_RATE = 4.0 / 53.0 / STEP_SCALE  # record #360's tuned lerp rate per tick (it is not derived from the window)

VALUE_EMBED_AVG_WINDOW = scaled_steps(250)
# value_embeds updates every 4th step throughout the window (TailAverages asserts it), so this is the
# tail-avg rate at the same tick spacing (record #360).
VALUE_EMBED_AVG_RATE = TAIL_AVG_RATE

BANK_BLEND_WINDOW = scaled_steps(298)
BANK_BLEND_PERIOD = 4  # ticks every 4th step (record #360)
# The per-step EMA timescale of the window, at 1/BANK_BLEND_PERIOD of the ticks.
BANK_BLEND_RATE = 2.0 / (BANK_BLEND_WINDOW // BANK_BLEND_PERIOD + 1)
BANK_BLEND = 0.55
RICHARDSON_W = 0.2007

DECONTRACTION_LABELS = ("embed", "lm_head", "mlp_bank", "qk_bank", "vo_bank")
# one EMA per group (time constant in steps, ticked when it changes), shipped as lerp(final, EMA, 0.5)
TAIL_GROUPS = {("qk_bank",): 128, ("vo_bank",): 128, ("mlp_bank",): 64, ("lm_head", "embed"): 128}


@dataclass
class TailAccumulator:
    """One fp32 accumulator over a set of weights, and how it is written back at the end."""
    tag: str
    labels: tuple[str, ...]
    window: int
    # rate(step, total_steps) -> lerp rate for this step's tick, or None for no tick
    rate: Callable[[int, int], float | None]
    # blend[label]: None replaces the weight by the accumulator, b lerps toward it by b
    blend: dict[str, float | None]
    bufs: dict[str, torch.Tensor] = field(default_factory=dict)
    seeded: set[str] = field(default_factory=set)


def tail_ema_rate(step: int, total_steps: int) -> float | None:
    """EMA over the last TAIL_EMA_WINDOW steps, ticked only on odd (Adam) steps.

    Two per-step EMA updates fold into one tick at the compounded rate 1 - (1 - r)^2. The window
    starts on an even step, which seeds; the final step ticks at r itself.
    """
    r = 2.0 / (TAIL_EMA_WINDOW + 1)
    if step % 2 == 0:
        return r if step == total_steps - TAIL_EMA_WINDOW else None
    return r if step == total_steps - 1 else -math.expm1(2.0 * math.log1p(-r))


def group_ema_rate(tau, period):
    return lambda s, n: (-math.expm1(-period / tau) if s >= period else 1.0) if s % period == period - 1 else None


class TailAverages:
    """All four accumulators, ticked after every optimizer step and shipped before the final eval."""

    def __init__(self, optimizer, rank: int, value_embed_updates: Callable[[int], bool], total_steps: int):
        """value_embed_updates(step): whether step's optimizer step wrote value_embeds (its update events).
        Ticking on any other step would re-sample a value the accumulator already holds."""
        # The value-embed average's rate assumes the period-4 update cadence over its whole window.
        assert total_steps - VALUE_EMBED_AVG_WINDOW >= NGRAM_ADAM_PERIOD4_START
        self.optimizer = optimizer
        self.rank = rank
        self.total_steps = total_steps
        self.accumulators = [TailAccumulator("ema", labels, math.ceil(3 * tau), group_ema_rate(tau, 2 if "embed" in labels else 1),
                                             dict.fromkeys(labels, 0.5)) for labels, tau in TAIL_GROUPS.items()] or [
            TailAccumulator("tail-ema", ("lm_head", "embed"), TAIL_EMA_WINDOW, tail_ema_rate,
                            {"lm_head": TAIL_EMA_BLEND, "embed": TAIL_EMA_BLEND}),
            TailAccumulator("tail-avg", ("vo_bank", "mlp_bank"), TAIL_AVG_WINDOW,
                            lambda s, n: TAIL_AVG_RATE if (n - 1 - s) % TAIL_AVG_PERIOD == 0 else None,
                            {"vo_bank": None, "mlp_bank": None}),
            TailAccumulator("bank-blend", ("qk_bank", "vo_bank", "mlp_bank"), BANK_BLEND_WINDOW,
                            # every BANK_BLEND_PERIOD-th step, plus the final two so the last iterate is always in
                            lambda s, n: BANK_BLEND_RATE if s % BANK_BLEND_PERIOD == 1 or s >= n - 2 else None,
                            {"qk_bank": BANK_BLEND, "vo_bank": BANK_BLEND - RICHARDSON_W,
                             "mlp_bank": BANK_BLEND - RICHARDSON_W}),
            TailAccumulator("value-embed-avg", ("value_embeds",), VALUE_EMBED_AVG_WINDOW,
                            lambda s, n: VALUE_EMBED_AVG_RATE if value_embed_updates(s) else None,
                            {"value_embeds": None}),
        ]
        # Allocate every buffer now: allocating on the first tick would land on the clock.
        for acc in self.accumulators:
            for label in acc.labels:
                acc.bufs[label] = torch.empty_like(self._own_shard(label), dtype=torch.float32)
        # Compile the lerp kernel now: its first launch would otherwise be the first tick, on the clock
        # (record #360 also compiles it before the clock). Triton specializes on whether the element
        # count divides by 16, so one scratch launch per class the buffers fall in covers every tick.
        bufs = [buf for acc in self.accumulators for buf in acc.bufs.values()]
        for divisible in {buf.numel() % 16 == 0 for buf in bufs}:
            scratch = torch.zeros(65536 if divisible else 65537, dtype=torch.float32, device=bufs[0].device)
            lerp_upcast_(scratch, scratch.bfloat16(), 0.5)

    def _param(self, label: str):
        return self.optimizer._param_by_label[label]

    def _own_shard(self, label: str) -> torch.Tensor:
        """The slice of the weight this rank updates in the optimizer."""
        p = self._param(label)
        cfg = self.optimizer.param_cfgs[p]
        base = p.data.view(cfg.reshape) if cfg.optim == "anvil" else p.data
        return base[self.rank * cfg.chunk_size:(self.rank + 1) * cfg.chunk_size]

    @torch.no_grad()
    def tick(self, step: int):
        """Fold this step's weights into every accumulator whose window and cadence include it."""
        for acc in self.accumulators:
            if step < self.total_steps - acc.window:
                continue
            rate = acc.rate(step, self.total_steps)
            if rate is None:
                continue
            for label in acc.labels:
                shard = self._own_shard(label)
                if label not in acc.seeded:
                    acc.bufs[label].copy_(shard)
                    acc.seeded.add(label)
                else:
                    lerp_upcast_(acc.bufs[label], shard, rate)

    @torch.no_grad()
    def ship(self):
        """Write the accumulators into the weights, gather the shards, and restore the norms."""
        norms_before = torch.stack([self._param(l).data.float().norm() for l in DECONTRACTION_LABELS])
        shipped = []
        for acc in self.accumulators_in_ship_order():
            for label in acc.labels:
                assert label in acc.seeded, f"[tail] {acc.tag} never ticked {label}"
                shard = self._own_shard(label)
                b = acc.blend[label]
                avg = acc.bufs[label]
                shard.copy_(avg.to(shard.dtype) if b is None else torch.lerp(shard.float(), avg, b).to(shard.dtype))
                if label not in shipped:
                    shipped.append(label)
        # Each rank wrote only its own shard: one gather per shipped weight reassembles it.
        for label in shipped + (["value_embeds"] if TAIL_GROUPS else []):
            p = self._param(label)
            cfg = self.optimizer.param_cfgs[p]
            full = p.data.view(cfg.reshape) if cfg.optim == "anvil" else p.data
            dist.all_gather_into_tensor(full, self._own_shard(label))
        norms_after = torch.stack([self._param(l).data.float().norm() for l in DECONTRACTION_LABELS])
        # Python-float ratios, so the product rounds to bf16 once, on the store.
        ratios = torch.where(norms_after > 0, norms_before / norms_after, torch.ones_like(norms_after)).tolist()
        for label, ratio in zip(DECONTRACTION_LABELS, ratios):
            self._param(label).data.mul_(ratio)

    def accumulators_in_ship_order(self):
        """bank-blend last: on vo/mlp it lerps from the already-shipped tail average."""
        by_tag = {acc.tag: acc for acc in self.accumulators}
        return self.accumulators if TAIL_GROUPS else [by_tag["tail-ema"], by_tag["tail-avg"], by_tag["value-embed-avg"], by_tag["bank-blend"]]

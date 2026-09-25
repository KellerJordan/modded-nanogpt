"""Keep FA3 forward unchanged; choose shorter backward windows from this run.

Two training steps capture complete-document prefixes. On the training clock,
compare Q/K/V gradients against the reference and take the worst error over
both steps and all ranks. A head keeps the full window unless a cheaper one
passes. The original backward is restored for the finishing stage.
"""
import json
import os
import time

import torch
import torch.distributed as dist

from . import cuda, reference

module = reference.module
ENABLED = os.environ.get("AB_ENABLE", "1") == "1"
START, STOP, THRESHOLD = 592, 1107, 0.2
LIMIT = 8192
LAYERS = (0, 1, 2, 5, 8)
WINDOWS = (64, 128, 256)
STATE = {}
HISTORY = []


def mode_at(step):
    if not ENABLED or step < START or step >= STOP:
        return 0
    return 1 if step < START + 2 else 2


def initialize(device):
    if not ENABLED:
        return
    for layer in LAYERS:
        heads, dv = (3, 128) if layer in (0, 2, 5) else (6, 64)
        buffers = [torch.empty((LIMIT, heads, dim), device=device, dtype=torch.bfloat16)
                   for dim in (64, 64, dv, dv, dv)]
        buffers += [torch.empty((heads, LIMIT), device=device),
                    torch.empty(257, device=device, dtype=torch.int32)]
        STATE[layer] = dict(
            buffers=buffers,
            table=torch.full((heads,), -1, device=device, dtype=torch.int32),
            errors=torch.zeros((len(WINDOWS), 3, heads), device=device),
        )


def reset():
    """Discard warmup observations without changing captured tensor addresses."""
    HISTORY.clear()
    for state in STATE.values():
        state["table"].fill_(-1)
        state["errors"].zero_()
        state.pop("chosen", None)
        state.pop("worst_errors", None)


def choose_windows(errors, full_window, block_m):
    # Each 128-key tile visits this many query tiles on average. In the dv=64
    # kernel, 64 and 128 have the same cost: prefer the more accurate window.
    codes = (-1,) + WINDOWS
    costs = [(128 + (full_window if w < 0 else min(w, full_window)) + block_m - 1)
             // block_m for w in codes]
    chosen = []
    for head in range(len(errors[0])):
        error = [0.0] + [row[head] for row in errors]
        allowed = [i for i, value in enumerate(error) if value <= THRESHOLD]
        chosen.append(codes[min(allowed, key=lambda i: (costs[i], error[i]))])
    return chosen


@torch.library.custom_op("nanogpt_ab::snapshot_backward",
                         mutates_args=("sq", "sk", "sv", "so", "sg", "slse", "scu"))
def snapshot_backward(g: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor,
                      cu: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor,
                      sv: torch.Tensor, so: torch.Tensor, sg: torch.Tensor,
                      slse: torch.Tensor, scu: torch.Tensor, maxlen: int,
                      scale: float, window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.shape[0] >= LIMIT and cu.numel() <= scu.numel()
    for dst, src in zip((sq, sk, sv, so, sg), (q, k, v, o, g)):
        dst.copy_(src[:LIMIT])
    slse.copy_(lse[:, :LIMIT])
    scu.fill_(-1)
    scu[:cu.numel()].copy_(cu)
    return reference.backward(g, q, k, v, o, lse, cu, maxlen, scale, window)


@snapshot_backward.register_fake
def _(g, q, k, v, o, lse, cu, sq, sk, sv, so, sg, slse, scu, maxlen, scale, window):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class HeadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, table, sq, sk, sv, so, sg, slse, scu,
                maxlen, scale, window, observe):
        out, lse = reference.forward(q, k, v, cu, maxlen, scale, window)
        ctx.save_for_backward(q, k, v, out, lse, cu, table, sq, sk, sv, so, sg, slse, scu)
        ctx.args = (maxlen, scale, window, observe)
        return out

    @staticmethod
    def backward(ctx, g):
        q, k, v, o, lse, cu, table = ctx.saved_tensors[:7]
        maxlen, scale, window, observe = ctx.args
        if observe:
            gradients = snapshot_backward(g, q, k, v, o, lse, cu,
                                          *ctx.saved_tensors[7:], maxlen, scale, window)
        else:
            gradients = cuda.backward(g, q, k, v, o, lse, cu, table, maxlen, scale, window)
        return gradients + (None,) * 13


def attention(q, k, v, cu, maxlen, scale, window, mode, layer):
    if mode == 0 or layer not in LAYERS:
        return reference.FA.flash_attn_varlen_func(
            q, k, v, cu, cu, maxlen, maxlen, softmax_scale=scale,
            causal=True, window_size=(window, 0))
    state = STATE[layer]
    return HeadBackward.apply(q, k, v, cu, state["table"], *state["buffers"],
                              maxlen, scale, window, mode == 1)


@torch.no_grad()
def measure(layer, window, scale):
    state = STATE[layer]
    q, k, v, o, g, lse, cu = state["buffers"]
    offsets = torch.unique_consecutive(cu.cpu())
    offsets = offsets[(offsets >= 0) & (offsets <= LIMIT)]
    assert len(offsets) > 1 and offsets[0] == 0
    end = int(offsets[-1])  # Exclude the final document if it crosses LIMIT.
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    maxlen = max(lengths)
    q, k, v, o, g = [x[:end] for x in (q, k, v, o, g)]
    lse, cu = lse[:, :end].contiguous(), offsets.to(q.device)
    ref = [x.float() for x in reference.backward(g, q, k, v, o, lse, cu, maxlen, scale, window)]
    denom = torch.stack([x.square().sum((0, 2)) for x in ref]).clamp_min(1e-30)
    errors = []
    for width in WINDOWS:
        table = torch.full_like(state["table"], min(width, window))
        got = cuda.backward(g, q, k, v, o, lse, cu, table, maxlen, scale, window)
        numerator = torch.stack([(a.float() - b).square().sum((0, 2)) for a, b in zip(got, ref)])
        errors.append((numerator / denom).sqrt())
    error = torch.stack(errors)
    state["errors"].copy_(torch.maximum(state["errors"], error))
    return dict(layer=layer, full_window=window, document_lengths=lengths, error=error.cpu().tolist())


@torch.no_grad()
def after_backward(step, manager):
    if mode_at(step) != 1:
        return
    started = time.perf_counter()
    raw = getattr(manager.model, "_orig_mod", manager.model)
    window = manager.ws_short * manager.block_size
    observations = []
    for layer in LAYERS:
        yarn = raw.yarn_paired_head if layer in (0, 2, 5) else raw.yarn
        observations.append(measure(layer, window, yarn.attn_scale))
    if step == START + 1:
        for layer in LAYERS:
            state = STATE[layer]
            if dist.is_initialized():
                dist.all_reduce(state["errors"], op=dist.ReduceOp.MAX)
            errors = state["errors"].amax(1).cpu().tolist()
            chosen = choose_windows(errors, window, 64 if layer in (0, 2, 5) else 128)
            state["table"].copy_(torch.tensor(chosen, device=state["table"].device, dtype=torch.int32))
            state["chosen"], state["worst_errors"] = chosen, errors
    torch.cuda.synchronize()
    HISTORY.append(dict(step=step, seconds=time.perf_counter() - started, observations=observations))


def configuration():
    from . import head
    return dict(attention_enabled=ENABLED, calibration_steps=[START, START + 1],
                attention_stop=STOP, threshold=THRESHOLD, head_sample_group=head.GROUP,
                head_min_rows=head.MIN_ROWS, extension=cuda.NAME)


def report(training_ms, val_loss):
    result = dict(**configuration(), training_ms=training_ms, val_loss=float(val_loss),
                  calibration=HISTORY, windows={str(l): s.get("chosen") for l, s in STATE.items()},
                  worst_errors={str(l): s.get("worst_errors") for l, s in STATE.items()})
    return "[approx-backward-result] " + json.dumps(result, sort_keys=True)

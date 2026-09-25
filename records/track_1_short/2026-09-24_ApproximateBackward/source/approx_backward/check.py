"""Numerical and CUDA-graph checks; these are not training benchmark runs.

From the repo root: python -m approx_backward.check
Use torchrun --standalone --nproc_per_node=8 -m approx_backward.check to also
check rank-specific sampling and the all-rank calibration error reduction.
"""
import json
import os

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
from . import attention, cuda, head, reference
import triton_kernels as tk


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))


def selected_rows(counter, count):
    indices = []
    for group in range(count // head.GROUP):
        z = group ^ (((counter + 72821) * 0x9e3779b9) & 0xffffffff)
        z = ((z ^ (z >> 16)) * 0x85ebca6b) & 0xffffffff
        z = ((z ^ (z >> 13)) * 0xc2b2ae35) & 0xffffffff
        z ^= z >> 16
        indices.append(group * head.GROUP + (z & (head.GROUP - 1)))
    return torch.tensor(indices, device="cuda")


def check_head(n_predict):
    t, d, vocab = max(head.MIN_ROWS, 512), 384, 256
    x = torch.randn(t, d, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(d, vocab, device="cuda").to(torch.float8_e4m3fn)
    targets = torch.randint(vocab, (t,), device="cuda", dtype=torch.int64)
    prefix = targets.roll(4)
    prefix[::7] = -1
    prefix[1::9] = targets[1::9]  # Duplicate prefix/next-token corrections.
    g = torch.rand(t, vocab, device="cuda") * 0.01
    rows = torch.arange(t, device="cuda")
    for offset in range(n_predict):
        valid = rows + offset < t
        g[rows[valid], targets[rows[valid] + offset]] -= 0.08
    valid = prefix >= 0
    g[rows[valid], prefix[valid]] -= 0.05
    g[rows[::11], targets[::11]] = 0.02  # Positive net gradients at some targets.
    g = g.to(torch.float8_e5m2)
    tk._SNS_IDX["TPOS"], tk._SNS_IDX["PPOS"] = targets, prefix
    def backward():
        return tk._ce_backward_gemms(g, w, x, 1.0, 1.0, 1.0, n_predict)
    for _ in range(3):
        backward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dx, dw = backward()
    counter = int(head.COUNTER)
    selected = selected_rows(counter, t)
    expected = head.GROUP * x.float()[selected].T @ g.float().clamp_min(0)[selected]
    expected += x.float().T @ g.float().clamp_max(0)
    exact_dx, _ = tk._ce_backward_gemms(g, w, x, 1.0, 1.0, 1.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(head.COUNTER) == counter + head.SAMPLE_STRIDE
    assert torch.equal(dx, exact_dx)
    error = relative(dw, expected)
    assert error < 0.003, error
    return dict(n_predict=n_predict, dx_exact=True, sampled_dw_relative_error=error)


def check_attention(dv):
    t, heads, scale, window = 1024, 3 if dv == 128 else 6, 0.13, 384
    offsets = [0, 255, 255, 638, t]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    q, k = [torch.randn(t, heads, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v, g = [torch.randn(t, heads, dv, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    o, lse = reference.forward(q, k, v, cu, 512, scale, window)
    windows = torch.full((heads,), -1, device="cuda", dtype=torch.int32)
    def backward():
        return cuda.backward(g, q, k, v, o, lse, cu, windows, 512, scale, window)
    for _ in range(3):
        backward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = backward()
    stock = reference.backward(g, q, k, v, o, lse, cu, 512, scale, window)
    graph.replay()
    full_errors = [relative(a, b) for a, b in zip(outputs, stock)]
    assert max(full_errors) < 0.003, full_errors
    codes = ([64, 128, -1] * 2)[:heads]
    windows.copy_(torch.tensor(codes, device="cuda", dtype=torch.int32))
    graph.replay()
    # Independent dense formula using the ORIGINAL forward lse and output.
    # Masking the forward or re-normalizing the retained weights would be wrong.
    expected = [torch.zeros_like(x, dtype=torch.float32) for x in (q, k, v)]
    for begin, end in zip(offsets, offsets[1:]):
        if begin == end:
            continue
        pos = torch.arange(end - begin, device="cuda")
        for h, code in enumerate(codes):
            width = window if code < 0 else code
            qs, ks, vs, gs, os_ = [x[begin:end, h].float() for x in (q, k, v, g, o)]
            p = (scale * qs @ ks.T - lse[h, begin:end, None]).exp()
            mask = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] <= width)
            p = p.masked_fill(~mask, 0)
            ds = p * (gs @ vs.T - (gs * os_).sum(-1, keepdim=True))
            expected[0][begin:end, h] = scale * ds @ ks
            expected[1][begin:end, h] = scale * ds.T @ qs
            expected[2][begin:end, h] = p.T @ gs
    dense_errors = [relative(a, b) for a, b in zip(outputs, expected)]
    assert max(dense_errors) < 0.008, dense_errors
    return dict(value_dim=dv, full_relative_errors=full_errors, restricted_dense_errors=dense_errors)


def check_compiled_attention():
    """Exercise the actual autograd wrapper and snapshot path under AOTAutograd."""
    t, heads, scale, window = 8192, 3, 0.13, 384
    cu = torch.arange(0, t + 1, 1024, device="cuda", dtype=torch.int32)
    q, k = [torch.randn(t, heads, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)]
    v = torch.randn(t, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(v)
    o, lse = reference.forward(q.detach(), k.detach(), v.detach(), cu, 1024, scale, window)
    records = []
    for mode in (1, 2):
        def f(q, k, v):
            return attention.attention(q, k, v, cu, 1024, scale, window, mode, 0)
        compiled = torch.compile(f, dynamic=False, fullgraph=True)
        q.grad = k.grad = v.grad = None
        out = compiled(q, k, v)
        out.backward(g)
        assert torch.equal(out, o)
        expected = reference.backward(g, q, k, v, o, lse, cu, 1024, scale, window)
        errors = [relative(x.grad, ref) for x, ref in zip((q, k, v), expected)]
        assert max(errors) < 0.003, errors
        if mode == 1:
            assert torch.equal(attention.STATE[0]["buffers"][0], q)
            assert torch.equal(attention.STATE[0]["buffers"][4], g)
        records.append(dict(mode=mode, forward_exact=True, full_gradient_errors=errors))
    return records


torch.set_num_threads(1)
torch.manual_seed(901 + int(os.environ.get("RANK", "0")))
head.init("cuda")
assert head.GROUP == 4
phase_steps = [591, 592, 593, 594, 1106, 1107, 1193]
assert [attention.mode_at(s) for s in phase_steps] == [0, 1, 1, 2, 2, 0, 0]
results = dict(head=[check_head(n) for n in (1, 2, 3)], attention=[check_attention(d) for d in (64, 128)])
attention.initialize("cuda")
results["compiled_attention"] = check_compiled_attention()
if int(os.environ.get("WORLD_SIZE", "1")) > 1 and os.environ.get("CHECK_DISTRIBUTED", "1") != "0":
    dist.init_process_group("nccl")
    for state in attention.STATE.values():
        state["errors"].fill_(dist.get_rank() + 1)
        dist.all_reduce(state["errors"], op=dist.ReduceOp.MAX)
        assert torch.all(state["errors"] == dist.get_world_size())
    results["distributed_error_max"] = True
attention.reset()
assert all(torch.all(s["errors"] == 0) and torch.all(s["table"] == -1) for s in attention.STATE.values())
head.reset()
assert int(head.COUNTER) == head.SAMPLE_RANK
print(json.dumps(dict(passed=True, device=torch.cuda.get_device_name(), rank=head.SAMPLE_RANK,
                     stride=head.SAMPLE_STRIDE, extension=cuda.NAME, results=results)), flush=True)
if dist.is_initialized():
    dist.destroy_process_group()

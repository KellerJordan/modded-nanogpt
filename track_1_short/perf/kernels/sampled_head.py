"""Sampled LM-head weight gradient (the idea of PR #371 / #382), in the backward of the sampled CPLM cross-entropy.

What it replaces: the dense dW = x^T @ g GEMM of the lm_head (n_rows x 768 x P) on the sampled-softmax steps with at
least MIN_ROWS rows. Why it is faster: the GEMM runs over n_rows / GROUP rows instead of n_rows.

The logit gradient splits as g = max(g, 0) + min(g, 0). The positive part (the softmax mass, dense over all P candidate
columns) is sampled: one row out of every GROUP, picked by a hash of a device-side counter, with weight GROUP. The
negative part lives only on the target columns (the MTP targets and the prefix target) and is added exactly. The <copy>
column of the CPLM mixture can carry either sign on any row, so it is left out of the sample and added exactly. The
resulting dW is unbiased; dX stays exact (dense GEMM).

Only SampledCPLMCrossEntropy.backward calls this, and only for at least MIN_ROWS rows (as in #382): the full-softmax
stages keep the dense backward. COUNTER is a device tensor the captured graphs bake: main() creates it once (init), and
the address census (perf/cuda_graphs/baked_addresses.py) checks that it survives the post-warmup reset.
"""
import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.transpose import transpose_copy

GROUP = 4             # one sampled row per GROUP rows, weight GROUP
MIN_ROWS = 32768      # smaller backwards stay dense
SAMPLE_STRIDE = 8     # the counter advances by this per call; ranks start at rank % SAMPLE_STRIDE
SAMPLE_RANK = 0
COUNTER = None
assert GROUP >= 1 and GROUP & (GROUP - 1) == 0, "GROUP must be a power of two"


def init(device, rank=0):
    """One counter per rank (different samples on different ranks); it advances by SAMPLE_STRIDE on every call,
    graph replays included."""
    global COUNTER, SAMPLE_RANK
    assert COUNTER is None
    SAMPLE_RANK = rank % SAMPLE_STRIDE
    COUNTER = torch.full((), SAMPLE_RANK, device=device, dtype=torch.int64)


@triton.jit
def _selected_row(group, counter, K: tl.constexpr):
    z = group.to(tl.uint32) ^ ((counter.to(tl.uint32) + 72821) * 0x9e3779b9)
    z = (z ^ (z >> 16)) * 0x85ebca6b
    z = (z ^ (z >> 13)) * 0xc2b2ae35
    z = z ^ (z >> 16)
    return group * K + (z & (K - 1)).to(tl.int32)


@triton.jit
def _sample_transpose(A, O, COUNTER, CC, T, D, K: tl.constexpr, POSITIVE: tl.constexpr,
                      BG: tl.constexpr = 64, BD: tl.constexpr = 128):
    """O[d, j] = A[row_j, d] for group j; with POSITIVE, max(., 0) and the CC (<copy>) column set to zero."""
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    row = _selected_row(group, tl.load(COUNTER), K).to(tl.int64)
    a = tl.load(A + row[:, None] * D + d[None, :], (row[:, None] < T) & (d[None, :] < D), 0.0).to(tl.float32)
    if POSITIVE:
        a = tl.maximum(a, 0.)
        a = tl.where(d[None, :] == tl.load(CC), 0., a)
    tl.store(O + d[None, :].to(tl.int64) * (T // K) + group[:, None], a,
             (group[:, None] < T // K) & (d[None, :] < D))


@triton.jit
def _negative_add(X, G, TPOS, PPOS, CC, O, XS, GS, T, V, D, N: tl.constexpr,
                  BT: tl.constexpr = 32, BD: tl.constexpr = 128):
    """O[idx, d] += x[row, d] * min(g[row, idx], 0) over the target columns (MTP + prefix), each once, CC excluded."""
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    cc = tl.load(CC)
    x = tl.load(X + row[:, None].to(tl.int64) * D + d[None, :], (row[:, None] < T) & (d[None, :] < D),
                0.0).to(tl.float32) * (tl.load(XS) * tl.load(GS))
    for q in tl.static_range(N + 1):
        if q < N:
            idx = tl.load(TPOS + row + q, row + q < T, -1)
        else:
            idx = tl.load(PPOS + row, row < T, -1)
        valid = (row < T) & (idx >= 0) & (idx < V) & (idx != cc)
        for earlier in tl.static_range(q):
            previous = tl.load(TPOS + row + earlier, row + earlier < T, -1)
            valid = valid & (idx != previous)
        val = tl.minimum(tl.load(G + row.to(tl.int64) * V + idx, valid, 0.0).to(tl.float32), 0.)
        tl.atomic_add(O + idx[:, None] * D + d[None, :], x * val[:, None],
                      valid[:, None] & (d[None, :] < D) & (val[:, None] != 0), sem='relaxed')


@triton.jit
def _copy_add(X, G, CC, O, XS, GS, T, V, D, BT: tl.constexpr = 128, BD: tl.constexpr = 128):
    """O[cc, d] += sum_row x[row, d] * g[row, cc]: the whole <copy> column, reduced in the block, one atomic per d."""
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    cc = tl.load(CC).to(tl.int64)
    x = tl.load(X + row[:, None].to(tl.int64) * D + d[None, :], (row[:, None] < T) & (d[None, :] < D),
                0.0).to(tl.float32)
    g = tl.load(G + row.to(tl.int64) * V + cc, row < T, 0.0).to(tl.float32)
    part = tl.sum(x * g[:, None], axis=0) * (tl.load(XS) * tl.load(GS))
    tl.atomic_add(O + cc * D + d, part, d < D, sem='relaxed')


@triton.jit
def _advance(COUNTER, STRIDE: tl.constexpr):
    tl.store(COUNTER, tl.load(COUNTER) + STRIDE)


@torch.library.custom_op("nanogpt::sampled_head_wg", mutates_args=("counter",))
def sampled_head_wg(g: torch.Tensor, x: torch.Tensor, tpos: torch.Tensor, ppos: torch.Tensor,
                    copy_col: torch.Tensor, xs: torch.Tensor, gs: torch.Tensor, counter: torch.Tensor,
                    n_predict: int, group: int) -> torch.Tensor:
    """dW [D, V] bf16 of x^T @ g: positive part sampled (weight group), negative part and <copy> exact."""
    t, v = g.shape
    d = x.shape[1]
    assert t % group == 0 and (t // group) % 16 == 0 and 1 <= n_predict <= 3
    gp = torch.empty((v, t // group), device=g.device, dtype=g.dtype)
    xp = torch.empty((d, t // group), device=x.device, dtype=x.dtype)
    _sample_transpose[(triton.cdiv(t // group, 64), triton.cdiv(v, 128))](g, gp, counter, copy_col, t, v, group, True)
    _sample_transpose[(triton.cdiv(t // group, 64), triton.cdiv(d, 128))](x, xp, counter, copy_col, t, d, group, False)
    out_t = torch._scaled_mm(gp, xp.T, scale_a=gs * group, scale_b=xs, out_dtype=torch.float32, use_fast_accum=False)
    _negative_add[(triton.cdiv(t, 32), triton.cdiv(d, 128))](x, g, tpos, ppos, copy_col, out_t, xs, gs, t, v, d,
                                                             n_predict)
    _copy_add[(triton.cdiv(t, 128), triton.cdiv(d, 128))](x, g, copy_col, out_t, xs, gs, t, v, d)
    out = torch.empty((d, v), device=x.device, dtype=torch.bfloat16)
    transpose_copy(out_t, out)
    _advance[(1,)](counter, SAMPLE_STRIDE)
    return out


@sampled_head_wg.register_fake
def _(g, x, tpos, ppos, copy_col, xs, gs, counter, n_predict, group):
    return x.new_empty((x.shape[1], g.shape[1]), dtype=torch.bfloat16)


def ce_backward(grad_input, w_t, x_f8, x_s, w_s, grad_s, n_predict, copy_col, target_pos, prefix_pos):
    """_ce_backward_gemms with the sampled dW: dX exact, dW from sampled_head_wg (layout [D, P]).

    target_pos / prefix_pos: each row's target and prefix-target position in the candidate set (the first n_rows
    entries of SampledLoss.target_pos / prefix_pos)."""
    x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
    w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
    grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)
    grad_x = torch._scaled_mm(grad_input, w_t.T, out_dtype=torch.bfloat16, scale_a=grad_scale, scale_b=w_scale,
                              use_fast_accum=False)
    grad_w = torch.ops.nanogpt.sampled_head_wg(
        grad_input, x_f8, target_pos, prefix_pos, copy_col, x_scale, grad_scale, COUNTER, n_predict, GROUP)
    return grad_x, grad_w

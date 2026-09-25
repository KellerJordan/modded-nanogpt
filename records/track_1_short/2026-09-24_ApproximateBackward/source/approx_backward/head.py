"""Sample positive head outer products; exact dX and negative corrections.

One row per K-token stratum is selected using a counter-driven hash, and its
positive contribution receives inverse-probability weight K. Unlike pooling,
this preserves feature/gradient pairing. Counter values advance on GPU replay.
"""
import os
import torch
import triton
import triton.language as tl
@triton.jit
def negative_add(X,G,TPOS,PPOS,O,XS,GS,T:tl.constexpr,V:tl.constexpr,D:tl.constexpr,N:tl.constexpr,
                 BT:tl.constexpr=32,BD:tl.constexpr=128):
    row=tl.program_id(0)*BT+tl.arange(0,BT)
    d=tl.program_id(1)*BD+tl.arange(0,BD)
    x=tl.load(X+row[:,None]*D+d[None,:],(row[:,None]<T)&(d[None,:]<D),0.0).to(tl.float32)*(tl.load(XS)*tl.load(GS))
    for q in tl.static_range(N+1):
        if q<N:idx=tl.load(TPOS+row+q,row+q<T,-1)
        else:idx=tl.load(PPOS+row,row<T,-1)
        valid=(row<T)&(idx>=0)&(idx<V)
        for earlier in tl.static_range(q):
            previous=tl.load(TPOS+row+earlier,row+earlier<T,-1)
            valid=valid&(idx!=previous)
        val=tl.minimum(tl.load(G+row*V+idx,valid,0.0).to(tl.float32),0.)
        tl.atomic_add(O+idx[:,None]*D+d[None,:],x*val[:,None],valid[:,None]&(d[None,:]<D)&(val[:,None]!=0),sem='relaxed')

MIN_ROWS = int(os.environ.get('HEAD_SAMPLE_MIN_ROWS', '32768'))
SAMPLE_RANK = int(os.environ.get('RANK', '0'))
SAMPLE_STRIDE = int(os.environ.get('WORLD_SIZE', '1'))
GROUP = int(os.environ.get('HEAD_SAMPLE_GROUP', '4'))
assert GROUP in (0, 4)
COUNTER = None


def init(device):
    global COUNTER
    COUNTER = torch.full((), SAMPLE_RANK, device=device, dtype=torch.int64)


def reset():
    if COUNTER is not None:
        COUNTER.fill_(SAMPLE_RANK)


@triton.jit
def selected_row(group, counter, K: tl.constexpr):
    z = group.to(tl.uint32) ^ ((counter.to(tl.uint32) + 72821) * 0x9e3779b9)
    z = (z ^ (z >> 16)) * 0x85ebca6b
    z = (z ^ (z >> 13)) * 0xc2b2ae35
    z = z ^ (z >> 16)
    return group * K + (z & (K - 1)).to(tl.int32)


@triton.jit
def sample_transpose(A, O, COUNTER, T: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
                     POSITIVE: tl.constexpr, BG: tl.constexpr = 64, BD: tl.constexpr = 128):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    row = selected_row(group, tl.load(COUNTER), K)
    a = tl.load(A + row[:, None] * D + d[None, :],
                (row[:, None] < T) & (d[None, :] < D), 0.0).to(tl.float32)
    if POSITIVE:
        a = tl.maximum(a, 0.)
    tl.store(O + d[None, :] * (T // K) + group[:, None], a,
             (group[:, None] < T // K) & (d[None, :] < D))


@triton.jit
def advance(COUNTER, STRIDE: tl.constexpr):
    tl.store(COUNTER, tl.load(COUNTER) + STRIDE)


@torch.library.custom_op('nanogpt::sampled_head_wg', mutates_args=('counter',))
def sampled_head_wg(g: torch.Tensor, x: torch.Tensor, tpos: torch.Tensor,
                    ppos: torch.Tensor, xs: torch.Tensor, gs: torch.Tensor,
                    counter: torch.Tensor, n_predict: int, group: int) -> torch.Tensor:
    from triton_kernels import transpose_copy
    t, v = g.shape
    d = x.shape[1]
    assert t % group == 0 and 1 <= n_predict <= 3
    gp = torch.empty((v, t // group), device=g.device, dtype=g.dtype)
    xp = torch.empty((d, t // group), device=x.device, dtype=x.dtype)
    sample_transpose[(triton.cdiv(t // group, 64), triton.cdiv(v, 128))](g, gp, counter, t, v, group, True)
    sample_transpose[(triton.cdiv(t // group, 64), triton.cdiv(d, 128))](x, xp, counter, t, d, group, False)
    out_t = torch._scaled_mm(gp, xp.T, scale_a=gs * group, scale_b=xs,
                             out_dtype=torch.float32, use_fast_accum=False)
    negative_add[(triton.cdiv(t, 32), triton.cdiv(d, 128))](x, g, tpos, ppos, out_t, xs, gs, t, v, d, n_predict)
    out = torch.empty((d, v), device=x.device, dtype=torch.bfloat16)
    transpose_copy(out_t, out)
    advance[(1,)](counter, SAMPLE_STRIDE)
    return out


@sampled_head_wg.register_fake
def _(g, x, tpos, ppos, xs, gs, counter, n_predict, group):
    return x.new_empty((x.shape[1], g.shape[1]), dtype=torch.bfloat16)

"""Triton kernels for the engram host table: the row update and the row gather."""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_update(grad_ptr, vals_ptr, accum_ptr, uniq_ptr, out_ptr,
                lr, decay, beta, seed,
                S: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    site = tl.arange(0, S)
    col = tl.arange(0, D)
    offs = row * (S * D) + site[:, None] * D + col[None, :]           # (S, D) tile of this row
    g = tl.load(grad_ptr + offs).to(tl.float32)
    v = tl.load(vals_ptr + offs).to(tl.float32)
    table_row = tl.load(uniq_ptr + row)
    a_ptr = accum_ptr + table_row * S + site
    a = tl.load(a_ptr)
    sq = tl.sum(g * g, axis=1) / D
    a = beta * a + (1.0 - beta) * sq
    tl.store(a_ptr, a)
    scale = lr / (tl.sqrt(a) + 1e-8)
    x = v * (1.0 - decay) - scale[:, None] * g
    noise = (tl.randint(seed, offs) & 0xFFFF).to(tl.int32)
    r = (x.to(tl.int32, bitcast=True) + noise) & (-65536)
    tl.store(out_ptr + offs, r.to(tl.float32, bitcast=True).to(tl.bfloat16))


def row_update(grad, vals, accum, uniq, lr, n_site, per_slot, decay, beta, seed, out):
    """New bf16 rows into `out` (U, n_site*per_slot); `accum` is updated in place at `uniq`."""
    U = vals.shape[0]
    assert grad.shape == vals.shape == out.shape and grad.is_contiguous() and vals.is_contiguous() and out.is_contiguous()
    assert uniq.is_contiguous() and accum.is_contiguous() and accum.shape[1] == n_site
    if U:
        _row_update[(U,)](grad, vals, accum, uniq, out, float(lr), float(decay), float(beta), int(seed) & 0x7FFFFFFF,
                          S=n_site, D=per_slot, num_warps=2)
    return out


# ---- row gather: leaf (U, S*Dp), inv (G*T,) slot-major -> out (T, S, G*Dp); atomic fp32 backward ----

@triton.jit
def _gather_fwd(leaf_ptr, inv_ptr, out_ptr, T, S: tl.constexpr, G: tl.constexpr, Dp: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    mt = t < T
    d = tl.arange(0, Dp)
    for g in tl.static_range(G):
        row = tl.load(inv_ptr + g * T + t, mask=mt, other=0)
        for s in tl.static_range(S):
            src = leaf_ptr + row[:, None] * (S * Dp) + s * Dp + d[None, :]
            val = tl.load(src, mask=mt[:, None], other=0.0)
            dst = out_ptr + t[:, None] * (S * G * Dp) + s * (G * Dp) + g * Dp + d[None, :]
            tl.store(dst, val, mask=mt[:, None])


@triton.jit
def _gather_bwd(gout_ptr, inv_ptr, acc_ptr, T, S: tl.constexpr, G: tl.constexpr, Dp: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    mt = t < T
    d = tl.arange(0, Dp)
    for g in tl.static_range(G):
        row = tl.load(inv_ptr + g * T + t, mask=mt, other=0)
        for s in tl.static_range(S):
            src = gout_ptr + t[:, None] * (S * G * Dp) + s * (G * Dp) + g * Dp + d[None, :]
            val = tl.load(src, mask=mt[:, None], other=0.0).to(tl.float32)
            dst = acc_ptr + row[:, None] * (S * Dp) + s * Dp + d[None, :]
            tl.atomic_add(dst, val, mask=mt[:, None])


class RowGatherTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, leaf, inv, T, S, G, Dp):
        ctx.save_for_backward(inv)
        ctx.dims = (leaf.shape[0], T, S, G, Dp)
        out = torch.empty(T, S, G * Dp, dtype=leaf.dtype, device=leaf.device)
        BT = 64
        _gather_fwd[(triton.cdiv(T, BT),)](leaf, inv, out, T, S=S, G=G, Dp=Dp, BT=BT, num_warps=4)
        return out

    @staticmethod
    def backward(ctx, gout):
        (inv,) = ctx.saved_tensors
        U, T, S, G, Dp = ctx.dims
        acc = torch.zeros(U, S * Dp, dtype=torch.float32, device=gout.device)
        BT = 64
        _gather_bwd[(triton.cdiv(T, BT),)](gout.contiguous(), inv, acc, T, S=S, G=G, Dp=Dp, BT=BT, num_warps=4)
        return acc.to(gout.dtype), None, None, None, None, None


def row_gather(leaf, inv, T, S, G, Dp):
    """(T, S, G*Dp) row slices per site from the (U, S*Dp) leaf; differentiable in leaf."""
    assert leaf.is_contiguous() and inv.numel() == G * T
    return RowGatherTriton.apply(leaf, inv, T, S, G, Dp)

"""The n-gram embedding lookup (see ngram_table.py) as one inductor-opaque op, forward and backward.

What it replaces: the plain torch spelling in GPT.forward,

    rows = cache[slots] + sink                                 # [2T, D]; sink is exact zeros
    x0_bigram = rows[:T] * sign_pool[bigram_sign(x)] + rows[T:] * sign_pool[trigram_sign(x)]

Why it is faster: aten.index is in AOTAutograd's recomputable set, so the partitioner inlines the
gathers into every consumer of x0_bigram (~21 of them) instead of materialising it once; record #360
measured that at 3.4 ms/step. A custom op is opaque to that pass: x0_bigram is written once, by one
kernel that also computes both sign hashes in registers. The backward is one kernel that writes the
sink's gradient (d x0_bigram times each channel's sign row), which is exactly the gradient of the
looked-up rows; the cache and the table never get one.

Invariants: the sign pool has a power-of-two row count (the hash's `& (rows - 1)` is the non-negative
remainder), and the sign rows are exactly +-1, so the products are exact and the only rounding is the
final bf16 store -- the same single rounding the torch spelling's bf16 add does. `sink` is never read:
it exists only as the autograd slot the row gradient lands on.

Provenance: record #360 (ANVIL2): bigram_kernels.py `bgx_x0_bigram`.
"""
import torch
import triton
import triton.language as tl

from track_1_short.ngram_table import BIGRAM_SIGN_MULS, TRIGRAM_SIGN_MULS

# Inductor's own `triton_poi_*` geometry at this tensor size (record #360).
XBLOCK, NUM_WARPS = 1024, 4


@triton.jit
def _sign_rows(inp_ptr, x1, xmask, RMASK: tl.constexpr,
               B0: tl.constexpr, B1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr):
    """Sign-pool rows of token t = x1 for both channels (ngram_table.py's sign hashes)."""
    cur = tl.load(inp_ptr + x1, xmask, eviction_policy="evict_last")
    pm1 = tl.load(inp_ptr + (x1 - 1), xmask & (x1 >= 1), other=0, eviction_policy="evict_last")
    pm2 = tl.load(inp_ptr + (x1 - 2), xmask & (x1 >= 2), other=0, eviction_policy="evict_last")
    bigram = tl.where(x1 >= 1, ((B1 * pm1) ^ (B0 * cur)) & RMASK, 0)
    trigram = tl.where(x1 >= 2, ((T2 * pm2) ^ (T1 * pm1) ^ (T0 * cur)) & RMASK, 0)
    return bigram, trigram


@triton.jit
def _ngram_embed_fwd_kernel(cache_ptr, slots_ptr, pool_ptr, inp_ptr, out_ptr, T, xnumel,
                            D: tl.constexpr, RMASK: tl.constexpr, XBLOCK: tl.constexpr,
                            B0: tl.constexpr, B1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex % D   # channel
    x1 = xindex // D  # token
    # Row offsets in int64: the cache may exceed 2**31 elements.
    bigram_slot = tl.load(slots_ptr + x1, xmask, eviction_policy="evict_last").to(tl.int64)
    trigram_slot = tl.load(slots_ptr + (T + x1), xmask, eviction_policy="evict_last").to(tl.int64)
    bigram_sign, trigram_sign = _sign_rows(inp_ptr, x1, xmask, RMASK, B0, B1, T0, T1, T2)
    a = tl.load(cache_ptr + bigram_slot * D + x0, xmask).to(tl.float32)
    b = tl.load(cache_ptr + trigram_slot * D + x0, xmask).to(tl.float32)
    sa = tl.load(pool_ptr + bigram_sign * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    sb = tl.load(pool_ptr + trigram_sign * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    tl.store(out_ptr + xindex, (a * sa + b * sb).to(tl.bfloat16), xmask)


@triton.jit
def _ngram_embed_bwd_kernel(g_ptr, pool_ptr, inp_ptr, out_ptr, xnumel,
                            D: tl.constexpr, RMASK: tl.constexpr, XBLOCK: tl.constexpr,
                            B0: tl.constexpr, B1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex % D
    x1 = xindex // D
    bigram_sign, trigram_sign = _sign_rows(inp_ptr, x1, xmask, RMASK, B0, B1, T0, T1, T2)
    g = tl.load(g_ptr + xindex, xmask).to(tl.float32)
    sa = tl.load(pool_ptr + bigram_sign * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    sb = tl.load(pool_ptr + trigram_sign * D + x0, xmask, eviction_policy="evict_last").to(tl.float32)
    tl.store(out_ptr + xindex, (g * sa).to(tl.bfloat16), xmask)             # bigram rows [0, T)
    tl.store(out_ptr + (xnumel + xindex), (g * sb).to(tl.bfloat16), xmask)  # trigram rows [T, 2T)


def _launch_shape(pool: torch.Tensor, inp: torch.Tensor, d: int, out_rows_per_token: int):
    """Output buffer, element count, grid and hash mask shared by both ops."""
    rows = pool.shape[0]
    assert pool.dtype == torch.bfloat16 and pool.is_contiguous() and pool.shape[1] == d
    assert rows & (rows - 1) == 0, "the sign pool needs a power-of-two row count"
    assert inp.dtype == torch.int32 and inp.is_contiguous() and inp.ndim == 1
    n = inp.shape[0] * d
    out = torch.empty((out_rows_per_token * inp.shape[0], d), device=inp.device, dtype=torch.bfloat16)
    return out, n, (triton.cdiv(n, XBLOCK),), rows - 1


_HASH_CONSTS = dict(B0=BIGRAM_SIGN_MULS[0], B1=BIGRAM_SIGN_MULS[1],
                    T0=TRIGRAM_SIGN_MULS[0], T1=TRIGRAM_SIGN_MULS[1], T2=TRIGRAM_SIGN_MULS[2])


@torch.library.custom_op("nanogpt::ngram_embed_fwd", mutates_args=())
def ngram_embed_fwd(cache: torch.Tensor, slots: torch.Tensor, pool: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """x0_bigram [T, D] = cache[slots[:T]] * pool[bigram_sign] + cache[slots[T:]] * pool[trigram_sign]."""
    d = cache.shape[1]
    assert cache.dtype == torch.bfloat16 and cache.is_contiguous()
    assert slots.dtype == torch.int32 and slots.is_contiguous() and slots.shape == (2 * inp.shape[0],)
    out, n, grid, rmask = _launch_shape(pool, inp, d, 1)
    _ngram_embed_fwd_kernel[grid](cache, slots, pool, inp, out, inp.shape[0], n, D=d, RMASK=rmask,
                                  XBLOCK=XBLOCK, num_warps=NUM_WARPS, num_stages=1, **_HASH_CONSTS)
    return out


@ngram_embed_fwd.register_fake
def _(cache, slots, pool, inp):
    return _launch_shape(pool, inp, cache.shape[1], 1)[0]


@torch.library.custom_op("nanogpt::ngram_embed_bwd", mutates_args=())
def ngram_embed_bwd(g: torch.Tensor, pool: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """d(sink) [2T, D] from d(x0_bigram) [T, D]: each channel's rows get g times their sign row."""
    d = g.shape[-1]
    g = g.contiguous().reshape(-1, d)
    assert g.shape[0] == inp.shape[0] and g.dtype == torch.bfloat16
    out, n, grid, rmask = _launch_shape(pool, inp, d, 2)
    _ngram_embed_bwd_kernel[grid](g, pool, inp, out, n, D=d, RMASK=rmask,
                                  XBLOCK=XBLOCK, num_warps=NUM_WARPS, num_stages=1, **_HASH_CONSTS)
    return out


@ngram_embed_bwd.register_fake
def _(g, pool, inp):
    return _launch_shape(pool, inp, g.shape[-1], 2)[0]


class _NgramEmbed(torch.autograd.Function):
    """The whole gradient goes to `sink` and nothing to `cache`: no table-sized gradient is ever built."""
    @staticmethod
    def forward(ctx, cache, slots, pool, inp, sink):
        ctx.save_for_backward(pool, inp)
        return torch.ops.nanogpt.ngram_embed_fwd(cache, slots, pool, inp)

    @staticmethod
    def backward(ctx, g):
        pool, inp = ctx.saved_tensors
        return None, None, None, None, torch.ops.nanogpt.ngram_embed_bwd(g, pool, inp)


def ngram_embedding(cache, slots, pool, inp, sink):
    """[T, D] x0_bigram. `sink` ([2T, D] leaf, training) receives the looked-up rows' gradient; None in eval."""
    if sink is None:
        return torch.ops.nanogpt.ngram_embed_fwd(cache, slots, pool, inp)
    return _NgramEmbed.apply(cache, slots, pool, inp, sink)

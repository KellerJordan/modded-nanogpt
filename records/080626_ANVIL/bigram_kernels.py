# bigram_kernels.py — post-step weight-cache refresh rewrite (M90Q) + bigram
# embedding backward rewrite (M90B). Standalone module imported by the trainer;
# both mechanisms are baked on/off by the trainer's configuration constants.
#
# M90Q — exact-scale fused dual-layout weight-cache refresh.
#   Replaces the eager quantize_mlp_fp8 tensor chain (abs temp + amax per proj,
#   double bf16 divides, contiguous fp8 temps, non-coalesced transposing copy_)
#   with ONE amax pass over the bank + ONE fused quantize kernel that emits
#   every destination layout (up row, up col[C1D], down col[C1A], down row[C1B])
#   directly. The arithmetic chain is replicated BITWISE vs the aten path:
#     scale: bf16 abs-amax -> clamp(min=1e-12) -> /448.0, all bf16 aten ops on a
#            [2L] vector (identical kernels to the eager path; the triton amax
#            pass only produces the fp32-exact bf16 max magnitudes).
#     quant: fp32(w) div.rn fp32(s) -> RN bf16 -> RN e4m3.  aten computes
#            bf16/bf16 at opmath fp32 with IEEE-RN divide then RN-casts to bf16,
#            then c10 casts bf16->e4m3 through fp32 (RTNE; overflow->NaN only at
#            >=480 which is unreachable here since |w|/s <= 448*(1+2^-8) < 480,
#            so triton's satfinite cast is bit-identical).
#   NOT the same thing as KX_A2P: scales are EXACT and refreshed every call from
#   the current bank (no lag, no headroom, no bootstrap, no epilogue partial-amax
#   variants) — bit-identical class.
#
# M90B — bigram embedding backward as bf16 index_add_.
#   Stock aten embedding_dense_backward zero-fills a full-table fp32 buffer,
#   scatter-adds in fp32 and casts down to bf16. The table is 377280x768 so
#   that is >1.7 GB of avoidable traffic per (odd) step. This replaces it with
#   a dynamo-traceable autograd.Function whose backward is a bf16 zeros +
#   index_add_ (atomic bf16 adds). Gradient support and amax are identical;
#   per-element values may differ by ~1 bf16 ulp (accumulation order/precision)
#   — NOT bit-identical class; gate with seeded loss run.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================== M90Q kernels

_AMAX_BLOCK = 4096
_AMAX_ITERS = 4


@triton.jit
def _m90_amax_kernel(w_ptr, pamax_ptr,
                     ROW_N: tl.constexpr, NT: tl.constexpr,
                     DOWN_ON: tl.constexpr,
                     BLOCK: tl.constexpr, ITERS: tl.constexpr):
    """Per-(layer,proj) partial abs-max over the flat bank rows.

    grid = (NROW, NT). Row pid maps to flat bank row rf (= pid when both projs
    are refreshed, = 2*pid when only the up-proj is). Slot written at rf so the
    trainer-side slicing is layout-stable across flag sets.
    """
    row = tl.program_id(0)
    t = tl.program_id(1)
    if DOWN_ON:
        rf = row
    else:
        rf = 2 * row
    base = rf * ROW_N
    m = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(ITERS):
        offs = t * (BLOCK * ITERS) + i * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(w_ptr + base + offs, mask=offs < ROW_N, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(v))
    tl.store(pamax_ptr + rf * NT + t, tl.max(m, axis=0))


@triton.jit
def _m90_quant_kernel(w_ptr, s_ptr,
                      up_row_ptr, up_colt_ptr, dn_row_ptr, dn_colt_ptr,
                      H: tl.constexpr, D: tl.constexpr, NTD: tl.constexpr,
                      DOWN_ON: tl.constexpr, UP_COL: tl.constexpr,
                      DN_ROW: tl.constexpr,
                      BH: tl.constexpr, BD: tl.constexpr):
    """ONE fused scale->bf16->e4m3 quantize emitting all destination layouts.

    grid = (NROW, cdiv(H,BH)*NTD). s_ptr is the [2L] bf16 exact scale vector
    (same bits the eager chain divides by). Row-major stores go to the
    contiguous caches; transposed stores go to the [L, D, H]-contiguous views
    of the col-major caches (kernel-module idiom `.transpose(1,2)` of the alloc).
    """
    r = tl.program_id(0)
    tile = tl.program_id(1)
    if DOWN_ON:
        rf = r
        layer = r // 2
        proj = r % 2
    else:
        rf = 2 * r
        layer = r
        proj = 0
    th = tile // NTD
    td = tile % NTD
    offs_h = th * BH + tl.arange(0, BH)
    offs_d = td * BD + tl.arange(0, BD)
    mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)
    mask_t = (offs_d[:, None] < D) & (offs_h[None, :] < H)
    v = tl.load(w_ptr + rf * (H * D) + offs_h[:, None] * D + offs_d[None, :],
                mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + rf).to(tl.float32)
    # aten bf16/bf16 divide == fp32 IEEE-RN divide + RN downcast to bf16
    q_bf = tl.math.div_rn(v, s).to(tl.bfloat16)
    q = q_bf.to(tl.float32).to(tl.float8e4nv)  # c10 bf16->e4m3 goes via fp32
    lbase = layer * (H * D)
    if proj == 0:
        tl.store(up_row_ptr + lbase + offs_h[:, None] * D + offs_d[None, :],
                 q, mask=mask)
        if UP_COL:
            tl.store(up_colt_ptr + lbase + offs_d[:, None] * H + offs_h[None, :],
                     tl.trans(q), mask=mask_t)
    else:
        tl.store(dn_colt_ptr + lbase + offs_d[:, None] * H + offs_h[None, :],
                 tl.trans(q), mask=mask_t)
        if DN_ROW:
            tl.store(dn_row_ptr + lbase + offs_h[:, None] * D + offs_d[None, :],
                     q, mask=mask)


def m90q_nt_amax(H: int = 3072, D: int = 768) -> int:
    """Partial-amax slot count per bank row the trainer must allocate."""
    return triton.cdiv(H * D, _AMAX_BLOCK * _AMAX_ITERS)


def m90q_refresh(bank: torch.Tensor,
                 s_bf: torch.Tensor, pamax: torch.Tensor,
                 up_scales: torch.Tensor, up_row: torch.Tensor,
                 down_scales: torch.Tensor | None = None,
                 up_colt: torch.Tensor | None = None,
                 dn_row: torch.Tensor | None = None,
                 dn_colt: torch.Tensor | None = None):
    """Exact-scale fused refresh of the FP8 MLP weight caches.

    bank      : [L, 2, H, D] bf16 contiguous (mlp_bank).
    s_bf      : [2L] bf16 scratch (exact per-row scales, kernel input).
    pamax     : [2L, m90q_nt_amax(H, D)] fp32 scratch.
    up_scales : [L] fp32 (trainer's _mlp_up_proj_scales) — refreshed here.
    up_row    : [L, H, D] e4m3 contiguous (_mlp_up_proj_f8).
    down_scales/dn_colt : [L] fp32 / [L, D, H] e4m3 contiguous view — pass the
                .transpose(1, 2) of _mlp_down_f8 when KX_C1A.
    up_colt   : [L, D, H] e4m3 contiguous view (_mlp_up_f8_col.transpose(1,2))
                when KX_C1D.
    dn_row    : [L, H, D] e4m3 contiguous (_mlp_down_f8_row) when KX_C1B.
    Bitwise-identical outputs to the eager quantize_mlp_fp8 chain.
    """
    L, P, H, D = bank.shape
    assert P == 2 and bank.is_contiguous()
    down_on = dn_colt is not None
    nrow = 2 * L if down_on else L
    NT = m90q_nt_amax(H, D)
    assert pamax.shape == (2 * L, NT) and s_bf.shape == (2 * L,)
    # ---- ONE amax pass over the rows we refresh
    _m90_amax_kernel[(nrow, NT)](
        bank, pamax, ROW_N=H * D, NT=NT, DOWN_ON=down_on,
        BLOCK=_AMAX_BLOCK, ITERS=_AMAX_ITERS, num_warps=8, num_stages=2,
    )
    # ---- exact scales, replicating the eager op chain bit-for-bit:
    # amax values are abs(bf16) hence fp32-exact; .to(bf16) is exact; clamp and
    # the /448.0 divide are the same aten bf16 kernels the eager path runs.
    pa = pamax.amax(dim=1)                         # [2L] fp32, exact bf16 values
    torch.div(pa.to(torch.bfloat16).clamp_(min=1e-12), 448.0, out=s_bf)
    up_scales.copy_(s_bf[0::2].float())
    if down_on and down_scales is not None:
        down_scales.copy_(s_bf[1::2].float())
    # ---- ONE fused quantize emitting every layout
    NTD = triton.cdiv(D, 64)
    ntiles = triton.cdiv(H, 64) * NTD
    _m90_quant_kernel[(nrow, ntiles)](
        bank, s_bf, up_row,
        up_colt if up_colt is not None else bank,
        dn_row if dn_row is not None else bank,
        dn_colt if dn_colt is not None else bank,
        H=H, D=D, NTD=NTD,
        DOWN_ON=down_on, UP_COL=up_colt is not None, DN_ROW=dn_row is not None,
        BH=64, BD=64, num_warps=4, num_stages=2,
    )


# ================================================================ M90B


class _M90BigramEmbed(torch.autograd.Function):
    """Embedding with bf16 index_add_ backward (no fp32 full-table zero-fill).

    Dynamo traces autograd.Functions, so this stays inside the fullgraph
    compile; inductor sees plain zeros + index_add_.
    """

    @staticmethod
    def forward(ctx, weight, idx):
        ctx.save_for_backward(idx)
        ctx.num_rows = weight.shape[0]
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        gw = torch.zeros((ctx.num_rows, g2.shape[-1]), dtype=g2.dtype,
                         device=g2.device)
        gw.index_add_(0, idx.reshape(-1).long(), g2)
        return gw, None


def m90b_bigram_embed(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return _M90BigramEmbed.apply(weight, idx)


# ================================================================ harness

def _eager_ship_chain(bank, up_row, up_col, dn_row, dn_col, up_scales, dn_scales):
    """Verbatim replica of the trainer's quantize_mlp_fp8 stock refresh at ship
    flags (C1A=C1B=C1D=1). Copied expression-for-expression."""
    E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    L = bank.shape[0]
    flat = bank[:, 0].view(L, -1)
    scales = flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
    up_scales[:] = scales.float()
    up_row[:] = (bank[:, 0] / scales.view(L, 1, 1)).to(torch.float8_e4m3fn)
    up_col.copy_((bank[:, 0] / scales.view(L, 1, 1)).to(torch.float8_e4m3fn))
    w2 = bank[:, 1]
    s2 = w2.reshape(L, -1).abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
    dn_scales[:] = s2.float()
    dn_col.copy_((w2 / s2.view(L, 1, 1)).to(torch.float8_e4m3fn))
    dn_row.copy_((w2 / s2.view(L, 1, 1)).to(torch.float8_e4m3fn))


def _alloc_caches(L, H, D, dev):
    up_row = torch.zeros(L, H, D, device=dev, dtype=torch.float8_e4m3fn)
    up_col = torch.zeros(L, H, D, device=dev, dtype=torch.float8_e4m3fn
                         ).transpose(1, 2).contiguous().transpose(1, 2)
    dn_row = torch.zeros(L, H, D, device=dev, dtype=torch.float8_e4m3fn)
    dn_col = torch.zeros(L, H, D, device=dev, dtype=torch.float8_e4m3fn
                         ).transpose(1, 2).contiguous().transpose(1, 2)
    up_s = torch.ones(L, dtype=torch.float32, device=dev)
    dn_s = torch.ones(L, dtype=torch.float32, device=dev)
    return up_row, up_col, dn_row, dn_col, up_s, dn_s


def _bitwise_eq(a, b, name):
    ok = bool(torch.eq(a.view(torch.uint8) if a.is_contiguous() else
                       a.contiguous().view(torch.uint8),
                       b.view(torch.uint8) if b.is_contiguous() else
                       b.contiguous().view(torch.uint8)).all())
    n = 0 if ok else int((a.contiguous().view(torch.uint8)
                          != b.contiguous().view(torch.uint8)).sum())
    print(f"  [{ 'PASS' if ok else 'FAIL' }] {name} bitwise"
          + ("" if ok else f"  ({n} byte mismatches)"))
    return ok


def _test_m90q(dev):
    print("== M90Q bitwise gate (frontier shape 12x2x3072x768) ==")
    torch.manual_seed(7)
    L, H, D = 12, 3072, 768
    ok_all = True
    for trial in range(4):
        bank = (torch.randn(L, 2, H, D, device=dev, dtype=torch.float32)
                * (10.0 ** torch.linspace(-3, 1, 2 * L, device=dev)
                   ).view(L, 2, 1, 1)).bfloat16()
        if trial == 1:
            # adversarial: plant exact-amax entries + near-max ratios so the
            # bf16-rounded scale exercises the >448 saturation region
            bank[:, :, 0, 0] = bank.abs().amax() * 1.001
            bank[:, :, 1, 1] = -bank[:, :, 0, 0]
        if trial == 2:
            bank[3, 0].zero_()  # clamp(min=1e-12) path
        if trial == 3:
            bank = (torch.rand(L, 2, H, D, device=dev) * 2 - 1).bfloat16() * 448
        r = _alloc_caches(L, H, D, dev)
        e = _alloc_caches(L, H, D, dev)
        _eager_ship_chain(bank, *e)
        s_bf = torch.zeros(2 * L, dtype=torch.bfloat16, device=dev)
        pamax = torch.zeros(2 * L, m90q_nt_amax(H, D), dtype=torch.float32, device=dev)
        m90q_refresh(bank, s_bf, pamax, r[4], r[0], down_scales=r[5],
                     up_colt=r[1].transpose(1, 2), dn_row=r[2],
                     dn_colt=r[3].transpose(1, 2))
        torch.cuda.synchronize()
        print(f" trial {trial}:")
        for i, nm in ((0, "up_row"), (2, "dn_row")):
            ok_all &= _bitwise_eq(r[i], e[i], nm)
        ok_all &= _bitwise_eq(r[1].transpose(1, 2).contiguous(),
                              e[1].transpose(1, 2).contiguous(), "up_col")
        ok_all &= _bitwise_eq(r[3].transpose(1, 2).contiguous(),
                              e[3].transpose(1, 2).contiguous(), "dn_col")
        ok_all &= _bitwise_eq(r[4].view(torch.uint8), e[4].view(torch.uint8), "up_scales")
        ok_all &= _bitwise_eq(r[5].view(torch.uint8), e[5].view(torch.uint8), "dn_scales")
    print(f"M90Q bitwise: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def _bench_m90q(dev, iters=200):
    print("== M90Q bench (eager ship chain vs fused) ==")
    torch.manual_seed(11)
    L, H, D = 12, 3072, 768
    bank = torch.randn(L, 2, H, D, device=dev).bfloat16() * 0.02
    e = _alloc_caches(L, H, D, dev)
    r = _alloc_caches(L, H, D, dev)
    s_bf = torch.zeros(2 * L, dtype=torch.bfloat16, device=dev)
    pamax = torch.zeros(2 * L, m90q_nt_amax(H, D), dtype=torch.float32, device=dev)

    def run_eager():
        _eager_ship_chain(bank, *e)

    def run_m90():
        m90q_refresh(bank, s_bf, pamax, r[4], r[0], down_scales=r[5],
                     up_colt=r[1].transpose(1, 2), dn_row=r[2],
                     dn_colt=r[3].transpose(1, 2))

    out = {}
    for name, fn in (("eager", run_eager), ("m90q", run_m90)):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        out[name] = t0.elapsed_time(t1) / iters
        print(f"  {name}: {out[name]*1000:.1f} us/call")
    print(f"  saving: {(out['eager']-out['m90q'])*1000:.1f} us/step"
          f"  (x1300 steps = {(out['eager']-out['m90q'])*1.3:.3f} s)")


def _test_m90b(dev):
    print("== M90B gate (377280x768 table, 8192-token batch) ==")
    torch.manual_seed(13)
    V, Dm, S = 377280, 768, 8192
    w = (torch.randn(V, Dm, device=dev) * 0.02).bfloat16()
    # realistic duplicate-heavy indices
    idx = torch.randint(0, V, (S,), device=dev, dtype=torch.int32)
    idx[: S // 4] = idx[S // 2: S // 2 + S // 4]  # force duplicates
    g = torch.randn(S, Dm, device=dev).bfloat16() * 0.1

    wa = w.clone().requires_grad_(True)
    F.embedding(idx, wa).backward(g)
    wb = w.clone().requires_grad_(True)
    m90b_bigram_embed(wb, idx).backward(g)
    ga, gb = wa.grad, wb.grad

    same_support = bool(((ga != 0) == (gb != 0)).all())
    exact = float((ga == gb).float().mean())
    ia = ga.view(torch.int16).int()
    ib = gb.view(torch.int16).int()
    sign_ok = bool(((ga >= 0) == (gb >= 0)).all())
    ulp = (ia - ib).abs()
    max_ulp = int(ulp.max())
    print(f"  support identical: {same_support}; exact-match frac: {exact:.6f}; "
          f"signs identical: {sign_ok}; max bf16 ulp diff: {max_ulp}")
    print(f"  grad amax: stock {ga.abs().amax().item():.6f} "
          f"new {gb.abs().amax().item():.6f}")
    ok = same_support and max_ulp <= 1
    # compile smoke: must trace under fullgraph
    def fn(w_, i_, g_):
        out = m90b_bigram_embed(w_, i_)
        return (out * g_).sum()
    cfn = torch.compile(fn, fullgraph=True)
    wc = w.clone().requires_grad_(True)
    cfn(wc, idx, g).backward()
    print(f"  fullgraph compile: OK (grad amax {wc.grad.abs().amax().item():.6f})")

    # bench: aten reference op vs bf16 chain
    def run_stock():
        torch.ops.aten.embedding_dense_backward(g, idx.long(), V, -1, False)

    def run_new():
        gw = torch.zeros(V, Dm, dtype=g.dtype, device=dev)
        gw.index_add_(0, idx.long(), g)

    for name, fn2 in (("stock_bwd", run_stock), ("m90b_bwd", run_new)):
        for _ in range(20):
            fn2()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(100):
            fn2()
        t1.record()
        torch.cuda.synchronize()
        print(f"  {name}: {t0.elapsed_time(t1) / 100 * 1000:.1f} us/call")
    print(f"M90B: {'PASS' if ok else 'FAIL (ulp bound exceeded)'}")
    return ok


if __name__ == "__main__":
    import sys
    dev = torch.device("cuda", 0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    ok = True
    if mode in ("test", "q"):
        ok &= _test_m90q(dev)
        _bench_m90q(dev)
    if mode in ("test", "b"):
        ok &= _test_m90b(dev)
    print("M90K HARNESS:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)

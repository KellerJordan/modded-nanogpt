"""Tiny-kernel fusions for AnvilAndAdam's optimizer TAIL -- the step plus the fp8 requantize, everything after
the main compiled graph.  Three launch-overhead removals, none of which changes any arithmetic: FLAT, FUSE and
SNSCP, each with its contract at its entry point below and its rationale in the record README.
"""

import torch

FT_BLOCK, FT_RING = 1024, 8   # fused-Adam kernel tile; pinned scalar-table ring depth
_SC_COLS = 8      # beta1, 1-beta1, beta2, 1-beta2, eps, step_size, eff_wd, active

# The fused kernel below reproduces _afe_run's statement sequence element for element, all in fp32 (bf16
# upcast exactly).  ATen's add_ / addcmul_ contract to FFMA under nvcc's default -fmad=true, which the
# tl.math.fma spellings mirror; ATen sqrt and tl.sqrt are both sqrt.rn.f32, both `/` are div.rn.f32, and the
# bf16 store is RNE.

_KERNEL = None

def _build_kernel():
    """Import triton and JIT-define the kernel lazily: importing this module is free."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl
    assert hasattr(tl, "sqrt") and hasattr(tl.math, "fma"), \
        "fused replicated Adam needs tl.sqrt (IEEE sqrt.rn.f32) and tl.math.fma to match ATen's " \
        "rounding and nvcc's -fmad=true contraction; do not substitute approximate spellings."

    @triton.jit
    def _afe_fused_adam(P, G, EA, ES, SEG, SC, n_elements,
                        BLOCK: tl.constexpr):
        """The whole replicated-Adam step for every param at once: SEG maps each element to its
        segment, SC that segment's eight scalars. Masking every load and store on `act` is the
        host half's `if tup is None: continue` -- that segment is skipped, moments included."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_elements
        seg = tl.load(SEG + offs, mask=m, other=0)
        base = seg * 8
        b1 = tl.load(SC + base + 0, mask=m, other=0.0); a1 = tl.load(SC + base + 1, mask=m, other=0.0)
        b2 = tl.load(SC + base + 2, mask=m, other=0.0); a2 = tl.load(SC + base + 3, mask=m, other=0.0)
        eps = tl.load(SC + base + 4, mask=m, other=0.0); ss = tl.load(SC + base + 5, mask=m, other=0.0)
        wd = tl.load(SC + base + 6, mask=m, other=0.0); act = tl.load(SC + base + 7, mask=m, other=0.0)
        m2 = m & (act != 0.0)
        g = tl.load(G + offs, mask=m2, other=0.0).to(tl.float32)
        ea = tl.load(EA + offs, mask=m2, other=0.0).to(tl.float32)
        es = tl.load(ES + offs, mask=m2, other=0.0).to(tl.float32)
        p = tl.load(P + offs, mask=m2, other=0.0).to(tl.float32)
        ea = ea * b1; ea = tl.math.fma(a1, g, ea)
        es = es * b2; t = a2 * g; es = tl.math.fma(t, g, es)
        dn = tl.sqrt(es); dn = dn + eps
        u = ea / dn; u = u * ss
        mk = (u * p) > 0.0; mkf = tl.where(mk, 1.0, 0.0); t2 = wd * p
        u = tl.math.fma(t2, mkf, u)
        pn = p - u          # p.add_(update, alpha=-1.0): alpha is exactly -1, no contraction
        tl.store(EA + offs, ea, mask=m2); tl.store(ES + offs, es, mask=m2)
        tl.store(P + offs, pn.to(P.dtype.element_ty), mask=m2)
    _KERNEL = (triton, _afe_fused_adam)
    return _KERNEL

def afe_flatten(opt):
    """FLAT: re-point the replicated-Adam params and moments at opt._far_flats / opt._far_views.  Pass 1
    validates, pass 2 mutates, so nothing is rebound unless every assert passed; the gradient views must tile
    each flat buffer exactly, as a gap would leave stray elements picking up segment 0's scalars.
    Bit-identical (bitwise copies, then views of the same shape, dtype and stride).  LIFETIME: nothing may
    REPLACE a param or moment afterwards -- hence load_state_dict copying in place, reset() leaving Adam state
    alone, and _BAKED publishing these views to the census."""

    far, flats, views = opt._far_params, opt._far_flats, opt._far_views
    assert far and set(far) == set(opt._afe_params), "[ft] FLAT needs the replicated Adam set"
    layout, end = [], dict.fromkeys(flats, 0)
    for p in far:
        v, st, n = views[p], opt.param_states[p], p.numel()
        ea, es, off = st["exp_avg"], st["exp_avg_sq"], v.storage_offset()
        assert v.dtype is p.dtype and v.numel() == n
        assert off == end[p.dtype], f"[ft] {p.dtype} flat buffer: gap/overlap at element {off}"
        assert p.data.is_contiguous() and ea.is_contiguous() and es.is_contiguous()
        assert ea.dtype is torch.float32 and es.dtype is torch.float32
        end[p.dtype] = off + n
        layout.append((p, st, off, n))
    for dt, g in flats.items():
        assert end[dt] == g.numel(), \
            f"[ft] {dt} flat buffer: {end[dt]} of {g.numel()} elements covered"

    # One quadruple per dtype over the gradient buffer's own index space, so the kernel addresses
    # P/G/EA/ES/SEG with one offset.  Segment order follows _afe_groups, identical on every rank.
    seg_of = {p: i for i, p in enumerate(q for _k, g in opt._afe_groups for q in g)}
    dev = far[0].device
    ft = {dt: dict(g=g, n=g.numel(), seg=torch.zeros(g.numel(), dtype=torch.int32),
                   p=torch.zeros(g.numel(), dtype=dt, device=dev),
                   ea=torch.zeros(g.numel(), dtype=torch.float32, device=dev),
                   es=torch.zeros(g.numel(), dtype=torch.float32, device=dev))
          for dt, g in flats.items()}
    for p, st, off, n in layout:
        b, sl = ft[p.dtype], slice(off, off + n)
        for k, src in (("p", p.data), ("ea", st["exp_avg"]), ("es", st["exp_avg_sq"])):
            b[k][sl].copy_(src.reshape(-1))
        b["seg"][sl] = seg_of[p]
        p.data, st["exp_avg"], st["exp_avg_sq"] = (b[k][sl].view(p.shape) for k in ("p", "ea", "es"))
    for b in ft.values():
        b["seg"] = b["seg"].to(dev)
    # Pinned ring for the per-step scalar table, one event per slot: WAR against the run-ahead host.
    opt._ft_flat, opt._ft_seg, opt._ft_sc_ring = ft, seg_of, 0
    opt._ft_sc_host = torch.zeros(FT_RING, len(seg_of), _SC_COLS, dtype=torch.float32,
                                  pin_memory=True)
    opt._ft_sc_np = opt._ft_sc_host.numpy()
    opt._ft_sc_dev = torch.zeros(len(seg_of), _SC_COLS, dtype=torch.float32, device=dev)
    opt._ft_sc_evt = [torch.cuda.Event() for _ in range(FT_RING)]
    _build_kernel()
    return True

def afe_run_fused(opt):
    """FUSE: one kernel launch per dtype in place of AnvilAndAdam._afe_run.  The host half is _afe_run's
    verbatim -- same waits, same step bump, same step_size / eff_wd arithmetic (Python doubles rounded to fp32
    once at the store, as `Scalar.to<float>()` does); that verbatim-ness is the correctness argument.  ONE
    partition, not two: _afe_groups' key and the per-step key compose into one 8-tuple, legal because
    _afe_groups is a dict's items() with distinct keys."""

    triton, kern = _build_kernel()
    ring = opt._ft_sc_ring
    opt._ft_sc_evt[ring].synchronize()          # WAR: slot's previous H2D has executed
    sc = opt._ft_sc_np[ring]
    sc[:, :] = 0.0
    parts = {}

    for key, group in opt._afe_groups:
        for p in group:
            tup = opt._reduce_futures.get(p)
            if tup is None:
                continue                        # no grad/reduce this step: stock path skips it too
            if tup[0] is not None:
                tup[0].wait()                   # stream-wait; idempotent per shared future
            st, cfg = opt.param_states[p], opt.param_cfgs[p]
            st["step"] += 1
            parts.setdefault(key + (st["step"], cfg.lr, cfg.weight_decay), []).append(p)
    for (betas, lr_mul, wd_mul, eps, _dt, t, lr_base, wd_base), items in parts.items():
        beta1, beta2 = betas
        lr = lr_base * lr_mul
        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        row = (beta1, 1 - beta1, beta2, 1 - beta2, eps,
               lr * (bias2 ** 0.5 / bias1), lr * lr * wd_base * wd_mul, 1.0)
        for p in items:
            sc[opt._ft_seg[p]] = row            # one fp32 rounding per scalar, at the store

    opt._ft_sc_ring = (ring + 1) % FT_RING
    if not parts:
        return
    opt._ft_sc_dev.copy_(opt._ft_sc_host[ring], non_blocking=True)
    opt._ft_sc_evt[ring].record()
    for b in opt._ft_flat.values():
        n = b["n"]
        kern[(triton.cdiv(n, FT_BLOCK),)](b["p"], b["g"], b["ea"], b["es"], b["seg"],
                                          opt._ft_sc_dev, n, BLOCK=FT_BLOCK, num_warps=4)

# Ordering contract for the SNS candidate H2Ds on their private copy stream: step N-1's step_optimizers() stamps the
# compute stream with sns_war_record() AFTER every reader of the _SNS_IDX buffers is enqueued, step N's
# sns_upload_async() waits on that stamp before copying (WAR safe), and TrainingManager.sns_gather waits on
# _sns_copy_evt before the first reader (RAW safe).  The buffers are permanently-live globals: no record_stream.
_sns_stream = _sns_copy_evt = _sns_war_evt = None

def sns_war_record():
    """Stamp the compute stream after the last reader of the SNS buffers; the first call builds the stream and events."""
    global _sns_stream, _sns_copy_evt, _sns_war_evt
    if _sns_stream is None:
        _sns_stream = torch.cuda.Stream()
        _sns_copy_evt, _sns_war_evt = torch.cuda.Event(), torch.cuda.Event()
    _sns_war_evt.record()

def sns_upload_async(dst_src_pairs):
    """Issue the H2Ds on the private copy stream, behind the WAR stamp the previous step left on the compute stream."""
    _sns_stream.wait_event(_sns_war_evt)
    with torch.cuda.stream(_sns_stream):
        for dst, src in dst_src_pairs:
            dst.copy_(src, non_blocking=True)
    _sns_copy_evt.record(_sns_stream)

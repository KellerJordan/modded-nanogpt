"""G3 for the MoE: the grouped "up" GEMM over the experts (grouped_lrs_kernel in moe_mlp.py) in Gluon, warp-specialized.

A port of the dense G3 (mlp_g3.py: _g3_*), same scheme: 2 consumer warpgroups (wgmma 64x256 each) + 1 loader warp
that prefetches A, B and, in the backward, the `aux` tile (post, e4m3) with TMA; epilogue in registers inside the consumer,
async TMA stores half a tile at a time, fp8 transpose through a 16-bit stmatrix thanks to the M permutation (4D descriptors
on A, aux and the row output).  See the long comment in mlp_g3.py for the permutation trick.

Differences from the dense version:
  * every 128-row tile belongs to ONE expert, tile_expert[pid_m]: the loader reads the B rows at `expert * N + off_n`
    (B = the stacked weights [E*N, K]); the next tile's expert is prefetched;
  * the dequantization factor is per tile: dq = dequant[expert] * dq2[0], read by the consumer before the mainloop;
  * backward: the epilogue is written in PTX in groups of 4 (bf16x2, fp8x2 satfinite, packed amax in registers and a single
    atomic_max at the end of the CTA, as in the forward): -25 % epilogue time compared with a direct transcription of the
    dense one.
prime_g3() compares the outputs (rows, transposes, amax) bit for bit with moe_mlp._grouped_lrs_triton at start-up.

Usage: grouped_lrs_g3(...) has the same contract as moe_mlp._grouped_lrs_impl; g3_eligible(...) says whether shapes and
dtypes qualify (M, N multiples of 128/256, K a multiple of 64 and >= 128, Hopper); prime_g3() compiles the variants in eager
mode and returns None, or the reason why G3 must not be used.
"""
import torch

# Guarded imports, as in mlp_g3.py: without Gluon _HAVE_GLUON stays False and grouped_lrs_g3 is unavailable.
try:
    from triton.experimental import gluon as _gluon
    from triton.experimental.gluon import language as _gl
    from triton.experimental.gluon.language.nvidia.hopper import (
        tma as _tma, mbarrier as _mbarrier, fence_async_shared as _fence, warpgroup_mma as _wgmma,
        warpgroup_mma_wait as _wgmma_wait)
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as _GTD
    _HAVE_GLUON = True
    _GLUON_IMPORT_ERROR = None
except Exception as _gluon_err:
    _HAVE_GLUON = False
    _GLUON_IMPORT_ERROR = f"{type(_gluon_err).__name__}: {str(_gluon_err)[:200]}"

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
G3_BM, G3_BN, G3_BK = 128, 256, 64
G3_STAGES_FWD, G3_STAGES_BWD = 6, 5

if _HAVE_GLUON:

    @_gluon.jit
    def _g3_loader(a4_desc, b_desc, aux4_desc, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, te_ptr, M, N,
                   BM: _gl.constexpr, BN: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                   STAGES: _gl.constexpr, NUM_SMS: _gl.constexpr, HAS_AUX: _gl.constexpr):
        pid = _gl.program_id(0)
        num_pid_n = _gl.cdiv(N, BN)
        num_tiles = _gl.cdiv(M, BM) * num_pid_n
        HW: _gl.constexpr = BN // 2
        NBYTES: _gl.constexpr = BM * BK + BN * BK
        AUX_AT: _gl.constexpr = STAGES if STAGES < KT - 1 else KT - 1
        num_pid_m = _gl.cdiv(M, BM)
        it = 0
        tc = 0
        e_cur = _gl.load(te_ptr + pid // num_pid_n)               # the expert of the current tile
        for tile in range(pid, num_tiles, NUM_SMS):
            pid_m = tile // num_pid_n
            off_m = pid_m * BM
            off_n = (tile % num_pid_n) * BN
            # the next tile's expert is read NOW (the latency of a global load hides inside the mainloop)
            e_nxt = _gl.load(te_ptr + _gl.minimum((tile + NUM_SMS) // num_pid_n, num_pid_m - 1))
            off_w = e_cur * N + off_n                             # this expert's rows in the stacked weights
            for ki in range(KT):
                s = it % STAGES
                ph = (it // STAGES) & 1
                _mbarrier.wait(empty.index(s), ph ^ 1)
                _mbarrier.expect(ready.index(s), NBYTES)
                dst = a_smem.index(s)._reinterpret(a4_desc.dtype, [BM // 16, 2, 8, BK], a4_desc.layout)
                _tma.async_copy_global_to_shared(a4_desc, [off_m // 16, 0, 0, ki * BK], ready.index(s), dst)
                _tma.async_copy_global_to_shared(b_desc, [off_w, ki * BK], ready.index(s), b_smem.index(s))
                it += 1
                if HAS_AUX:
                    if ki == AUX_AT:
                        # aux tile of the current tile: after the first STAGES steps the consumers have already freed the
                        # previous tile's (no deadlock), and KT - AUX_AT mainloop steps remain for this one to arrive
                        _mbarrier.wait(aux_empty, (tc & 1) ^ 1)
                        _mbarrier.expect(aux_ready, 2 * 64 * BN)
                        for w in _gl.static_range(2):
                            for h in _gl.static_range(2):
                                d4 = aux_smem.index(w).slice(h * HW, HW, dim=1)._reinterpret(
                                    aux4_desc.dtype, [4, 2, 8, HW], aux4_desc.layout)
                                _tma.async_copy_global_to_shared(
                                    aux4_desc, [(off_m + 64 * w) // 16, 0, 0, off_n + h * HW], aux_ready, d4)
            tc += 1
            e_cur = e_nxt

    @_gluon.jit
    def _g3_split_cols(c, W: _gl.constexpr):
        c3 = _gl.permute(_gl.reshape(c, [64, 2, W // 2]), [0, 2, 1])
        return _gl.split(c3)

    @_gluon.jit
    def _g3_token_pairs(x, t_stage, W: _gl.constexpr, F8: _gl.constexpr):
        # x [64, W] fp32 (rows in order r').  The transposed output: token pairs (2g, 2g+1) = rows (g, g+8) of the same thread -> one uint16 per pair.
        x4 = _gl.reshape(x, [4, 2, 8, W])
        xp = _gl.permute(x4, [0, 2, 3, 1])                   # (w, g, n, e)
        xe, xo = _gl.split(xp)                               # [4, 8, W]: token pari / dispari
        if F8 == 0:
            q16 = _gl.inline_asm_elementwise("cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;", "=h,r,r", [xe, xo],
                                             _gl.int16, True, 1)
        else:
            q16 = _gl.inline_asm_elementwise("cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;", "=h,r,r", [xe, xo],
                                             _gl.int16, True, 1)
        t_stage.store(_gl.permute(_gl.reshape(q16, [32, W]), [1, 0]))

    @_gluon.jit
    def _g3_quad_split(ah, W: _gl.constexpr):
        # ah [64, W] (rows in order r' = 16w + 8e + g, columns n = 2j + p) -> four tensors [4, 8, W/2] (w, g, j), one per (e, p)
        a5 = _gl.permute(_gl.reshape(ah, [4, 2, 8, W // 2, 2]), [0, 2, 3, 1, 4])      # (w, g, j, e, p)
        ap0, ap1 = _gl.split(a5)                                                      # p = 0 / 1
        x00, x10 = _gl.split(ap0)                                                     # e = 0 / 1
        x01, x11 = _gl.split(ap1)
        return x00, x01, x10, x11

    @_gluon.jit
    def _g3_pmax(a, b):
        return _gl.inline_asm_elementwise("max.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], _gl.int32, True, 1)

    @_gluon.jit
    def _g3_rm_init(mma_layout: _gl.constexpr, BN: _gl.constexpr):
        z = _gl.zeros([64, BN], _gl.int32, mma_layout)
        z0, z1 = _g3_split_cols(z, BN)
        q00, q01, q10, q11 = _g3_quad_split(z0, BN // 2)
        return _gl.reduce(q00, 2, _g3_pmax)

    @_gluon.jit
    def _g3_stores(r0, r1, t0, t1, row_stage, t_stage, W: _gl.constexpr):
        # r0/r1: column pairs (n = 2j, 2j+1) of rows e = 0/1 -> row staging as int16 [64, W/2];
        # t0/t1: token pairs (g, g+8) for the even/odd columns -> transposed staging as before.
        rj = _gl.reshape(_gl.permute(_gl.join(r0, r1), [0, 3, 1, 2]), [64, W // 2])       # (w, e, g, j) -> [64, W/2]
        l16: _gl.constexpr = _gl.NVMMASharedLayout.get_default_for([64, W // 2], _gl.int16)
        row_stage._reinterpret(_gl.int16, [64, W // 2], l16).store(rj)
        q16 = _gl.reshape(_gl.join(t0, t1), [4, 8, W])                                    # (w, g, n = 2j + p)
        t_stage.store(_gl.permute(_gl.reshape(q16, [32, W]), [1, 0]))

    @_gluon.jit
    def _g3_fwd_half(ah, inv_ps, row_stage, t_stage, W: _gl.constexpr):
        # ah: [64, W] fp32, already scaled by the dequant factor.  The same chain as the Triton kernel (bf16 -> relu -> square
        # -> scale -> min -> fp8), done in groups of 4 elements in PTX: packed cvt.rn.bf16x2, relu and the amax max in bf16x2
        # (exact: same values, order irrelevant), fp8 through cvt e4m3x2 on column pairs (row output) and token pairs
        # (transposed output).  Returns the packed (bf16x2) max of relu(c) per row group, [4, 8]: amax = (max relu)^2, exact
        # because squaring is monotonic.
        x00, x01, x10, x11 = _g3_quad_split(ah, W)
        r0, r1, t0, t1, mm = _gl.inline_asm_elementwise(
            """{
            .reg .b32 pa, pb, z, fa, fb, fc, fd;
            .reg .b16 b0, b1, b2, b3;
            cvt.rn.bf16x2.f32 pa, $6, $5;
            cvt.rn.bf16x2.f32 pb, $8, $7;
            mov.b32 z, 0;
            max.bf16x2 pa, pa, z;
            max.bf16x2 pb, pb, z;
            max.bf16x2 $4, pa, pb;
            shl.b32 fa, pa, 16;
            and.b32 fb, pa, 0xffff0000;
            shl.b32 fc, pb, 16;
            and.b32 fd, pb, 0xffff0000;
            mul.f32 fa, fa, fa;
            mul.f32 fb, fb, fb;
            mul.f32 fc, fc, fc;
            mul.f32 fd, fd, fd;
            mul.f32 fa, fa, $9;
            mul.f32 fb, fb, $9;
            mul.f32 fc, fc, $9;
            mul.f32 fd, fd, $9;
            min.f32 fa, fa, 0f43E00000;
            min.f32 fb, fb, 0f43E00000;
            min.f32 fc, fc, 0f43E00000;
            min.f32 fd, fd, 0f43E00000;
            cvt.rn.satfinite.e4m3x2.f32 b0, fb, fa;
            cvt.rn.satfinite.e4m3x2.f32 b1, fd, fc;
            cvt.rn.satfinite.e4m3x2.f32 b2, fc, fa;
            cvt.rn.satfinite.e4m3x2.f32 b3, fd, fb;
            mov.b16 $0, b0;
            mov.b16 $1, b1;
            mov.b16 $2, b2;
            mov.b16 $3, b3;
            }""",
            "=h,=h,=h,=h,=r,r,r,r,r,r", [x00, x01, x10, x11, inv_ps],
            (_gl.int16, _gl.int16, _gl.int16, _gl.int16, _gl.int32), True, 1)
        _g3_stores(r0, r1, t0, t1, row_stage, t_stage, W)
        return _gl.reduce(mm, 2, _g3_pmax)

    @_gluon.jit
    def _g3_bwd_half(ah, rh, ps, inv_ds, row_stage, t_stage, W: _gl.constexpr):
        # ah: [64, W] fp32 (acc*dq), rh: [64, W] e4m3 (post).  dpre = bf16(2*acc*sqrt(post*ps)), then e5m2 of dpre/dscale.
        # As in the forward: in groups of 4 elements in PTX (packed bf16x2, fp8 x2 satfinite for rows and tokens); the explicit
        # clamp of the Triton kernel is the cvt's satfinite (same values: RN rounding then saturation, or saturation then RN,
        # agree on finite values).  Returns the packed (bf16x2) max of |dpre|.
        y = 2.0 * ah * _gl.sqrt(rh.to(_gl.float32) * ps)
        x00, x01, x10, x11 = _g3_quad_split(y, W)
        r0, r1, t0, t1, mm = _gl.inline_asm_elementwise(
            """{
            .reg .b32 pa, pb, fa, fb, fc, fd;
            .reg .b16 b0, b1, b2, b3;
            cvt.rn.bf16x2.f32 pa, $6, $5;
            cvt.rn.bf16x2.f32 pb, $8, $7;
            and.b32 fa, pa, 0x7fff7fff;
            and.b32 fb, pb, 0x7fff7fff;
            max.bf16x2 $4, fa, fb;
            shl.b32 fa, pa, 16;
            and.b32 fb, pa, 0xffff0000;
            shl.b32 fc, pb, 16;
            and.b32 fd, pb, 0xffff0000;
            mul.f32 fa, fa, $9;
            mul.f32 fb, fb, $9;
            mul.f32 fc, fc, $9;
            mul.f32 fd, fd, $9;
            cvt.rn.satfinite.e5m2x2.f32 b0, fb, fa;
            cvt.rn.satfinite.e5m2x2.f32 b1, fd, fc;
            cvt.rn.satfinite.e5m2x2.f32 b2, fc, fa;
            cvt.rn.satfinite.e5m2x2.f32 b3, fd, fb;
            mov.b16 $0, b0;
            mov.b16 $1, b1;
            mov.b16 $2, b2;
            mov.b16 $3, b3;
            }""",
            "=h,=h,=h,=h,=r,r,r,r,r,r", [x00, x01, x10, x11, inv_ds],
            (_gl.int16, _gl.int16, _gl.int16, _gl.int16, _gl.int32), True, 1)
        _g3_stores(r0, r1, t0, t1, row_stage, t_stage, W)
        return _gl.reduce(mm, 2, _g3_pmax)

    @_gluon.jit
    def _g3_consumer(wg: _gl.constexpr, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, row_stage, t_stage,
                     row4_desc, t16_desc, te_ptr, dq_ptr, dq2_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                     BM: _gl.constexpr, BN: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                     STAGES: _gl.constexpr, NUM_SMS: _gl.constexpr, FORWARD: _gl.constexpr):
        pid = _gl.program_id(0)
        num_pid_n = _gl.cdiv(N, BN)
        num_tiles = _gl.cdiv(M, BM) * num_pid_n
        HW: _gl.constexpr = BN // 2
        mma_layout: _gl.constexpr = _gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                              instr_shape=[16, BN, 32])
        dq2 = _gl.load(dq2_ptr)
        ps = _gl.load(ps_ptr)
        if FORWARD:
            scale2 = 1.0 / ps
        else:
            scale2 = 1.0 / _gl.load(dscale_ptr)
        rm = _g3_rm_init(mma_layout, BN)
        it = 0
        tc = 0
        for tile in range(pid, num_tiles, NUM_SMS):
            pid_m = tile // num_pid_n
            off_m = pid_m * BM + wg * 64
            off_n = (tile % num_pid_n) * BN
            dq = _gl.load(dq_ptr + _gl.load(te_ptr + pid_m)) * dq2      # loaded before the mainloop: the latency hides
            acc = _gl.zeros([64, BN], _gl.float32, mma_layout)
            for ki in range(KT):
                s = it % STAGES
                ph = (it // STAGES) & 1
                _mbarrier.wait(ready.index(s), ph)
                a_s = a_smem.index(s).slice(wg * 64, 64, dim=0)
                acc = _wgmma(a_s, b_smem.index(s).permute((1, 0)), acc, use_acc=(ki > 0), is_async=True)
                acc = _wgmma_wait(num_outstanding=1, deps=(acc,))
                if ki > 0:
                    _mbarrier.arrive(empty.index((it - 1) % STAGES))
                it += 1
            acc = _wgmma_wait(num_outstanding=0, deps=(acc,))
            _mbarrier.arrive(empty.index((it - 1) % STAGES))

            if not FORWARD:
                _mbarrier.wait(aux_ready, tc & 1)
                raw = aux_smem.index(wg).load(mma_layout)
                _gl.thread_barrier()
                _mbarrier.arrive(aux_empty)
            acc = acc * dq
            a0, a1 = _g3_split_cols(acc, BN)
            if not FORWARD:
                r0, r1 = _g3_split_cols(raw, BN)
            _tma.store_wait(0)                                # the previous tile's stores have read the staging
            _gl.thread_barrier()
            # left half: compute, stage, async TMA store (row through the permuted 4D descriptor + int16 transposed)
            if FORWARD:
                m0 = _g3_fwd_half(a0, scale2, row_stage.index(wg * 2 + 0), t_stage.index(wg * 2 + 0), HW)
            else:
                m0 = _g3_bwd_half(a0, r0, ps, scale2, row_stage.index(wg * 2 + 0), t_stage.index(wg * 2 + 0), HW)
            _fence()
            _gl.thread_barrier()
            _tma.async_copy_shared_to_global(
                row4_desc, [off_m // 16, 0, 0, off_n],
                row_stage.index(wg * 2 + 0)._reinterpret(row4_desc.dtype, [4, 2, 8, HW], row4_desc.layout))
            _tma.async_copy_shared_to_global(t16_desc, [off_n, off_m // 2], t_stage.index(wg * 2 + 0))
            if FORWARD:
                m1 = _g3_fwd_half(a1, scale2, row_stage.index(wg * 2 + 1), t_stage.index(wg * 2 + 1), HW)
            else:
                m1 = _g3_bwd_half(a1, r1, ps, scale2, row_stage.index(wg * 2 + 1), t_stage.index(wg * 2 + 1), HW)
            _fence()
            _gl.thread_barrier()
            _tma.async_copy_shared_to_global(
                row4_desc, [off_m // 16, 0, 0, off_n + HW],
                row_stage.index(wg * 2 + 1)._reinterpret(row4_desc.dtype, [4, 2, 8, HW], row4_desc.layout))
            _tma.async_copy_shared_to_global(t16_desc, [off_n + HW, off_m // 2], t_stage.index(wg * 2 + 1))
            # amax: packed (bf16x2) max of relu(c) (forward) or |dpre| (backward) accumulated in registers; reduced across
            # threads and atomic_max'ed once at the end of the CTA (forward: amax = square of that max, exact because
            # squaring is monotonic)
            rm = _g3_pmax(rm, _g3_pmax(m0, m1))
            tc += 1
        _tma.store_wait(0)
        ml = _gl.max(_gl.max(rm << 16, axis=1), axis=0)
        mh = _gl.max(_gl.max(rm & -65536, axis=1), axis=0)
        vf = _gl.maximum(ml, mh).to(_gl.float32, bitcast=True)
        if FORWARD:
            _gl.atomic_max(amax_ptr, vf * vf)
        else:
            _gl.atomic_max(amax_ptr, vf)

    @_gluon.jit(do_not_specialize_on_alignment=["te_ptr", "dq_ptr", "dq2_ptr", "ps_ptr", "amax_ptr", "dscale_ptr"])
    def _g3_kernel(a4_desc, b_desc, aux4_desc, row4_desc, t16_desc, te_ptr, dq_ptr, dq2_ptr, ps_ptr, dscale_ptr,
                   amax_ptr, M, N,
                   BM: _gl.constexpr, BN: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                   STAGES: _gl.constexpr, NUM_SMS: _gl.constexpr, FORWARD: _gl.constexpr):
        HW: _gl.constexpr = BN // 2
        a_smem = _gl.allocate_shared_memory(a4_desc.dtype, [STAGES, BM, BK],
                                            _gl.NVMMASharedLayout.get_default_for([BM, BK], a4_desc.dtype))
        b_smem = _gl.allocate_shared_memory(b_desc.dtype, [STAGES, BN, BK], b_desc.layout)
        if FORWARD:
            OUT_DT: _gl.constexpr = _gl.float8e4nv
            aux_smem = _gl.allocate_shared_memory(_gl.float8e4nv, [1, 64, 128],
                                                  _gl.NVMMASharedLayout.get_default_for([64, 128], _gl.float8e4nv))
        else:
            OUT_DT: _gl.constexpr = _gl.float8e5
            aux_smem = _gl.allocate_shared_memory(_gl.float8e4nv, [2, 64, BN],
                                                  _gl.NVMMASharedLayout.get_default_for([64, BN], _gl.float8e4nv))
        row_stage = _gl.allocate_shared_memory(OUT_DT, [4, 64, HW],
                                               _gl.NVMMASharedLayout.get_default_for([64, HW], OUT_DT))
        t_stage = _gl.allocate_shared_memory(t16_desc.dtype, [4, HW, 32], t16_desc.layout)
        ready = _gl.allocate_shared_memory(_gl.int64, [STAGES, 1], _mbarrier.MBarrierLayout())
        empty = _gl.allocate_shared_memory(_gl.int64, [STAGES, 1], _mbarrier.MBarrierLayout())
        aux_bars = _gl.allocate_shared_memory(_gl.int64, [2, 1], _mbarrier.MBarrierLayout())
        aux_ready = aux_bars.index(0)
        aux_empty = aux_bars.index(1)
        for i in _gl.static_range(STAGES):
            _mbarrier.init(ready.index(i), count=1)
            _mbarrier.init(empty.index(i), count=2)
        _mbarrier.init(aux_ready, count=1)
        _mbarrier.init(aux_empty, count=2)
        HAS_AUX: _gl.constexpr = not FORWARD
        _gl.warp_specialize([
            (_g3_consumer, (0, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, row_stage, t_stage,
                            row4_desc, t16_desc, te_ptr, dq_ptr, dq2_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                            BM, BN, BK, KT, STAGES, NUM_SMS, FORWARD)),
            (_g3_consumer, (1, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, row_stage, t_stage,
                            row4_desc, t16_desc, te_ptr, dq_ptr, dq2_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                            BM, BN, BK, KT, STAGES, NUM_SMS, FORWARD)),
            (_g3_loader, (a4_desc, b_desc, aux4_desc, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty,
                          te_ptr, M, N, BM, BN, BK, KT, STAGES, NUM_SMS, HAS_AUX)),
        ], [4, 1], [232, 40])

    _LAYOUT_CACHE = {}

    def _g3_desc(t, block, gl_dtype):
        # the layout depends only on (block, dtype): building it on every call cost ~15 us of CPU per launch
        key = (tuple(block), gl_dtype)
        lay = _LAYOUT_CACHE.get(key)
        if lay is None:
            lay = _LAYOUT_CACHE[key] = _gl.NVMMASharedLayout.get_default_for(block, gl_dtype)
        return _GTD.from_tensor(t, block, lay)

    def _g3_perm4(t):
        # view [M, X] -> [M/16, 2, 8, X] with strides (16X, X, 2X, 1): token = 16b + e + 2g in the TMA descriptor
        M, X = t.shape
        return t.as_strided((M // 16, 2, 8, X), (16 * X, X, 2 * X, 1), t.storage_offset())

    _NUM_SMS_CACHE = {}

    def _num_sms(dev):
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if idx not in _NUM_SMS_CACHE:
            _NUM_SMS_CACHE[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
        return _NUM_SMS_CACHE[idx]


_CAP_CACHE = {}


def _is_hopper(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    cap = _CAP_CACHE.get(idx)
    if cap is None:
        cap = _CAP_CACHE[idx] = torch.cuda.get_device_capability(idx)[0]
    return cap == 9


def g3_eligible(a_f8, b_f8, n_per_expert, forward, aux=None):
    """Conditions on shapes/dtypes/layout for which the G3 path applies (otherwise the Triton kernel runs)."""
    if not _HAVE_GLUON or not a_f8.is_cuda or not _is_hopper(a_f8.device):
        return False
    M, K = a_f8.shape
    N = n_per_expert
    if M % G3_BM or N % G3_BN or K % G3_BK or K < 2 * G3_BK or M < G3_BM or b_f8.shape[0] % N or b_f8.shape[1] != K:
        return False
    if not (a_f8.is_contiguous() and b_f8.is_contiguous()) or b_f8.dtype != E4M3:
        return False
    if forward:
        return a_f8.dtype == E4M3
    return (a_f8.dtype == E5M2 and aux is not None and aux.dtype == E4M3 and aux.is_contiguous()
            and tuple(aux.shape) == (M, N))


def grouped_lrs_g3(a_f8, b_f8, tile_expert, dequant, dq2, n_per_expert, *, forward, post_scale, amax, aux=None,
                   dpre_scale=None):
    """Same contract as moe_mlp._grouped_lrs_impl.  Forward (a = x e4m3): (post e4m3 [M, N], its transpose).
    Backward (a = g e5m2, aux = post e4m3): (dpre e5m2 [M, N], its transpose)."""
    assert _HAVE_GLUON, "Gluon is not available"
    M, K = a_f8.shape
    EN, K2 = b_f8.shape
    N = n_per_expert
    assert K == K2 and EN % N == 0, (a_f8.shape, b_f8.shape, N)
    assert g3_eligible(a_f8, b_f8, N, forward, aux), (a_f8.shape, a_f8.dtype, b_f8.shape, N)
    assert forward == (aux is None) and (forward or dpre_scale is not None)
    assert tile_expert.dtype == torch.int32 and tile_expert.numel() == M // G3_BM
    assert dequant.dtype == torch.float32 and dequant.numel() == EN // N
    for t in (dq2, post_scale, amax, *(() if forward else (dpre_scale,))):
        assert t.dtype == torch.float32 and t.numel() == 1, (t.dtype, t.shape)
    BM, BN, BK = G3_BM, G3_BN, G3_BK
    stages = G3_STAGES_FWD if forward else G3_STAGES_BWD
    dev = a_f8.device
    out_dt = E4M3 if forward else E5M2
    out_gl = _gl.float8e4nv if forward else _gl.float8e5
    out_row = torch.empty((M, N), device=dev, dtype=out_dt)
    out_t = torch.empty((N, M), device=dev, dtype=out_dt)
    a_gl = _gl.float8e4nv if forward else _gl.float8e5
    a4 = _g3_desc(_g3_perm4(a_f8), [BM // 16, 2, 8, BK], a_gl)
    b_d = _g3_desc(b_f8, [BN, BK], _gl.float8e4nv)
    row4 = _g3_desc(_g3_perm4(out_row), [4, 2, 8, BN // 2], out_gl)
    t16 = _g3_desc(out_t.view(torch.int16), [BN // 2, 32], _gl.int16)
    aux4 = a4 if forward else _g3_desc(_g3_perm4(aux), [4, 2, 8, BN // 2], _gl.float8e4nv)   # forward: unused
    nsm = _num_sms(dev)
    grid = (min(nsm, (M // BM) * (N // BN)),)
    _g3_kernel[grid](a4, b_d, aux4, row4, t16, tile_expert, dequant, dq2, post_scale,
                     post_scale if forward else dpre_scale, amax, M, N,
                     BM=BM, BN=BN, BK=BK, KT=K // BK, STAGES=stages, NUM_SMS=nsm, FORWARD=forward,
                     num_warps=4)
    return out_row, out_t


def prime_g3(reference=None, K=768, n_experts=2, N=512):
    """Compile (eagerly) the two G3 variants on a small problem, BEFORE torch.compile / a CUDA graph capture: K is a
    compile-time constant, so use the production one.  If `reference` (moe_mlp._grouped_lrs_triton) is given, compare rows,
    transposes and amax with it bit for bit.  Returns None if all is well, else a short string saying why G3 must not be
    used."""
    if not _HAVE_GLUON:
        return f"Gluon is not importable ({_GLUON_IMPORT_ERROR})"
    if not torch.cuda.is_available() or not _is_hopper(torch.device("cuda")):
        return "the GPU is not Hopper (sm_90)"
    try:
        dev = torch.device("cuda")
        g = torch.Generator(device=dev)
        g.manual_seed(1234)
        M = 128 * n_experts
        te = torch.arange(n_experts, device=dev, dtype=torch.int32)
        x = (torch.randn(M, K, device=dev, generator=g) * 6).clamp(-448, 448).to(E4M3)
        w1 = (torch.randn(n_experts * N, K, device=dev, generator=g) * 6).clamp(-448, 448).to(E4M3)
        w2 = (torch.randn(n_experts * N, K, device=dev, generator=g) * 6).clamp(-448, 448).to(E4M3)
        gr = (torch.randn(M, K, device=dev, generator=g) * 6).clamp(-57344.0, 57344.0).to(E5M2)
        dq = torch.full((n_experts,), 1.0 / 400.0, device=dev)
        dq2 = torch.full((1,), 1.0 / 400.0, device=dev)
        ps = torch.full((1,), 0.5, device=dev)
        ds = torch.full((1,), 0.015625, device=dev)
        a_g, a_t = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
        pf, pt = grouped_lrs_g3(x, w1, te, dq, dq2, N, forward=True, post_scale=ps, amax=a_g)
        d_g, d_t = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
        df, dt = grouped_lrs_g3(gr, w2, te, dq, dq2, N, forward=False, post_scale=ps, amax=d_g, aux=pf, dpre_scale=ds)
        torch.cuda.synchronize(dev)
        if reference is None:
            return None
        rpf, rpt = reference(x, w1, te, dq, dq2, N, forward=True, post_scale=ps, amax=a_t)
        rdf, rdt = reference(gr, w2, te, dq, dq2, N, forward=False, post_scale=ps, amax=d_t, aux=pf, dpre_scale=ds)
        torch.cuda.synchronize(dev)
        eq = lambda a, b: torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        if (eq(pf, rpf) and eq(pt, rpt) and eq(df, rdf) and eq(dt, rdt) and torch.equal(a_g, a_t)
                and torch.equal(d_g, d_t)):
            return None
        return "outputs differ from the Triton kernel"
    except Exception as e:                      # Gluon missing/incompatible: the caller stays on Triton
        return f"{type(e).__name__}: {str(e)[:200]}"

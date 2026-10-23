"""G3: the Gluon (Hopper) path for the two fp8 MLP GEMMs of linear_relu_square.

What it replaces: the Triton kernel in mlp.py for the two rows training runs (the fp8 forward with its
relu^2 / e4m3 epilogue and the fp8 backward that emits dpre). It produces the same BYTES as that kernel;
mlp.py's _g3_prime compares the two bit for bit at start-up and G3 is only used if they match.
Why it is faster: the epilogue runs concurrently with the next tile's mainloop (warp-specialised loader,
2 consumer warpgroups), the aux tile arrives by async TMA, and the fp8 transpose is a 16-bit stmatrix
thanks to a token permutation done for free by a 4D TMA descriptor. The forward additionally has a
ping-pong variant (G3PP) with a stationary B panel.

Self-contained on purpose (no import of mlp.py): mlp.py imports this module and owns the dispatch, the
start-up comparison against the Triton kernel and the stage cache. The custom ops keep the "nanogpt::"
namespace. The module-level flags below are written only by eager start-up code (mlp.py's _g3_prime and
_g3pp_selfcheck) and are read-only under torch.compile.

Provenance: record #92's G3 kernel, plus the ping-pong forward (G3PP).
"""
import os

import torch
import triton

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


# -----------------------------------------------------------------------------
# G3: Gluon path for the two fp8 MLP GEMMs (the forward and the backward's emit_dpre), producing the same bytes as the
# Triton kernel in mlp.py.  _g3_prime_base compares the two bit for bit at start-up and G3 is only used if they match.
#
# Why: in the Triton kernel the epilogue runs AFTER the mainloop on the same warpgroup, so the tensor core sits idle.  G3 has
# 2 consumer warpgroups (wgmma 64x256 each) plus 1 dedicated loader warp that prefetches A, B and, in the backward, the `aux`
# tile (fp8 post) into smem with async TMA; the epilogue runs in registers inside the consumer.  The gain comes from:
#   (1) the aux tile is no longer a synchronous TMA load on the critical path (backward: it was ~100 us of 343);
#   (2) the fp8 transpose is no longer a byte-wise store (128 st.shared.b8 per thread): see below;
#   (3) the TMA store happens half a tile at a time.
#
# THE M-PERMUTATION TRICK.  Tokens enter the wgmma in row order r' = g + 8e + 16b (token = 16b + 2g + e), obtained FOR FREE
# with a 4D TMA descriptor on operand A ([M/16, 2, 8, K] with strides (16K, K, 2K, 1)).  The rows (g, g+8) that one MMA
# thread owns are then the ADJACENT tokens (2g, 2g+1): a single cvt e4m3x2/e5m2x2 on (even token, odd token) gives a
# uint16, and the transpose [n][token] comes out of a 16-bit stmatrix.trans (hardware).  The row output returns to the
# natural order through an identical 4D descriptor on the store side; the transposed output is an int16 view [N, M/2].
# The K-sum of every element is unchanged (same sequence of wgmma k32), so the result is identical bit for bit.
# -----------------------------------------------------------------------------

_G3_BM, _G3_BN, _G3_BK = 128, 256, 64
_G3_STAGES_FWD, _G3_STAGES_BWD = 6, 5

if _HAVE_GLUON:

    @_gluon.jit
    def _g3_loader(a4_desc, b_desc, aux4_desc, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, M, N,
                   BM: _gl.constexpr, BN: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                   STAGES: _gl.constexpr, NUM_SMS: _gl.constexpr, HAS_AUX: _gl.constexpr):
        pid = _gl.program_id(0)
        num_pid_n = _gl.cdiv(N, BN)
        num_tiles = _gl.cdiv(M, BM) * num_pid_n
        HW: _gl.constexpr = BN // 2
        NBYTES: _gl.constexpr = BM * BK + BN * BK
        AUX_AT: _gl.constexpr = STAGES if STAGES < KT - 1 else KT - 1
        it = 0
        tc = 0
        for tile in range(pid, num_tiles, NUM_SMS):
            off_m = (tile // num_pid_n) * BM
            off_n = (tile % num_pid_n) * BN
            for ki in range(KT):
                s = it % STAGES
                ph = (it // STAGES) & 1
                _mbarrier.wait(empty.index(s), ph ^ 1)
                _mbarrier.expect(ready.index(s), NBYTES)
                dst = a_smem.index(s)._reinterpret(a4_desc.dtype, [BM // 16, 2, 8, BK], a4_desc.layout)
                _tma.async_copy_global_to_shared(a4_desc, [off_m // 16, 0, 0, ki * BK], ready.index(s), dst)
                _tma.async_copy_global_to_shared(b_desc, [off_n, ki * BK], ready.index(s), b_smem.index(s))
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
        # ah: [64, W] fp32 (acc*dq), rh: [64, W] e4m3 (post).  dpre = bf16(2*acc*sqrt(post*ps)) -> e5m2.
        c = (2.0 * ah * _gl.sqrt(rh.to(_gl.float32) * ps)).to(_gl.bfloat16)
        cf = c.to(_gl.float32)
        x = _gl.maximum(_gl.minimum(cf * inv_ds, 57344.0), -57344.0)
        row_stage.store(x.to(_gl.float8e5))
        _g3_token_pairs(x, t_stage, W, 1)
        return _gl.abs(cf)

    @_gluon.jit
    def _g3_consumer(wg: _gl.constexpr, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, row_stage, t_stage,
                     row4_desc, t16_desc, dq_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                     BM: _gl.constexpr, BN: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                     STAGES: _gl.constexpr, NUM_SMS: _gl.constexpr, FORWARD: _gl.constexpr):
        pid = _gl.program_id(0)
        num_pid_n = _gl.cdiv(N, BN)
        num_tiles = _gl.cdiv(M, BM) * num_pid_n
        HW: _gl.constexpr = BN // 2
        mma_layout: _gl.constexpr = _gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                              instr_shape=[16, BN, 32])
        dq = _gl.load(dq_ptr)
        ps = _gl.load(ps_ptr)
        if FORWARD:
            scale2 = 1.0 / ps
        else:
            scale2 = 1.0 / _gl.load(dscale_ptr)
        if FORWARD:
            rm = _g3_rm_init(mma_layout, BN)
        it = 0
        tc = 0
        for tile in range(pid, num_tiles, NUM_SMS):
            off_m = (tile // num_pid_n) * BM + wg * 64
            off_n = (tile % num_pid_n) * BN
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
            # amax: backward = max |dpre| of the tile, one atomic_max per warpgroup and tile.  Forward = packed (bf16x2) max of
            # relu(c) accumulated in registers; reduced across threads and atomic_max'ed once at the end of the CTA (amax =
            # square of that max, exact: squaring is monotonic)
            if FORWARD:
                rm = _g3_pmax(rm, _g3_pmax(m0, m1))
            else:
                _gl.atomic_max(amax_ptr, _gl.max(_gl.max(_gl.maximum(m0, m1), axis=1), axis=0))
            tc += 1
        _tma.store_wait(0)
        if FORWARD:
            ml = _gl.max(_gl.max(rm << 16, axis=1), axis=0)
            mh = _gl.max(_gl.max(rm & -65536, axis=1), axis=0)
            vf = _gl.maximum(ml, mh).to(_gl.float32, bitcast=True)
            _gl.atomic_max(amax_ptr, vf * vf)

    @_gluon.jit(do_not_specialize_on_alignment=["dq_ptr", "ps_ptr", "amax_ptr", "dscale_ptr"])
    def _g3_kernel(a4_desc, b_desc, aux4_desc, row4_desc, t16_desc, dq_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
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
                            row4_desc, t16_desc, dq_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                            BM, BN, BK, KT, STAGES, NUM_SMS, FORWARD)),
            (_g3_consumer, (1, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty, row_stage, t_stage,
                            row4_desc, t16_desc, dq_ptr, ps_ptr, dscale_ptr, amax_ptr, M, N,
                            BM, BN, BK, KT, STAGES, NUM_SMS, FORWARD)),
            (_g3_loader, (a4_desc, b_desc, aux4_desc, a_smem, b_smem, aux_smem, ready, empty, aux_ready, aux_empty,
                          M, N, BM, BN, BK, KT, STAGES, NUM_SMS, HAS_AUX)),
        ], [4, 1], [232, 40])

    # ------------------------------------------------------------------------------------------------------------------
    # G3PP: PING-PONG variant of G3 for the FORWARD GEMM.  _g3pp_selfcheck compares it bit for bit against plain G3 at
    # start-up and the ping-pong is only used if the two agree.
    #
    # Why: in G3 the two warpgroups work in lockstep on the same 128x256 tile and the epilogue (58 us of the 225 us forward)
    # leaves the tensor core idle.  Measurements (H100, 700 W cap): G3's mainloop is limited by the L2->SM BANDWIDTH (~7 TB/s:
    # 1.25 GB of loads in 170 us), so the "obvious" ping-pong (each warpgroup on its own 64x256 tile, B not shared) raises the
    # traffic to 2.1 GB and the forward from 225 to 309 us.  The design that works keeps B STATIONARY:
    #   - each CTA pins a block of BNP=128 columns of N (22 blocks x 6 CTAs = 132 SMs) and keeps the B panel [128, K] in smem
    #     (96 KB for K=768); it streams the rows of A in items of 128 tokens x 128 columns -> L2->SM traffic 0.83 GB (-33%);
    #   - each item has two [64,128] accumulators (two wgmma m64n128 per k step, same B);
    #   - the two consumer warpgroups ALTERNATE on the items (item j -> warpgroup j % 2) with a strict alternation of the
    #     mainloop (mbarrier ord_bars: the mainloop of item j starts once the other warpgroup has issued the last wgmma of
    #     item j-1): the tensor core is always busy with one of them while the other runs the epilogue;
    #   - 1 loader warp (B panel, then A in stages of BK=128 = 16 KB with G3's M permutation through the 4D descriptor) and 1
    #     storer warp that issues the TMA stores (row + transposed) from the staging and frees the slot once they have READ smem
    #     (a ring of R=3 half-items of 16 KB shared by the two consumers: no store_wait in the consumers);
    #   - the K-sum of every element is the same sequence of wgmma k32 as G3 and the epilogue is the same math
    #     (_g3_fwd_half), so rows, transposes and amax match G3 bit for bit (also in offline tests with NaN and ps = 0/inf
    #     inputs; the start-up self-check covers random inputs).
    # The backward does not use the ping-pong: its epilogue is ALU-heavy and in the ping-pong it runs on ONE warpgroup at a
    # time (on two in G3), so at equal time there was no gain.
    # ------------------------------------------------------------------------------------------------------------------
    @_gluon.jit
    def _g3pp_rm_init(mma_layout: _gl.constexpr, W: _gl.constexpr):
        z = _gl.zeros([64, W], _gl.int32, mma_layout)
        q00, q01, q10, q11 = _g3_quad_split(z, W)
        return _gl.reduce(q00, 2, _g3_pmax)

    @_gluon.jit
    def _g3pp_loader(a4_desc, b_desc, a_smem, b_smem, ready, empty, b_ready, M, N,
                     BM: _gl.constexpr, BNP: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                     STAGES: _gl.constexpr, QN: _gl.constexpr):
        pid = _gl.program_id(0)
        nb = pid // QN
        q = pid % QN
        off_n = nb * BNP
        num_mb = _gl.cdiv(M, BM)
        n_items = _gl.cdiv(num_mb - q, QN)
        # B panel: one k block per barrier, so the first item starts as soon as its first block arrives
        for k in _gl.static_range(KT):
            _mbarrier.expect(b_ready.index(k), BNP * BK)
            _tma.async_copy_global_to_shared(b_desc, [off_n, k * BK], b_ready.index(k), b_smem.index(k))
        it = 0
        for j in range(n_items):
            off_m = (q + j * QN) * BM
            for ki in range(KT):
                s = it % STAGES
                ph = (it // STAGES) & 1
                _mbarrier.wait(empty.index(s), ph ^ 1)
                _mbarrier.expect(ready.index(s), BM * BK)
                dst = a_smem.index(s)._reinterpret(a4_desc.dtype, [BM // 16, 2, 8, BK], a4_desc.layout)
                _tma.async_copy_global_to_shared(a4_desc, [off_m // 16, 0, 0, ki * BK], ready.index(s), dst)
                it += 1

    @_gluon.jit
    def _g3pp_storer(row4_desc, t16_desc, row_stage, t_stage, stg_full, stg_free, M, N,
                     BM: _gl.constexpr, BNP: _gl.constexpr, QN: _gl.constexpr, R: _gl.constexpr):
        pid = _gl.program_id(0)
        nb = pid // QN
        q = pid % QN
        off_n = nb * BNP
        num_mb = _gl.cdiv(M, BM)
        n_items = _gl.cdiv(num_mb - q, QN)
        n = 0
        for j in range(n_items):
            off_m = (q + j * QN) * BM
            for h in _gl.static_range(2):
                slot = n % R
                _mbarrier.wait(stg_full.index(slot), (n // R) & 1)
                _tma.async_copy_shared_to_global(
                    row4_desc, [(off_m + 64 * h) // 16, 0, 0, off_n],
                    row_stage.index(slot)._reinterpret(row4_desc.dtype, [4, 2, 8, BNP], row4_desc.layout))
                _tma.async_copy_shared_to_global(t16_desc, [off_n, (off_m + 64 * h) // 2], t_stage.index(slot))
                _tma.store_wait(0)
                _mbarrier.arrive(stg_free.index(slot))
                n += 1

    @_gluon.jit
    def _g3pp_consumer(wg: _gl.constexpr, a_smem, b_smem, ready, empty, b_ready, ord_bars,
                       row_stage, t_stage, stg_full, stg_free, dq_ptr, ps_ptr, amax_ptr, M, N,
                       BM: _gl.constexpr, BNP: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                       STAGES: _gl.constexpr, QN: _gl.constexpr, R: _gl.constexpr):
        pid = _gl.program_id(0)
        q = pid % QN
        num_mb = _gl.cdiv(M, BM)
        n_items = _gl.cdiv(num_mb - q, QN)
        mma_layout: _gl.constexpr = _gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                              instr_shape=[16, BNP, 32])
        dq = _gl.load(dq_ptr)
        ps = _gl.load(ps_ptr)
        scale2 = 1.0 / ps
        rm = _g3pp_rm_init(mma_layout, BNP)
        cnt = 0
        for j in range(wg, n_items, 2):
            it = j * KT
            # strict alternation of the mainloop: item j starts after the other warpgroup has issued the last wgmma of item j-1
            if wg == 0:
                if cnt > 0:
                    _mbarrier.wait(ord_bars.index(1), (cnt - 1) & 1)
            else:
                _mbarrier.wait(ord_bars.index(0), cnt & 1)
            acc0 = _gl.zeros([64, BNP], _gl.float32, mma_layout)
            acc1 = _gl.zeros([64, BNP], _gl.float32, mma_layout)
            for ki in _gl.static_range(KT):
                s = (it + ki) % STAGES
                ph = ((it + ki) // STAGES) & 1
                _mbarrier.wait(ready.index(s), ph)
                _mbarrier.wait(b_ready.index(ki), 0)
                a_s = a_smem.index(s)
                b_t = b_smem.index(ki).permute((1, 0))
                acc0 = _wgmma(a_s.slice(0, 64, dim=0), b_t, acc0, use_acc=(ki > 0), is_async=True)
                acc1 = _wgmma(a_s.slice(64, 64, dim=0), b_t, acc1, use_acc=(ki > 0), is_async=True)
                acc0, acc1 = _wgmma_wait(num_outstanding=2, deps=(acc0, acc1))
                if ki > 0:
                    _mbarrier.arrive(empty.index((it + ki - 1) % STAGES))
            _mbarrier.arrive(ord_bars.index(wg))
            acc0, acc1 = _wgmma_wait(num_outstanding=0, deps=(acc0, acc1))
            _mbarrier.arrive(empty.index((it + KT - 1) % STAGES))
            # epilogue: half an item (64 tokens) at a time into slot n = 2j + h of the staging ring; the storer issues the TMA stores
            acc0 = acc0 * dq
            acc1 = acc1 * dq
            # The staging-ring slots are taken IN ORDER of n (item j-1 of the other warpgroup before item j): stg_free's parity
            # check only distinguishes 2 phases and is valid only for in-order acquisitions; otherwise, with few slots and a
            # late storer, a warpgroup could move on to a slot still in use (wrong outputs or a hang).  ord_bars[2 + wg] =
            # "wg has acquired both slots of its item".
            if j >= 1:
                _mbarrier.wait(ord_bars.index(3 - wg), ((j - 1) // 2) & 1)
            n0 = 2 * j
            slot0 = n0 % R
            _mbarrier.wait(stg_free.index(slot0), ((n0 // R) & 1) ^ 1)
            m0 = _g3_fwd_half(acc0, scale2, row_stage.index(slot0), t_stage.index(slot0), BNP)
            _fence()
            _gl.thread_barrier()
            _mbarrier.arrive(stg_full.index(slot0))
            n1 = n0 + 1
            slot1 = n1 % R
            _mbarrier.wait(stg_free.index(slot1), ((n1 // R) & 1) ^ 1)
            _mbarrier.arrive(ord_bars.index(2 + wg))
            m1 = _g3_fwd_half(acc1, scale2, row_stage.index(slot1), t_stage.index(slot1), BNP)
            _fence()
            _gl.thread_barrier()
            _mbarrier.arrive(stg_full.index(slot1))
            # amax: packed (bf16x2) max of relu(c) in registers, a single atomic_max at the end of the CTA (amax = square of
            # that max, exact: squaring is monotonic)
            rm = _g3_pmax(rm, _g3_pmax(m0, m1))
            cnt += 1
        ml = _gl.max(_gl.max(rm << 16, axis=1), axis=0)
        mh = _gl.max(_gl.max(rm & -65536, axis=1), axis=0)
        vf = _gl.maximum(ml, mh).to(_gl.float32, bitcast=True)
        _gl.atomic_max(amax_ptr, vf * vf)

    @_gluon.jit(do_not_specialize_on_alignment=["dq_ptr", "ps_ptr", "amax_ptr"])
    def _g3pp_kernel(a4_desc, b_desc, row4_desc, t16_desc, dq_ptr, ps_ptr, amax_ptr, M, N,
                     BM: _gl.constexpr, BNP: _gl.constexpr, BK: _gl.constexpr, KT: _gl.constexpr,
                     STAGES: _gl.constexpr, QN: _gl.constexpr, R: _gl.constexpr):
        a_smem = _gl.allocate_shared_memory(a4_desc.dtype, [STAGES, BM, BK],
                                            _gl.NVMMASharedLayout.get_default_for([BM, BK], a4_desc.dtype))
        b_smem = _gl.allocate_shared_memory(b_desc.dtype, [KT, BNP, BK], b_desc.layout)
        OUT_DT: _gl.constexpr = _gl.float8e4nv
        row_stage = _gl.allocate_shared_memory(OUT_DT, [R, 64, BNP],
                                               _gl.NVMMASharedLayout.get_default_for([64, BNP], OUT_DT))
        t_stage = _gl.allocate_shared_memory(t16_desc.dtype, [R, BNP, 32], t16_desc.layout)
        ready = _gl.allocate_shared_memory(_gl.int64, [STAGES, 1], _mbarrier.MBarrierLayout())
        empty = _gl.allocate_shared_memory(_gl.int64, [STAGES, 1], _mbarrier.MBarrierLayout())
        b_ready = _gl.allocate_shared_memory(_gl.int64, [KT, 1], _mbarrier.MBarrierLayout())
        ord_bars = _gl.allocate_shared_memory(_gl.int64, [4, 1], _mbarrier.MBarrierLayout())
        stg_full = _gl.allocate_shared_memory(_gl.int64, [R, 1], _mbarrier.MBarrierLayout())
        stg_free = _gl.allocate_shared_memory(_gl.int64, [R, 1], _mbarrier.MBarrierLayout())
        for i in _gl.static_range(STAGES):
            _mbarrier.init(ready.index(i), count=1)
            _mbarrier.init(empty.index(i), count=1)
        for i in _gl.static_range(KT):
            _mbarrier.init(b_ready.index(i), count=1)
        for i in _gl.static_range(4):
            _mbarrier.init(ord_bars.index(i), count=1)
        for i in _gl.static_range(R):
            _mbarrier.init(stg_full.index(i), count=1)
            _mbarrier.init(stg_free.index(i), count=1)
        _gl.warp_specialize([
            (_g3pp_consumer, (0, a_smem, b_smem, ready, empty, b_ready, ord_bars, row_stage, t_stage,
                              stg_full, stg_free, dq_ptr, ps_ptr, amax_ptr, M, N,
                              BM, BNP, BK, KT, STAGES, QN, R)),
            (_g3pp_consumer, (1, a_smem, b_smem, ready, empty, b_ready, ord_bars, row_stage, t_stage,
                              stg_full, stg_free, dq_ptr, ps_ptr, amax_ptr, M, N,
                              BM, BNP, BK, KT, STAGES, QN, R)),
            (_g3pp_loader, (a4_desc, b_desc, a_smem, b_smem, ready, empty, b_ready, M, N,
                            BM, BNP, BK, KT, STAGES, QN)),
            (_g3pp_storer, (row4_desc, t16_desc, row_stage, t_stage, stg_full, stg_free, M, N, BM, BNP, QN, R)),
        ], [4, 1, 1], [232, 40, 40])

    def _g3pp_ok(M, N, K):
        """Shapes G3PP covers: B panel [128, K] in smem (K <= 768), rows in items of 128, columns in blocks of 128 with at
        least one block per available SM."""
        return (M % 128 == 0 and N % 128 == 0 and K % _G3PP_BK == 0 and K // _G3PP_BK <= 768 // _G3PP_BK
                and 0 < N // 128 <= _G3_NUM_SMS[0])

    def _g3pp_launch(a_f8, b_f8, dequant, post_scale, amax, out_row, out_t):
        M, K = a_f8.shape
        N = b_f8.shape[0]
        BM, BNP, BK = 128, 128, _G3PP_BK
        nblk = N // BNP
        qn = max(1, min(_G3_NUM_SMS[0] // nblk, M // BM))          # CTAs per column block (6 with N=2816 on 132 SMs)
        a4 = _g3_desc(_g3_perm4(a_f8), [BM // 16, 2, 8, BK], _gl.float8e4nv)
        b_d = _g3_desc(b_f8, [BNP, BK], _gl.float8e4nv)
        row4 = _g3_desc(_g3_perm4(out_row), [4, 2, 8, BNP], _gl.float8e4nv)
        t16 = _g3_desc(out_t.view(torch.int16), [BNP, 32], _gl.int16)
        _g3pp_kernel[(nblk * qn,)](a4, b_d, row4, t16, dequant, post_scale, amax, M, N,
                                   BM=BM, BNP=BNP, BK=BK, KT=K // BK, STAGES=_G3PP_STAGES, QN=qn, R=_G3PP_R,
                                   num_warps=4)
        return out_row, out_t

    def _g3_desc(t, block, gl_dtype):
        return _GTD.from_tensor(t, block, _gl.NVMMASharedLayout.get_default_for(block, gl_dtype))

    def _g3_perm4(t):
        # view [M, X] -> [M/16, 2, 8, X] with strides (16X, X, 2X, 1): token = 16b + e + 2g in the TMA descriptor
        M, X = t.shape
        return t.as_strided((M // 16, 2, 8, X), (16 * X, X, 2 * X, 1), t.storage_offset())

    def _g3_launch(forward, a_f8, b_f8, aux_f8, dequant, post_scale, dpre_scale, amax):
        M, K = a_f8.shape
        N = b_f8.shape[0]
        BM, BN, BK = _G3_BM, _G3_BN, _G3_BK
        stages = _G3_STAGES_FWD if forward else _G3_STAGES_BWD
        dev = a_f8.device
        out_dt = torch.float8_e4m3fn if forward else torch.float8_e5m2
        out_gl = _gl.float8e4nv if forward else _gl.float8e5
        out_row = torch.empty((M, N), device=dev, dtype=out_dt)
        out_t = torch.empty((N, M), device=dev, dtype=out_dt)
        if forward and _G3_PP[0] and _g3pp_ok(M, N, K):        # ping-pong forward; the backward always runs plain G3
            return _g3pp_launch(a_f8, b_f8, dequant, post_scale, amax, out_row, out_t)
        a_gl = _gl.float8e4nv if forward else _gl.float8e5
        a4 = _g3_desc(_g3_perm4(a_f8), [BM // 16, 2, 8, BK], a_gl)
        b_d = _g3_desc(b_f8, [BN, BK], _gl.float8e4nv)
        row4 = _g3_desc(_g3_perm4(out_row), [4, 2, 8, BN // 2], out_gl)
        t16 = _g3_desc(out_t.view(torch.int16), [BN // 2, 32], _gl.int16)
        if forward:
            aux4 = a4                                        # unused (HAS_AUX=False)
        else:
            aux4 = _g3_desc(_g3_perm4(aux_f8), [4, 2, 8, BN // 2], _gl.float8e4nv)
        grid = (min(_G3_NUM_SMS[0], triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        _g3_kernel[grid](a4, b_d, aux4, row4, t16, dequant, post_scale, dpre_scale, amax, M, N,
                         BM=BM, BN=BN, BK=BK, KT=K // BK, STAGES=stages, NUM_SMS=_G3_NUM_SMS[0], FORWARD=forward,
                         num_warps=4)
        return out_row, out_t

    # Custom ops, opaque to dynamo/inductor: the launch always happens in eager mode (also inside a compiled graph) and writes no
    # global Python state; amax is mutated in place (atomic_max).
    @torch.library.custom_op("nanogpt::lrs_g3_fwd", mutates_args=("post_amax",))
    def _g3_fwd_op(x_f8: torch.Tensor, w_f8: torch.Tensor, dequant: torch.Tensor, post_scale: torch.Tensor,
                   post_amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _g3_launch(True, x_f8, w_f8, None, dequant, post_scale, post_scale, post_amax)

    @_g3_fwd_op.register_fake
    def _(x_f8, w_f8, dequant, post_scale, post_amax):
        return (x_f8.new_empty((x_f8.shape[0], w_f8.shape[0]), dtype=torch.float8_e4m3fn),
                x_f8.new_empty((w_f8.shape[0], x_f8.shape[0]), dtype=torch.float8_e4m3fn))

    @torch.library.custom_op("nanogpt::lrs_g3_bwd", mutates_args=("dpre_amax",))
    def _g3_bwd_op(g_f8: torch.Tensor, w2_f8: torch.Tensor, aux_f8: torch.Tensor, dequant: torch.Tensor,
                   post_scale: torch.Tensor, dpre_scale: torch.Tensor,
                   dpre_amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _g3_launch(False, g_f8, w2_f8, aux_f8, dequant, post_scale, dpre_scale, dpre_amax)

    @_g3_bwd_op.register_fake
    def _(g_f8, w2_f8, aux_f8, dequant, post_scale, dpre_scale, dpre_amax):
        return (g_f8.new_empty((g_f8.shape[0], w2_f8.shape[0]), dtype=torch.float8_e5m2),
                g_f8.new_empty((w2_f8.shape[0], g_f8.shape[0]), dtype=torch.float8_e5m2))

# Number of SMs for G3's persistent grid (a 1-element list: written only by prime_stage_cache, eagerly).  _g3_ok is decided
# ONCE in prime_stage_cache (compile + bit-for-bit comparison against the Triton kernel); under compile it is read-only.
_G3_NUM_SMS = [132]
# G3PP (see above): [1] = ping-pong forward on, [0] = plain G3 forward.  A 1-element list, written only by _g3_prime and
# _g3pp_selfcheck (eager), read-only under compile.  BK = 128 (16 KB stages: fewer barriers per MAC), R = 3 half-item
# staging slots, 5 stages in the A ring (smem 224 KB).
_G3_PP = [1]
_G3PP_BK, _G3PP_R, _G3PP_STAGES = 128, 3, 5
_g3_ok = [False]
# Public status of the Gluon path, final once prime_stage_cache() has run (it runs before torch.compile):
#   G3_ACTIVE / G3PP_ACTIVE: the fp8 MLP GEMMs run the Gluon kernels / their ping-pong forward variant;
#   G3_FALLBACK_REASON / G3PP_FALLBACK_REASON: None when active, else why the path fell back (to the Triton kernel / to plain
#   G3).  The fallback is also printed on rank 0 by _g3_prime; the traced code reads _g3_ok and _G3_PP, never these.
G3_ACTIVE = G3PP_ACTIVE = False
G3_FALLBACK_REASON = G3PP_FALLBACK_REASON = "prime_stage_cache() has not run"


def _g3_eligible(forward, a_f8, b_f8, aux, dequant_scale_ptr, post_scale, amax):
    """All conditions are trace-time (shapes, dtypes, layout): the G3 path covers EXACTLY the rows used in training."""
    if not (_HAVE_GLUON and _g3_ok[0]):
        return False
    if a_f8 is None or b_f8 is None or dequant_scale_ptr is None or post_scale is None or amax is None:
        return False
    M, K = a_f8.shape
    N = b_f8.shape[0]
    if M % _G3_BM or N % _G3_BN or K % _G3_BK or K < 2 * _G3_BK or M < _G3_BM:
        return False
    if not (a_f8.is_contiguous() and b_f8.is_contiguous() and b_f8.shape[1] == K):
        return False
    if b_f8.dtype != torch.float8_e4m3fn:
        return False
    if forward:
        return a_f8.dtype == torch.float8_e4m3fn
    return (a_f8.dtype == torch.float8_e5m2 and aux is not None and aux.dtype == torch.float8_e4m3fn
            and aux.is_contiguous() and tuple(aux.shape) == (M, N))

def _g3pp_selfcheck(dev):
    """G3PP (ping-pong forward) against plain G3, bit for bit: a small shape and one with N=2816, K=768 (the training
    specialization: it compiles here, in eager mode, and not during CUDA graph capture).  If anything differs or raises,
    the ping-pong is switched off and plain G3 stays."""
    global G3PP_ACTIVE, G3PP_FALLBACK_REASON
    reason = None
    try:
        e4 = torch.float8_e4m3fn
        g = torch.Generator(device=dev)
        g.manual_seed(4321)
        for (M, N, K) in ((256, 512, 768), (768, 2816, 768)):
            x = (torch.randn(M, K, device=dev, generator=g) * 6).clamp(-448, 448).to(e4)
            w = (torch.randn(N, K, device=dev, generator=g) * 6).clamp(-448, 448).to(e4)
            dq = torch.full((1,), 1.0 / 400.0, device=dev)
            ps = torch.full((), 0.7, device=dev)
            res = []
            for v in (0, 1):
                _G3_PP[0] = v
                am = torch.zeros(1, device=dev)
                pf, pt = torch.ops.nanogpt.lrs_g3_fwd(x, w, dq, ps, am)
                res.append([pf.view(torch.uint8), pt.view(torch.uint8), am])
            torch.cuda.synchronize(dev)
            if not all(torch.equal(a, b) for a, b in zip(res[0], res[1])):
                reason = f"outputs differ from plain G3 (M={M}, N={N}, K={K})"
                break
    except Exception as _e:
        reason = f"{type(_e).__name__}: {str(_e)[:200]}"
    _G3_PP[0] = 0 if reason else 1
    G3PP_ACTIVE, G3PP_FALLBACK_REASON = reason is None, reason
    if reason is not None and int(os.environ.get("RANK", "0")) == 0:
        print(f"[linear_relu_square] G3PP (ping-pong) disabled, plain G3 stays: {reason}", flush=True)

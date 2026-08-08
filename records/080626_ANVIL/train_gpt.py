"""NanoGPT speedrun record-attempt trainer: GPT-2-scale (124M) on 8xH100, single node.

Target: mean final validation cross-entropy <= 3.28 (all runs count), minimizing the
wall-clock time of the training section. Timing convention (standard): compilation,
kernel warmup, CUDA-graph capture and every validation pass (including the final
one) are untimed; the clock runs from the post-warmup torch.cuda.synchronize() to
the synchronize at each validation break. First-batch fetch, shard loading, the
prefix-table build and the terminal weight ships are all inside the timed section.

Main method departures from the stock baseline:
  * Optimizer: ANVIL — a twin-rail velocity matrix optimizer over sharded
    parameter banks: fast/slow velocity rails blended per step, a quintic
    spectral-map cascade on the velocity Gram, per-lane energy equalization,
    sign-aligned decay, exact-fp32 commits on bf16 storage via a mantissa
    sidecar, and a bank tail-blend ship step;
    replicated-full Adam for lm_head/embed; fused flat all_reduce for the small
    replicated params and explicitly scheduled communication orders.
  * fp8 MLP forward AND backward (static clip-free activation scales, delayed
    grad/post-activation scales, dual-layout weight caches); fp8 lm_head cache
    feeding a fused softcapped cross-entropy with an auxiliary prefix-token CE term.
  * Hashed-bigram embedding channel with a sparse-gradient sink (compact
    segment-sum + sparse all_to_all comms) and value-embedding planes with a
    selected-load backward (value_embed_op.py, sha-pinned import).
  * 3-stage batch/window/seq-len schedule with a terminal batch taper and a short
    extension phase; YaRN window rescaling with partial in-loop factor-table
    rebuilds repaired off-clock; tail-EMA / tail-average weight shipping applied
    after the clock stops, before the final validation.
  * Manual CUDA-graph capture of the post-step fp8 quantize segment.

Live environment variables: KX_SEED (seeded reproduction runs), KX_STEPS (scheduled
step count), DATA_PATH (data root). Everything else is baked into constants below.
"""
import os
import sys

# Read this file and every kernel module it ships with, for the run log.
code = ""
for _src_file in (sys.argv[0], 'fused_kernels.py', 'value_embed_op.py',
                  'value_embed_kernel.py', 'bigram_kernels.py', 'dc_triton_kernels.py'):
    _p = _src_file if _src_file == sys.argv[0] else os.path.join(os.path.dirname(sys.argv[0]), _src_file)
    with open(_p, 'r') as f:
        if code:
            code += f"\n\n{'-'*40}\n# {os.path.basename(_p)}\n{'-'*40}\n\n"
        code += f.read()
with open(os.path.join(os.path.dirname(sys.argv[0]), 'dc_triton_kernels.py'), 'r') as f:
    code += f"\n\n{'-'*40}\n# dc_triton_kernels.py\n{'-'*40}\n\n"
    code += f.read()
# DC-attention correction kernel selector: 0 = stock kernel set (shipping config);
# 1 would select an atomic-free v2 set (dc_triton_kernels_v2.py), unused here.
_KX_DCK2 = int("0")

import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import triton
import numpy as np

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from kernels import get_kernel
from torch import Tensor, nn

# Fused Triton / compiled kernels (nanogpt::* custom-op namespace): Gram matmuls
# for the whitening cascade, the fp8 MLP forward/backward, the fused softcapped CE,
# tiled transpose helpers and the dual-layout fp8 quantizers.
if '1' == "1" or '1' == "1":
    from fused_kernels import (XXT, XTX, ba_plus_cAA, FusedLinearReLUSquareFunction,
                     FusedSoftcappedCrossEntropy, transpose_add, transpose_copy,
                     quantize_dual_layout, quantize_mlp_weights_dual,
                     quantize_weights_dual_ntiles, prime_stage_cache,
                     quantize_transposed, AttnProjBwdFP8, ptp_init)
    import fused_kernels as _fk  # module alias: carries the prefix-CE runtime weight hook

# ---- wall-clock bundle flags ----
_KX_R2 = True        # off-clock TEMA ship + fp32 TEMA scratch + shard prescan + partial Yarn rebuild
_KX_LMF8T = True      # col-major lm_head fp8 cache for the CE forward GEMM
_KX_PROF = int('0')    # probe-only: chrome-trace steps [N, N+8) on rank 0 (wall meaningless)
_R2_YARN_ROWS = 0                                       # set >0 before the timed loop when _KX_R2
_R2_PRESCAN_MAX = 0                                     # 0 = stock lazy shard loading (no pre-clock prescan)
_r2_tema_scratch: dict = {}
from dc_triton_kernels import (
    dc_attention_postonly_nodd_correction_add_base_triton,
)
# Fused triton kernel: relu(x @ W1.T)^2 @ W2.T
# https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply

dynamo.config.recompile_limit = 64

KX_SLG_LOAD = int('1')
assert KX_SLG_LOAD in (0, 1), "KX_SLG_LOAD must be 0 or 1"
if KX_SLG_LOAD:
    import hashlib as _slg_hashlib

    _SLG_OP_PATH = Path(__file__).with_name("value_embed_op.py")
    _SLG_OP_SHA256 = "27cf093a81516bb3bd6062c45d28a63662826c16c31d4646c435426afbb080be"
    _slg_op_sha256 = _slg_hashlib.sha256(_SLG_OP_PATH.read_bytes()).hexdigest()
    assert _slg_op_sha256 == _SLG_OP_SHA256, (
        f"SLG trainer op mismatch: got {_slg_op_sha256}, expected {_SLG_OP_SHA256}"
    )
    from value_embed_op import value_embedding_planes_selected_load

FP8_MLP_FWD = True     # FP8 MLP down-projection forward
PREFIX_CE = True          # PR#337 port: auxiliary prefix-token CE through the fused kernel (train only; val untouched)
PREFIX_CE_WEIGHT = float("0.2")

_ptp_tab_cache = None
def _ptp_table_alloc(device):
    """PREFIX_CE: allocate the prefix-token table with every row -1 ("no prefix" =
    exact no-op in the CE kernel, which is what warmup runs with), and warm the
    tokenizer download/parse — both untimed. The table CONTENTS are built by
    _ptp_table_fill() right after the clock starts, so the construction is charged
    to training time, matching the placement this mechanism's record (#88) was
    accepted with. Static vocab map (PR#337); no data-pipeline involvement."""
    global _ptp_tab_cache
    if _ptp_tab_cache is None or _ptp_tab_cache.device != device:
        import tiktoken
        tiktoken.get_encoding("gpt2")  # download + parse untimed; fill pays construction only
        _ptp_tab_cache = torch.full((50304,), -1, dtype=torch.int64, device=device)
    return _ptp_tab_cache

def _ptp_table_fill():
    """Token -> longest proper byte-prefix token id, copy_'d into the pre-allocated
    device table IN PLACE (the tensor feeds compiled code; its address must never
    change). Sorted-stack build: sorting the vocab's byte strings puts every proper
    prefix before its extensions, so a stack of live ancestors yields the longest
    one in O(V log V). Runs after the timer starts."""
    import tiktoken
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks
    table = [-1] * 50304
    stack: list = []
    stack_ids: list = []
    for b, tid in sorted(ranks.items()):
        while stack and not b.startswith(stack[-1]):
            stack.pop()
            stack_ids.pop()
        if stack_ids:
            table[tid] = stack_ids[-1]
        stack.append(b)
        stack_ids.append(tid)
    _ptp_tab_cache.copy_(torch.tensor(table, dtype=torch.int64))
TAIL_AVG_WINDOW = int('250')          # tail weight-average window (0=off); assemblies compared at final val
# Tail-average rule: lerp_(p, 0.02) per step, i.e. an EMA with a ~50-step timescale
# (NOT a uniform mean over the TAIL_AVG_WINDOW window), over the label set returned
# by _tavg_label_list (vo_bank, mlp_bank). Shipped before the final validation.
# DUAL_MOMENTUM: dual-timescale Nesterov momentum (PR #339) on the
# bank params. Format "fast_beta,slow_beta,fast_w,engage_step". Rail 0 of the
# fp32 twin-rail velocity state is the FAST rail (its stock update rule is exactly
# vel.lerp_(g, 1-beta) with beta = the scheduled beta, so pre-engage it runs the
# stock trajectory bit-identically via a beta scalar tensor holding the scheduled
# beta and blend weight 1.0); rail 1 is the zero-init SLOW rail, accumulating
# from step 0 with slow_beta so it is warm at engage_step.
DUAL_MOMENTUM = '0.85,0.98,0.4385,514'
_DUAL_MOMENTUM = None
if DUAL_MOMENTUM:
    _bmx_f = DUAL_MOMENTUM.split(",")
    assert len(_bmx_f) == 4, "DUAL_MOMENTUM format: fast_beta,slow_beta,fast_w,engage_step"
    _DUAL_MOMENTUM = (float(_bmx_f[0]), float(_bmx_f[1]), float(_bmx_f[2]), int(_bmx_f[3]))
_DUAL_MOMENTUM_SLOW_BETA = _DUAL_MOMENTUM[1] if _DUAL_MOMENTUM is not None else 0.0
BIGRAM_BF16_BWD = True  # bf16 index_add bigram embedding backward
KX_TRI = int("0")      # trigram hash embedding channel: 0=off (shipping), 1=dense comms, 2=sparse comms
# ---- BIGRAM_SPARSE_GRAD / KX_BGLITE: sparse-gradient wiring for the hashed-bigram channel ----
# Stock (and BIGRAM_BF16_BWD) materialize a DENSE [bigram_vocab, 768] gradient on every backward:
# a 579 MB zero-fill + the index_add + a 579 MB read-modify-write inside AccumulateGrad on
# the second step of each Adam pair + a dense row gather at exchange time -- ~2.7 GB/step
# of bandwidth for a table where <0.05% of the rows are touched.
# Both knobs replace that with a GRAD SINK: the bigram lookup is DETACHED and a persistent
# zeros [T, bigram_dim] leaf (requires_grad=True, passed as a forward ARGUMENT so dynamo
# sees a fresh graph input) is added to it. The forward value is unchanged (adding zeros);
# autograd never builds a table gradient; and after backward `sink.grad` is EXACTLY the
# per-token value stream the stock backward would have index_add_'d into the table -- the
# sign multiply sits DOWNSTREAM of the add, so autograd applies it for us.
#   KX_BGLITE=1 : A-lite stepping stone. Harvested values are index_add'd into ONE
#                 persistent DENSE [V, 768] buffer, zeroed once per Adam cycle and handed
#                 to the stock comms path as param.grad. Kills the per-backward zero-fill
#                 and the AccumulateGrad pass; keeps the dense gather/merge. Works at any
#                 world size (dense grad is what the stock path wants).
#   BIGRAM_SPARSE_GRAD=1/2 : full sparse wiring (needs _sparse_comms_active()). Values are
#                 segment-summed into a COMPACT [n_unique, 768] buffer aligned with the
#                 sparse-comms row list (np.flatnonzero(row_update_mask) IS that
#                 sorted-unique list), the exchange payload is two contiguous slices of it
#                 (no gather at all), and the merge lands in a persistent DENSE SHARD
#                 [V/world, 768] buffer that feeds the STOCK _adam_update.
#                 1 = bf16 segment-sum (same accumulation dtype as the stock bf16
#                 index_add), 2 = fp32 segment-sum (exchange + shard dtypes unchanged).
# Neither is bit-identical to stock: multi-hit row sums are re-associated (one pass over
# the whole Adam cycle instead of per-step index_add + accumulate) -- same clean
# numerics class as BIGRAM_BF16_BWD.
BIGRAM_SPARSE_GRAD = int('1')     # 0=off, 1=bf16 compact segment-sum, 2=fp32 compact segment-sum
_bigram_sink_on = BIGRAM_SPARSE_GRAD > 0 or False               # any grad-sink wiring active (forward is shared)
assert BIGRAM_SPARSE_GRAD in (0, 1, 2), "BIGRAM_SPARSE_GRAD: 0=off, 1=bf16 segment-sum, 2=fp32 segment-sum"
assert not (BIGRAM_SPARSE_GRAD and False), "BIGRAM_SPARSE_GRAD and KX_BGLITE are two wirings of the same channel -- pick one"
assert not (0 >= 3), "BIGRAM_SPARSE_GRAD/KX_BGLITE own the bigram table's backward; KX_TRI>=3 puts a SECOND lookup into that same table (rows would be dropped) -- not wired"
KX_TLRM = float("0")   # trigram_embed lr_mul override (0 = use BIGRAM_LR_MUL)
KX_TVOC = int("1")     # trigram own-table vocab DIVISOR (1=full 377k; 4/8 => smaller table, cheaper comms)
KX_AUX = int("0")       # dedicated aux n-gram table, vocab/KX_AUX rows; tri+tet+skp hash streams, single fused lookup
KX_TRISYNC = int("0")  # K-step FedAvg for trigram_embed (KX_TRI=1 only): 0=off; K>0 => NO grad comms (rank-local Adam, comms "local_kavg"), weight all_reduce(AVG) every K steps + before any val eval + at the final step
KX_NGF = int("0")       # fused concatenated-own-rows n-gram channel: tri+tet+skp streams, ONE 3V-row table (per-stream row blocks), single fused lookup
KX_NGFK = int("0")     # fused triton op for the KX_NGF channel (external fused-ngram op; not shipped): three hashes + 0/V/2V-offset gathers + sum + sign in ONE fwd kernel, ONE 3-atomic-add bwd kernel
KX_NGD = int("0")       # narrow-dim NGF: when >0 the ngram_fused table is (3V, KX_NGD) and its output is injected into ONLY the first KX_NGD dims of x0_bigram; 0 = full bigram_dim (current behavior)
FP8_POST_HEADROOM = float("1.03")  # mlp post-act amax headroom (stock 1.03)
_KX_GLR_CONST = float("1.0")  # global lr multiplier, hoisted to a module constant (read per param per step)
PRINT_EVERY = int('25')  # step-print thinning: print every N steps (0=stock every step)
_KX_ASC = float("0")    # attention logit softcap via FA3 native param (0=off/stock)
_pref_batch = None  # holds the prefetched batch tuple (or None -> synchronous fetch)
if BIGRAM_BF16_BWD:
    import bigram_kernels as _m90
# REPLICATED_FULL_ADAM:
# lm_head + embed move from sharded Adam (RS + shard update + AG) to
# "replicated_full" (one async AVG all_reduce of the full grad, full-param Adam
# on every rank, NO gather). NCCL AR is bitwise-identical across ranks -> ranks
# stay in lockstep. Distinct comms name keeps them OUT of the FLAT_ALLREDUCE flat buffer.
REPLICATED_FULL_ADAM = int('1')
VEMB_LR_MUL = float('70')  # value_embeds lr_mul
BIGRAM_LR_MUL = float('70')  # bigram_embed lr_mul
KX_C2S = False          # FP8 attention forwards, static scales + cached weights (v2)
KX_XS8 = float("32.0")       # static amax bound for attn inputs (RMS-normed)
# KX_QKD: reduced-width QK attention. 0 = off (bit-identical to stock).
# N>0: query/key head dim becomes N while value/output head dim stays head_dim
# (128). Halves (at N=64) the QK^T FLOPs and the q/k projection GEMM widths.
# qk_bank rows shrink 256 -> 2N per head-pair group; bank (64, 2N, 768), chunk
# (8, 2N, 768) — all optimizer geometry flows from qk_bank.reshape. FA3 accepts
# mixed q/k=64, v=128 head dims (verified fwd on H100). N=64 is the validated
# setting.
KX_QKD = int("0")
# KX_WSL:
# per-layer attention-window multipliers "layer:mult,...". Applied to per-stage bm_size in
# tokens, rounded to a 128 multiple, floor 128. Mults <1 on SHORT-window layers only (keeps
# key_offset's b==ws_long test sound). Empty = off = bit-identical.
KX_WSL = {int(kv.split(":")[0]): float(kv.split(":")[1]) for kv in "".split(",") if ":" in kv}
_KX_QKLRMW_N, _KX_QKLRMW_M = (0, 1.0)
# KX_QKMW: MIXED-WIDTH QK. Only meaningful on top of KX_QKD=64.
# The two long-window (ws_long) carrier layers {3,10} -- which are also the layers
# that carry key_offset -- keep the FULL d_qk=128 head; the other 8 attention layers
# stay at d_qk=64. Packing trick: qk_bank is allocated at the STOCK full-width
# geometry (64, 256, 768) and the narrow layers use only rows [:128] of each
# head-pair group; rows [128:256] are zero-initialised and provably stay exactly
# zero (never read in the forward => exactly-zero grad => zero momentum => the
# cascade's odd polynomial and the ANVIL per-lane rescale both map zero
# rows to zero rows => zero update; sign-aligned decay multiplies zero by zero). So
# optimizer chunking, ANVIL state, comms, scatter/work order and
# shape_mult (max(1, 256/768)**0.5 == max(1, 128/768)**0.5 == 1) are all UNTOUCHED
# -- the mechanism lives entirely in forward weight assembly + module routing.
# Default 0 = off = bit-identical.
# KX_QKSCAF ("scaffold curriculum"): stacks on KX_QKMW. Rationale: formation
# at halved qk capacity may be what costs loss. So:
# let ALL 10 attention layers form their qk basis at the FULL d_qk=128 during stage 1,
# then TRUNCATE the 8 narrow-destined layers back to d_qk=64 at the stage-1 -> stage-2
# boundary. No weights are moved at the switch: the scaffold assembly applies a fixed
# per-group ROW PERMUTATION chosen so that the dims that survive truncation are exactly
# the bank rows the stock narrow slice reads afterwards (see the assembly site in
# GPT.forward for the row algebra). The optimizer/bank are COMPLETELY untouched -- the
# bank is already allocated full-size under KX_QKMW; post-truncation the scaffold-only
# rows stop being read, take exactly-zero gradient and decay exactly like QKMW's pad
# rows. Requires KX_QKMW=1 (hence KX_QKD=64, KX_QKROT=0, no fp8 attention caches --
# all asserted just above / at the S3 assert below). Default 0 = off = bit-identical.
# KX_RSTAG / KX_RNOPE (rotary head-diversity): at KX_QKD=64 the NARROW head has
# only _rot = d_qk//2 = 32 rotating dims == 16 distinct frequencies, and every one of the
# 6 heads gets the SAME 16 (the Yarn factor tables are (2*max_seq, d_qk) and broadcast
# over the head axis at the application site). 192 of the 384 per-layer qk dims therefore
# encode 16 frequencies redundantly. Both knobs give the NARROW NON-PAIRED Yarn (self.yarn)
# a real head axis -- factors become (2*max_seq, H, d_qk):
#   KX_RSTAG=1  per-head STAGGERED ladders: head h takes exponents (k*H + h)/(H*steps-1)
#               of the (1/1024)**e band, so the 6 heads jointly cover H*steps = 96 distinct
#               frequencies evenly spaced over EXACTLY the original [1, 1/1024] band
#               (h=0,k=0 -> 1; h=H-1,k=steps-1 -> 1/1024). Per-head spacing is 1/15.83 in
#               exponent instead of the stock 1/15 -- the only price of keeping the band ends.
#   KX_RNOPE=n  the last n heads get angular_freq = 0 on ALL dims -> cos=1, sin=0 ->
#               identity rotary: n pure CONTENT heads per narrow non-paired layer.
# They compose (stagger the first H-n heads, identity the rest). SCOPE: self.yarn only.
# The PAIRED narrow Yarn is deliberately untouched -- its factor rows pack two heads per
# row-block over a doubled sequence (heads {0,3},{1,4},{2,5} share a flash-attn head after
# the reshape), so per-head ladders there would break RoPE's relative-position property on
# the cross-head terms. Wide (KX_QKMW) and scaffold (KX_QKSCAF) Yarns untouched by design.
# NOTE if stacking with KX_QKSCAF: the scaffold-phase Yarns carry the stock single ladder,
# so the narrow layers' rotary semantics CHANGE at the scaffold truncation switch.
# Default 0/0 = off = bit-identical (no head axis is built; every stock line runs unchanged).
KX_RSTAG = int("0")
KX_RNOPE = int("0")
FP8_LMHEAD_CACHE = True           # trainer-side flag for the lm_head FP8 reuse
FLAT_ALLREDUCE = True     # fused flat all_reduce for replicated Adam params
# [AFE] Fused foreach update for the small replicated-comms Adam params (scalars/
# gates/lambdas/mudd_*): one torch._foreach_* sequence per hyperparameter group
# instead of ~13 separate compiled-region entries + two 0-D fill_s each on odd
# steps. Elementwise math per param is identical to _adam_update_step (bias
# correction + sign-aligned decay included); the two paths are numerically interchangeable.
# Comm re-orchestration inside the optimizer phase. Pure permutation of
# scatter_order/work_order: every collective keeps its exact inputs, op, group and
# dtype, so the run is bit-for-bit identical to COMM_ORDER=0 (time-only change).
#   0 = stock order, 1 = trace-optimal order (shipping), 2 = alternative traced order
COMM_ORDER = int('1')
GRAD_FIREWALL = True  # scale-poison + grad firewall (nan/inf scrub on delayed fp8 scales)
# Per-param Adam period for value_embeds only (0/1/2 = stock every-odd-step).
# NOT numerics-preserving: it changes how many steps of gradient accumulate into one
# Adam update -> loss-screen risk class, kept on its own knob, default off.
KX_VEPER = int("0")
# Manual CUDA-graph capture of the optimizer-phase compute segments.
# Collectives, future.wait()s, transpose_add/copy and the sparse
# bigram path all stay eager; only the collective-free compute between them is captured:
#   G_qk / G_vo / G_mlp  per-bank cascade + ANVIL, replayed every step
#   G_quant              the quantize_mlp_fp8 body, replayed every step
#   G_tiny               the ~12 replicated Adam updates, odd steps (needs FLAT_ALLREDUCE=1)
# Bit-for-bit identical to CUDA_GRAPH_TIER=0 by construction: same kernels, same order, same fp32
# scalar values (fed through one pinned staging vector instead of per-launch fill_s).
#   0 = off (this file byte-identical in behaviour to the pre-CG version)
#   1 = full tier: banks + quant (+ tiny when FLAT_ALLREDUCE=1)
#   2 = minimal-risk tier: G_quant only (zero scalars, zero variants, fully eager body)
CUDA_GRAPH_TIER = int('2')  # shipping tier: 2 (G_quant only)
assert CUDA_GRAPH_TIER in (0, 1, 2), "CUDA_GRAPH_TIER must be 0, 1 or 2"
FP8_MLP_DPRE = True     # FP8 MLP backward dpre GEMM (e5m2 grads)
FP8_GRAD_SCALE = float('0.015625')  # static e5m2 grad scale (2^-8)
FP8_MLP_DW2 = True     # FP8 dW2 via epilogue dual-layout emit
FP8_MLP_DX = True     # FP8 dx (dpre @ W1), col-major W1 cache
FP8_MLP_DW1 = True     # FP8 dW1 via dpre_t/x_t epilogue layouts
FP8_DROP_PRE = True # drop pre; bwd reconstructs sqrt(post_f8)
KX_GE4 = False     # e4m3 grads (dynamic g / delayed dpre scales)
FP8_LAGGED_QUANT = True     # fused lagged weight quantize + per-SM amax
FP8_DPRE_HEADROOM = float('1.25')
assert True, "FP8 MLP wgrad/dx arms require FP8_MLP_FWD=1 FP8_MLP_DPRE=1"
if FP8_DROP_PRE:
    assert FP8_MLP_FWD, "FP8_DROP_PRE reconstructs from the C1A fp8 post"
# ---- S3: fp8 attention BACKWARD (bf16 fwd untouched); disabled in this
# configuration (KX_S3D/KX_S3W off), plumbing retained. ----
KX_S3D = False      # fp8 dx_qkv (K=2304, dx_f8 op; fused aten e5m2 g cast)
KX_S3W = int("0")        # fp8 dW_qkv (K=T, wg1_f8 op): 1=primary, 2=route B (DCE-independent)
KX_S3_GS = float("0.015625")  # static e5m2 attention-g scale (2^-6; ladder flat 2^-6..2^-5)
FP8_ATTN_WGRAD_E4M3 = True       # e4m3 wgrad g @ delayed per-layer scale (default ON: dW_qkv v-rows feed the vo_bank)
KX_S3_HR = float("1.10")      # delayed-scale headroom
FP8_ATTN_XSCALE = 0.0625  # static e4m3 scale for attn_in_normed: 448*2^-4 = 28.0 > sqrt(768) = 27.713 — clip-free BY PROOF, no amax kernel anywhere on x
# STATIC-XS-MLP: static 2^-4 x-scale for the
# fp8 MLP forward. Same tensor class as the S3 attn x above (post-rms_norm, unit RMS
# per 768-dim row => max|x| <= sqrt(768) = 27.713 < 28.0 = 448*2^-4), so the identical
# clip-free proof applies. Deletes the per-layer per-microstep amax reduction + the
# 3-op _mlp_dequant_scale_buf RMW chain; per-layer dequant scales (up_proj_scale *
# 2^-4) precompute in quantize_mlp_fp8.
FP8_STATIC_XSCALE = True
assert 0 in (0, 1, 2), "KX_S3W must be 0, 1 or 2"
assert True, \
    "S3 (fp8 attention backward) and C2S/C2F (fp8 attention forward) are exclusive arms"
assert True, (
    "KX_QKMW does not cover the S3 fp8 attention-backward caches (_s3_qkv_col / _s3_gt "
    "are sized from a single self.attn.qkv_rows; mixed widths need per-layer nq + padded "
    "caches). Run KX_QKMW with KX_S3D=0.")
# PR #317 residual variant. Upstream merged ONLY the XSA algebraic rewrite from that PR
# (already present in this file, inherited from the base); the sparse-attention-gate
# removal was declined because record #85 had already re-laid-out those gates as
# MUDD-generated, token-conditional coefficients on layers {3,10}. KX_317 exposes that
# declined half as a switchable variant. It changes NO coefficient indices and NO parameter
# shapes -- only whether the layer-{3,10} attn gate is applied -- so KX_317=0 is
# byte-identical in behaviour to the file without this knob.
#   0 = off (stock #85: token-conditional gate on layers 3 and 10)
#   1 = faithful #317 variant: drop the attn gate entirely on both layers.
#       NOTE the gate's init value is 0.25 (attn_gate_bias 2.5 x _mudd_gate_scale 0.1),
#       so mode 1 multiplies the layer-3 / layer-10 attention branch by 4x at init.
#       That is a much larger perturbation than the pre-#85 SAG removal, where the gate
#       was sigmoid(0)=0.5 (a 2x change). Expect a step/loss retune to be needed.
#   2 = scale-preserving control: keep the 0.25 factor but freeze it, i.e. ablate only
#       the token-conditionality of the gate. Isolates "is the gate's dynamics earning
#       its keep" from "is the 0.25 branch scale load-bearing". Run 2 alongside 1.
KX_317 = int("0")
assert 0 in (0, 1, 2), "KX_317 must be 0, 1 or 2"
HOST_OPT = True  # host-gap bundle: buffered log handle, pinned-buffer reuse,
                                              # gc off in the timed loop, lm_head-refresh/tavg micro-cleanups.
                                              # Every HOST_OPT site is numerics-preserving by construction.
_tavg_bufs: dict = {}
# Tail-EMA2 ladder: short-window EMAs for Adam-side params {lm_head, embed, value_embeds,
# bigram_embed} (which ship RAW here). Comma list of windows, e.g. "40,80". Assembly is
# an UNTIMED final-break ladder on one trajectory — training math untouched. Empty = off.
_TAIL_EMA2_WINDOWS = [int(x) for x in "".split(",") if x.strip()]
_tavg2_bufs: dict = {}
_TAVG2_LABELS = ('lm_head', 'embed', 'value_embeds', 'bigram_embed')
_tavg_n: dict = {}   # per-label sample count (kept for the ship log line)
_hg_tavg_scratch: dict = {}   # HOST_OPT: persistent fp32 scratch for the tavg EMA upcast
def _tavg_label_list(opt):
    """Labels covered by the TAIL_AVG_WINDOW tail average (the ANVIL-shipped banks)."""
    return ["vo_bank", "mlp_bank"]
    return labels
_lk_slow: dict = {}
# [SOV]: comma list of work_order labels whose optimizer work item (reduce-wait,
# update math, gather launch) runs on a dedicated side stream, overlapping the dense
# items on the compute stream. Math identical to the inline path; excludes the
# tied lm_head/embed pair (quantize refresh_lm reads lm_head on the main stream).
_sov_labels = set(x for x in "".split(",") if x)  # empty = side-stream path disabled (shipping)

# -----------------------------------------------------------------------------
# Distributed training setup
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
grad_scale = 1 / grad_accum_steps # consistent grad magnitudes between different num_devices
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng
# Transposed layout by @ChrisJMcCormick allows for faster gradient accumulation.

@torch.library.custom_op("nanogpt::mm_t", mutates_args=())
def mm_t_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """Computes y = x @ w with F8 weights stored as (in_features, out_features)."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]  # x: (batch, in), w: (in, out)

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)

        # _scaled_mm requires column-major B. w_f8 is row-major (in, out).
        # .T.contiguous().T creates a column-major view without changing logical shape.
        w_f8_col_major = w_f8.T.contiguous().T

        out = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_t_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_t_backward", mutates_args=())
def mm_t_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()

        x_scale = grad.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)

        # grad_x = grad @ w.T
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        # grad_w = x.T @ grad
        # Result is (in, out), naturally matching weight storage. No final .T needed.
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, grad_w

    grad_x, grad_w = impl(g, x_f8, w_f8)

    return grad_x, grad_w

@mm_t_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward_t(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_t_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context_t(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_t_op.register_autograd(backward_t, setup_context=setup_context_t)

# -----------------------------------------------------------------------------
# Polar Express

# ANVIL map schedule: six quintic spectral maps t -> a + b*t + c*t^2 acting on
# the velocity Gram, derived by a CEM + minimax-polish pipeline and
# constraint-verified on a 6000-point grid. Composite envelope on
# sigma >= 3e-3: [0.9971, 1.0095]; tail gain 950; every intermediate bounded by
# 1.296; worst per-map cancellation ratio 31.3. Composed, they push every
# singular value of the normalized input onto ~1.
_ANVIL_STAGES = int("6")
ANVIL_MAPS = [
    (4.447393659248992,  -8.6592834371539,    4.484453950130224),
    (3.0003381342927278, -3.180035633097396,  0.959298544318381),
    (3.5997737315685896, -5.129961939372116,  1.94624091228252),
    (3.238755176943844,  -3.988313284248892,  1.4073635628857994),
    (2.558664852492382,  -2.5626431459988033, 0.9815933969681315),
    (2.38665846260798,   -2.223181615281231,  0.835162681686761)
][:_ANVIL_STAGES]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def anvil_cascade(grad_chunk: torch.Tensor, velocity: torch.Tensor, momentum_t: torch.Tensor,
                   split_baddbmm: bool = False,
                   bimax_bf_t: torch.Tensor | None = None,
                   bimax_w_t: torch.Tensor | None = None):
    """
    Twin-rail velocity update + ANVIL whitening cascade, fused in one graph.

    Velocity is a single fp32 state tensor of shape [2, *chunk]: rail 0 is the
    fast rail (scheduled beta), rail 1 the slow rail (constant beta). Both rails
    are EMAs of the reduced gradient, advanced together in one broadcast lerp;
    the shipped estimate is a weighted rail blend passed through a Nesterov-style
    lookahead. All of that runs in fp32; the result is cast to bf16 for the
    whitening cascade, avoiding materialization of the fp32 intermediate between
    graph breaks.

    Whitening: each cascade stage applies X <- X @ (a*I + b*A + c*A^2) with
    A = X^T X the Gram of the current iterate (mirrored to A = X X^T with left
    multiplication for wide chunks), driving all singular values toward 1
    (the Polar Express / Newton-Schulz iteration family, arXiv:2505.16932;
    coefficients re-derived, see ANVIL_MAPS). The first Gram is
    taken on the raw bf16 blend; its trace supplies the normalization that puts
    the spectrum inside the cascade's convergence region.

    momentum_t is a 0-D CPU tensor to avoid triggering graph recompilations when the value changes.
    """
    # ---- twin-rail velocity (fp32) ----
    # The fp32 upcast of the reduced grad happens INSIDE this compiled graph
    # (the caller used to do it eagerly), so inductor fuses the cast into the
    # first rail read instead of materializing a standalone fp32 copy. Same
    # .float() cast, same values everywhere below; the Nesterov lerp_ mutates
    # this fp32 copy exactly as it mutated the eager copy before, and the raw
    # reduced grad is never read again after the cascade.
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    if bimax_bf_t is None:
        # Single-rail mode (dual-timescale blend disabled): only the fast rail
        # advances, and it is the whole velocity estimate.
        fast_rail = velocity[0]
        fast_rail.lerp_(grad_chunk, 1 - momentum)
        g = grad_chunk.lerp_(fast_rail, momentum)
    else:
        # Dual-timescale rails (DUAL_MOMENTUM, PR #339). bimax_bf_t carries the
        # scheduled beta before engage_step (and with blend weight 1.0 the
        # shipped estimate is exactly the fast rail => stock trajectory
        # bit-identically) and fast_beta after; bimax_w_t carries 1.0 before
        # engage_step, fast_w after. Both are 0-D scalar tensors (filled per
        # step in step_optimizers) so the engage flip never recompiles this
        # graph. Rail 1 always accumulates the raw reduced grad with the
        # constant slow beta, so it is warm long before engage_step.
        # Both rails advance in the single [2,*chunk] velocity tensor; the per-rail
        # weights stay 0-D scalars all the way into lerp_ (a materialized CPU weight
        # tensor here becomes a blocking H2D inside the compiled graph that drains
        # the host run-ahead every step — measured +1.4 ms/step).
        velocity[0].lerp_(grad_chunk, 1 - bimax_bf_t.to(grad_chunk.dtype))
        velocity[1].lerp_(grad_chunk, 1 - _DUAL_MOMENTUM_SLOW_BETA)
        w = bimax_w_t.to(grad_chunk.dtype)
        m_eff = w * velocity[0] + (1 - w) * velocity[1]
        g = grad_chunk.lerp_(m_eff, momentum)

    X = g.bfloat16().contiguous()
    is_tall = g.size(-2) > g.size(-1)

    # ---- Gram-trace normalization + whitening cascade (bf16) ----
    # The first Gram is computed on the UNNORMALIZED blend; tr(A_raw) equals
    # ||X||_F^2 algebraically, so d = sqrt(tr A_raw)*(1+2e-2)+1e-6 is the margin
    # divisor that caps the spectral norm just below 1. X/d then has Gram
    # A_raw/d^2, so stage 1 reuses the rescaled Gram and only stages 2..5
    # recompute it from the running iterate.
    if is_tall:
        # Tall: use Triton kernels with X^T @ X (small) and right multiplication
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        XTX(X, out=A)  # A = X.T @ X on the raw blend
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * (1 + 2e-2) + 1e-6
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        # Run the cascade stages
        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XTX(X, out=A)  # A = X.T @ X, Gram of the current iterate
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)

            # Referencing X twice causes pytorch to make a defensive copy,
            # resulting in a cudaMemcpyAsync in baddbmm.
            # For large matrices (i.e., the mlp weights), it's faster to split
            # the operation into two kernels to avoid this.
            if split_baddbmm:
                XB_matmul(X, B, out=C)  # C = X @ B
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)  # C = a * X + X @ B

            X, C = C, X  # Swap references to avoid unnecessary copies
    else:
        # Wide: use Triton kernels with X @ X^T (small) and left multiplication
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        XXT(X, out=A)  # A = X @ X.mT on the raw blend
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * (1 + 2e-2) + 1e-6
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        # Run the cascade stages
        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XXT(X, out=A)  # A = X @ X.mT, Gram of the current iterate
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

            if split_baddbmm:
                BX_matmul(B, X, out=C)  # C = B @ X
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

            X, C = C, X  # Swap references to avoid unnecessary copies

    return X

# -----------------------------------------------------------------------------
# Sparse Comms for bigram embedding gradient reduce-scatter
def _sparse_comms_active():
    # we count on this in order for sparse communication to be worthwhile
    return world_size == 8 and grad_accum_steps == 1

# BIGRAM_SPARSE_GRAD rides the sparse-comms row machinery (compact buffer is aligned with the
# flatnonzero row list); without it there is no row list to align to. KX_BGLITE has no
# such dependency -- it produces a plain dense grad, so it works under "sharded" too.
assert not (BIGRAM_SPARSE_GRAD and not _sparse_comms_active()), \
    "BIGRAM_SPARSE_GRAD needs the sparse-comms path (world_size==8, grad_accum_steps==1); use KX_BGLITE=1 otherwise"

@torch.no_grad
def sparse_comms_start(idxes_np, N, rank, world, send_idxes_buffer, bgsp_out=None):
    """Start the sparse row exchange: upload the sorted-unique touched-row list,
    partition it by owning rank, and launch the async send-count all_to_all."""
    rows_per_rank = N // world

    # queue upload of indexes to gpu
    send_idxes = send_idxes_buffer[:idxes_np.shape[0]]
    send_idxes.copy_(torch.from_numpy(idxes_np))
    send_idxes = send_idxes.to(device, non_blocking=True)

    # calculate how many gradient rows we will send to every rank
    insertion_points = np.searchsorted(
        idxes_np,
        np.arange(0, rows_per_rank * (world + 1), rows_per_rank, dtype=np.int32),
    )
    send_counts = torch.from_numpy(insertion_points[1:] - insertion_points[:-1])
    # zero-out own send-count - we won't send our own gradient rows to ourselves as it's a waste:
    # in sparse_comms_merge_gradients, we'll use the slice of the gradient that already includes them as the base tensor
    send_counts[rank] = 0

    # BIGRAM_SPARSE_GRAD: hand back the FULL (pre-cat) device-side sorted-unique row list plus the
    # rank partition. The compact grad buffer is aligned with exactly this list, so the
    # send payload is cat(compact[:ip[rank]], compact[ip[rank+1]:]) -- the same two
    # contiguous slices this cat takes of the index list, hence no gather is ever needed.
    if bgsp_out is not None:
        bgsp_out["full_idx"] = send_idxes
        bgsp_out["lo"] = int(insertion_points[rank])
        bgsp_out["hi"] = int(insertion_points[rank + 1])
        bgsp_out["n"] = int(idxes_np.shape[0])

    # remove indexes owned by our rank from the send list
    send_idxes = torch.cat([send_idxes[: insertion_points[rank]], send_idxes[insertion_points[rank + 1] :]])

    # share the send counts so that each rank will know how many rows
    # to expect from every other rank
    recv_counts = torch.empty_like(send_counts)
    recv_counts_fut = dist.all_to_all_single(recv_counts, send_counts, async_op=True).get_future()
    return send_idxes, send_counts, recv_counts, recv_counts_fut

@torch.no_grad
def sparse_comms_share_indexes(send_idxes, send_counts, recv_counts):
    # cpu tensors, so these ops are cheap and don't force a host<->device sync
    total_recv_count = recv_counts.sum().item()
    recv_counts = recv_counts.tolist()
    send_counts = send_counts.tolist()

    # queue sharing of row indexes
    recv_idxes = torch.empty(total_recv_count, dtype=torch.int32, device=device)
    idxes_fut = dist.all_to_all_single(
        recv_idxes,
        send_idxes,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        async_op=True,
    ).get_future()

    sparse_state = {
        "send_idxes": send_idxes,
        "send_counts": send_counts,
        "recv_counts": recv_counts, # list for sharing
    }
    return recv_idxes, sparse_state, idxes_fut

@torch.compile
@torch.no_grad
def sparse_comms_share_gradients(grad, idxes, send_counts, recv_counts):
    # gather the rows that we want to send
    send_vals = grad[idxes]

    d = grad.shape[1]

    send_sizes = [i*d for i in send_counts]
    recv_sizes = [i*d for i in recv_counts]

    recv_vals = torch.empty(sum(recv_sizes), device=send_vals.device, dtype=grad.dtype)

    val_fut = dist.all_to_all_single(
        recv_vals,
        send_vals.view(-1),
        input_split_sizes=send_sizes,
        output_split_sizes=recv_sizes,
        async_op=True,
    ).get_future()

    return recv_vals, val_fut

@torch.no_grad
def sparse_comms_merge_gradients(grad, recv_idx, recv_vals, rank, world):
    d = grad.shape[1]
    rows_per_rank = grad.shape[0] // world

    grad.index_add_(0, recv_idx, recv_vals.view(-1, d))

    # return the slice of the gradient for parameters our rank updates
    return grad[rows_per_rank * rank : rows_per_rank * (rank + 1)].mul_((1 / world))


@torch.no_grad
def bgsp_share_gradients(compact, lo, hi, send_counts, recv_counts, dtype):
    """BIGRAM_SPARSE_GRAD twin of sparse_comms_share_gradients. `compact` is [n_unique, d], row i
    holding the summed gradient of global row full_idx[i]; [lo:hi) is the block this rank
    owns. sparse_comms_start built send_idxes as cat(full_idx[:lo], full_idx[hi:]), so the
    payload is the same two slices of `compact` -- a contiguous copy, never a gather."""
    d = compact.shape[1]
    send_vals = torch.cat([compact[:lo], compact[hi:]]).to(dtype)

    send_sizes = [i * d for i in send_counts]
    recv_sizes = [i * d for i in recv_counts]

    recv_vals = torch.empty(sum(recv_sizes), device=compact.device, dtype=dtype)

    val_fut = dist.all_to_all_single(
        recv_vals,
        send_vals.view(-1),
        input_split_sizes=send_sizes,
        output_split_sizes=recv_sizes,
        async_op=True,
    ).get_future()

    return recv_vals, val_fut

@torch.no_grad
def bgsp_merge_gradients(shard, compact, full_idx, lo, hi, recv_idx, recv_vals, rank, world):
    """BIGRAM_SPARSE_GRAD twin of sparse_comms_merge_gradients: merge into a persistent DENSE SHARD
    buffer [N/world, d] instead of the full [N, d] table. Own-rank rows come straight from
    the compact buffer (they were never sent), received rows are index_add'd on top. Both
    index sets are global row ids inside our shard, so they shift by -rank*rows_per_rank."""
    d = shard.shape[1]
    base = rank * shard.shape[0]
    shard.zero_()
    if hi > lo:
        shard.index_add_(0, full_idx[lo:hi] - base, compact[lo:hi].to(shard.dtype))
    shard.index_add_(0, recv_idx - base, recv_vals.view(-1, d).to(shard.dtype))
    return shard.mul_((1 / world))


# -----------------------------------------------------------------------------
# Combined NorMuon + Adam Optimizer

@dataclass(slots=True)
class ParamConfig:
    """Per-parameter configuration for AnvilAndAdam optimizer."""
    label: str
    optim: str  # "adam" or "anvil"
    comms: str  # "none", "replicated", "sharded" or "sharded_sparse"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    # Adam-specific
    eps: float | None = None
    # NorMuon-specific
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


class AnvilAndAdam:
    """
    Combined optimizer: ANVIL for the banked 2D projection matrices, Adam for
    embeddings/scalars/gate weights.

    ANVIL (Averaged, Normalized Velocity with Isotropic Lanes) is our rework of
    the Muon/NorMuon optimizer stack used by the current record. Lineage of each
    component is noted inline; the re-derived cascade coefficient schedule and
    the bank tail-blend ship are new in this submission.

    - Twin-rail velocity: two EMA rails of the reduced gradient held in one
      [2, *chunk] fp32 state tensor - a fast rail on the scheduled beta and a
      slow rail on a constant long-horizon beta - blended into a single
      Nesterov-lookahead velocity estimate. (Dual-timescale momentum after
      modded-nanogpt PR #339 "bi-Maxwell"; the packed single-tensor state and
      scalar-tensor scheduling are ours.)
    - Momentum whitening: a cascade of quintic spectral maps on the velocity
      Gram drives every singular value toward 1 - the orthogonalized-momentum
      idea of Muon (Keller Jordan, https://kellerjordan.github.io/posts/muon/),
      via the Polar Express iteration family (Amsel, Persson, Musco, Gower,
      https://arxiv.org/pdf/2505.16932). The six coefficient triples here are
      re-derived from scratch (CEM + minimax polish; see ANVIL_MAPS), and the
      Gram-trace normalization reuses the first Gram. Runs stably in bf16.
    - Per-lane energy equalization: an EMA of per-row/column update energy
      rescales lanes to equal RMS while preserving the chunk's Frobenius norm
      (the NorMuon low-rank variance rescale, https://arxiv.org/pdf/2510.05491,
      in Adafactor's factored-second-moment family).
    - Sign-aligned decay: decoupled weight decay applied only where it agrees
      with the update direction (cautious weight decay, the gated variant of
      decoupled decay from the cautious-optimizer line).
    - Mantissa-extended updates: bf16 parameters carry a uint16 mantissa sidecar
      so updates accumulate with full fp32 precision (from the record lineage).

    Whitened-velocity updates suit only the 2D projection matrices in the
    attention and MLP layers; embeddings, scalars, and individual weight vectors
    (e.g., bias terms or gate weights) use Adam instead:
    - Standard Adam with bias correction
    - The same sign-aligned decay gate

    Lineage: the whitening cascade family is related to Newton-Schulz/polar methods.

    Configuration:
    Unlike torch.optim.Optimizer, this class uses per-parameter configs from a `param_table` dict
    and does not include parameter "groups". All parameters require a .label attribute, and a
    corresponding entry in the param_table to specify their hyperparameters (lr_mul, wd_mul, adam_betas, etc.).

    Communication and ordering:
    Gradient communication is explicitly scheduled rather than hook-driven.
    Reductions are launched in `scatter_order`, while update math and final
    gathers are executed in `work_order`. These orders are independent and
    must each contain every parameter label exactly once.

    Two communication modes are supported per parameter:
    - 'replicated': Gradients are all-reduced and each rank computes the full update.
    - 'sharded': Gradients are reduce-scattered, each rank updates its shard,
      and results are all-gathered.

    Adam parameters may be freely sharded. NorMuon operates on full matrices; sharding is
    supported by grouping matrices into parameter banks. NorMuon parameters must have a
    `.reshape` attribute that reshapes the bank so that the leading dimension is divisible
    by world_size.

    # Contributors include @YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
    # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
    """
    def __init__(self, named_params, param_table: dict, scatter_order: list, work_order: list,
                 adam_defaults: dict, anvil_defaults: dict):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Store defaults for each optimizer type
        self.adam_defaults = adam_defaults
        self.anvil_defaults = anvil_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        # Collect params by label and build config
        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table  # all params must have valid label
            assert label not in self._param_by_label  # exactly one param per label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        # Assert scatter_order and work_order match present labels exactly
        present = self._param_by_label.keys()
        assert set(scatter_order) == present and set(work_order) == present

        # Handle world_size=1: overwrite comms to "none"
        if self.world_size == 1:
            for p_cfg in self.param_cfgs.values():
                p_cfg.comms = "none"

        # Initialize state for all params
        self._init_state()

        # 0-D CPU tensors to avoid recompilation
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # DUAL_MOMENTUM shared 0-D CPU scalars (filled per step in step_optimizers):
        # fast-buffer beta (scheduled momentum pre-engage, fast_beta after) and the
        # fast/slow blend weight (1.0 pre-engage => stock trajectory, fast_w after).
        self._bimax_bf_t = torch.tensor(0.95, dtype=torch.float32, device="cpu")
        self._bimax_w_t = torch.tensor(1.0, dtype=torch.float32, device="cpu")

        # Track async operations
        self._reduce_futures: dict[nn.Parameter, tuple] = {}
        self._sparse_async_data: dict[nn.Parameter, list] = {}
        # Labels to skip entirely this step (no reduce, no update, no gather, and the
        # gradient is left in place so it accumulates into the next update).
        # Empty unless KX_VEPER is active. Set by TrainingManager before step().
        self._skip_labels: set = set()

        # Embed/lm_head tying state
        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

        # ---- KX_TRISYNC: local_kavg params (trigram_embed FedAvg mode) ----
        # No grad comms for these; Adam runs on the rank-LOCAL gradient at full
        # parameter extent, and the WEIGHT is all_reduce(AVG)'d every KX_TRISYNC
        # steps (flag set per step by TrainingManager.step_optimizers) plus a hard
        # guard before any val eval in the main loop. Moments stay rank-local.
        self._trisync_now = False
        self._lkavg_params: list[nn.Parameter] = [
            p for p in self.param_cfgs if self.param_cfgs[p].comms == "local_kavg"]
        if self._lkavg_params and _sov_labels:
            assert not any(self.param_cfgs[p].label in _sov_labels for p in self._lkavg_params), \
                "KX_TRISYNC: local_kavg labels must not run on the KX_SOV side stream"

        # Fused flat all_reduce for small replicated params: one NCCL call per grad
        # dtype instead of ~13 latency-bound ones. Elementwise AVG per param unchanged.
        self._far_params: list[nn.Parameter] = []
        self._far_flats: dict[torch.dtype, Tensor] = {}
        self._far_views: dict[nn.Parameter, Tensor] = {}
        # BIGRAM_SPARSE_GRAD: this step's compact-gradient payload, published by
        # TrainingManager.bgsp_build() before step() runs (None on non-Adam steps).
        self._bgsp_state: dict | None = None
        if self.world_size > 1:
            self._far_params = [p for p in self.param_cfgs
                                if self.param_cfgs[p].comms == "replicated"]
            by_dt: dict[torch.dtype, list[nn.Parameter]] = {}
            for p in self._far_params:
                by_dt.setdefault(p.dtype, []).append(p)
            for dt, ps in by_dt.items():
                total = sum(p.numel() for p in ps)
                flat = torch.zeros(total, dtype=dt, device=device)
                self._far_flats[dt] = flat
                off = 0
                for p in ps:
                    self._far_views[p] = flat[off:off + p.numel()].view(p.shape)
                    off += p.numel()

        # ---- [AFE]: fused foreach path for replicated-comms Adam params ----
        # Static label lists (grouped by everything that feeds the update scalars);
        # per-step scalars (lr/wd schedule, step count) are re-derived inside
        # _afe_run so the values match the stock per-param path exactly.
        # Wall fix: the builder below was missing (groups stayed empty and
        # _afe_run was unreachable). One group per (adam_betas, lr_mul, wd_mul,
        # eps, dtype) over the far (replicated-comms) Adam params -- exactly the
        # inputs _afe_run reads from cfg0 for a whole partition; the per-STEP
        # scalars (step count, scheduled lr, weight_decay) are sub-partitioned
        # inside _afe_run itself, and its st["step"] += 1 bump matches
        # _adam_update's placement one-for-one. Group and partition order follow
        # param registration order (identical on every rank). The G_tiny
        # CUDA-graph branch in step() keeps precedence when armed: it
        # `continue`s before the AFE branch is reached, so each far param is
        # handled by exactly one of the two paths per step.
        self._afe_groups: list = []
        self._afe_params: set = set()
        self._afe_done = False
        if self._far_params:
            _afe_by_key: dict[tuple, list] = {}
            for p in self._far_params:
                _cfg = self.param_cfgs[p]
                if _cfg.optim != "adam":
                    continue  # far list is Adam-only in this table; guard anyway
                _afe_by_key.setdefault(
                    (_cfg.adam_betas, _cfg.lr_mul, _cfg.wd_mul, _cfg.eps, p.dtype), []
                ).append(p)
            self._afe_groups = list(_afe_by_key.items())
            self._afe_params = {p for _k, _ps in self._afe_groups for p in _ps}

        # ---- Wall fix: per-matrix eff_lr staging for the mlp eager path ----
        # Replaces chunk_size sequential _sign_aligned_decay_update launches
        # (each behind a 0-D fill_) with ONE chunk-wide launch. The per-matrix
        # eff_lr products are written with the exact fill_ rounding (numpy fp32
        # store of the same python-double product) into a pinned RING slot and
        # copied H2D non_blocking OUTSIDE the compiled call; per-slot CUDA
        # events keep the run-ahead host from rewriting a slot before its DMA
        # executed (same WAR-race discipline as _cg_stage). The CUDA-graph path
        # (per_matrix_lr_ts) is untouched.
        self._pml_dev: dict[str, Tensor] = {}
        self._pml_host: dict[str, Tensor] = {}
        self._pml_hostnp: dict = {}
        self._pml_evts: dict[str, list] = {}
        self._pml_ring: dict[str, int] = {}
        for p, _cfg in self.param_cfgs.items():
            if _cfg.optim == "anvil" and _cfg.per_matrix_lr_mul is not None:
                _PML_RING = 8
                self._pml_dev[_cfg.label] = torch.zeros(
                    _cfg.chunk_size, 1, 1, dtype=torch.float32, device=p.device)
                _pml_h = torch.zeros(_PML_RING, _cfg.chunk_size, dtype=torch.float32, pin_memory=True)
                self._pml_host[_cfg.label] = _pml_h
                self._pml_hostnp[_cfg.label] = _pml_h.numpy()
                self._pml_evts[_cfg.label] = [torch.cuda.Event() for _ in range(_PML_RING)]
                self._pml_ring[_cfg.label] = 0

        # ---- CUDA_GRAPH_TIER: CUDA-graph plumbing (inert until _cg_build_graphs() arms it) ----
        self._cg_active = False          # True only after at least one bank/tiny capture
        self._cg_graphs: dict = {}       # label -> CUDAGraph for G_qk/G_vo/G_mlp
        self._cg_tiny = None             # CUDAGraph for the replicated-Adam block
        self._cg_tiny_params: list = []  # far params covered by G_tiny (work_order order)
        self._cg_tiny_done = False       # per-step latch (G_tiny replays once per odd step)
        self._cg_rs_bufs: dict = {}      # label -> persistent reduce_scatter output (stable address)
        self._cg_bank_labels: list = []  # labels with staged scalars (set by builder)
        self._cg_ptrs: list = []         # (name, tensor, data_ptr at capture) for the rebind assert
        self._cg_ptr_checked = False

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        """Build config for a single parameter from param_table."""
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        if comms == "sharded_sparse" and not _sparse_comms_active():
            comms = "sharded"
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms.startswith("sharded") else None
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                chunk_size=chunk_size,
            )
        elif optim == "anvil":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"ANVIL param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            # Shape-based LR multiplier for NorMuon
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul
            if label == "mlp_bank":
                # mlp_bank lr trim multiplier (the bank carries both the largest shape lr,
                # 4x on c_proj, and the fp8 wgrad quantization noise floor). 1.0 = neutral.
                lr_mul = lr_mul * float("1.0")
            if label == "qk_bank":
                # qk_bank lr multiplier. 1.0 = stock.
                lr_mul = lr_mul * float("1.0")

            # Per-matrix LR multipliers for MLP c_proj (2x LR on odd indices)
            per_matrix_lr_mul = None
            if label == "mlp_bank":
                rank = dist.get_rank() if dist.is_initialized() else 0
                start_idx = rank * chunk_size
                per_matrix_lr_mul = []
                for i in range(chunk_size):
                    global_idx = start_idx + i
                    is_c_proj = (global_idx % 2 == 1)
                    per_matrix_lr_mul.append(2.0 if is_c_proj else 1.0)

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.anvil_defaults["lr"],
                initial_lr=self.anvil_defaults["lr"],
                weight_decay=self.anvil_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.anvil_defaults["momentum"],
                beta2=self.anvil_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        """Initialize optimizer state for all parameters."""
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                # Sharded params use chunk state, replicated use full state
                if p_cfg.comms.startswith("sharded"):
                    chunk = param[:p_cfg.chunk_size]
                else:
                    chunk = param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "anvil":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])

                # Twin-rail velocity state (fp32 for precision): one allocation of
                # shape [2, *chunk]. Rail 0 is the fast rail (scheduled beta) and
                # is the whole velocity estimate until the dual-timescale blend
                # engages; rail 1 is the slow rail (constant beta), zero-init,
                # accumulating from step 0 so it is warm at engage_step. A single
                # tensor keeps one stable data_ptr for the CUDA-graph captures.
                velocity = torch.zeros(
                    (2, *chunk_shape), dtype=torch.float32, device=param.device
                )

                # Second momentum buffer - reduced along one dimension
                if chunk_shape[-2] >= chunk_shape[-1]:
                    lane_shape = (*chunk_shape[:-1], 1)
                else:
                    lane_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                lane_energy = torch.zeros(
                    lane_shape, dtype=torch.float32, device=param.device
                )

                # Mantissa buffer for precision tracking
                mantissa = torch.zeros(
                    chunk_shape, dtype=torch.uint16, device=param.device
                )

                self.param_states[param] = dict(
                    velocity=velocity,
                    lane_energy=lane_energy,
                    mantissa=mantissa,
                )

    # -----------------------------------
    # Reduce/Gather operations

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        """Launch async reduce for a parameter based on its comms policy."""
        p_cfg = self.param_cfgs[param]

        if p_cfg.comms == "none":
            if p_cfg.optim == "anvil":
                # ANVIL needs reshaped gradient even without communication
                grad = grad.view(p_cfg.reshape)
            self._reduce_futures[param] = (None, grad)
        elif p_cfg.comms == "replicated":
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
        elif p_cfg.comms == "sharded":
            if p_cfg.optim == "anvil":
                # ANVIL: reshape before reduce_scatter
                grad_reshaped = grad.view(p_cfg.reshape)
                # CUDA_GRAPH_TIER: reduce into the persistent stable-address buffer the bank
                # graphs were captured against (same contents either way; the dtype
                # guard falls back to the stock per-step alloc if grads ever change dtype).
                grad_chunk = self._cg_rs_bufs.get(p_cfg.label)
                if grad_chunk is None or grad_chunk.dtype != grad.dtype:
                    grad_chunk = torch.empty(
                        (p_cfg.chunk_size, *grad_reshaped.shape[1:]),
                        dtype=grad.dtype,
                        device=grad.device
                    )
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad_reshaped.contiguous(), op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                # Adam: simple reduce_scatter
                grad_chunk = torch.empty_like(grad[:p_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
        elif p_cfg.comms == "replicated_full":
            # REPLICATED_FULL_ADAM: full-extent grad, one async AVG all_reduce; replicated Adam in
            # Phase 2 runs identical math on every rank -- no gather.
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
        elif p_cfg.comms == "local_kavg":
            # KX_TRISYNC: NO dist op on the gradient. The Adam update consumes the
            # rank-local grad at full parameter extent (the replicated path minus
            # its all_reduce). Weight averaging happens on the K-step cadence in
            # step() and via the pre-val guard in the main loop.
            self._reduce_futures[param] = (None, grad)
        elif p_cfg.comms == "sharded_sparse":
            sparse_state = self._sparse_async_data[param]
            send_idxes = sparse_state["send_idxes"]
            send_counts = sparse_state["send_counts"]
            recv_counts = sparse_state["recv_counts"]
            if BIGRAM_SPARSE_GRAD and p_cfg.label == "bigram_embed":
                # BIGRAM_SPARSE_GRAD: payload is two contiguous slices of the compact buffer.
                st = self._bgsp_state
                recv_vals, val_fut = bgsp_share_gradients(
                    st["compact"], st["lo"], st["hi"], send_counts, recv_counts, st["dtype"]
                )
            else:
                recv_vals, val_fut = sparse_comms_share_gradients(
                    grad, send_idxes, send_counts, recv_counts
                )
            self._reduce_futures[param].extend((val_fut, recv_vals))

    def _bgsp_merge(self, recv_idxes, recv_vals, rank) -> Tensor:
        """BIGRAM_SPARSE_GRAD merge: own rows straight from the compact buffer + received rows, into
        the persistent dense SHARD buffer. Returns the same grad_chunk the stock path
        returns (dense [chunk_size, d], already scaled by 1/world) for the stock Adam."""
        st = self._bgsp_state
        return bgsp_merge_gradients(
            st["shard"], st["compact"], st["full_idx"], st["lo"], st["hi"],
            recv_idxes, recv_vals, rank, self.world_size,
        )

    def _launch_gather(self, param: nn.Parameter, p_slice: Tensor) -> "torch.futures.Future":
        """Launch async all_gather for a sharded parameter."""
        p_cfg = self.param_cfgs[param]
        if p_cfg.optim == "anvil":
            full_param = param.data.view(p_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(
                full_param, p_slice.contiguous(), async_op=True
            ).get_future()
        else:
            return dist.all_gather_into_tensor(
                param, p_slice.contiguous(), async_op=True
            ).get_future()

    # -----------------------------------
    # State management

    def reset(self):
        """Reset NorMuon momentum buffers and split_embed state (called on training reset)."""
        self.split_embed = False
        self._skip_labels = set()
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "anvil":
                # In-place zeroing only: every tensor here may be baked into a
                # captured CUDA graph, so its data_ptr must never change.
                p_state = self.param_states[param]
                p_state["velocity"].zero_()  # both rails
                p_state["mantissa"].zero_()
                p_state["lane_energy"].zero_()

    def copy_lm_state_to_embed(self):
        """
        Copy the optimizer state from the lm_head to the embed at the untie point.
        This requires an all-gather + reshard because of different sharding:
        - lm_head (768, 50304) is sharded to (96, 50304) per rank (along model_dim)
        - embed (50304, 768) is sharded to (6288, 768) per rank (along vocab_size)

        We all-gather the lm_head momentum, transpose it, then each rank takes their
        embed shard to get the correct momentum state.
        """
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]
        lm_cfg = self.param_cfgs[lm_head]
        embed_cfg = self.param_cfgs[embed]

        embed_state['step'] = lm_state['step'] # Preserve step count for bias correction

        # Copy optimizer state with all-gather + transpose + reshard
        if REPLICATED_FULL_ADAM:
            # REPLICATED_FULL_ADAM: both states are full-extent -- plain transpose, no reshard.
            for key in ["exp_avg", "exp_avg_sq"]:
                embed_state[key].copy_(lm_state[key].T)

        # Mark as split
        self.split_embed = True

    def state_dict(self):
        """Return the optimizer state as a dict."""
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from a dict."""
        # Build id->param mapping
        id_to_param = {id(p): p for p in self.param_cfgs}

        # Load state, preserving dtypes
        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        # In-place restore when shapes match (values identical to the old
                        # rebinding `.to()` path -- copy_ performs the same dtype/device
                        # conversion). Keeping the original tensor object preserves its
                        # data_ptr, which CUDA_GRAPH_TIER's captured graphs bake in (the
                        # `.to()` of a matching deepcopy returns the deepcopy itself and
                        # silently rebinds every optimizer state address at reset).
                        if isinstance(p_state[k], torch.Tensor) and p_state[k].shape == v.shape:
                            p_state[k].copy_(v)
                        else:
                            target_dtype = p_state[k].dtype
                            p_state[k] = v.to(dtype=target_dtype, device=p_state[k].device)
                    else:
                        p_state[k] = v

    # -----------------------------------
    # CUDA_GRAPH_TIER helpers (CUDA-graph capture of the optimizer compute segments).
    # All of this is dead code until _cg_build_graphs() arms it.

    def _cg_stage(self, do_adam: bool):
        """Write every graph-consumed scalar into a pinned staging slot and issue the
        single non_blocking H2D copy for this step. Every graphed consumer
        owns its own device slot -- the four shared 0-D CPU tensors are never read by
        any graph (the shared-mutable-scalar trap).

        RING + EVENTS: the host in this trainer runs several
        steps ahead of the GPU (every future.wait() is a stream wait --
        same reason the HOST_OPT data path needs a pinned ring). A single pinned buffer is
        therefore a WAR race: cudaMemcpyAsync reads host memory at DMA *execution*
        time, by which point the run-ahead host had already overwritten it with step
        N+k's scalars -- every graph consumed a future/torn schedule (momentum
        warmup/cooldown, lr/wd cooldown, lambda), a systematic loss offset with zero
        timing signature. (Eager was immune: CPU 0-D fill_ + .item() are consumed
        host-side at enqueue.) Each step now writes its own ring slot, guarded by a
        CUDA event recorded after that slot's copy: the slot cannot be rewritten until
        its previous DMA has executed, so the GPU always reads exactly the values
        staged for that step. sync on a never-recorded event is an immediate no-op;
        the host only ever blocks if it runs > _CG_RING optimizer steps ahead."""
        slot = self._cg_ring
        self._cg_ring = (slot + 1) % len(self._cg_evts)
        self._cg_evts[slot].synchronize()  # DMA issued RING steps ago has read this slot
        h = self._cg_hostnp[slot]
        s = self._cg_slot
        for label in self._cg_bank_labels:
            cfg = self.param_cfgs[self._param_by_label[label]]
            # numpy stores round exactly like fill_(python_float): double -> fp32
            h[s[label + ".mom"]] = cfg.momentum
            h[s[label + ".wd"]] = cfg.wd_mul * cfg.weight_decay * cfg.lr
            if cfg.per_matrix_lr_mul is None:
                h[s[label + ".lr"]] = cfg.lr_mul * cfg.lr
            else:
                for i, m in enumerate(cfg.per_matrix_lr_mul):
                    h[s[f"{label}.lr{i}"]] = cfg.lr_mul * m * cfg.lr
        if do_adam and self._cg_tiny is not None:
            for p in self._cg_tiny_params:
                cfg = self.param_cfgs[p]
                st = self.param_states[p]
                st["step"] += 1  # host-side bump moves here from _adam_update
                t = st["step"]
                beta1, beta2 = cfg.adam_betas
                lr = cfg.lr * cfg.lr_mul
                bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
                h[s[cfg.label + ".ss"]] = lr * (bias2 ** 0.5 / bias1)
                h[s[cfg.label + ".awd"]] = lr * lr * cfg.weight_decay * cfg.wd_mul
        self._cg_dev.copy_(self._cg_host[slot], non_blocking=True)
        self._cg_evts[slot].record()  # completes when this slot's DMA has executed

    def _cg_record_ptrs(self, named_tensors):
        """Remember capture-time data_ptrs so _cg_assert_ptrs can detect rebinds."""
        for name, t in named_tensors:
            self._cg_ptrs.append((name, t, t.data_ptr()))

    def _cg_assert_ptrs(self):
        """Verify that no captured state tensor was rebound between
        capture and first replay (load_state_dict was the known offender; the copy_
        fix above plus capture-after-reset placement should make this a no-op).
        Checked exactly once, on the first replay of any graph."""
        for name, t, ptr in self._cg_ptrs:
            assert t.data_ptr() == ptr, (
                f"[cg] captured tensor '{name}' was rebound after capture "
                f"(data_ptr {t.data_ptr():#x} != captured {ptr:#x}); replaying would "
                f"write to orphaned memory -- aborting")
        self._cg_ptr_checked = True

    # -----------------------------------
    # Unified optimizer step with explicit ordering

    @torch.no_grad()
    def step(self, do_adam: bool = True):
        """
        Combined optimizer step with explicit ordering.

        Args:
            do_adam: If True, update Adam params. NorMuon params always updated.

        Flow:
        1. Scatter phase: Launch reduces in scatter_order
        2. Work phase: Process updates in work_order
           - Wait for reduce, compute update, launch gather
        3. Finalize phase: Wait for gathers

        While the embeddings are tied:
        - Comms and update math are only done on lm_head.
        - We add embed.grad.T into lm_head.grad before comms.
        - After lm_head gather, we copy lm_head.data.T --> embed.data
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        lm_param, embed_param = self._lm_head_param, self._embed_param

        # ===== CUDA_GRAPH_TIER: stage all host scalars in one pinned->device copy =====
        self._cg_tiny_done = False
        self._afe_done = False
        if self._cg_active:
            self._cg_stage(do_adam)

        # ===== KX_TRISYNC: K-step FedAvg of local_kavg weights (async launch) =====
        # Launched at the top of the optimizer phase so it overlaps with the grad
        # reduces and small-param work; awaited immediately BEFORE the table's own
        # Adam update in Phase 2 (the averaged weight must land before the local
        # update writes it), or in Phase 3 if no update runs this step. NCCL
        # enqueue is stream-ordered after all prior compute, so the last write to
        # the weight (previous Adam step) is already visible.
        _trisync_futs = None
        if self._trisync_now and self._lkavg_params and self.world_size > 1:
            _trisync_futs = [
                dist.all_reduce(p.data, op=dist.ReduceOp.AVG, async_op=True).get_future()
                for p in self._lkavg_params]

        # ===== Phase 0: fused flat all_reduce for replicated params =====
        if self._far_params and do_adam:
            futs: dict[torch.dtype, object] = {}
            # One foreach dispatch instead of a python loop of copy_ calls.
            # Same views, same sources, same per-element copies -- bit-identical.
            _far_pairs = [(self._far_views[p], p.grad)
                          for p in self._far_params if p.grad is not None]
            if _far_pairs:
                torch._foreach_copy_([v for v, _ in _far_pairs],
                                     [g for _, g in _far_pairs])
            for dt, flat in self._far_flats.items():
                futs[dt] = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True).get_future()
            for p in self._far_params:
                if p.grad is not None:
                    self._reduce_futures[p] = (futs[p.dtype], self._far_views[p])

        # ===== Phase 1: Launch reduces in scatter_order =====
        for label in self.scatter_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]

            if p_cfg.optim == "adam" and not do_adam:
                continue
            if label in self._skip_labels:
                continue  # KX_VEPER: keep accumulating this param's grad, no comms
            if param.grad is None:
                # BIGRAM_SPARSE_GRAD: the bigram table deliberately has NO autograd grad (its lookup
                # is detached); this step's payload lives in the compact buffer instead.
                if not (BIGRAM_SPARSE_GRAD and label == "bigram_embed" and self._bgsp_state is not None):
                    continue
            if param in self._far_views and param in self._reduce_futures:
                continue  # already handled by the fused flat reduce

            # lm_head when tied: aggregate embed.grad.T (tiled Triton transpose-add)
            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    transpose_add(embed_param.grad, param.grad)

            # Skip embed when tied (copied from lm_head after gather)
            if label == "embed" and not self.split_embed:
                continue

            self._launch_reduce(param, param.grad)

        # ===== Phase 2: Process updates in work_order =====
        gather_futures = []
        lm_head_gather_future = None

        for label in self.work_order:
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue

            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            # Wait for reduce
            if p_cfg.comms != "sharded_sparse":
                future, grad_chunk = self._reduce_futures[param]
                if future is not None:
                    future.wait()
            else:
                idxes_fut, recv_idxes, recv_fut, recv_vals = self._reduce_futures[param]
                idxes_fut.wait()
                recv_fut.wait()

                grad_chunk = self._bgsp_merge(recv_idxes, recv_vals, rank) if (BIGRAM_SPARSE_GRAD and label == "bigram_embed") \
                    else sparse_comms_merge_gradients(param.grad, recv_idxes, recv_vals, rank, world_size)

            # Apply update based on optim type
            if p_cfg.optim == "adam":
                # CUDA_GRAPH_TIER G_tiny: one replay covers every replicated (FLAT_ALLREDUCE) Adam param.
                # Their updates are mutually independent, so batching them at the first
                # far param in work_order is compute-order-neutral and bitwise identical.
                if self._cg_tiny is not None and do_adam and param in self._far_views:
                    if not self._cg_tiny_done:
                        waited = set()
                        for fp in self._cg_tiny_params:
                            tup = self._reduce_futures.get(fp)
                            # The graph updates ALL far params; a far param without a
                            # registered reduce this step would silently consume a stale
                            # gradient -- fail loudly instead.
                            assert tup is not None, "[cg] G_tiny: far param missing its grad/reduce this step"
                            if tup[0] is not None and id(tup[0]) not in waited:
                                tup[0].wait()  # stream-wait; idempotent per dtype-future
                                waited.add(id(tup[0]))
                        if not self._cg_ptr_checked:
                            self._cg_assert_ptrs()
                        self._cg_tiny.replay()
                        self._cg_tiny_done = True
                    continue  # replicated: no gather; state["step"] bumped in _cg_stage
                # [AFE]: one fused foreach pass covers every replicated Adam param.
                # Same batching argument as G_tiny above: the updates are mutually
                # independent, so running them all at the first such param in
                # work_order is compute-order-neutral. Reduce futures for the whole
                # set are waited inside _afe_run (stream-waits, idempotent).
                if self._afe_params and param in self._afe_params:
                    if not self._afe_done:
                        self._afe_run()
                        self._afe_done = True
                    continue  # replicated: no gather
                if _trisync_futs is not None and p_cfg.comms == "local_kavg":
                    # KX_TRISYNC: the averaged table must land before this step's
                    # rank-local Adam update writes the weight (stream-wait).
                    for _f in _trisync_futs:
                        _f.wait()
                    _trisync_futs = None
                p_slice = self._adam_update(param, grad_chunk, p_cfg, rank)
            else:
                _cg_g = self._cg_graphs.get(label) if self._cg_graphs else None
                if _cg_g is not None:
                    # The graph was captured against the persistent RS buffer; the eager
                    # future.wait() above has already stream-ordered its contents.
                    assert grad_chunk is self._cg_rs_bufs[label]
                    if not self._cg_ptr_checked:
                        self._cg_assert_ptrs()
                    _cg_g.replay()
                    param_view = param.data.view(p_cfg.reshape)
                    p_slice = param_view[rank * p_cfg.chunk_size:(rank + 1) * p_cfg.chunk_size]
                else:
                    p_slice = self._anvil_update(param, grad_chunk, p_cfg, rank)
            # Launch gather for sharded params
            if p_cfg.comms.startswith("sharded") and self.world_size > 1:
                gather_fut = self._launch_gather(param, p_slice)
                if label == "lm_head":
                    lm_head_gather_future = gather_fut
                else:
                    gather_futures.append(gather_fut)

        # ===== Phase 3: Wait for gathers, sync embed if tied =====
        # Wait for lm_head gather first so we can copy to embed while other gathers complete
        if lm_head_gather_future is not None:
            lm_head_gather_future.wait()
        if getattr(self, "_sov_lm_fut", None) is not None:
            self._sov_lm_fut.wait()
            self._sov_lm_fut = None

        # When tied: copy lm_head.T to embed (tiled Triton transpose for coalesced writes)
        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            transpose_copy(lm_param.data, embed_param.data)

        # Wait for remaining gathers
        for fut in gather_futures:
            fut.wait()

        # KX_TRISYNC backstop: sync step where the local_kavg param took no Adam
        # update this step (non-adam step or no grad) -- still complete the weight
        # average before the next forward reads the table.
        if _trisync_futs is not None:
            for _f in _trisync_futs:
                _f.wait()

        self._reduce_futures.clear()
        self._sparse_async_data.clear()

        # Clear grads for updated params
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue  # Don't clear Adam grads on even steps
            if p_cfg.label in self._skip_labels:
                continue  # KX_VEPER: this param did not update -> keep accumulating
            param.grad = None

    def _sov_run(self, rank, do_adam, gather_futures):
        """[SOV] side-stream twin of the Phase-2 work loop for the _sov_labels set."""
        # [SOV] copy of the inline work-item path for the side-stream label set.
        # record_stream pins backward-produced grads for the caching allocator.
        for label in self.work_order:
            if label not in _sov_labels:
                continue
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue
            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is not None:
                param.grad.record_stream(torch.cuda.current_stream())
            if p_cfg.comms != "sharded_sparse":
                future, grad_chunk = self._reduce_futures[param]
                if future is not None:
                    future.wait()
            else:
                idxes_fut, recv_idxes, recv_fut, recv_vals = self._reduce_futures[param]
                idxes_fut.wait()
                recv_fut.wait()
                grad_chunk = self._bgsp_merge(recv_idxes, recv_vals, rank) if (BIGRAM_SPARSE_GRAD and label == "bigram_embed") \
                    else sparse_comms_merge_gradients(param.grad, recv_idxes, recv_vals, rank, world_size)
            if p_cfg.optim == "adam":
                p_slice = self._adam_update(param, grad_chunk, p_cfg, rank)
            else:
                p_slice = self._anvil_update(param, grad_chunk, p_cfg, rank)
            if p_cfg.comms.startswith("sharded") and self.world_size > 1:
                gf = self._launch_gather(param, p_slice)
                if label == "lm_head":
                    self._sov_lm_fut = gf
                else:
                    gather_futures.append(gf)

    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        # Get parameter slice
        if p_cfg.comms.startswith("sharded"):
            p_slice = param[rank * p_cfg.chunk_size:(rank + 1) * p_cfg.chunk_size]
        else:
            p_slice = param

        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]

        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        AnvilAndAdam._adam_update_step(
            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )

        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Compiled Adam update step."""
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    def _afe_run(self):
        """[AFE] Fused replicated-Adam update: one torch._foreach_* sequence per
        (adam_betas, lr_mul, wd_mul, eps, dtype) group instead of a compiled-region
        entry + two 0-D fill_s per param. The elementwise math per param is the
        _adam_update/_adam_update_step sequence verbatim (EMA updates, bias-corrected
        step size, sign-aligned decay mask, final add). Params without a reduce this step
        (e.g. KX_VEPER skip) are left out exactly like the stock
        `param not in self._reduce_futures` guard; any resulting divergence in step
        count or schedule scalars is handled by sub-partitioning, never approximated."""
        for _key, group in self._afe_groups:
            # Partition by the per-step scalar inputs so every foreach call uses one
            # exact (step_size, eff_wd) pair, matching the per-param fill_ values.
            parts: dict[tuple, list] = {}
            for p in group:
                tup = self._reduce_futures.get(p)
                if tup is None:
                    continue  # no grad/reduce this step: stock path skips it too
                future, grad = tup
                if future is not None:
                    future.wait()  # stream-wait; idempotent per shared future
                cfg = self.param_cfgs[p]
                st = self.param_states[p]
                st["step"] += 1
                parts.setdefault((st["step"], cfg.lr, cfg.weight_decay), []).append((p, grad, st))
            for (t, lr_base, wd_base), items in parts.items():
                cfg0 = self.param_cfgs[items[0][0]]
                beta1, beta2 = cfg0.adam_betas
                lr = lr_base * cfg0.lr_mul
                bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
                step_size = lr * (bias2 ** 0.5 / bias1)
                eff_wd = lr * lr * wd_base * cfg0.wd_mul
                params = [p.data for p, _, _ in items]
                grads = [g for _, g, _ in items]
                exp_avgs = [s["exp_avg"] for _, _, s in items]
                exp_avg_sqs = [s["exp_avg_sq"] for _, _, s in items]
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denom, cfg0.eps)
                update = torch._foreach_div(exp_avgs, denom)
                torch._foreach_mul_(update, step_size)
                # Sign-aligned decay gate: mask = (update * p) > 0. No foreach
                # comparison op exists, so the mask stays per-tensor (cheap eager
                # launches; the compiled-region entries are what AFE removes).
                masks = [torch.gt(u * pd, 0) for u, pd in zip(update, params)]
                torch._foreach_addcmul_(update, params, masks, value=eff_wd)
                torch._foreach_add_(params, update, alpha=-1.0)

    # -----------------------------------
    # NorMuon update

    def _anvil_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        """Apply the ANVIL update to a parameter. Returns the updated p_slice.

        Eager wrapper: fills the shared 0-D CPU scalar tensors, then runs the compute
        segment. CUDA_GRAPH_TIER captures `_anvil_compute` directly,
        against per-bank 0-D CUDA views of the staging vector instead of the shared
        CPU tensors (a graph must never read a shared mutable scalar)."""
        self._momentum_t.fill_(p_cfg.momentum)
        self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.lr)
        self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)

        pml_lr_dev = None
        if p_cfg.per_matrix_lr_mul is not None:
            # Stage this step's per-matrix eff_lr vector (see __init__).
            # numpy fp32 stores round python doubles exactly like fill_, so each
            # matrix sees the same fp32 lr value the old per-matrix fill_ loop
            # produced; the event ring makes the pinned rewrite WAR-safe under
            # host run-ahead.
            _slot = self._pml_ring[p_cfg.label]
            self._pml_ring[p_cfg.label] = (_slot + 1) % len(self._pml_evts[p_cfg.label])
            _evt = self._pml_evts[p_cfg.label][_slot]
            _evt.synchronize()  # the DMA issued RING uses ago has read this slot
            _h = self._pml_hostnp[p_cfg.label]
            for _i, _m in enumerate(p_cfg.per_matrix_lr_mul):
                _h[_slot, _i] = p_cfg.lr_mul * _m * p_cfg.lr
            pml_lr_dev = self._pml_dev[p_cfg.label]
            pml_lr_dev.copy_(self._pml_host[p_cfg.label][_slot].view(-1, 1, 1),
                             non_blocking=True)
            _evt.record()  # completes when this slot's DMA has executed

        return self._anvil_compute(
            param, grad_chunk, p_cfg, rank,
            self._momentum_t, self._eff_lr_t, self._eff_wd_t, None,
            pml_lr_dev=pml_lr_dev)

    def _anvil_compute(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig,
                         rank: int, momentum_t, eff_lr_t, eff_wd_t,
                         per_matrix_lr_ts, pml_lr_dev=None) -> Tensor:
        """Collective-free ANVIL compute segment (the CUDA_GRAPH_TIER capture unit).

        momentum_t / eff_lr_t / eff_wd_t are 0-D fp32 scalar tensors: the shared CPU
        tensors on the eager path (pre-filled by the wrapper, stock behaviour), or
        per-bank CUDA views of _cg_dev under capture.
        per_matrix_lr_ts is None on the eager path (the shared eff_lr_t
        is re-filled per matrix, stock behaviour) or a list of 0-D CUDA views."""
        chunk_shape = grad_chunk.shape

        p_state = self.param_states[param]

        # The fp32 upcast for the velocity rails moved INSIDE the compiled
        # cascade (first line there), so inductor fuses it into the rail reads
        # instead of materializing an eager full-chunk fp32 copy here. The raw
        # reduced grad is never read again after the cascade (only v_chunk is).

        # Fused twin-rail velocity update + whitening cascade
        is_large_matrix = chunk_shape[-2] > 1024
        v_chunk = anvil_cascade(
            grad_chunk, p_state["velocity"], momentum_t,
            split_baddbmm=is_large_matrix,
            bimax_bf_t=self._bimax_bf_t if _DUAL_MOMENTUM is not None else None,
            bimax_w_t=self._bimax_w_t if _DUAL_MOMENTUM is not None else None,
        )

        # Variance reduction
        red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2
        v_chunk = AnvilAndAdam._rail_equalizer(
            v_chunk, p_state["lane_energy"], p_cfg.beta2, red_dim
        )

        # Update parameter, in place, with cautious weight decay
        param_view = param.data.view(p_cfg.reshape)
        p_slice = param_view[rank * p_cfg.chunk_size:(rank + 1) * p_cfg.chunk_size]

        # MLP has per-matrix LR multipliers (c_proj gets 2x LR)
        if p_cfg.per_matrix_lr_mul is not None:
            if per_matrix_lr_ts is None:
                # Eager path: ONE chunk-wide launch. pml_lr_dev is the
                # [chunk,1,1] fp32 CUDA vector staged by the wrapper -- the same
                # fp32 values the old per-matrix fill_ loop produced -- broadcast
                # per matrix inside the compiled kernel; per-element arithmetic
                # is identical. (The old loop's eff_wd_t re-fill is dropped: the
                # wrapper already filled the identical value.)
                AnvilAndAdam._sign_aligned_decay_update_mlr(
                    p_slice.view(torch.uint16), p_state["mantissa"], v_chunk,
                    eff_wd_t, pml_lr_dev
                )
            else:
                # Graphed path: the three launches read stable 0-D
                # views of the staging vector; no fills inside the capture.
                for mat_idx in range(p_cfg.chunk_size):
                    AnvilAndAdam._sign_aligned_decay_update(
                        p_slice[mat_idx].view(torch.uint16), p_state["mantissa"][mat_idx], v_chunk[mat_idx],
                        eff_wd_t, per_matrix_lr_ts[mat_idx]
                    )
        else:
            AnvilAndAdam._sign_aligned_decay_update(
                p_slice.view(torch.uint16), p_state["mantissa"], v_chunk,
                eff_wd_t, eff_lr_t
            )

        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _sign_aligned_decay_update(p, mantissa, grad, wd_tensor, lr_tensor):
        """
        Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors.
        Mantissa is tracked to enable higher precision updates on bfloat16 parameters.
        bfloat16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits total
        float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
        """
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        shadow_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        shadow = shadow_raw.view(torch.float32)  # aliases shadow_raw
        aligned = (grad * shadow) >= 0
        shadow.copy_(shadow - (shadow * aligned * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((shadow_raw >> 16).to(torch.uint16))
        mantissa.copy_(shadow_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _sign_aligned_decay_update_mlr(p, mantissa, grad, wd_tensor, lr_vec):
        """Chunk-wide variant of _sign_aligned_decay_update for banks with
        per-matrix learning rates (wall fix). lr_vec is a [chunk,1,1] fp32
        CUDA tensor staged outside this graph (pinned ring + non_blocking H2D);
        it broadcasts one lr value per matrix. Every per-element operation and
        its order match _sign_aligned_decay_update exactly -- the same fp32 lr
        value multiplies the same fp32 grad/shadow terms -- so the update is
        numerically identical to the old chunk_size sequential launches."""
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_vec  # fp32 already; one value per leading (matrix) index
        shadow_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        shadow = shadow_raw.view(torch.float32)  # aliases shadow_raw
        aligned = (grad * shadow) >= 0
        shadow.copy_(shadow - (shadow * aligned * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((shadow_raw >> 16).to(torch.uint16))
        mantissa.copy_(shadow_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _rail_equalizer(v_chunk, lane_energy, beta2, red_dim):
        """Per-lane energy equalization of the whitened velocity chunk.

        Each lane (a row or column - whichever runs along the LONGER matrix
        dimension, i.e. reduced over red_dim) tracks an EMA of its mean squared
        update in `lane_energy`. The update is rescaled by the inverse RMS of
        that energy - equalizing energy across lanes - and then renormalized so
        each matrix keeps its pre-equalization Frobenius norm. The two
        normalizations are fused algebraically to minimize full-size memory ops.
        """
        lane_power = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        lane_len = v_chunk.size(red_dim)
        pre_norm_sq = lane_power.sum(dim=(-2, -1), keepdim=True).mul_(lane_len)
        pre_norm = pre_norm_sq.sqrt_()
        lane_energy.lerp_(lane_power.to(dtype=lane_energy.dtype), 1 - beta2)
        lane_gain = lane_energy.clamp_min(1e-10).rsqrt_()
        post_power = (lane_power * lane_len) * lane_gain.float().square()
        post_norm = post_power.sum(dim=(-2, -1), keepdim=True).sqrt_()
        eq_scale = lane_gain * (pre_norm / post_norm.clamp_min_(1e-10))
        return v_chunk.mul_(eq_scale.type_as(v_chunk))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    """Half-truncated rotary embedding with YaRN rescaling at attention-window
    transitions. Precomputes bf16 cos/sin factor tables (paired variant packs two
    heads per row block over a doubled sequence); apply() rescales angular_freq and
    rebuilds only the training-indexable rows in-loop (r2_ensure_full restores the
    rest off-clock before any eval), and retunes attn_scale."""
    def __init__(self, head_dim, max_seq_len, paired=False, rot=None, heads=0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        # KX_RSTAG / KX_RNOPE: number of heads to give the factor tables a real head axis.
        # 0 (every Yarn but the narrow non-paired one, and that one too unless a knob is
        # set) = stock: 1-D angular_freq, 2-D factors broadcast over heads.
        self.heads = heads
        if heads:
            assert not paired, "per-head rotary (KX_RSTAG/KX_RNOPE) is not defined for paired Yarns"
            assert 0 <= 0 < heads, f"KX_RNOPE must be in [0,{heads})"
        # KX_QKSCAF: explicit rotating-dim count, overriding the half-truncation
        # default. Used by the scaffold Yarns (head_dim=128 tables carrying the NARROW
        # head's 16-frequency band, zero-padded across the remaining 96 dims) so the
        # dims that survive the stage-2 truncation keep exactly the narrow rotary
        # semantics. None = stock (bit-identical: `None or X` is X).
        self.rot_override = rot
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        if self.heads:
            # KX_RSTAG/KX_RNOPE: factors are (2*max_seq, H, head_dim) -- the head axis is
            # real, so it is sliced through instead of broadcast ([None, :T, :, :]).
            factor1, factor2 = (
                self.factor1[None, : x_BTHD.size(-3)],
                self.factor2[None, : x_BTHD.size(-3)],
            )
        else:
            factor1, factor2 = (
                self.factor1[None, : x_BTHD.size(-3), None, :],
                self.factor2[None, : x_BTHD.size(-3), None, :],
            )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        """Rebuild angular_freq and the full factor tables from scratch (init and
        post-warmup reset)."""
        # Rotating dim count out of head_dim (must be even; stock
        # head_dim//2 half-truncation, which is the shipping value).
        _rot = self.rot_override or (int("0") or (self.head_dim // 2))
        assert _rot % 4 == 0 and _rot <= self.head_dim
        if self.heads:
            angular_freq = self._head_angular_freq(_rot)              # (H, head_dim)
        else:
            angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=_rot//2, dtype=torch.float32, device=device)
            angular_freq = angular_freq.repeat_interleave(2)
            # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
            angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim - _rot)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if self.heads:
            # (2*max_seq, H, head_dim) tables, filled one head at a time so the fp32
            # theta/cos/sin temporaries stay the size of ONE stock table.
            f1 = torch.empty(t.numel(), self.heads, self.head_dim, dtype=torch.bfloat16, device=device)
            f2 = torch.empty_like(f1)
            for _h in range(self.heads):
                _theta = torch.outer(t, angular_freq[_h])
                f1[:, _h].copy_(_theta.cos())
                f2[:, _h].copy_(_theta.sin())
            self.factor1 = nn.Buffer(f1, persistent=False)
            self.factor2 = nn.Buffer(f2, persistent=False)
        elif not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self._r2_valid = 2 * self.max_seq_len  # rows of factor1/2 valid for the CURRENT angular_freq (partial-rebuild bookkeeping)
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = float("0.085")  # tuned at d=128; KX_QKD<128 halves qk dot variance -> sweep upward

    def _head_angular_freq(self, _rot: int):
        """KX_RSTAG/KX_RNOPE: (H, head_dim) per-head frequency ladders.

        KX_RSTAG=0: every head gets the stock ladder (exactly the values reset() would
        have built), so RNOPE alone only zeroes rows. KX_RSTAG=1: head h takes exponents
        (k*H + h)/(H*steps-1), k=0..steps-1 -- H interleaved ladders whose union is
        H*steps frequencies evenly spaced over the original [1, 1/1024] band.
        KX_RNOPE=n then zeroes the last n heads' rows (cos=1, sin=0 => identity rotary).
        """
        H, steps = self.heads, _rot // 2
        _exp = torch.linspace(0, 1, steps=steps, dtype=torch.float32, device=device).expand(H, steps)
        angular_freq = (1 / 1024) ** _exp
        angular_freq = angular_freq.repeat_interleave(2, dim=-1)       # (H, _rot)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(H, self.head_dim - _rot)], dim=-1)
        return angular_freq.contiguous()

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        # Partial rebuild: at in-loop window transitions rebuild only the rows training can
        # index (packed-sequence T <= max stage tokens-per-rank); stock rebuilds all
        # 2*max_seq_len val-sized rows on-clock. Rows [0,K) get bitwise the values a full
        # rebuild would give (row-elementwise math); the rest are restored off-clock by
        # r2_ensure_full() before any eval.
        _full = 2 * self.max_seq_len
        _rows = min(_full, _R2_YARN_ROWS) if (_R2_YARN_ROWS > 0) else _full
        self._rebuild_rows(0, _rows)
        self._r2_valid = min(getattr(self, "_r2_valid", _full), _rows)
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def _rebuild_rows(self, lo: int, hi: int):
        """Rebuild factor1/factor2 rows [lo,hi) from the CURRENT angular_freq. Row math
        is elementwise per position, so any [lo,hi) slice is bitwise what a full rebuild
        would put there."""
        if hi <= lo:
            return
        t = torch.arange(lo, hi, dtype=torch.float32, device=self.angular_freq.device)
        if self.heads:
            for _h in range(self.heads):
                theta = torch.outer(t, self.angular_freq[_h])
                self.factor1[lo:hi, _h].copy_(theta.cos())
                self.factor2[lo:hi, _h].copy_(theta.sin())
        elif not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1[lo:hi].copy_(theta.cos())
            self.factor2[lo:hi].copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1[lo:hi].copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2[lo:hi].copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[lo:hi][..., 1::2] *= -1

    def r2_ensure_full(self):
        """Restore rows [valid, full) from the current angular_freq.
        Called off-clock before any eval; no-op when the table is already full."""
        _full = 2 * self.max_seq_len
        _v = getattr(self, "_r2_valid", _full)
        if _v >= _full:
            return
        self._rebuild_rows(_v, _full)
        self._r2_valid = _full

@dataclass(slots=True)
class AttnArgs:
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor | None
    aux_v: torch.Tensor | None
    xsa_alpha: torch.Tensor | None
    train_max_seq_len: torch.Tensor
    attn_temp: torch.Tensor | None = None
    qk_gain: torch.Tensor | None = None
    w_f8: torch.Tensor | None = None
    w_scale: torch.Tensor | None = None
    # ---- S3 (fp8 attention backward) plumbing; all None unless KX_S3* ----
    qkv_col_f8: torch.Tensor | None = None   # [2304,768] e4m3 COL-major cache of lam0*W_qkv
    qkv_ws: torch.Tensor | None = None       # its 0-D per-layer scale
    gs_t: torch.Tensor | None = None         # 0-D static e5m2 g scale (KX_S3_GS)
    x_t_f8: torch.Tensor | None = None       # [768,T] e4m3 (prefilled fwd-side)
    xs_t: torch.Tensor | None = None         # 0-D static x scale (2^-4)
    g_scale: torch.Tensor | None = None      # wgrad g scale (delayed under GE4)
    g_amax: torch.Tensor | None = None       # its amax slot (GE4)
    gt_buf: torch.Tensor | None = None       # [2304,T] fp8 PREALLOC transposed-g dst
    y_t_f8: torch.Tensor | None = None       # [768,T] e4m3 (S3O; prefilled in attn fwd)
    y_scale: torch.Tensor | None = None      # delayed per-layer y scale (S3O)
    y_amax: torch.Tensor | None = None       # its amax slot (S3O)
    o_g_scale: torch.Tensor | None = None    # o-site wgrad g scale (S3O)
    o_g_amax: torch.Tensor | None = None     # its amax slot (S3O + GE4)
    o_gt_buf: torch.Tensor | None = None     # [768,T] fp8 PREALLOC transposed-g_o dst

flash_attn_interface = get_kernel('kernels-community/flash-attn3', version=1).flash_attn_interface


def dc_gate(
    x: Tensor,
    dc_w: tuple[Tensor, Tensor],
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    dc_w1, dc_w2 = dc_w
    assert dc_w1.shape == dc_w2.shape == (x.size(0), x.size(1), num_heads)
    post_w1 = F.rms_norm(dc_w1.float(), (num_heads,), eps=1.0e-6).type_as(x)
    return post_w1.contiguous(), dc_w2.type_as(x).contiguous()

class CausalSelfAttention(nn.Module):
    """Stateless causal attention over bank-provided fused qkvo weights: QK-norm,
    rotary via the passed Yarn, FA3 varlen sliding-window attention, optional
    key_offset induction shift, value-embedding aux_v injection, DC correction,
    gated XSA subtraction and per-head output gates. The `paired` variant interleaves
    adjacent heads' streams (doubled sequence, halved effective window)."""
    def __init__(self, dim: int, head_dim: int, num_heads: int, paired: bool = False, full_qk: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # KX_QKD: q/k head dim (d_qk) may be narrower than the v/o head dim.
        # With KX_QKD=0 these equal the stock values (qkv_rows == 3 * dim), so
        # every slice/view below is bit-identical to the baseline.
        # KX_QKMW: full_qk=True pins this module at the FULL v/o head dim regardless
        # of KX_QKD, so the two wide (ws_long) layers keep stock 128-wide q/k.
        self.d_qk = head_dim if full_qk else (head_dim)
        self.qk_rows = num_heads * self.d_qk       # rows of q (= rows of k) in qkvo_w
        self.qkv_rows = 2 * self.qk_rows + self.hdim  # q+k+v rows (o rows follow)
        # Weights are stored in parameter banks and passed via forward()

    def _split_qkv(self, y, B: int, T: int):
        """Split fused qkv projection output into per-head q, k, v.

        y: (..., qkv_rows). Off (KX_QKD=0): the original uniform-head-dim
        view+chunk, untouched. On: unequal split — q/k at d_qk, v at head_dim."""
        return y.view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor, dc_w: tuple[Tensor, Tensor] | None = None):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn

        q, k, v = self._split_qkv(F.linear(x, sa_lambdas[0] * qkvo_w[:self.qkv_rows].type_as(x)), B, T)
        max_len = train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        q, k = norm(q), norm(k) # QK norm @Grad62304977
        if attn_args.qk_gain is not None:
            # Learnable per-dim per-layer qk gain post-norm pre-rotary; the
            # model re-weights positional-vs-content channels continuously across window
            # switches. Zero-init => bit-identical start.
            _g = (1.0 + attn_args.qk_gain).type_as(q)
            q, k = q * _g, k * _g
        if attn_args.attn_temp is not None:
            q = q * attn_args.attn_temp.view(1, 1, self.num_heads, 1).type_as(q)

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                # NOTE: the stationary band starts at the ROTATING dim count,
                # not d_qk//2 — with a wider rotating band the old bound overwrote rotating dims.
                _koff = int("0") or (self.d_qk // 2)
                k[:, 1:, :, _koff:] = k[:, :-1, :, _koff:]

            if aux_v is not None:
                v = v + aux_v.view_as(v)

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.d_qk * 2)
            k = k.view(B, T, self.num_heads // 2, self.d_qk * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.d_qk)
            k = k.view(B, T * 2, self.num_heads // 2, self.d_qk)

            if aux_v is not None:
                v = v + aux_v.view_as(v)

            seqlens = 2 * seqlens
            max_len = 2 * max_len

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0),
                                                        **({}))
        if dc_w is not None:
            dc_weights = dc_gate(x, dc_w, self.num_heads)
            y = dc_attention_postonly_nodd_correction_add_base_triton(
                y, q, k, v, dc_weights, None,
                scaling=yarn.attn_scale,
                window=112,
                seq_lens=seqlens,
            )
        y = y.view(B, T, self.num_heads, self.head_dim)
        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(α)
        # of y aligned with v̂. Non-paired only (v shape doesn't line up for paired layers).
        if attn_args.xsa_alpha is not None and not self.paired:
            dot = (y * v).sum(-1, keepdim=True)
            denom = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(B, T, self.num_heads, 1)
            y = y - alpha * (dot / denom) * v
        if attn_gate_w is not None:
            y = y * attn_gate_w.type_as(y).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.qkv_rows:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int
    # KX_QKSCAF: scaffold phase active for this step (stage 1 only). Rides the same
    # per-step config object as ws_short/ws_long, so it is a dynamo-specialised python
    # bool exactly like they are -- flipping it recompiles, which is why it is keyed to
    # the stage-1 -> stage-2 boundary that already recompiles. Always False when the
    # knob is off => constant-folded => stock graph.
    qkscaf: bool = False
    ptp_w: float = -1.0   # per-stage prefix-CE weight (piecewise-const ramp); -1 => use PREFIX_CE_WEIGHT constant

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # there are only 50257 unique GPT-2 tokens; extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977, originates from Karpathy's experiments.
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        # Transposed weight storage for faster gradient accumulation
        use_fp8 = True
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, use_fp8=use_fp8, x_s=100/448, w_s=1.6/448, grad_s=grad_scale * 0.75/448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            # tie embed and lm_head at init
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.init_attn(model_dim, head_dim, num_heads, num_layers, max_seq_len)
        self.init_mlp(model_dim)
        self.init_misc(model_dim, num_layers)
        self.init_mudd(num_layers, model_dim)

        self.xsa_layers = [1, 3, 4, 7]
        # DC-attention layer placement (10 = stock). The mudd
        # post-gate coefficient block [29:41] follows the layer; layout unchanged.
        self.dc_layers = [int("10")]
        # KX_317=1 drops the attn gate on both layers (PR #317's declined SAG-removal half).
        # Coefficient layout is deliberately NOT compacted: the mudd-gate w2 rows 12:18 stay
        # in place so every other coefficient keeps its exact index and bias init, and
        # mudd_gate_w2 / mudd_gate_b2 keep their (2,41,64) / (2,41) shapes (no optimizer
        # param-group, sharding, FLAT_ALLREDUCE flat-buffer or RNG-stream churn). The rows simply
        # go dead: they are Adam params (elementwise, no cross-row coupling), init to zero
        # in w2, and mudd_gate_b2 has wd_mul=0, so dead rows cannot perturb live ones.
        # Compacting 41->35 / 26->20 would save ~0.3 us/step of gate GEMM -- not worth it.
        self.attn_gate_layers = [3, 10]
        self.init_mudd_gate(model_dim)

        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def init_attn(self, model_dim, head_dim, num_heads, num_layers, max_seq_len):
        # Cache layers for skip / backout snapshots taken at end of loop iter.
        self.cache_layers = [3, 7]

        # Attention modules (no learned params -- weights come from qk_bank/vo_bank)
        # Paired-head layer set (0,2,5,9)
        self.paired_head_layers = [int(x) for x in "0,2,5,9".split(",") if x.strip() != ""]
        self.attn = CausalSelfAttention(model_dim, head_dim, num_heads, paired=False)
        self.attn_paired = CausalSelfAttention(model_dim, head_dim, num_heads, paired=True)
        # KX_QKD: rotary tables are built for the q/k head dim (d_qk), which the
        # narrowed q/k tensors carry. Off: d_qk == head_dim, identical to stock.
        _qk_dim = head_dim
        # KX_RSTAG/KX_RNOPE: only the narrow NON-PAIRED Yarn gets a head axis (heads=0 =>
        # `if self.heads:` is False everywhere => stock construction, bit-identical).
        self.yarn = Yarn(_qk_dim, max_seq_len, heads=(0))
        self.yarn_paired_head = Yarn(_qk_dim, max_seq_len, paired=True)
        # KX_QKMW: the wide (ws_long carrier) layers get their own full-width, non-paired
        # attention module + rotary table (32 freqs, rot band head_dim//2 = 64 -- exactly
        # the stock d=128 rotary). Model-layer indices; the matching attention-bank
        # indices drop the attn-less layer 6.
        # KX_QKMWL: list override of the wide set, e.g. "3,10,7"
        # or "3". Adding wide layers is always safe (extra full-width, no ws_long
        # semantics attached); removing 3/10 leaves a narrow ws_long carrier —
        # mechanically identical to uniform QKD64 on that layer.
        _qkmwl = ""
        self._qkmw_wide_layers = [int(x) for x in _qkmwl.split(",")] if _qkmwl else [3, 10]
        assert all(0 <= l <= 10 and l != 6 and l not in self.paired_head_layers for l in self._qkmw_wide_layers), \
            "KX_QKMWL: wide layers must be non-paired attention layers"
        self._qkmw_wide_attn = {i - (i > 6) for i in self._qkmw_wide_layers}   # {3, 9} stock

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))

        # parameter banks for attention and value embedding gate weights
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates
        self.gate_filler_nones = [None] * (num_layers - 6)

        # Parameter banks for sharded optimization, by @chrisjmccormick
        # Attention is skipped in layer 6 by @YouJiacheng
        num_attn_layers = num_layers - 1
        hdim = num_heads * head_dim

        # QK bank: per-head-pair Muon groups for Q, K weights
        # Each pair of adjacent heads gets its own independent polar express orthogonalization
        self._num_attn_layers = num_attn_layers
        num_qk_groups = num_attn_layers * 2 * (num_heads // 2)  # 10 * 2 * 3 = 60
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=world_size)  # 64
        # KX_QKD: each head-pair group holds 2 heads' worth of q (or k) rows, so
        # group rows = 2 * d_qk (stock 256; 128 at KX_QKD=64). Optimizer chunking,
        # ANVIL state and shape_mult all derive from .reshape.
        # KX_QKMW: allocate at the STOCK full width (2 * head_dim = 256 rows/group) even
        # though KX_QKD=64 -- the narrow layers occupy rows [:2*KX_QKD] of each group and
        # the remainder is dead zero. Keeps every optimizer/comms geometry at stock.
        _qk_grp = (_qk_dim) * 2
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, _qk_grp, model_dim))
        self.qk_bank.reshape = (num_qk_padded, _qk_grp, model_dim)

        # VO bank: per-layer Muon groups for V and O weights
        num_vo_real = num_attn_layers * 2  # 20
        num_vo_padded = next_multiple_of_n(num_vo_real, n=world_size)  # 24
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, hdim))
        self.vo_bank.reshape = (num_vo_padded, hdim, hdim)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        # Multiplier on the qk init bound (would restore logit variance at the source
        # under narrowed qk). 1.0 = bit-identical stock (shipping).
        _qki = float("1.0")
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound * _qki, bound * _qki)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()

    def init_mlp(self, model_dim):        
        # MLP bank: stores c_fc and c_proj for all MLP layers
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        mlp_hdim = 4 * model_dim
        self.mlp_bank = nn.Parameter(torch.empty(12, 2, mlp_hdim, model_dim))  # (12, 2, 3072, 768)
        self.mlp_bank.reshape = (24, mlp_hdim, model_dim)  # Shape for sharding: (24, 3072, 768)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

    def init_misc(self, model_dim, num_layers):
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, args.bigram_dim)
        nn.init.zeros_(self.bigram_embed.weight)
        bigram_sign_table = torch.randn(args.bigram_sign_table_rows, args.bigram_dim).sign().to(torch.bfloat16)
        self.register_buffer('bigram_sign_table', bigram_sign_table)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))

        pad = (-num_layers * 2 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )

    def init_mudd(self, num_layers: int, model_dim: int):
        """
        Multiway Dynamic Dense Connections @lishengping. https://arxiv.org/abs/2502.12170
        Expressive and efficient mechanism for data dependent skip connections.
        Given current activation x, return n skip coefficients computed via ~mlp(x).
        Trimmed for speedrun: invoked at start of last layer and post-loop only.

        Start of last layer produces 14 coefficients:
          mu[0..2]  = v_mudd source coefs  (cache[0], cache[7], x)   -> added into V
          mu[3..5]  = residual source coefs (cache[0], cache[7], x)  -> residual recombination
          mu[6..7]  = per-pair ve_gate (2 channels, tiled to num_heads)
          mu[8..9]  = resid_attn / post_attn lambdas (dynamic)
          mu[10..11]= x0 / bigram injection lambdas (dynamic)
          mu[12..13]= resid_mlp / post_mlp lambdas (dynamic)

        Post-loop produces 5 residual coefs over
          {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        """
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 64
        max_num_coef = 14

        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))

        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))

        # Bias init in pre-scaled domain (effective = bias * _mudd_scale).
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        # Per-pair ve_gate baseline (matches max of `2*sigmoid` used at other layers):
        bs_init[0, 6]  = 2.0 / self._mudd_scale       # ve_gate lane 0
        bs_init[0, 7]  = 2.0 / self._mudd_scale       # ve_gate lane 1
        # Layer-0 layer-10 dynamic lambdas (effective values match per-layer defaults):
        bs_init[0, 8]  = 1.1**0.5 / self._mudd_scale  # resid_attn[10]
        bs_init[0, 9]  = 1.0 / self._mudd_scale       # post_attn[10]
        bs_init[0, 10] = 0.0                          # x0_lambda[10] (init 0)
        bs_init[0, 11] = 0.05 / self._mudd_scale      # bigram_lambda[10]
        bs_init[0, 12] = 1.1**0.5 / self._mudd_scale  # resid_mlp[10]
        bs_init[0, 13] = 1.0 / self._mudd_scale       # post_mlp[10]
        # Layer-1 (post-loop): -0.5 backout absorbed into residual h7 coef.
        bs_init[1, 1]  = -0.5 / self._mudd_scale      # post-loop residual h7 coef
        self.mudd_b2 = nn.Parameter(bs_init)

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def quantize_mlp_fp8(self, refresh_lm: bool = True):
        """Refresh the FP8 weight caches + delayed activation/grad scales (post-step).

        refresh_lm=False (HOST_OPT only) skips the lm_head FP8 copy; callers must pass
        False only on steps that did not mutate lm_head.weight.
        """
        E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        E5M2_MAX = torch.finfo(torch.float8_e5m2).max
        with torch.no_grad():
            dev = self.mlp_bank.device
            if not hasattr(self, "_mlp_up_proj_f8"):
                self._mlp_up_proj_f8 = torch.zeros_like(self.mlp_bank[:, 0], dtype=torch.float8_e4m3fn)
                self._mlp_up_proj_scales = torch.ones(12, dtype=torch.float32, device=dev)
                self._mlp_dequant_scale_buf = torch.ones(1, dtype=torch.float32, device=dev)
                if FP8_STATIC_XSCALE:
                    # STATIC-XS-MLP: per-layer static dequant scales
                    # (up_proj_scale * 2^-4, refreshed below in one tensor op,
                    # off the fwd critical chain) + 0-dim static x-scale tensor
                    # for quantize_dual_layout (same constant/proof as _s3_xs_t).
                    self._mlp_static_dq = torch.ones(12, dtype=torch.float32, device=dev)
                    self._mlp_xs_t = torch.tensor(FP8_ATTN_XSCALE, dtype=torch.float32, device=dev)
                nsm = torch.cuda.get_device_properties(dev).multi_processor_count
                if FP8_LAGGED_QUANT:
                    nt = quantize_weights_dual_ntiles(3072, 768)
                    self._w_up_pamax = torch.zeros(12, nt, dtype=torch.float32, device=dev)
                    self._w_dn_pamax = torch.zeros(12, nt, dtype=torch.float32, device=dev)
                    self._a2p_calls = 0
                if FP8_MLP_DX:
                    self._mlp_up_f8_col = torch.zeros_like(self.mlp_bank[:, 0], dtype=torch.float8_e4m3fn
                                                           ).transpose(1, 2).contiguous().transpose(1, 2)
                self._mlp_dpre_scale = torch.full((12,), FP8_GRAD_SCALE, dtype=torch.float32, device=dev)
                self._mlp_dpre_amax = torch.zeros((12, nsm) if FP8_LAGGED_QUANT else (12,),
                                                  dtype=torch.float32, device=dev)
                self._mlp_gs_t = torch.tensor(FP8_GRAD_SCALE, dtype=torch.float32, device=dev)
            # ---- up-projection weight caches (row; + col under FP8_MLP_DX) ----
            if FP8_LAGGED_QUANT:
                boot = self._a2p_calls < 16      # PR#342 bootstrap: exact scales first
                self._a2p_calls += 1
                if boot:
                    flat = self.mlp_bank[:, 0].reshape(12, -1)
                    self._mlp_up_proj_scales[:] = (flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX).float()
                quantize_mlp_weights_dual(
                    self.mlp_bank[:, 0], self._mlp_up_proj_scales, self._w_up_pamax,
                    row=self._mlp_up_proj_f8,
                    col_t=self._mlp_up_f8_col.transpose(1, 2) if FP8_MLP_DX else None,
                    headroom=1.12, update_scales=not boot)
            if FP8_STATIC_XSCALE:
                # STATIC-XS-MLP: all 12 fwd dequant scales in one tiny op
                # (in-place: address-stable for the G_quant CG capture list).
                torch.mul(self._mlp_up_proj_scales, FP8_ATTN_XSCALE, out=self._mlp_static_dq)
            if FP8_MLP_FWD:
                if not hasattr(self, "_mlp_down_f8"):
                    self._mlp_down_f8 = torch.zeros_like(self.mlp_bank[:, 1], dtype=torch.float8_e4m3fn
                                                         ).transpose(1, 2).contiguous().transpose(1, 2)
                    self._mlp_down_scales = torch.ones(12, dtype=torch.float32, device=dev)
                    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
                    # Delayed scaling for the post activation (per layer): the fused
                    # kernel epilogue tracks amax; scale refreshes here.
                    self._mlp_post_scale = torch.ones(12, dtype=torch.float32, device=dev)
                    self._mlp_post_amax = torch.zeros((12, nsm) if FP8_LAGGED_QUANT else (12,),
                                                      dtype=torch.float32, device=dev)
                    if FP8_MLP_DPRE:
                        self._mlp_down_f8_row = torch.zeros_like(self.mlp_bank[:, 1], dtype=torch.float8_e4m3fn)
                        self._mlp_dq_bwd = torch.ones(12, dtype=torch.float32, device=dev)
                if FP8_LAGGED_QUANT:
                    if boot:
                        w2f = self.mlp_bank[:, 1].reshape(12, -1)
                        self._mlp_down_scales[:] = (w2f.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX).float()
                    quantize_mlp_weights_dual(
                        self.mlp_bank[:, 1], self._mlp_down_scales, self._w_dn_pamax,
                        row=self._mlp_down_f8_row if FP8_MLP_DPRE else None,
                        col_t=self._mlp_down_f8.transpose(1, 2),
                        headroom=1.12, update_scales=not boot)
                # ---- delayed activation scale (reduce the per-SM amax slots first) ----
                pa = self._mlp_post_amax.amax(dim=1) if FP8_LAGGED_QUANT else self._mlp_post_amax
                if GRAD_FIREWALL:
                    pa = torch.nan_to_num(pa, nan=0.0, posinf=0.0, neginf=0.0).clamp(max=1e12)
                torch.clamp(pa * (FP8_POST_HEADROOM / E4M3_MAX), min=1e-6, out=self._mlp_post_scale)
                self._mlp_post_amax.zero_()
                if FP8_MLP_DPRE:
                    torch.mul(self._mlp_down_scales, FP8_GRAD_SCALE, out=self._mlp_dq_bwd)
                # ---- delayed dpre scale ----
                da = self._mlp_dpre_amax.amax(dim=1) if FP8_LAGGED_QUANT else self._mlp_dpre_amax
                if GRAD_FIREWALL:
                    da = torch.nan_to_num(da, nan=0.0, posinf=0.0, neginf=0.0).clamp(max=1e12)
                fmt_max = E5M2_MAX
                new_s = (da * (FP8_DPRE_HEADROOM / fmt_max)).clamp(min=1e-12)
                self._mlp_dpre_scale.copy_(torch.where(da > 0, new_s, self._mlp_dpre_scale))
                self._mlp_dpre_amax.zero_()
            if FP8_LMHEAD_CACHE:
                _do_lm = True
                if not hasattr(self, "_lm_f8"):
                    self._lm_f8 = torch.zeros_like(self.lm_head.weight, dtype=torch.float8_e4m3fn)
                elif HOST_OPT:
                    _do_lm = refresh_lm  # refresh only after steps that mutate lm_head (Adam runs on odd steps)
                if _do_lm:
                    self._lm_f8.copy_(self.lm_head.weight.div(self.lm_head.w_s).to(torch.float8_e4m3fn))
                    if _KX_LMF8T:
                        # _KX_LMF8T: keep a second, column-major layout of the same fp8
                        # bytes. The CE forward's w_f8.T.contiguous().T (a per-step fp8
                        # transpose on the serial forward path) becomes a zero-copy view;
                        # the transposing copy runs here, in the comm-overlapped refresh
                        # (and inside the captured G_quant body under CUDA_GRAPH_TIER).
                        if not hasattr(self, "_lm_f8_cm"):
                            _w = self.lm_head.weight
                            self._lm_f8_cm = torch.empty_strided(_w.shape, (1, _w.shape[0]), dtype=torch.float8_e4m3fn, device=_w.device)
                        # The aten copy_ into the col-major cache is a
                        # strided elementwise scatter (~0.54 ms/step in the
                        # late-step direct_copy bucket). Identity: _lm_f8_cm has
                        # strides (1, 768), so _lm_f8_cm.T is a CONTIGUOUS
                        # (50304, 768) view of the same storage, and
                        # _lm_f8_cm.copy_(_lm_f8) == _lm_f8_cm.T = _lm_f8.T —
                        # exactly the tiled transpose_copy kernel's contract
                        # (coalesced reads AND writes). fp8 is 1 byte, so the
                        # uint8 reinterpret moves the identical bytes (a pure
                        # transpose is dtype-blind at equal itemsize) without
                        # relying on fp8 tl.load support; 768 and 50304 are
                        # exact multiples of the 64x128 tile, so the masked
                        # `other=0.0` fill path never even triggers.
                        transpose_copy(self._lm_f8.view(torch.uint8),
                                       self._lm_f8_cm.T.view(torch.uint8))

    def init_mudd_gate(self, model_dim: int):
        self._mudd_gate_scale = nn.Parameter(torch.tensor(0.1))
        mudd_gate_dim = 64
        assert self.num_heads == 6
        # Fixed gate layouts. The post gate is generated at the start of layer 4.
        # pre:  xsa[1,3] 12 + attn[3] 6 + inject[0..3] 8 = 26
        # post: xsa[4,7] 12 + attn[10] 6 + inject[4,5,7,8,9] 10 + skip 1 + dc[10] 12 = 41
        self._mudd_gate_pre_num_coef = 26
        # KX_DCL2: add a SECOND DC-attention layer at the given index (0=off, stock).
        # Appends 12 gate coefficients (w1 rows 41:47 bias 1.0, w2 rows 47:53 bias 0)
        # after the frozen 41-slot layout, so all existing indices are unchanged.
        self._kx_dcl2 = int("0")
        self._mudd_gate_post_num_coef = 41 + (12 if self._kx_dcl2 else 0)
        # KX_AGX: append 5x6 post attn-gate coefs after the frozen layout (and 3x6 pre
        # coefs after pre's frozen 26); existing indices unchanged (same trick as DCL2).
        self._agx_post_base = self._mudd_gate_post_num_coef
        max_num_coef = max(self._mudd_gate_post_num_coef, 0)
        self.mudd_gate_w1 = nn.Parameter(torch.empty(2, mudd_gate_dim, model_dim))
        self.mudd_gate_w2 = nn.Parameter(torch.zeros(2, max_num_coef, mudd_gate_dim))
        for j in range(2):
            nn.init.kaiming_uniform_(self.mudd_gate_w1.data[j], a=math.sqrt(5))

        bs_init = torch.zeros(2, max_num_coef)
        _mudd_gate_scale = 0.1
        attn_gate_bias = 0.25 / _mudd_gate_scale
        bigram_gate_bias = 0.05 / _mudd_gate_scale
        skip_gate_bias = 0.5 / _mudd_gate_scale
        dc_w1_gate_bias = 1.0 / _mudd_gate_scale
        bs_init[0, 12:18].fill_(attn_gate_bias)     # pre attn[3]
        bs_init[0, 19:26:2].fill_(bigram_gate_bias) # pre bigram gates for layers 0..3
        bs_init[1, 12:18].fill_(attn_gate_bias)     # post attn[10]
        bs_init[1, 19:28:2].fill_(bigram_gate_bias) # post bigram gates for layers 4,5,7,8,9
        bs_init[1, 28].fill_(skip_gate_bias)
        bs_init[1, 29:35].fill_(dc_w1_gate_bias)    # post dc[10] w1, output starts at 1.0
        if self._kx_dcl2:
            bs_init[1, 41:47].fill_(dc_w1_gate_bias)  # second dc layer w1 gate
        self.mudd_gate_b2 = nn.Parameter(bs_init)

    def forward_mudd_gate(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_gate_w1[id]))
        return (F.linear(x, self.mudd_gate_w2[id, :num_coef]) + self.mudd_gate_b2[id, :num_coef]) * self._mudd_gate_scale.type_as(x)

    def unpack_pre_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates, bigram_gates):
        xsa_alphas[1] = gate[..., 0:6] # 6 means 6 heads
        xsa_alphas[3] = gate[..., 6:12]
        attn_gates[3] = gate[..., 12:18]
        for layer, offset in zip((0, 1, 2, 3), range(18, 26, 2)):
            x0_gates[layer] = gate[..., offset:offset + 1]
            bigram_gates[layer] = gate[..., offset + 1:offset + 2]

    def unpack_post_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates, bigram_gates):
        xsa_alphas[4] = gate[..., 0:6]
        xsa_alphas[7] = gate[..., 6:12]
        attn_gates[10] = gate[..., 12:18]
        for layer, offset in zip((4, 5, 7, 8, 9), range(18, 28, 2)):
            x0_gates[layer] = gate[..., offset:offset + 1]
            bigram_gates[layer] = gate[..., offset + 1:offset + 2]
        return gate[..., 28:29]

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig, bg_sink: Tensor | None = None):
        """Full forward pass over one packed batch; returns per-token loss (training:
        fused softcapped CE with MTP + prefix-CE; eval: plain softcapped CE).
        bg_sink is the zeros leaf that harvests the bigram-channel gradient."""
        assert input_seq.ndim == 1

        # ---- Schedule and layer topology ----
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        # set block masks and key shift
        bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short, ws_long]
        if KX_WSL:
            for _i, _m in KX_WSL.items():
                if bm_sizes[_i] is not None:
                    bm_sizes[_i] = max(128, int(round(bm_sizes[_i] * _m / 128)) * 128)
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==ws_long for b in bm_sizes] # apply partial key offset to long windows

        use_mlp_fp8 = self.training and True
        if use_mlp_fp8:
            mlp_up_proj_f8 = self._mlp_up_proj_f8.unbind(0)
            mlp_up_proj_scales = [self._mlp_up_proj_scales[i:i+1] for i in range(12)]

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)
        veg = self.ve_gate_bank.unbind(0)
        attn_gates = [None] * self.num_layers
        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]
        dc_weights = [None] * self.num_layers
        xsa_alphas = [None] * self.num_layers
        x0_gates = [None] * self.num_layers
        bigram_gates = [None] * self.num_layers
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers
        assert len(dc_weights) == self.num_layers
        qk_all = self.qk_bank[:self._num_qk_groups].view(self._num_attn_layers, -1, self.qk_bank.shape[-1])
        vo_flat = self.vo_bank[:self._num_attn_layers * 2].view(self._num_attn_layers, 2, *self.vo_bank.shape[1:]).flatten(1, 2)
        attn_weights = torch.cat([qk_all, vo_flat], dim=1).unbind(0)
        use_attn_f8 = False
        attn_f8 = self._attn_f8.unbind(0) if use_attn_f8 else None
        attn_ws = [self._attn_ws[j] for j in range(self._num_attn_layers)] if use_attn_f8 else None
        # S3 (fp8 attention backward): per-layer cache/buffer plumbing
        use_s3 = False
        use_s3w = bool(0) and use_s3
        use_s3o = False
        _s3_nq = self.attn.qkv_rows
        _s3_d = self.num_heads * self.head_dim
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer
        
        # Use sign-trick to better compress multiple bigrams into a shared bigram embedding row
        # (details in https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy)
        sign_idx = torch.zeros_like(input_seq)
        sign_idx[1:] = (input_seq[:-1] ^ input_seq[1:]) % self.bigram_sign_table.shape[0]  # (8192,)
        bigram_signs = self.bigram_sign_table[sign_idx]                                    # (seq, bigram_dim)
        if bg_sink is not None:
            # BIGRAM_SPARSE_GRAD/KX_BGLITE grad sink. Value-identical (bg_sink is zeros) but the
            # detach cuts autograd's path to the table, so NO dense [V, 768] grad is
            # ever built; bg_sink.grad after backward is the per-token value stream
            # (the * bigram_signs below is downstream of the add, so autograd folds
            # the signs into the sink grad exactly as the stock index_add sees them).
            _bge = self.bigram_embed(bigram_input_seq).detach() + bg_sink
        elif BIGRAM_BF16_BWD:
            _bge = _m90.m90b_bigram_embed(self.bigram_embed.weight, bigram_input_seq)
        x0_bigram = (_bge * bigram_signs)[None]             # (1, seq, bigram_dim)

        # Value embeddings - always computed (not precomputed)
        if KX_SLG_LOAD:
            _ve0, _ve1, _ve2, _ve3, _ve4 = value_embedding_planes_selected_load(
                self.value_embeds, input_seq
            )
            # Shifted .01 ... 234 structure on token value embeddings by @photomz
            ve = [None, _ve0, _ve1, *self.gate_filler_nones, _ve2, _ve3, _ve4]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        pre_gate = self.forward_mudd_gate(x0, id=0, num_coef=self._mudd_gate_pre_num_coef)
        self.unpack_pre_mudd_gate(
            pre_gate,
            xsa_alphas,
            attn_gates,
            x0_gates,
            bigram_gates,
        )

        # Initialize residual stream with pre-layer-0 bigram injection
        x = x0.clone()
        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_gates[0]
        skip_gate_out = None
        post_skip_gate = None

        # cache[k] is the layer-k snapshot used downstream by MUDD.
        # cache[0] = residual stream after bigram injection (input to layer 0).
        cache = {0: x}
        for i in range(self.num_layers):
            is_paired = i in self.paired_head_layers
            # KX_QKMW: the ws_long carriers run the full-width (d_qk=128) module + rotary.
            # Off: `is_wide` is a compile-time-constant False and this is the stock pick.
            is_wide = False
            # KX_QKSCAF: during the scaffold phase every NON-wide layer also runs at
            # d_qk=128 -- same pairing as always, only the width (and the rotary/attn
            # scale that go with it) change. Off: `is_scaf` is a compile-time-constant
            # False and the two picks below are the stock QKMW lines.
            is_scaf = schedule_cfg.qkscaf and not is_wide
            if is_scaf:
                yarn = self.yarn_scaffold_paired if is_paired else self.yarn_scaffold
                attn = self.attn_scaffold_paired if is_paired else self.attn_wide
            else:
                yarn = self.yarn_wide if is_wide else (self.yarn_paired_head if is_paired else self.yarn)
                attn = self.attn_wide if is_wide else (self.attn_paired if is_paired else self.attn)
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]
            if use_mlp_fp8:
                up_proj_f8, up_proj_scale = mlp_up_proj_f8[i], mlp_up_proj_scales[i]
            mu = None

            if i == 4:
                post_gate = self.forward_mudd_gate(x, id=1, num_coef=self._mudd_gate_post_num_coef)
                post_skip_gate = self.unpack_post_mudd_gate(
                    post_gate,
                    xsa_alphas,
                    attn_gates,
                    x0_gates,
                    bigram_gates,
                )
                dc_weights[self.dc_layers[0]] = (post_gate[..., 29:35], post_gate[..., 35:41])
                if self._kx_dcl2:
                    dc_weights[self._kx_dcl2] = (post_gate[..., 41:47], post_gate[..., 47:53])

            # process attn. skip on layer 6 @YouJiacheng
            if i == 6:
                assert post_skip_gate is not None
                skip_gate_out = torch.sigmoid(skip_lambda) * post_skip_gate
                x = x + skip_gate_out * cache[3]
            else:
                qkvo_w = attn_weights[i - (i > 6)]
                attn_in_normed = norm(cache.get(7, x))
                B, T = attn_in_normed.size(0), attn_in_normed.size(1)
                if use_s3w:
                    # S3 (the load-bearing line): prefill x_t = quantized
                    # transpose of attn_in_normed into the PREALLOCATED flat
                    # slice. Static 2^-4 scale — clip-free by the sqrt(768)
                    # proof, NO amax reduction anywhere on x (census check:
                    # no new standalone abs/amax kernel may appear).
                    quantize_transposed(
                        attn_in_normed.detach().view(-1, _s3_d),
                        self._s3_xs_t,
                        self._s3_xt[i - (i > 6)][: _s3_d * B * T].view(_s3_d, B * T),
                        fmt=torch.float8_e4m3fn)

                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=14)
                    v_mudd = (mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x).view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        self.num_heads // 2, dim=-1
                    ).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    # gate pattern g(x[:6] + ve[:6]) by @photomz
                    gate_in = torch.cat([attn_in_normed[..., :6], ve[i][None, ..., :6]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None

                attn_args = AttnArgs(
                    attn_temp=None,
                    qk_gain=None,
                    sa_lambdas=sa_lambdas[i],
                    seqlens=seqlens,
                    bm_size=bm_sizes[i],
                    yarn=yarn,
                    key_offset=key_offset[i],
                    attn_gate_w=attn_gates[i] if i in self.attn_gate_layers else None,
                    aux_v=aux_v,
                    xsa_alpha=xsa_alphas[i],
                    train_max_seq_len=train_max_seq_len,
                    w_f8=attn_f8[i - (i > 6)] if use_attn_f8 else None,
                    w_scale=attn_ws[i - (i > 6)] if use_attn_f8 else None,
                    qkv_col_f8=self._s3_qkv_col[i - (i > 6)] if use_s3 else None,
                    qkv_ws=self._s3_qws[i - (i > 6)] if use_s3 else None,
                    gs_t=self._s3_gs_t if use_s3 else None,
                    x_t_f8=(self._s3_xt[i - (i > 6)][: _s3_d * B * T].view(_s3_d, B * T)
                            if use_s3w else None),
                    xs_t=self._s3_xs_t if use_s3w else None,
                    g_scale=((self._s3_g_scale[i - (i > 6)])
                             if use_s3w else None),
                    g_amax=(self._s3_g_amax[i - (i > 6)]
                            if use_s3w and (True) else None),
                    gt_buf=(self._s3_gt[i - (i > 6)][: _s3_nq * B * T].view(_s3_nq, B * T)
                            if use_s3w else None),
                    y_t_f8=(self._s3_yt[i - (i > 6)][: _s3_d * B * T].view(_s3_d, B * T)
                            if use_s3o else None),
                    y_scale=self._s3_y_scale[i - (i > 6)] if use_s3o else None,
                    y_amax=self._s3_y_amax[i - (i > 6)] if use_s3o else None,
                    o_g_scale=((self._s3_go_scale[i - (i > 6)] if FP8_ATTN_WGRAD_E4M3 else self._s3_gs_t)
                               if use_s3o else None),
                    o_g_amax=self._s3_go_amax[i - (i > 6)] if (use_s3o and True) else None,
                    o_gt_buf=(self._s3_got[i - (i > 6)][: _s3_d * B * T].view(_s3_d, B * T)
                              if use_s3o else None),
                )
                dc_w = dc_weights[i] if i in self.dc_layers and not is_paired else None
                attn_out = attn(attn_in_normed, attn_args, qkvo_w, dc_w)

                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0] 
                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram
                else:
                    x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]
                    if i != 0:
                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_gates[i]

            # process mlp
            normed = norm(x)
            if use_mlp_fp8:
                if FP8_STATIC_XSCALE:
                    # A1 STATIC-XS-MLP: static 2^-4 x-scale — clip-free by the
                    # same sqrt(768) = 27.713 < 28.0 = 448*2^-4 proof as the S3
                    # attn path above (normed is post-rms_norm, same tensor
                    # class). NO amax reduction, NO scalar RMW chain: dequant
                    # scales were precomputed per-layer in quantize_mlp_fp8
                    # (census check: no standalone abs/amax kernel on x here).
                    if FP8_MLP_DW1:
                        x_scale = self._mlp_xs_t
                        x_f8, x_f8_t = quantize_dual_layout(
                            normed.detach().view(-1, normed.shape[-1]), x_scale,
                            fmt=torch.float8_e4m3fn)
                    mlp_args = (c_fc, c_proj, up_proj_f8, self._mlp_static_dq[i:i+1], x_f8)
                if FP8_MLP_FWD:
                    mlp_args = mlp_args + (self._mlp_down_f8[i], self._mlp_down_scales[i],
                                           self._mlp_post_scale[i], self._mlp_post_amax[i])
                    if FP8_MLP_DPRE:
                        mlp_args = mlp_args + (self._mlp_down_f8_row[i], FP8_GRAD_SCALE, self._mlp_dq_bwd[i])
                        mlp_args = mlp_args + (
                            x_f8_t, x_scale,
                            self._mlp_up_f8_col[i] if FP8_MLP_DX else None,
                            self._mlp_up_proj_scales[i] if FP8_MLP_DX else None,
                            self._mlp_gs_t,
                            self._mlp_dpre_scale[i],
                            self._mlp_dpre_amax[i])
            else:
                mlp_args = (c_fc, c_proj)

            if mu is not None:
                x = mu[12] * x + mu[13] * ReLUSqrdMLP(normed, *mlp_args)
            else:
                x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(normed, *mlp_args)

            if i in self.cache_layers:
                cache[i] = x

        # Post-loop MUDD: 5 residual coefs over {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        mu = self.forward_mudd(x, id=1, num_coef=5)
        ve_bank0 = ve[1][None].to(dtype=x.dtype)  # (1, T, D), same VE as layer-1 attn
        x = x + mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * cache[9] + mu[3] * ve_bank0 + mu[4] * cache[3]

        x = norm(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training:
            lm_f8 = self._lm_f8 if (hasattr(self, "_lm_f8")) else None
            lm_f8_row = None
            if hasattr(self, "_lm_f8_cm"):
                lm_f8_row = lm_f8          # row-major original: backward's w_f8.T stays a col-major view
                lm_f8 = self._lm_f8_cm     # col-major: forward's .T.contiguous().T is zero-copy
            _ptp_t, _ptp_w = (None, 0.0)
            if PREFIX_CE:
                # table built eagerly at startup; plain global-tensor lookup is dynamo-traceable
                _ptp_t = _ptp_tab_cache[target_seq]
                # constant sentinel: the LIVE weight is read eagerly inside the opaque
                # ce_fwd_bwd op (kernel-module._PTP_W_RUNTIME) so ramp flips never recompile the graph
                _ptp_w = 1.0
            if lm_f8_row is not None:
                loss_per_token = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights, self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s, grad_scale, 23.0, 5.0, 7.5, lm_f8, _ptp_t, _ptp_w, lm_f8_row)
            else:
                # graph-hash identity: with the col-major cache off the apply must carry
                # the SAME arg count as the baseline (a 15th constant arg changes the FX
                # graph and can move inductor/autotune kernel choices).
                loss_per_token = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights, self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s, grad_scale, 23.0, 5.0, 7.5, lm_f8, _ptp_t, _ptp_w)
        else:
            logits = self.lm_head(x)
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            logits_for_loss = logits.float()
            loss_per_token = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="none")
        return loss_per_token
# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 49152: 128}

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

_HG_PIN_RING = 8
_hg_pin_bufs: list = []
_hg_pin_i = 0

def _hg_pinned_like(x):
    """HOST_OPT: hand out a slice of a preallocated pinned buffer instead of a fresh
    cudaHostAlloc every step (stock does `torch.empty_like(x, pin_memory=True)`).

    Constraint: a slot may only be rewritten once its previous non_blocking H2D has
    been consumed. The ring depth (8) exceeds the host run-ahead: every training step
    the host blocks in the optimizer's `future.wait()`, which is ordered after that
    step's H2D + forward, so slot k is free long before it comes round again.
    """
    global _hg_pin_i
    if not _hg_pin_bufs:
        # Max per-rank token count over every consumer of this helper: the val batch
        # (4*64K*8 / 8 = 262144) dominates the largest train stage (bs24 -> 49152).
        _cap = max(args.val_batch_size, max(s.batch_size for s in TRAINING_STAGES)) // (world_size * grad_accum_steps)
        for _ in range(_HG_PIN_RING):
            _hg_pin_bufs.append(torch.empty(_cap, dtype=torch.int32, pin_memory=True))
    buf = _hg_pin_bufs[_hg_pin_i]
    _hg_pin_i = (_hg_pin_i + 1) % _HG_PIN_RING
    return buf[:x.numel()].view(x.shape)

def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32)
    out = _hg_pinned_like(x) if HOST_OPT else torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    # Shard loading follows the stock convention: every shard (including the first)
    # is loaded by the async loader during the timed run. _R2_PRESCAN_MAX = 0 keeps
    # the prescan machinery disabled; _r2_shards stays empty so _r2_next_getter
    # always falls through to the stock async path.
    _r2_shards = None
    if align_to_bos and _R2_PRESCAN_MAX > 0:
        _r2_shards = {}
        for _f in files[:_R2_PRESCAN_MAX]:
            _t = _load_data_shard(_f)
            _sh = Shard(_t, world_size)
            _sh._ready.wait()
            _sh._maybe_switch()
            _r2_shards[_f] = _sh

    def _r2_next_getter(_it):
        """Next-shard getter: prescanned dict hit, else the stock async loader."""
        _nf = next(_it)  # StopIteration propagates as in stock
        if _r2_shards is not None and _nf in _r2_shards:
            _sh = _r2_shards[_nf]
            return lambda: _sh
        return Shard.load_async(_nf, world_size)

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    if _r2_shards is not None:
        _f0 = next(file_iter)
        shard = _r2_shards[_f0]
        tokens = shard.tokens
        next_shard_getter = _r2_next_getter(file_iter)
    else:
        tokens = _load_data_shard(next(file_iter))
        if align_to_bos:
            shard = Shard(tokens, world_size)
            next_shard_getter = Shard.load_async(next(file_iter), world_size)
        else:
            pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = TRAIN_MAX_NUM_DOCS.get(num_tokens_local, next_multiple_of_n(num_tokens_local // 300, n=128))

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    if _r2_shards is not None:
                        next_shard_getter = _r2_next_getter(file_iter)
                    else:
                        next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        _yield_tuple = (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
            _bigram_inputs.numpy(),
        )
        new_params = yield _yield_tuple

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

# -----------------------------------------------------------------------------
# Training Management

@dataclass(slots=True)
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule
    num_scheduled_iterations: int = int(os.environ.get("KX_STEPS", "1178"))  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = int('23')  # number of steps to continue training at final lr and ws
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    # Descriptive run_id for this iteration:
    #   - explicit sparse connectivity refactor (no generic loop)
    #   - (1 + m_r9) * x self-reference fuse on layer 9
    #   - backout_lambda fully removed (slot dropped from self.scalars; absorbed into MUDD bias init)
    val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False
    run_evals: bool = False  # run additional evaluations after training is completed
    # bigram hash embedding
    bigram_vocab_size: int = 50304 * 15 // 2
    bigram_dim: int = 768
    bigram_sign_table_rows: int = 8192  # prefer a power of 2 (values ~500-15000 gave similar results)

args = Hyperparameters()
args.val_loss_every = 0

@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int
    duration: float = None

class TrainingSchedule:
    """
    Training schedule initialized via TRAINING_STAGES
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        3. YaRN updates to RoPE on window changes
        4. Split embed and lm head at 2/3 of training
        5. Batch size schedule of 8 -> 16 -> 24
        6. Post training extension of long windows from 13 to 20
        7. Seq len updates from 896 to 2048 at 1/3 of training
    """

    def __init__(self, stages: list[TrainingStage], scheduled_iterations: int, extension_iterations: int,
                 cooldown_frac: float = 0.5, split_embed_stage: int = 2, ws_post_yarn_ext: int = 20):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        # increase final validation ws, used for YaRN extension and short window size @classiclarryd
        self.ws_post_yarn_ext = ws_post_yarn_ext

        self.total_steps = self.scheduled_iterations + extension_iterations

        # Build stage boundaries (last is extension stage)
        ends = [0, *[round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])], self.total_steps]
        assert self.scheduled_iterations == ends[-2]
        self.boundaries = list(pairwise(ends))

        # Split embed at specified stage (ensure odd step for Adam)
        self.split_step = self.boundaries[split_embed_stage][0] | 1

        # Precompute MTP weights for all steps
        self.mtp_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        # Returns stage and % of the way through that stage
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr(self, step: int) -> float:
        # learning rate schedule: tied to batch size schedule, with cooldown at the end
        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            w = (1 - t) ** COOLDOWN_POW
            lr = lr * w + LR_FLOOR * (1 - w)
        return lr

# window_sizes are in units of `block_size` tokens (defined in TrainingManager)
TAIL_EMA_CFG = '298,0.65'                # "win,blend" TailEMA terminal blend (PR#347)
TAIL_EMA_WINDOW = int(TAIL_EMA_CFG.split(",")[0]) if TAIL_EMA_CFG else 0
TAIL_EMA_BLEND = float(TAIL_EMA_CFG.split(",")[1]) if TAIL_EMA_CFG else 0.0
TAIL_EMA_LABELS = tuple(s.strip() for s in "lm_head,embed".split(",") if s.strip())
_tema_bufs: dict = {}
# ---- Bank tail-blend ship: fp32 EMA over the three matrix banks (window K,
# rate 2/(K/4+1) at every-4th-step cadence, lazy alloc + accumulate from
# step >= N-K), blended into the shipped state at final-ship time at
# strength B: p = lerp(p_shipped, ema, B). Measured on same-trajectory
# ladders at k298@0.5; the blend basin is flat over k270-330 x 0.35-0.65.
_SHIPBLEND = True
_SHIPBLEND_B = float("0.5")
_SHIPBLEND_K = int("298")
_SHIPBLEND_LABELS = ("qk_bank", "vo_bank", "mlp_bank")
_shipblend_bufs: dict = {}
_shipblend_n: dict = {}
LR_FLOOR = float("0.20")   # lr cooldown floor (stock 0.15)
COOLDOWN_POW = float("1.0")    # warmdown shape exponent (stock linear)
MOMENTUM_COOLDOWN_STEPS = int("50")       # terminal rail-beta cooldown steps (stock 50)
STAGE2_LR_MUL = float("1.52")     # stage-2 lr mult (stock (16/8)^0.6)
STAGE3_LR_MUL = float("1.73")     # stage-3 lr mult (stock (24/8)^0.5)
# stage window sizes, block units (stock 1,3 / 3,7 / 5,11)
STAGE_WINDOW_SIZES = [int(x) for x in "1,3,3,7,5,11".split(",")]
# KX_WSVAL: extend ws_SHORT by N blocks at FINAL VALIDATION ONLY (0=off, stock).
# The stock final-val extension (ws_post_yarn_ext) widens only ws_long, on the
# 2 long-window layers; ws_short stays at the extension stage's 6 blocks on the other 8
# of the 10 attention layers. Widening it is untimed (the training clock is stopped for
# the whole validation section) and rules-legal (the venue permits evaluation at any
# sequence length / attention geometry; the token streams are untouched).
# A single integer is the one-shot form (ws_short += N for the final val). A comma list
# (e.g. "0,2,4,8") instead runs a LADDER: the final-val section re-validates once per
# value on the same trained/shipped weights and prints one `[wsval] N ... val_loss:` line
# each, so the whole ladder is self-paired on one seed and costs one run. Include 0 in the
# list to get the un-widened reference point in the same log.
_kx_wsval_list = [int(x) for x in "0".split(",") if x.strip() != ""]
# KX_YWSVAL: final-val ws_LONG ladder (comma list of absolute block counts, e.g.
# "13,16,20,24"). Untimed, self-paired on one trajectory; stock ships ws_long=20 un-YaRNed.
_kx_ywsval_list = [int(x) for x in "".split(",") if x.strip() != ""]
_kx_wsval = _kx_wsval_list[0] if len(_kx_wsval_list) == 1 else 0  # one-shot value; 0 in ladder mode
# terminal batch taper: "frac,bs" carves a 4th stage of the given duration
# fraction off the end of stage 3, at the given (smaller) batch size
BATCH_TAPER = '0.06,16'
STAGE_DURATIONS = [float(x) for x in '0.2854,0.3209,0.3931'.split(",")]
STAGE_DURATIONS = [d / sum(STAGE_DURATIONS) for d in STAGE_DURATIONS]
_bs1 = int("8")
_bs2 = int("16")
_bs3 = int("24")
_seq1 = int("896")
# _seq3: max sequence length for stage 3 onward (stage 3 + optional taper + the
# extension stage), stock 2048. Exact mirror of _seq1: `train_max_seq_len` is *only*
# the document-truncation cap used by Shard.next_batch -- each rank still packs exactly
# `batch_size / world_size` tokens per step, so the token budget, the batch geometry and
# every tensor shape are unchanged (the "2048" inside `_bs3 * 2048 * 8` is the token
# unit of the batch size, NOT the sequence length, and is deliberately left alone).
# Attention windows are in 128-token blocks and set separately by STAGE_WINDOW_SIZES. Raising it only
# lowers the sequences-per-batch count, so the TRAIN_MAX_NUM_DOCS[49152]=128 cap stays
# satisfied; the assert guards the (untested) shrinking direction.
_seq3 = int('3072')
assert _seq3 >= 512, "KX_SEQ3 < 512 can exceed TRAIN_MAX_NUM_DOCS[49152]=128 sequences per rank-batch"
TRAINING_STAGES = [
    TrainingStage(duration=STAGE_DURATIONS[0], train_max_seq_len=_seq1, batch_size=_bs1 * 2048 * 8, window_sizes=(STAGE_WINDOW_SIZES[0], STAGE_WINDOW_SIZES[1]), lr_mul=(_bs1 / 8) ** 0.6,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0]),
    TrainingStage(duration=STAGE_DURATIONS[1], train_max_seq_len=2048, batch_size=_bs2 * 2048 * 8, window_sizes=(STAGE_WINDOW_SIZES[2], STAGE_WINDOW_SIZES[3]), lr_mul=STAGE2_LR_MUL,  # stock (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0]),
    TrainingStage(duration=STAGE_DURATIONS[2], train_max_seq_len=_seq3, batch_size=_bs3 * 2048 * 8, window_sizes=(STAGE_WINDOW_SIZES[4], STAGE_WINDOW_SIZES[5]), lr_mul=STAGE3_LR_MUL,  # stock (24/8)**0.5
                  mtp_weights_start=([1.0]),
                  mtp_weights_end=([1.0])),
    # extension stage (continues stage-3 geometry, so it follows KX_SEQ3)
    TrainingStage(train_max_seq_len=_seq3, batch_size=_bs3 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
]
# Mid-stage-3 window transition — would split stage 3 into (5,11) then (6,13) halves
# at the SAME bs24 geometry, so ~half of s3 trains the val window at ACTIVE LR (as
# shipped only the min-LR EXT steps see (6,13)). Value = fraction of s3 spent at (6,13).
_kx_s3w = float("0")
if BATCH_TAPER:
    _tfrac, _tbs = BATCH_TAPER.split(",")
    _tfrac, _tbs = float(_tfrac), int(_tbs)
    TRAINING_STAGES[2].duration = STAGE_DURATIONS[2] - _tfrac
    TRAINING_STAGES.insert(3, TrainingStage(
        duration=_tfrac, train_max_seq_len=_seq3, batch_size=_tbs * 2048 * 8,
        window_sizes=(STAGE_WINDOW_SIZES[4], STAGE_WINDOW_SIZES[5]), lr_mul=STAGE3_LR_MUL * (_tbs / _bs3) ** 0.5,
        mtp_weights_start=[1.0], mtp_weights_end=[1.0]))
    TRAINING_STAGES[4].batch_size = _tbs * 2048 * 8  # extension continues at taper batch

# EXT_BATCH: decouple EXTENSION-stage batch size from stage 3 (0 = stock bs24).
# Wall-matched swap: EXT7@bs24 ~= EXT18@bs8. Warmup sampler covers
# the new ext-boundary shape bucket, so no in-window compile.
EXT_BATCH = int('8')
if EXT_BATCH:
    TRAINING_STAGES[-1].batch_size = EXT_BATCH * 2048 * 8

ptp_init(device)  # default-path CE dummy: must pre-exist any compiled forward (dynamo-safe)
if PREFIX_CE:
    _fk._PTP_W_RUNTIME = float(PREFIX_CE_WEIGHT)  # baseline before first schedule update; RAMP overwrites per stage
    # Eager alloc (dynamo cannot trace tiktoken's registry): the all(-1) table lives
    # on-device before the first compiled forward; contents are filled on the clock
    # by _ptp_table_fill() below. The forward does a plain tensor lookup only.
    _ptp_table_alloc(device)

training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=float("0.80"),
                                     split_embed_stage=int('4'),
                                     ws_post_yarn_ext=int("20"))

def get_rail_beta(step: int, beta_warmup_steps=None, beta_cooldown_steps=None, beta_min=0.85, beta_max=float('0.93')):
    """Scheduled fast-rail velocity beta: linear warmup from beta_min to beta_max,
    flat at beta_max for the bulk of the run, then a linear terminal cooldown
    back down to beta_min over the last beta_cooldown_steps."""
    if beta_warmup_steps is None:
        beta_warmup_steps = 240
    if beta_warmup_steps == 0:
        beta_warmup_steps = 1  # avoid div-by-zero; step 0 gets max beta immediately
    if beta_cooldown_steps is None:
        beta_cooldown_steps = max(MOMENTUM_COOLDOWN_STEPS, 1)
    beta_cd_start = training_schedule.total_steps - beta_cooldown_steps
    if step < beta_warmup_steps:
        frac = step / beta_warmup_steps
        beta = beta_min + frac * (beta_max - beta_min)
    elif step > beta_cd_start:
        frac = (step - beta_cd_start) / beta_cooldown_steps
        beta = beta_max - frac * (beta_max - beta_min)
    else:
        beta = beta_max
    return beta

class TrainingManager():
    """
    Manages the AnvilAndAdam for all parameters with explicit ordering.
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Explicit scatter_order and work_order for communication scheduling (no backward hooks)
        4. Muon has a linear momentum warmup and cooldown schedule
        5. Learning rates follow a linear decay schedule
        6. Embed is tied to lm_head until split step (2/3 of training), then untied @classiclarryd
    """
    def __init__(self, model):
        self.model = model
        self.block_size = 128

        # - Ordering dictates when to launch reduce/reduce_scatter operations
        # - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
        # - lr_mul and wd_mul are per-parameter learning rate and weight decay multipliers
        self.param_table = {
            "qk_bank":        {"optim": "anvil", "comms": "sharded",    "adam_betas": None},
            "vo_bank":        {"optim": "anvil", "comms": "sharded",    "adam_betas": None},
            "mlp_bank":       {"optim": "anvil", "comms": "sharded",    "adam_betas": None},
            "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
            "lm_head":        {"optim": "adam",    "comms": ("replicated_full" if REPLICATED_FULL_ADAM else "sharded"),    "adam_betas": [0.5,  0.95], "lr_mul": float("1.0"), "wd_mul": 150. / float("1.0")**2},
            "bigram_embed":   {"optim": "adam",    "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": BIGRAM_LR_MUL,  "wd_mul": float("5.0")},
            "post_lambdas":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "resid_lambdas":  {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "value_embeds":   {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": VEMB_LR_MUL,  "wd_mul": float("5.0")},
            "embed":          {"optim": "adam",    "comms": ("replicated_full" if REPLICATED_FULL_ADAM else "sharded"),    "adam_betas": [0.5,  0.95], "lr_mul": float("1.0"), "wd_mul": 150. / float("1.0")**2},
        }

        # ---- MUDD parameter overrides ----
        self.param_table.update({
            "mudd_w1":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_w2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_b2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25, "wd_mul": 0.0},
            "mudd_gate_w1": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
            "mudd_gate_w2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
            "mudd_gate_b2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
            "_mudd_gate_scale": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
        })

        # - Process smaller/faster params first while large reduces complete
        # - lm_head must complete before embed sync (when tied)
        self.work_order = ['scalars', 'smear_gate', 've_gate_bank', 'mudd_b2', 'mudd_gate_b2', '_mudd_gate_scale', 'post_lambdas', 'resid_lambdas', 'mudd_w2', 'mudd_gate_w2', 'value_embeds', 'bigram_embed', 'mudd_w1', 'mudd_gate_w1', 'lm_head', 'embed', 'qk_bank', 'vo_bank', 'mlp_bank']

        # Stock: dict order defines scatter priority.
        self.scatter_order = list(self.param_table)

        # ---- COMM_ORDER: comm re-orchestration (pure permutation, bitwise identical) ----
        # NCCL runs one stream in enqueue order, and `future.wait()` on a NCCL future is
        # a *stream* wait (cudaStreamWaitEvent on the compute stream), not a host block;
        # symmetrically, every collective is stream-ordered after the compute enqueued
        # before it. So the two lists jointly schedule two serial resources. Rules used:
        #   - start the compute stream as early as possible => the bank whose RS is
        #     cheapest (qk) is both scattered first and worked first;
        #   - the longest compute (mlp PE) goes second so it covers the 368 MB
        #     value_embeds RS, which is scattered right behind the banks;
        #   - every collective a work item waits on is enqueued no later than that item's
        #     position in work_order (stock violates this badly: mudd_* are appended to
        #     param_table last, i.e. behind the 368 MB value_embeds RS, yet mudd_b2 is
        #     work item #6 -> ~1.2 ms of dead compute stream on every odd step);
        #   - the *last* all_gather enqueued is the smallest one (vo, 27 MB) instead of
        #     the largest (mlp, 108 MB), since that one is the exposed tail.
        if COMM_ORDER:
            def _cr_perm(labels, head, tail):
                """Permute `labels`: `head` first (in head order), `tail` last, the rest
                keep their stock relative order. Labels absent from `labels` are ignored,
                so this stays valid if the model gains/loses a param."""
                labels = list(labels)
                h = [l for l in head if l in labels]
                t = [l for l in tail if l in labels and l not in h]
                mid = [l for l in labels if l not in h and l not in t]
                return h + mid + t

            self.scatter_order = _cr_perm(
                self.scatter_order,
                head=["qk_bank", "vo_bank", "mlp_bank",   # smallest->largest, all early
                      "value_embeds",                     # longest RS, consumed mid-phase
                      "lm_head", "embed", "bigram_embed"],
                tail=[])                                  # 13 tiny all_reduces last
            self.work_order = _cr_perm(
                self.work_order,
                head=["qk_bank",        # cheapest RS -> compute starts ~0.05 ms in
                      "mlp_bank",       # longest compute -> covers the value_embeds RS
                      "lm_head", "embed",  # lm_head before embed sync (Phase 3)
                      "value_embeds",   # 368 MB AG enqueued before the tail
                      "bigram_embed"],
                tail=["vo_bank"])       # smallest AG is the exposed tail
            # ---- Alternative comm-order candidates (pure perms,
            # bitwise identical by the same argument as COMM_ORDER itself) ----
            _crv = ""
            if _crv == "2":
                # C2 lm_head-early: the binding constraint is the tail
                # AGs; enqueue the 77MB lm_head AG (+ its Phase-3 embed transpose_copy
                # dependency) before the mlp PE block so bank compute covers it.
                self.work_order = _cr_perm(
                    self.work_order,
                    head=["qk_bank", "lm_head", "mlp_bank", "embed", "value_embeds", "bigram_embed"],
                    tail=["vo_bank"])
                self.scatter_order = _cr_perm(
                    self.scatter_order,
                    head=["qk_bank", "vo_bank", "lm_head", "mlp_bank", "value_embeds", "embed", "bigram_embed"],
                    tail=[])
            elif _crv == "3":
                # C3 VE-RS-earliest: the 368MB VE RS can start behind
                # the two small bank RSs and finish before its work item unaided.
                self.scatter_order = _cr_perm(
                    self.scatter_order,
                    head=["qk_bank", "vo_bank", "value_embeds", "mlp_bank", "lm_head", "embed", "bigram_embed"],
                    tail=[])

        adam_defaults = dict(
            lr=float("0.008"),
            eps=1e-10,
            weight_decay=0.005,
        )

        anvil_defaults = dict(
            lr=float("0.023"),
            momentum=0.95,
            beta2=float("0.9"),
            weight_decay=float('1.5'),
        )

        self.optimizer = AnvilAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=self.scatter_order,  # Dict order defines scatter priority (COMM_ORDER permutes)
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            anvil_defaults=anvil_defaults,
        )

        # Split embed from lm_head at 2/3 of training (on an odd step so Adam updates)
        self.split_step = training_schedule.split_step

        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = training_schedule.ws_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            ws_short = self.ws_short * self.block_size,
            ws_long = self.ws_long * self.block_size,
            train_max_seq_len = self.train_max_seq_len,
            qkscaf = self.qkscaf,
            ptp_w = getattr(self, "_ptp_w_now", -1.0),
        )

    def _is_adam_step(self, step: int):
        """Adam params are only updated on odd steps."""
        return step % 2 == 1

    def get_transition_steps(self):
        return [start for start, _ in training_schedule.boundaries[1:]]

    def advance_schedule(self, step: int):
        """Apply this step's schedule: window sizes (YaRN on ws_long changes), batch
        geometry (queued for the loader via .send), MTP weights, prefix-CE ramp."""
        # KX_QKSCAF: the scaffold phase is stage 1 and nothing else. The flip is a python
        # bool that changes traced shapes for the 8 narrow layers, so it MUST ride an
        # existing recompile boundary -- it is keyed to get_transition_steps()[0], the
        # stage-1 -> stage-2 step, which already recompiles (batch size 8->16 tokens*2048,
        # train_max_seq_len 896->2048 and both window sizes all change there). Off: the
        # boundary is 0 and this is a constant False.
        self.qkscaf = step < self._qkscaf_until
        stage, _ = training_schedule.lookup(step)
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)
            self.model.yarn_paired_head.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)

        new_batch_size = stage.batch_size
        new_train_max_seq_len = stage.train_max_seq_len
        if new_batch_size != self.batch_size or new_train_max_seq_len != self.train_max_seq_len:
            self.train_loader_send_args = (new_batch_size, new_train_max_seq_len, grad_accum_steps)
            self.batch_size = new_batch_size
            self.train_max_seq_len = new_train_max_seq_len
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = training_schedule.mtp_weights[step]
            # PR#337 upstream schedule: stage1 flat 0.25; stage2 ramp 0.25->0 (quartered
            # piecewise to bound compile variants); stage3/ext 0. Values sampled at
            # warmup transition steps compile ahead; mid-stage flips pay one small
            # variant compile each (loss screens: warm caches; wall runs: warm first).
        _b = training_schedule.boundaries
        _s1e, _s2e = _b[0][1], _b[1][1]
        if step < _s1e:
            self._ptp_w_now = 0.25
        elif step < _s2e:
            _t = (step - _s1e) / max(1, _s2e - _s1e)
            self._ptp_w_now = [0.20, 0.15, 0.10, 0.05][min(3, int(_t * 4))]
        else:
            self._ptp_w_now = 0.0
        _fk._PTP_W_RUNTIME = self._ptp_w_now

    def step_optimizers(self, step: int):
        step_lr = training_schedule.get_lr(step)
        rail_beta = get_rail_beta(step)
        do_adam = self._is_adam_step(step)

        if _DUAL_MOMENTUM is not None:
            # DUAL_MOMENTUM engage switch, expressed as scalar-tensor fills so the
            # anvil_cascade graph never recompiles: before engage_step the fast
            # rail runs the stock scheduled-beta update with blend weight 1
            # (bit-identical trajectory for that rail); from engage_step onward
            # it runs fast_beta with blend weight fast_w against the slow rail.
            _bmx_eng = step >= _DUAL_MOMENTUM[3]
            self.optimizer._bimax_bf_t.fill_(_DUAL_MOMENTUM[0] if _bmx_eng else rail_beta)
            self.optimizer._bimax_w_t.fill_(_DUAL_MOMENTUM[2] if _bmx_eng else 1.0)


        # Update learning rates and momentum for all params
        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr * _KX_GLR_CONST
            if _KX_QKLRMW_N > 0 and p_cfg.label == "qk_bank" and step < _KX_QKLRMW_N:
                # qk-early-protect: lower qk_bank lr for step<N (the formation phase
                # punishes qk aggression). Disabled in this configuration
                # (_KX_QKLRMW_N=0). Host float via p_cfg.lr -> eff_lr fill,
                # recompile-safe.
                p_cfg.lr = p_cfg.lr * _KX_QKLRMW_M
            if p_cfg.optim == "anvil":
                p_cfg.momentum = rail_beta

        # KX_VEPER: give value_embeds its own (longer) Adam period. Its gradient keeps
        # accumulating on the skipped steps exactly like Adam's stock every-other-step
        # accumulation, so no signal is dropped -- but the *content* of the gradient an
        # Adam update sees changes (sum over KX_VEPER steps instead of 2), so this is a
        # loss-affecting change and is deliberately independent of COMM_ORDER.

        # KX_TRISYNC: K-step FedAvg cadence for local_kavg tables (trigram_embed).
        # Fires on step % K == 0 independent of the Adam parity (the seam inside
        # optimizer.step() awaits the async weight all_reduce either right before
        # the table's Adam update or in Phase 3). The pre-val guard in the main
        # loop is the hard correctness backstop; this is the periodic averaging.

        # BIGRAM_SPARSE_GRAD/KX_BGLITE: fold this Adam cycle's harvested per-token bigram gradient
        # values into the payload the optimizer consumes (compact rows, or the persistent
        # dense buffer). Must run after the last backward and before Phase 1.
        if _bigram_sink_on:
            self.bgsp_build(do_adam)

        # Step optimizer with do_adam flag
        self.optimizer.step(do_adam=do_adam)

        # At split step: copy lm_head optimizer state to embed and mark as split
        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)

        # Reset NorMuon momentum buffers and split_embed state
        self.optimizer.reset()

        stage, _ = training_schedule.lookup(0)
        # KX_QKSCAF: first stage boundary == the truncation step (0 when the knob is off,
        # which makes `self.qkscaf` a constant False everywhere).
        self._qkscaf_until = 0
        self.qkscaf = False
        self.ws_short, self.ws_long = stage.window_sizes
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()
        if _sparse_comms_active():
            self.row_update_mask = np.zeros(args.bigram_vocab_size, dtype=np.uint8)
            self.sparse_counts_state = None
            # buffer we use for fast GPU uploads of send indexes
            self.send_idxes_buffer = torch.empty(args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)

        if _bigram_sink_on:
            # Persistent buffers survive the post-warmup reset (stable addresses, no
            # re-allocation); only the per-cycle state is cleared here.
            if getattr(self, "_bgsp_sinks", None) is None:
                _bdt = self.model.bigram_embed.weight.dtype
                self._bgsp_sinks = {}
                if BIGRAM_SPARSE_GRAD:
                    # Compact capacity: an Adam cycle is at most 2 steps (odd-step
                    # Adam), so the unique-row count cannot exceed
                    # twice the largest per-rank token count. Asserted at build time.
                    _cap_tok = max(s.batch_size for s in TRAINING_STAGES) // (world_size * grad_accum_steps)
                    _cap = min(args.bigram_vocab_size, 2 * _cap_tok + 64)
                    self._bgsp_buf = torch.zeros(_cap, args.bigram_dim, device=device,
                                                 dtype=(torch.float32 if BIGRAM_SPARSE_GRAD == 2 else _bdt))
                    self._bgsp_shard = torch.zeros(args.bigram_vocab_size // world_size,
                                                   args.bigram_dim, dtype=_bdt, device=device)
                    print0(f"[bgsp] BIGRAM_SPARSE_GRAD={BIGRAM_SPARSE_GRAD} compact_cap={_cap} rows "
                           f"({self._bgsp_buf.numel() * self._bgsp_buf.element_size() / 2**20:.0f} MB) "
                           f"shard={self._bgsp_shard.shape[0]} rows "
                           f"({self._bgsp_shard.numel() * 2 / 2**20:.0f} MB); dense table grad "
                           f"({args.bigram_vocab_size * args.bigram_dim * 2 / 2**20:.0f} MB/backward) retired; "
                           f"grad-sink forward supersedes BIGRAM_BF16_BWD={int(BIGRAM_BF16_BWD)} on the train path",
                           console=True)
            for _s in self._bgsp_sinks.values():
                _s.grad = None
            self._bgsp_pending = []
            self._bgsp_idx = None
            self._bglite_zero_next = False
            self.optimizer._bgsp_state = None


    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def sparse_index_update(self, step, bigram_indexes, trigram_indexes=None):
        """Accumulate this step's touched bigram rows; on Adam steps, freeze the
        sorted-unique row list and start the sparse count exchange."""
        if not _sparse_comms_active():
            return

        self.row_update_mask[bigram_indexes] = 1

        if self._is_adam_step(step):
            with torch.no_grad():
                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)
                # BIGRAM_SPARSE_GRAD: np.flatnonzero over the accumulated row mask IS the sorted,
                # unique row list for this Adam cycle -- exactly the layout the compact
                # gradient buffer uses, so capture it (and the rank partition) here.
                _bgo = {} if BIGRAM_SPARSE_GRAD else None
                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(
                    bigram_idx_np, args.bigram_vocab_size, rank, world_size, self.send_idxes_buffer, _bgo
                )
                if _bgo is not None:
                    self._bgsp_idx = _bgo
                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)

    def sparse_index_share(self, step):
        if not _sparse_comms_active() or not self._is_adam_step(step):
            return

        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state
        self.sparse_counts_state = None

        recv_counts_fut.wait()
        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts)
        self.optimizer._reduce_futures[model.bigram_embed.weight] = [idxes_fut, recv_idxes]
        self.optimizer._sparse_async_data[model.bigram_embed.weight] = sparse_state

        self.row_update_mask.fill(0)


    # ---- BIGRAM_SPARSE_GRAD / KX_BGLITE: grad-sink plumbing ----

    def bgsp_sink(self, n_tokens: int):
        """The zeros [T, bigram_dim] leaf added to the (detached) bigram lookup. One per
        distinct train token-count -- a handful, all of them exercised in warmup, so the
        extra compile variant never lands on the clock. Passed as a forward ARGUMENT, so
        dynamo sees an ordinary graph input and AOTAutograd returns its grad."""
        s = self._bgsp_sinks.get(n_tokens)
        if s is None:
            s = torch.zeros(n_tokens, args.bigram_dim, device=device,
                            dtype=self.model.bigram_embed.weight.dtype, requires_grad=True)
            self._bgsp_sinks[n_tokens] = s
        return s

    def bgsp_harvest(self, bigram_indexes, sink):
        """Called immediately after loss.backward(). Takes ownership of sink.grad (no
        copy, no zero-fill) and clears the slot so the next backward's AccumulateGrad
        installs a fresh tensor instead of accumulating across steps."""
        vals = sink.grad
        assert vals is not None, "[bgsp] sink got no gradient -- the bigram channel is not wired into the loss"
        sink.grad = None
        self._bgsp_pending.append((bigram_indexes, vals))

    def bgsp_build(self, do_adam):
        """Between the last backward and the optimizer's Phase 1: turn this Adam cycle's
        harvested per-token value streams into the payload the optimizer consumes."""
        if not do_adam:
            self.optimizer._bgsp_state = None
            return
        st = self._bgsp_idx
        assert st is not None and self._bgsp_pending, "[bgsp] Adam step with no row list / no harvested values"
        n = st["n"]
        if n > self._bgsp_buf.shape[0]:
            # The timed loop runs consecutive steps, so a cycle is 1-2 steps and n is
            # bounded by the pre-sized capacity. WARMUP samples NON-adjacent steps, so a
            # cycle there can span 3+ sampled steps: grow once (untimed) and keep it.
            _grow = min(args.bigram_vocab_size, n + 4096)
            print0(f"[bgsp] growing compact buffer {self._bgsp_buf.shape[0]} -> {_grow} rows", console=True)
            self._bgsp_buf = torch.zeros(_grow, args.bigram_dim, device=device,
                                         dtype=self._bgsp_buf.dtype)
        compact = self._bgsp_buf[:n]
        full_idx = st["full_idx"]
        compact.zero_()
        for _idx, _vals in self._bgsp_pending:
            # Every touched row is in the mask by construction, so searchsorted over the
            # sorted-unique list is an exact global-row -> compact-row map.
            _pos = torch.searchsorted(full_idx, _idx, out_int32=True)
            compact.index_add_(0, _pos, _vals if _vals.dtype == compact.dtype else _vals.to(compact.dtype))
        self._bgsp_pending = []
        self._bgsp_idx = None
        self.optimizer._bgsp_state = {
            "compact": compact, "full_idx": full_idx, "lo": st["lo"], "hi": st["hi"],
            "shard": self._bgsp_shard, "dtype": self._bgsp_shard.dtype,
        }


# -----------------------------------------------------------------------------
# CUDA_GRAPH_TIER: manual CUDA-graph capture of the optimizer-phase compute segments.
# Split architecture: every collective, future.wait(), transpose_add/copy and
# the sparse bigram path stays EAGER; only the collective-free compute between them is
# captured. Replays are launched on the compute stream, so PG-NCCL's fork/join events
# order them against the eager collectives exactly as the eager kernels were -- the
# COMM_ORDER comm schedule is untouched.

def _cg_static_check():
    """A captured graph bakes every python-level branch it recorded.
    Refuse capture for any knob that could flip a captured branch mid-run (the step
    loop then stays eager for that family -- correct, just slower). Returns
    (bank_reasons, tiny_reasons, quant_reasons); empty list = family capturable."""
    bank_reasons = []
    if world_size == 1:
        bank_reasons.append("world_size==1 (comms=none path reads per-step param.grad, unstable address)")
    tiny_reasons = []
    if world_size == 1:
        tiny_reasons.append("world_size==1 (no _far_views)")
    quant_reasons = []
        # Note: the `boot = _a2p_calls < 16` branch is frozen False long before
        # capture (warmup + post-reset make >=20 quantize calls), so listing it as a
        # quant reason is conservative for the bank/tiny tiers only.
    quant_reasons.append("FP8_LAGGED_QUANT=1 (host-side 16-call bootstrap counter inside quantize_mlp_fp8)")
    # CUDA_GRAPH_TIER=1 composed with the fp8-wgrad family NaN'd mid-run
    # reproducibly while each feature alone is clean. Audit found NO wgrad-dependent
    # branch and NO backward-produced tensor inside any captured segment (bank graphs
    # read only the persistent RS outputs, optimizer state, param slices and _cg_dev),
    # so the suspect channel is allocator-level: the graph private pool x
    # expandable_segments interaction under the wgrad configs' larger memory footprint
    # (extra fp8 caches + backward transients). Until root-caused, ALL captures are
    # refused under these flags -> graceful eager fallback (the fp8 wgrad win outranks
    # the CG win).
    wgrad_on = [n for n, v in (("FP8_MLP_DW2", FP8_MLP_DW2), ("FP8_MLP_DX", FP8_MLP_DX), ("FP8_MLP_DW1", FP8_MLP_DW1),
                               ("FP8_DROP_PRE", FP8_DROP_PRE), ("KX_GE4", False),
                               # S3 attention-backward features: same allocator risk class
                               # (extra fp8 caches + backward transients); refuse
                               # capture until the allocator interaction is
                               # root-caused. In this configuration the MLP wgrad trio
                               # already forces eager, so this costs nothing new.
                               ("KX_S3D", False), ("KX_S3W", bool(0))) if v]
    if wgrad_on:
        r = ("+".join(wgrad_on) + " set: CG x fp8-wgrad composition NaN'd mid-run "
             "reproducibly; capture refused pending root-cause (suspect graph private "
             "pool x expandable_segments) -- running eager")
        bank_reasons.append(r)
        tiny_reasons.append(r)
            # The NaN incident composed BANK captures with wgrad.
            # G_quant reads only persistent post-step state (fp8 weight caches, optimizer
            # state, far flats) — no backward-produced tensor, no wgrad-dependent branch.
            # Banks/tiny stay refused above.
        quant_reasons.append(r)
    return bank_reasons, tiny_reasons, quant_reasons


class _CGGraphs:
    """Replay handles for the captured graphs. G_qk/G_vo/G_mlp/G_tiny live on the
    optimizer (replayed inside step()); G_quant lives here so the timed loop can
    replay it in place of model.quantize_mlp_fp8()."""
    def __init__(self, opt):
        self.opt = opt
        self.quant = {}  # {True/False: graph} (refresh_lm parity variants) or {"any": graph}

    @property
    def has_quant(self):
        return bool(self.quant)

    def replay_quant(self, refresh_lm: bool = True):
        if not self.opt._cg_ptr_checked:
            self.opt._cg_assert_ptrs()
        self.quant.get(refresh_lm, self.quant.get("any")).replay()


@torch.no_grad()
def _cg_build_graphs(training_manager, model):
    """Capture all enabled graph families. Runs in the untimed post-reset window:
    strictly after training_manager.reset()'s load_state_dict (final addresses) and
    after the post-reset quantize_mlp_fp8() (fp8 caches exist), strictly before the
    training clock starts. Any capture failure falls back to eager with a warning."""
    opt = training_manager.optimizer
    mraw = getattr(model, "_orig_mod", model)
    holder = _CGGraphs(opt)

    bank_r, tiny_r, quant_r = _cg_static_check()
    banks_ok = CUDA_GRAPH_TIER == 1 and not bank_r
    tiny_ok = CUDA_GRAPH_TIER == 1 and not tiny_r
    quant_ok = not quant_r
    print0(f"[cg] CUDA_GRAPH_TIER={CUDA_GRAPH_TIER} banks={'ON' if banks_ok else 'off'} "
           f"tiny={'ON' if tiny_ok else 'off'} quant={'ON' if quant_ok else 'off'}", console=True)
    if CUDA_GRAPH_TIER == 2:
        print0("[cg]   tier 2: G_quant only by design (banks/tiny not attempted)", console=True)
    for r in (bank_r if CUDA_GRAPH_TIER == 1 else []):
        print0(f"[cg]   WARNING: banks stay eager: {r}", console=True)
    for r in (tiny_r if CUDA_GRAPH_TIER == 1 else []):
        print0(f"[cg]   G_tiny stays eager: {r}", console=True)
    for r in quant_r:
        print0(f"[cg]   WARNING: G_quant stays eager: {r}", console=True)
    if not (banks_ok or tiny_ok or quant_ok):
        return holder

    # Provenance: the exact frozen-branch set the captures bake.
    print0("[cg] static set: " + " ".join(f"{k}={v}" for k, v in (
        ("FLAT_ALLREDUCE", int(FLAT_ALLREDUCE)),
        ("FP8_MLP_FWD", int(FP8_MLP_FWD)), ("FP8_MLP_DPRE", int(FP8_MLP_DPRE)), ("FP8_MLP_DX", int(FP8_MLP_DX)),
        ("FP8_MLP_DW1", int(FP8_MLP_DW1)), ("KX_C7", int(FP8_LMHEAD_CACHE)), ("KX_C2S", int(False)),
        ("FP8_LAGGED_QUANT", int(FP8_LAGGED_QUANT)), ("KX_HG", int(HOST_OPT)), ("KX_LHXD", int(False)))))

    # ---- scalar staging: one pinned host vector + one device vector ----
    if banks_ok or tiny_ok:
        slots: dict = {}
        bank_labels: list = []
        if banks_ok:
            for label in opt.work_order:
                cfg = opt.param_cfgs[opt._param_by_label[label]]
                if cfg.optim == "anvil" and cfg.comms == "sharded" and label in opt._cg_rs_bufs:
                    bank_labels.append(label)
                    slots[label + ".mom"] = len(slots)   # rail beta
                    slots[label + ".wd"] = len(slots)    # eff_wd
                    if cfg.per_matrix_lr_mul is None:
                        slots[label + ".lr"] = len(slots)  # eff_lr
                    else:
                        for i in range(cfg.chunk_size):    # per-matrix lr
                            slots[f"{label}.lr{i}"] = len(slots)
        far_list: list = []
        if tiny_ok:
            far_list = [opt._param_by_label[l] for l in opt.work_order
                        if opt._param_by_label[l] in opt._far_views]
            for p in far_list:
                lbl = opt.param_cfgs[p].label
                slots[lbl + ".ss"] = len(slots)   # Adam step_size
                slots[lbl + ".awd"] = len(slots)  # Adam eff_wd
        opt._cg_slot = slots
        # Pinned RING (x per-slot events, see _cg_stage): one row per in-flight step so
        # the run-ahead host can never overwrite a row before its H2D DMA executes.
        _CG_RING = 8
        opt._cg_host = torch.zeros((_CG_RING, max(len(slots), 1)), dtype=torch.float32, pin_memory=True)
        opt._cg_hostnp = opt._cg_host.numpy()
        opt._cg_evts = [torch.cuda.Event() for _ in range(_CG_RING)]
        opt._cg_ring = 0
        opt._cg_dev = torch.zeros(max(len(slots), 1), dtype=torch.float32, device=device)
        opt._cg_views = {k: opt._cg_dev[i] for k, i in slots.items()}
        opt._cg_bank_labels = bank_labels
        opt._cg_tiny_params = far_list
        # Seed the device scalars with current (post-warmup) cfg values so capture-time
        # warmup runs on sane magnitudes. do_adam=False: no step-count bumps here.
        opt._cg_stage(do_adam=False)
        torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()  # one pool shared by all graphs
    side = torch.cuda.Stream()

    def _capture(name, seg, touched):
        """Capture recipe: snapshot touched state, warm the segment on a side stream
        (JIT/inductor recompiles for the CPU->CUDA scalar move land here, untimed),
        capture into the shared pool, restore state exactly."""
        snap = [(t, t.clone()) for t in touched]
        g = None
        try:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    seg()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            # thread_local error mode: the PG-NCCL watchdog thread keeps issuing
            # cudaEventQuery while we capture; global mode would fail the capture on
            # those (same reason inductor's cudagraph trees use thread_local).
            with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
                seg()   # records; does not execute
        except Exception as e:
            print0(f"[cg] WARNING: capture of {name} failed ({type(e).__name__}: {e}); "
                   f"{name} stays eager", console=True)
            g = None
        finally:
            torch.cuda.synchronize()
            for t, s in snap:
                t.copy_(s)
            torch.cuda.synchronize()
        if g is not None:
            print0(f"[cg] captured {name}", console=True)
        return g

    captured = []

    # ---- G_qk / G_vo / G_mlp ----
    if banks_ok:
        for label in opt._cg_bank_labels:
            param = opt._param_by_label[label]
            cfg = opt.param_cfgs[param]
            st = opt.param_states[param]
            gbuf = opt._cg_rs_bufs[label]
            v = opt._cg_views
            mom_v, wd_v = v[label + ".mom"], v[label + ".wd"]
            lr_v = v.get(label + ".lr")
            per_lr = ([v[f"{label}.lr{i}"] for i in range(cfg.chunk_size)]
                      if cfg.per_matrix_lr_mul is not None else None)
            def seg(param=param, cfg=cfg, gbuf=gbuf, mom_v=mom_v, lr_v=lr_v,
                    wd_v=wd_v, per_lr=per_lr):
                opt._anvil_compute(param, gbuf, cfg, rank,
                                     mom_v, lr_v, wd_v, per_lr)

            # Capture snapshot/restore set: the FAST velocity rail (rail 0), the
            # lane energies and the mantissa sidecar. The slow rail sits outside
            # the snapshot, so the capture-warmup passes advance it exactly as
            # ordinary pre-clock steps would.
            touched = [st["velocity"][0]] + [st[k] for k in ("lane_energy",
                                       "mantissa") if k in st]
            p_slice = param.data.view(cfg.reshape)[rank * cfg.chunk_size:(rank + 1) * cfg.chunk_size]
            touched.append(p_slice)
            g = _capture(f"G_{label}", seg, touched)
            if g is not None:
                opt._cg_graphs[label] = g
                captured.append(f"G_{label}")
                opt._cg_record_ptrs([(f"{label}.param", param.data), (f"{label}.rs", gbuf)]
                                    + [(f"{label}.{k}", st[k]) for k in
                                       ("velocity", "lane_energy", "mantissa") if k in st])

    # ---- G_tiny (replicated Adam block; odd steps; FLAT_ALLREDUCE is the enabler) ----
    if tiny_ok and opt._cg_tiny_params:
        far_list = opt._cg_tiny_params

        def tiny_seg():
            for p in far_list:
                cfg = opt.param_cfgs[p]
                st = opt.param_states[p]
                AnvilAndAdam._adam_update_step(
                    p, opt._far_views[p], st["exp_avg"], st["exp_avg_sq"],
                    cfg.adam_betas[0], cfg.adam_betas[1], cfg.eps,
                    opt._cg_views[cfg.label + ".ss"], opt._cg_views[cfg.label + ".awd"])

        touched = []
        for p in far_list:
            st = opt.param_states[p]
            touched += [p.data, st["exp_avg"], st["exp_avg_sq"]]
        g = _capture("G_tiny", tiny_seg, touched)
        if g is not None:
            opt._cg_tiny = g
            captured.append(f"G_tiny[{len(far_list)}]")
            ptrs = [(f"far.{dt}", fl) for dt, fl in opt._far_flats.items()]
            for p in far_list:
                lbl = opt.param_cfgs[p].label
                st = opt.param_states[p]
                ptrs += [(f"{lbl}.param", p.data), (f"{lbl}.exp_avg", st["exp_avg"]),
                         (f"{lbl}.exp_avg_sq", st["exp_avg_sq"])]
            opt._cg_record_ptrs(ptrs)

    # ---- G_quant ----
    if quant_ok:
        qattrs = [a for a in ("_mlp_up_proj_f8", "_mlp_up_proj_scales", "_mlp_up_f8_col",
                              # _mlp_static_dq is written by G_quant every
                              # replay (torch.mul out=); _mlp_xs_t is alloc-only
                              # (never touched post-capture, like _mlp_gs_t).
                              "_mlp_static_dq",
                              "_mlp_down_f8", "_mlp_down_f8_row", "_mlp_down_scales",
                              "_mlp_post_scale", "_mlp_post_amax", "_mlp_dq_bwd",
                              "_mlp_dpre_scale", "_mlp_dpre_amax", "_lm_f8",
                              "_attn_f8", "_attn_ws",
                              # S3: every tensor the KX_S3D refresh block
                              # touches — a missed buffer under captured replay is a
                              # silent WAR race. _s3_xt/_s3_gt/
                              # _s3_yt/_s3_got are NOT touched by G_quant (they
                              # are written in the compiled fwd/bwd), only
                              # address-stability matters for them (alloc-once).
                              "_s3_qkv_col", "_s3_qws", "_s3_g_scale", "_s3_g_amax",
                              "_s3_y_scale", "_s3_y_amax", "_s3_go_scale", "_s3_go_amax",
                              # The per-SM amax accumulators are written by
                              # the quantize kernels every replay — they must be on the
                              # snapshot/restore + pointer-check lists.
                              "_w_up_pamax", "_w_dn_pamax",
                              # _KX_LMF8T: the col-major lm_head cache is written
                              # inside the captured body (same class as _lm_f8).
                              "_lm_f8_cm")
                  if hasattr(mraw, a)]
        touched = [getattr(mraw, a) for a in qattrs]
        # HOST_OPT refreshes the lm_head fp8 copy only on Adam (odd) steps -> the branch
        # is per-step, so capture BOTH parity variants; otherwise one graph suffices.
        variants = ([(True, 'G_quant_lm'), (False, 'G_quant_nolm')])
        qgraphs = {}
        for rlm, name in variants:
            g = _capture(name, lambda rlm=rlm: mraw.quantize_mlp_fp8(refresh_lm=rlm), touched)
            if g is None:
                qgraphs = {}
                break  # never mix a captured parity with an eager one
            qgraphs[rlm if len(variants) > 1 else "any"] = g
        if qgraphs:
            holder.quant = qgraphs
            captured += [nm for _, nm in variants]
            opt._cg_record_ptrs([("mlp_bank", mraw.mlp_bank.data),
                                 ("lm_head.weight", mraw.lm_head.weight.data)]
                                + [(a, getattr(mraw, a)) for a in qattrs])

    opt._cg_active = bool(opt._cg_graphs) or opt._cg_tiny is not None
    if opt._cg_active:
        opt._cg_record_ptrs([("cg.dev", opt._cg_dev)])
    print0(f"[cg] captured: {captured if captured else 'nothing (fully eager)'}; "
           f"reserved {torch.cuda.memory_reserved() // (1 << 20)} MiB", console=True)
    return holder


# -----------------------------------------------------------------------------
# int main

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
_logf = None
if master_process and True:
    # HOST_OPT: one persistent block-buffered handle instead of an open()+close() per step
    # inside the timed loop; explicit flush at the untimed val breaks + at exit.
    import atexit as _atexit
    _logf = open(logfile, "a", buffering=1 << 20)
    _atexit.register(lambda: (_logf.flush(), _logf.close()))

def print0(s, console=False):
    if master_process:
        if _logf is not None:
            # byte-identical output to the reopen path (same print(), same file, append order)
            if console:
                print(s)
            print(s, file=_logf)
            return
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")
if KX_SLG_LOAD:
    print0(
        "[slg] ON value_embeds gather backward: indexed adjoint load, stock grid/atomics",
        console=True,
    )

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

# KX_SEED (reproduction runs only): common-random-numbers init. Init is otherwise UNSEEDED
# (fresh network every run — the dominant term in the measured per-run sigma~1.6mb).
# Seeded A/B pairs share init+data, leaving only atomic/reduction noise. Default-off.
_train_seed = int(os.environ.get("KX_SEED", "0"))
if _train_seed:
    torch.manual_seed(_train_seed)
    torch.cuda.manual_seed_all(_train_seed)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
model.qk_bank.data = model.qk_bank.data.bfloat16()
model.vo_bank.data = model.vo_bank.data.bfloat16()
model.mlp_bank.data = model.mlp_bank.data.bfloat16()
model.mudd_w1.data = model.mudd_w1.data.bfloat16()
model.mudd_w2.data = model.mudd_w2.data.bfloat16()
model.mudd_b2.data = model.mudd_b2.data.bfloat16()
model.mudd_gate_w1.data = model.mudd_gate_w1.data.bfloat16()
model.mudd_gate_w2.data = model.mudd_gate_w2.data.bfloat16()
model.mudd_gate_b2.data = model.mudd_gate_b2.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)
dist.broadcast(model.bigram_sign_table, 0)  # buffer, not in parameters()
model.quantize_mlp_fp8()
# Resolve the fused kernels' num_stages cache EAGERLY, before torch.compile ever
# traces the MLP host code. Under compile the OutOfResources retry loop can neither
# run nor record (module-state mutation inside the autograd-Function HOP is a dynamo
# hard error), so the compiled path is a read-only cache
# lookup; this call is what fills it for the fp8 emit variants.
# Untimed (pre-warmup) and one-time; the Triton binaries it builds are reused.
prime_stage_cache()

_kx_probe = int("0")
_probe_events: dict[str, list] = {}

_kx_census = int("0")

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
training_manager = TrainingManager(model)
if COMM_ORDER or 0 > 2:
    print0(f"[cr] COMM_ORDER={COMM_ORDER} KX_VEPER={0}")
    print0(f"[cr] scatter_order={training_manager.scatter_order}")
    print0(f"[cr] work_order={training_manager.work_order}")


########################################
#            Warmup kernels            #
########################################
print0("Compiling model and warming up kernels (~7 minutes on first execution)", console=True)
# Warmup the training kernels, then re-initialize the state so we aren't cheating
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizer=training_manager.get_state()) # save the initial state
train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps)
val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

transition_steps = training_manager.get_transition_steps()
# first and last pair of steps in each transition
warmup_steps = sorted({0, 1} | {s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 2}
                      | {training_schedule.split_step + offset for offset in [1, 2, 3]
                         if 2 <= training_schedule.split_step + offset <= args.num_scheduled_iterations + args.num_extension_iterations - 1})
print0(f"Sampling steps {warmup_steps} for warmup", console=True)
for step in warmup_steps:
    training_manager.advance_schedule(step)
    model.eval()
    with torch.no_grad():
        inputs, targets, cum_seqlens, bigram_inputs, *_ = next(val_loader)
        model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).mean()
    model.train()
    for idx in range(grad_accum_steps):
        send_args = training_manager.train_loader_send_args
        _batch = train_loader.send(send_args)
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = _batch[:5]
        training_manager.sparse_index_update(step, bigram_cpu, None)
        if _bigram_sink_on:
            _bg_sink = training_manager.bgsp_sink(bigram_inputs.shape[0])
            loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(), _bg_sink).sum() * grad_scale
        training_manager.sparse_index_share(step)
        loss.backward()
        if _bigram_sink_on:
            training_manager.bgsp_harvest(bigram_inputs, _bg_sink)
        del loss
    training_manager.step_optimizers(step)
    model.quantize_mlp_fp8()
print0("Resetting Model", console=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
training_manager.reset(initial_state["optimizer"])
del val_loader, train_loader, initial_state
model.quantize_mlp_fp8()
model.train()

# ---- CUDA_GRAPH_TIER: capture CUDA graphs in the untimed post-reset window ----
# Placement is load-bearing: strictly AFTER training_manager.reset()'s load_state_dict
# (every optimizer state tensor must already hold its final address) and after
# the post-reset quantize_mlp_fp8() (fp8 caches allocated), strictly BEFORE the clock
# starts at torch.cuda.synchronize()/t0 below, so warmup+capture cost is untimed.
_cg = None
if CUDA_GRAPH_TIER:
    _cg = _cg_build_graphs(training_manager, model)

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps)

gc.collect()
if HOST_OPT:
    # HOST_OPT: no cyclic garbage may be collected inside the timed loop (gen-2 pauses are
    # multi-ms and desynchronise ranks); freeze the post-warmup heap and collect
    # manually at every untimed val break so cycles still cannot accumulate.
    gc.freeze()
    gc.disable()

def _extra_val():
    """One extra validation pass over the model's *current* weights and the current
    TrainingManager forward args. Used by last-step diagnostics only (window ladders);
    untimed, since `training_time_ms` is never advanced again after the last-step val."""
    if _KX_R2:
        for _m in model.modules():
            if isinstance(_m, Yarn):
                _m.r2_ensure_full()
    _n = grad_accum_steps * args.val_tokens // args.val_batch_size
    vl = 0
    vloader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
    with torch.no_grad():
        for _ in range(_n):
            iv, tv, cv, bv, _ = next(vloader)
            vl += model(iv, tv, cv, bv, training_manager.get_forward_args()).mean()
    vl /= _n
    dist.reduce(vl, 0, op=dist.ReduceOp.AVG)
    return vl.item()

if _KX_R2:
    # Partial Yarn rebuild: training-indexable row bound for partial rebuilds = max
    # tokens-per-rank-per-accum-step over all stages (the packed rotary T). Set only
    # now so every pre-clock reset()/apply() built full tables.
    _R2_YARN_ROWS = max(s.batch_size for s in TRAINING_STAGES) // (world_size * grad_accum_steps)
    print0(f"[r2] yarn partial-rebuild rows={_R2_YARN_ROWS}", console=True)

training_time_ms = 0
# The first .send() on the lazy train_loader generator (first-shard read + pinned
# alloc + BOS scan) is paid INSIDE the timed region at step 0, matching the stock
# convention. _pref_batch stays None here so step 0 takes the synchronous-fetch path.
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
if PREFIX_CE:
    _ptp_table_fill()  # prefix-table construction charged to training time (record #88 convention)
# begin training
train_steps = training_schedule.total_steps
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    training_manager.advance_schedule(step)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            training_manager.apply_final_ws_ext()
        if last_step and TAIL_EMA_WINDOW > 0 and _tema_bufs:
            # TailEMA terminal ship, ON the clock: the terminal blend is an O(params)
            # lerp costing single-digit ms; keeping it inside the timed section means
            # the counted weights are exactly the weights that exist when the clock
            # stops. Labels are disjoint
            # from the tavg ship below, so relative order is value-irrelevant.
            with torch.no_grad():
                for _lbl, _bar in _tema_bufs.items():
                    _p = training_manager.optimizer._param_by_label[_lbl]
                    _p.data.copy_(torch.lerp(_p.data.float(), _bar, TAIL_EMA_BLEND).to(_p.dtype))
            print0(f"[tema] shipped win={TAIL_EMA_WINDOW} blend={TAIL_EMA_BLEND} labels={sorted(_tema_bufs)}", console=True)
        if _KX_R2:
            # Restore any partially-rebuilt Yarn factor rows before eval
            # touches val-length sequences. On the clock (runs before the timer read).
            for _m in model.modules():
                if isinstance(_m, Yarn):
                    _m.r2_ensure_full()
        if last_step and TAIL_AVG_WINDOW > 0 and 'full' == "full" and _tavg_bufs:
            # Ship the full tail-average: swap the assembled weights in BEFORE the final
            # counted validation and BEFORE the timer read, so the assembly is paid
            # on the clock. (Placement before validation is load-bearing: after the
            # whole validation section the reported final val would run on the raw
            # weights and the ship would only ever affect the saved checkpoint.)
            # FP8 weight caches (quantize_mlp_fp8: _mlp_up_proj_f8 / _mlp_down_f8 /
            # _mlp_down_f8_row / _lm_f8 / _attn_f8) need no refresh here: every consumer
            # is gated on self.training (GPT.forward `use_mlp_fp8`/`use_attn_f8`,
            # CastedLinearT.forward), and validation runs under model.eval(), reading the
            # bf16 nn.Parameters directly. Re-running quantize_mlp_fp8() here would in
            # fact be harmful, since it recomputes _mlp_post_scale from an already-zeroed
            # _mlp_post_amax. So copying into p.data is sufficient for *all* labels
            # (embed/lm_head/value_embeds/bigram/
            # mudd included); there is no eval-time cache derived from any of them.
            with torch.no_grad():
                for _lbl, _bar in _tavg_bufs.items():
                    _p = training_manager.optimizer._param_by_label[_lbl]
                    _p.data.copy_(_bar.to(_p.dtype))
            print0(f"[tavg] shipped: mode={'lerp0.02'} "
                   f"k={max(_tavg_n.values()) if _tavg_n else 0} labels={sorted(_tavg_bufs)}", console=True)
            _tavg_bufs.clear()
        if last_step and _SHIPBLEND and _shipblend_bufs:
            # Tail-blend ship: AFTER the stock TEMA/tavg ships above (both stay
            # byte-identical), blend the bank tail-average INTO the shipped
            # state -- p = lerp(p_shipped, ema, B). No restore: this IS the shipped
            # state; the counted final val below, the saved checkpoint and all
            # post-loop evals see it. Mantissa staleness is the same class the
            # stock tavg ship already accepts (no optimizer step ever consumes
            # the sidecar after this point). On the clock: the timer read happens
            # after every ship in this section.
            with torch.no_grad():
                for _lbl, _bar in _shipblend_bufs.items():
                    _p = training_manager.optimizer._param_by_label[_lbl]
                    _p.data.copy_(torch.lerp(_p.data.float(), _bar, _SHIPBLEND_B).to(_p.dtype))
            print0(f"[shipblend] blended banks k={_SHIPBLEND_K} blend={_SHIPBLEND_B} "
                   f"n={max(_shipblend_n.values()) if _shipblend_n else 0} labels={sorted(_shipblend_bufs)}", console=True)
        # stop the clock (after every last_step weight ship above)
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens, bigram_inputs, *_ = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).mean()
        val_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        if HOST_OPT:
            # HOST_OPT: clock is stopped here -> both the log flush and the manual GC are untimed.
            if _logf is not None:
                _logf.flush()
            gc.collect()
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step and len(_kx_wsval_list) > 1:
        # ws_short diagnostic ladder: re-run the final validation once per short-window value, on the
        # same weights the run just reported -- i.e. AFTER the tail-average ship, which
        # happens inside the validation section above, so this measures the shipping
        # config. Untimed: the training clock stopped in that section and is never read
        # again at last_step (same guarantee the [tavg] mode= block below relies on).
        model.eval()
        _ws_base = training_manager.ws_short
        for _v in _kx_wsval_list:
            _ws_new = _ws_base + _v
            if _ws_new >= training_manager.ws_long:
                # never let ws_short reach ws_long: key_offset = [b == ws_long for b in
                # bm_sizes] would then switch on for every layer (a different mechanism).
                _ws_new = training_manager.ws_long - 1
                print0(f"[wsval] clamped +{_v} to ws_short={_ws_new} (must stay < ws_long={training_manager.ws_long})", console=True)
            training_manager.ws_short = _ws_new
            print0(f"[wsval] +{_v} ws_short:{_ws_new} ({_ws_new * training_manager.block_size} tok) val_loss:{_extra_val():.4f}", console=True)
        training_manager.ws_short = _ws_base
        model.train()

    if last_step and _kx_ywsval_list:
        # ws_long diagnostic ladder: re-run final validation at alternative ws_LONG values on
        # the shipped (tavg-assembled) weights. Untimed for the same reason as the
        # ladder above. Values are absolute block counts; must stay > ws_short (key_offset
        # flips per-layer at b == ws_long).
        model.eval()
        _wl_base = training_manager.ws_long
        for _v in _kx_ywsval_list:
            if _v <= training_manager.ws_short:
                print0(f"[ywsval] skipped ws_long={_v} (must stay > ws_short={training_manager.ws_short})", console=True)
                continue
            training_manager.ws_long = _v
            print0(f"[ywsval] ws_long:{_v} ({_v * training_manager.block_size} tok) val_loss:{_extra_val():.4f}", console=True)
        training_manager.ws_long = _wl_base
        model.train()

    if last_step and _TAIL_EMA2_WINDOWS and _tavg2_bufs:
        # tavg2 ladder: swap each window's Adam-side EMA in, val, swap back. Untimed.
        model.eval()
        for _w in _TAIL_EMA2_WINDOWS:
            _saved = {}
            with torch.no_grad():
                for _lbl in _TAVG2_LABELS:
                    _p = training_manager.optimizer._param_by_label.get(_lbl)
                    if _p is None or (_w, _lbl) not in _tavg2_bufs:
                        continue
                    _saved[_lbl] = _p.data.clone()
                    _p.data.copy_(_tavg2_bufs[(_w, _lbl)].to(_p.dtype))
            print0(f"[tavg2] win={_w} val_loss:{_extra_val():.4f}", console=True)
            with torch.no_grad():
                for _lbl, _v in _saved.items():
                    training_manager.optimizer._param_by_label[_lbl].data.copy_(_v)
        model.train()

    if last_step and TAIL_AVG_WINDOW > 0 and _tavg_bufs:
        # Assembly comparison: compare final-val under raw / tail-averaged
        # weights. All untimed (training clock already stopped at last_step val).
        def _run_val():
            """One full validation pass on the current weights (untimed)."""
            vl = 0
            vloader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
            with torch.no_grad():
                for _ in range(grad_accum_steps * args.val_tokens // args.val_batch_size):
                    iv, tv, cv, bv, _ = next(vloader)
                    vl += model(iv, tv, cv, bv, training_manager.get_forward_args()).mean()
            vl /= (grad_accum_steps * args.val_tokens // args.val_batch_size)
            dist.reduce(vl, 0, op=dist.ReduceOp.AVG)
            return vl.item()
        model.eval()
        raw = {}
        opt = training_manager.optimizer
        for lbl, bar in _tavg_bufs.items():
            p = opt._param_by_label[lbl]
            raw[lbl] = p.data.clone()
        for mode in ("full", "split"):
            with torch.no_grad():
                for lbl, bar in _tavg_bufs.items():
                    p = opt._param_by_label[lbl]
                    p.data.copy_(bar.to(p.dtype))
            v = _run_val()
            print0(f"[tavg] mode={mode} val_loss:{v:.4f}", console=True)
            with torch.no_grad():
                for lbl in _tavg_bufs:
                    opt._param_by_label[lbl].data.copy_(raw[lbl])
        model.train()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for idx in range(grad_accum_steps):
        if _pref_batch is not None:
            _batch = _pref_batch
            _pref_batch = None
        else:
            _batch = train_loader.send(training_manager.train_loader_send_args)
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = _batch[:5]
        training_manager.sparse_index_update(step, bigram_cpu, None)
        if _bigram_sink_on:
            _bg_sink = training_manager.bgsp_sink(bigram_inputs.shape[0])
            loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(), _bg_sink).sum() * grad_scale
        training_manager.sparse_index_share(step)
        loss.backward()
        if _bigram_sink_on:
            training_manager.bgsp_harvest(bigram_inputs, _bg_sink)
        if grad_accum_steps == 1 and step + 1 < train_steps:
            # Prefetch batch k+1 while the GPU chews on k's bwd. send_args derived from a
            # PURE schedule lookup (advance_schedule mutates yarn state — must not be
            # called early). Geometry compare vs the batch just fetched keeps the send
            # protocol identical to stock; consumed batch sequence is bit-identical.
            _ns, _ = training_schedule.lookup(step + 1)
            _pref_args = ((_ns.batch_size, _ns.train_max_seq_len, grad_accum_steps)
                          if (_ns.batch_size != training_manager.batch_size or _ns.train_max_seq_len != training_manager.train_max_seq_len)
                          else None)
            _pref_batch = train_loader.send(_pref_args)
        del loss
    training_manager.step_optimizers(step)
    if _cg is not None and _cg.has_quant:
        # CUDA_GRAPH_TIER: one replay covers the whole quantize body (~20 eager dispatches -> 1
        # launch). Same program position and stream as the eager call, so it stays
        # ordered after Phase 3's gather waits and before the next forward.
        _cg.replay_quant(refresh_lm=(training_manager._is_adam_step(step) if HOST_OPT else True))
    elif HOST_OPT:
        # HOST_OPT: refresh the lm_head FP8 copy only after steps that mutate lm_head
        # (lm_head is an Adam param; AnvilAndAdam.step skips Adam on even steps).
        model.quantize_mlp_fp8(refresh_lm=training_manager._is_adam_step(step))
    if TAIL_EMA_WINDOW > 0 and step >= training_schedule.total_steps - TAIL_EMA_WINDOW:
        # TailEMA accumulation: post-step fp32 EMA, seeded on first tick.
        with torch.no_grad():
            for _lbl in TAIL_EMA_LABELS:
                _p = training_manager.optimizer._param_by_label.get(_lbl)
                if _p is None:
                    continue
                if _KX_R2:
                    # Persistent fp32 scratch replaces the per-step
                    # .float() temporary (same pattern as _hg_tavg_scratch; bf16->fp32
                    # copy_ is exact => bitwise-identical EMA).
                    _s = _r2_tema_scratch.get(_lbl)
                    if _s is None:
                        _s = _r2_tema_scratch[_lbl] = torch.empty_like(_p.data, dtype=torch.float32)
                    _s.copy_(_p.data)
                    _src = _s
                if _lbl not in _tema_bufs:
                    _tema_bufs[_lbl] = _src.clone()
                else:
                    _tema_bufs[_lbl].lerp_(_src, 2.0 / (TAIL_EMA_WINDOW + 1))
    if TAIL_AVG_WINDOW > 0 and step >= training_schedule.total_steps - TAIL_AVG_WINDOW:
        with torch.no_grad():
            for _lbl in _tavg_label_list(training_manager.optimizer):
                _p = training_manager.optimizer._param_by_label.get(_lbl)
                if _p is None:
                    continue
                if HOST_OPT:
                    # HOST_OPT: ATen's lerp requires end.dtype == self.dtype, so the fp32 upcast
                    # must stay; only the per-step fp32 temporary is removed by reusing a
                    # persistent scratch. bf16->fp32 copy_ is exact => tavg numerics
                    # unchanged.
                    _s = _hg_tavg_scratch.get(_lbl)
                    if _s is None:
                        _s = _hg_tavg_scratch[_lbl] = torch.empty_like(_p.data, dtype=torch.float32)
                    _s.copy_(_p.data)
                    if _lbl not in _tavg_bufs:
                        _tavg_bufs[_lbl] = _s.clone()
                        _tavg_n[_lbl] = 1
                    else:
                        _tavg_n[_lbl] += 1
                        _tavg_bufs[_lbl].lerp_(_s, 0.02)

    if (_SHIPBLEND and step >= training_schedule.total_steps - _SHIPBLEND_K
            and (step % 4 == 1 or step >= training_schedule.total_steps - 2)):
        # Tail-blend accumulation at every-4th-step cadence: quarters the fp32
        # bank traffic versus per-step accumulation (which measured +0.33 ms/step)
        # at an equivalent effective window -- the lerp rate 2/(K/4+1) matches the
        # K-step window EMA horizon at this cadence. fp32 seed-clone on the first
        # tick; steps < N-K are byte-identical to the no-blend path.
        with torch.no_grad():
            for _lbl in _SHIPBLEND_LABELS:
                _p = training_manager.optimizer._param_by_label.get(_lbl)
                if _p is None:
                    continue
                _src = _p.data.float()
                if _lbl not in _shipblend_bufs:
                    _shipblend_bufs[_lbl] = _src.clone()
                    _shipblend_n[_lbl] = 1
                else:
                    _shipblend_n[_lbl] += 1
                    _shipblend_bufs[_lbl].lerp_(_src, 2.0 / (_SHIPBLEND_K // 4 + 1))

    # logging (PRINT_EVERY thinning: skip the f-string + double print on most steps;
    # final-step and every-Nth prints unchanged, so nothing record-relevant changes)
    if not PRINT_EVERY or (step + 1) % PRINT_EVERY == 0 or step + 1 >= train_steps - 1:
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

if args.run_evals:
    model.eval()
    if _KX_R2:
        for _m in model.modules():
            if isinstance(_m, Yarn):
                _m.r2_ensure_full()
    from evals import hellaswag
    hellaswag.evaluate(model=model, 
                       schedule_cfg=training_manager.get_forward_args(), 
                       seq_len=args.val_batch_size // (grad_accum_steps * world_size),
                       get_bigram_hash=get_bigram_hash, 
                       print0=print0)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()

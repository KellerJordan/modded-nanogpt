"""EXACT_LEVERS: exact speed levers on top of #393 (same model, data, schedule and arithmetic). All four are ON by default; EXACT_LEVERS=none restores
#393's exact code path, and a comma-separated subset switches levers individually (for A/B checks).

Found in 8x H100 profiles of #393 (details in the PR description). None changes the arithmetic: the same kernels
compute the same values; what goes is copies, zero fills and waits. Values: unset or "all" (default), "none", or a subset:
  fa3copy      q, k, v reach FA3 as the [tokens, heads, dim] tensors the QKV kernel made, without the q[None] / q[0]
               round trip, whose select_backward materialized a copy of dq / dk (and dv on three layers) after every
               FA3 backward: 17 copy kernels per step (model/attention.py)
  rsearly      the ANVIL banks' reduce-scatters are launched right after the backward graph's replay, ahead of the
               Adam gradient accumulation and the n-gram hold copy that a collective otherwise waits behind
               (perf/cuda_graphs/step_graphs.py, optim/anvil.py)
  bankinplace  the dense MLPs' dW1 / dW2 are written straight into a persistent mlp_bank gradient buffer (as the
               experts' already are into moe_bank_grad), so autograd no longer stacks the bank's gradient: no slot
               copies and no zero fill of the padding slots, which stay zero from allocation (model/gpt.py,
               perf/kernels/mlp.py)
  prescatter   each step scatter-adds its n-gram sink gradient into the cycle's per-slot sum right away (the cycle's
               first step zeroes it), instead of the event step zeroing it and scattering every held step's gradient:
               the same zero and scatter kernels on the same inputs in the same order, spread over the cycle's steps,
               and no ring copy to hold a gradient past its step (ngram_table.py, step_graphs.hold_ngram_grad)
  moeanvil     moe_bank uses anvil_opt's fused polar-express cascade parameterized by bank.margin instead of the
               torch.compile anvil_cascade fallback (optim/anvil.py, perf/kernels/anvil_opt.py)
  moeg3bwd     MoE grouped weight-gradient GEMM uses 128x128 tiles instead of 128x256 for better SM occupancy
               (perf/kernels/moe_mlp.py)
  cplmbf16     CPLM copy backward stores dq / dk directly in bf16 from fp32 accumulators instead of writing and
               re-reading fp32 temporaries (perf/kernels/cplm.py)
  tailopt      pre-warms TailAverages.ship() ATen kernels during __init__ before the clock starts (tail_average.py)
  b3opt        fused equalizer+decay update (optim/anvil.py), 128x128 dual FP8 weight quant + fused/1-launch
               scale reduction (perf/kernels/mlp.py, perf/kernels/fp8_attention_quant.py)
"""
import os

KNOWN = (
    "fa3copy", "rsearly", "bankinplace", "prescatter",
    "moeanvil", "moeg3bwd", "cplmbf16", "tailopt", "b3opt",
)
_RAW = os.environ.get("EXACT_LEVERS", "all").replace(" ", "").lower()
LEVERS = frozenset() if _RAW in ("none", "off", "0") else frozenset(x for x in _RAW.split(",") if x)
if "all" in LEVERS:
    LEVERS = frozenset(KNOWN) | (LEVERS - {"all"})
assert LEVERS <= set(KNOWN), f"unknown EXACT_LEVERS {sorted(LEVERS - set(KNOWN))} (known: {KNOWN})"

FA3_NO_SELECT = "fa3copy" in LEVERS
RS_EARLY = "rsearly" in LEVERS
MLP_GRAD_IN_PLACE = "bankinplace" in LEVERS
NGRAM_PRESCATTER = "prescatter" in LEVERS
MOE_ANVIL_OPT = "moeanvil" in LEVERS
MOE_G3_BWD_PTX = "moeg3bwd" in LEVERS
CPLM_BF16 = "cplmbf16" in LEVERS
TAIL_OPT = "tailopt" in LEVERS
B3_OPT = "b3opt" in LEVERS

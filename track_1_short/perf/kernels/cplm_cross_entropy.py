"""CPLM (copy-sink mixture) variants of the fused softcapped CE: full softmax and sampled candidate set.

The CE kernel (perf/kernels/cross_entropy.py) computes, per row, the softcapped log-sum-exp, the loss and
the e5m2 logit gradient in ONE pass; the backward only runs the fp8 GEMMs. For CPLM the next-token (k = 0)
cross-entropy is replaced by the copy-sink mixture NLL

    p(y) = p_lm(y) * (1 + alpha * a_sink / (1 - alpha)) + alpha * p_copy,   alpha = p_lm(<copy>)

whose logit gradient is still  w0 * [g_lse * softmax(z)_v + g_t * [v == y] + g_c * [v == copy]]  with
closed-form row scalars. The copy branch (p_copy, a_sink) does not read the logits, so it is computed first
and passed in; the kernel also returns d nll / d p_copy and d nll / d a_sink for the copy head.
The <copy> column is read from a device int (its position in the candidate set changes per step, and the
step is CUDA-graph captured). Source = cross_entropy.CE_KERNEL_SOURCE + anchored patches (each asserted).
"""
import torch

from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE, CE_KERNEL_SOURCE, CE_KERNEL_VOCAB_SIZE, CUDA_INCLUDE_DIRS, SOFTCAP_A, SOFTCAP_B,
    SOFTCAP_C, _ce_backward_gemms, _ce_logit_gemm,
)
from track_1_short.perf.kernels.sampled_cross_entropy import sampled_densify
from track_1_short.sampled_softmax import ALL_CANDIDATE_COUNTS

COPY_TOKEN_ID = 50257  # first padding row of the 50304-wide vocab
AUX_NOCOPY = __import__("os").environ.get("CPLM_AUX_NOCOPY") == "1"
GATE_OFFSET = float(__import__("os").environ.get("CPLM_GATE_OFFSET", "0"))  # raw-logit offset on <copy>
# CPLM_EXT_GATE=1: the gate alpha is a per-row input (a sigmoid gate computed outside), not the <copy> column's softmax
# mass; the kernel returns d nll / d alpha. p = p_lm * (1 - alpha + alpha * a_sink) + alpha * p_copy over the full
# vocab softmax p_lm (the <copy> column is then an ordinary, never-targeted vocab row).
EXT_GATE = __import__("os").environ.get("CPLM_EXT_GATE") == "1"
# CPLM_LM_AUX=beta: add beta * (plain LM CE) to the k = 0 term, so the LM keeps learning tokens the copy branch explains
# (with a large early gate the young LM got almost no gradient on copyable tokens and ended worse).
LM_AUX = float(__import__("os").environ.get("CPLM_LM_AUX", "0"))
# CPLM_GATE_FLOOR=eps: the softmax gate alpha (the <copy> column's mass) is used as a = eps + (1 - eps) * alpha, so the copy
# branch always gets at least eps of the mass and the pointer always gets gradient. With p~ = p_lm / (1 - alpha) (the LM over
# the real vocab) the mixture is p = p~ * (1 - a * (1 - a_sink)) + a * p_copy; at eps = 0 this is the original mixture.
GATE_FLOOR = float(__import__("os").environ.get("CPLM_GATE_FLOOR", "0"))
# CPLM_MTP_COPY=1: the MTP terms (k >= 1) are copy-sink mixtures too, with the same gate and sink and their own p_copy[k]
# (the pointer's attention copying the token k after each source); p_copy / d_pc are then [n_predict, N]. With
# AUX_NOCOPY only the prefix term remains a plain CE over the real vocabulary.
MTP_COPY = __import__("os").environ.get("CPLM_MTP_COPY") == "1"
# CPLM_STATS=1 (diagnostics): without EXT_GATE the kernel writes the row's gate alpha into d_al (otherwise unused), and the
# autograd fns store summaries in the model's stats buffer (slots 0-3: mean alpha, mean log alpha, frac alpha > 1e-2,
# mean copy share alpha * p_copy / p).
STATS = __import__("os").environ.get("CPLM_STATS") == "1"
# CPLM_RAW_COPY=1: the <copy> gate logit skips the softcap: z_copy = l + RAW_C0 (linear; RAW_C0 = the softcap's value at
# l = 0, so the gate starts where it did). Cached as z_copy / A in the fp16 row; its logit gradient is grad_z (slope 1).
RAW_COPY = __import__("os").environ.get("CPLM_RAW_COPY") == "1"
RAW_C0 = 23.0 / (1.0 + __import__("math").exp(-5.0 / 7.5))


def _patch(src, old, new):
    assert src.count(old) == 1, old[:80]
    return src.replace(old, new)


def cplm_source():
    s = CE_KERNEL_SOURCE
    s = _patch(s, "__global__ void ce_fwd_bwd_kernel(", "__global__ void cplm_ce_fwd_bwd_kernel(")
    s = _patch(s, """    double grad_s_param)
{""", """    double grad_s_param,
    const float* __restrict__ p_copy_ptr,
    const float* __restrict__ a_sink_ptr,
    float* __restrict__ d_pc_ptr,
    float* __restrict__ d_as_ptr,
    const int* __restrict__ copy_col_ptr,
    const float* __restrict__ lam_ptr,
    const float* __restrict__ alpha_ptr,
    float* __restrict__ d_al_ptr)
{""")
    s = _patch(s, """  float lse = A + __logf(block_sum);
""", """  float lse = A + __logf(block_sum);

  // CPLM: copy-sink mixture on the next-token (k = 0) term. g_* = d nll / d (z_target, z_copy, lse).
  int copy_col = copy_col_ptr[0];
  float w0 = mtp_weights[0];
  int64_t t0 = targets[blockIdx.x];
  float pl = __expf(A * __half2float(smem[t0]) - lse);
  float al = lam_ptr[0] * __expf(A * __half2float(smem[copy_col]) - lse);  // copy-gate warmup scale
  float pc = p_copy_ptr[blockIdx.x];
  float as_ = a_sink_ptr[blockIdx.x];
  float r = 1.0f / fmaxf(1.0f - al, 1e-6f);
  float pmix = pl * (1.0f + al * as_ * r) + al * pc;
  float inv = 1.0f / (pmix + 1e-9f);
  float g_t = -pl * (1.0f + al * as_ * r) * inv;
  float g_c = -al * (pl * as_ * r * r + pc) * inv;
  float g_lse = -(g_t + g_c);
  float nll0 = -__logf(pmix + 1e-9f);
""")
    s = _patch(s, """          total_loss += weight * (lse - z_target);""",
               """          total_loss += weight * (k == 0 ? nll0 : (lse - z_target));""")
    s = _patch(s, """    losses[blockIdx.x] = total_loss;
""", """    losses[blockIdx.x] = total_loss;
    d_pc_ptr[blockIdx.x] = w0 * (-al * inv);
    d_as_ptr[blockIdx.x] = w0 * (-pl * al * r * inv);
""")
    s = _patch(s, """  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }
""", """  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }
  S_w += w0 * (g_lse - 1.0f);  // the k = 0 softmax coefficient is w0 * g_lse instead of w0
""")
    s = _patch(s, """  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && (prefix_weight > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];""",
               """  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && (prefix_weight > 0.0f);
  bool is_copy = (threadIdx.x == n_predict + 1);  // one more thread corrects the <copy> column
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx || is_copy) {
    int i = threadIdx.x;
    int64_t target = is_copy ? (int64_t)copy_col : (is_pfx ? ptgt_row : targets[blockIdx.x + i]);""")
    s = _patch(s, """        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }""", """        if (targets[target_idx] == target) {
          term2 += (k == 0) ? (-w0 * g_t) : mtp_weights[k];
        }""")
    s = _patch(s, """    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += prefix_weight;
    }
""", """    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += prefix_weight;
    }
    if (target == copy_col) {
      term2 += -w0 * g_c;
    }
""")
    if EXT_GATE:
        s = _patch(s, """  float pl = __expf(A * __half2float(smem[t0]) - lse);
  float al = lam_ptr[0] * __expf(A * __half2float(smem[copy_col]) - lse);  // copy-gate warmup scale""",
                   """  float al = alpha_ptr[blockIdx.x];
  float pl_raw = __expf(A * __half2float(smem[t0]) - lse);
  float pl = pl_raw * (1.0f - al);  // the mixture below is then p_raw * (1 - al + al * as) + al * pc""")
        s = _patch(s, "  float g_c = -al * (pl * as_ * r * r + pc) * inv;", "  float g_c = 0.0f;")
        s = _patch(s, """    d_as_ptr[blockIdx.x] = w0 * (-pl * al * r * inv);
""", """    d_as_ptr[blockIdx.x] = w0 * (-pl * al * r * inv);
    d_al_ptr[blockIdx.x] = w0 * (-(pl_raw * (as_ - 1.0f) + pc) * inv);
""")
    if AUX_NOCOPY:
        # The auxiliary CE terms (MTP k >= 1, prefix) normalize over the real vocabulary only: <copy> is an output
        # of the copy-sink mixture, not a class those heads predict. Otherwise they push its logit down on every
        # token and, early on (large MTP / prefix weights), collapse the copy gate in some seeds.
        s = _patch(s, """  float nll0 = -__logf(pmix + 1e-9f);
""", """  float nll0 = -__logf(pmix + 1e-9f);
  float pcol = __expf(A * __half2float(smem[copy_col]) - lse);           // the slot's mass, without lambda
  float lse_aux = lse + __logf(fmaxf(1.0f - pcol, 1e-6f));             // log-sum-exp over the real vocabulary
""")
        s = _patch(s, "(k == 0 ? nll0 : (lse - z_target))", "(k == 0 ? nll0 : (lse_aux - z_target))")
        s = _patch(s, "total_loss += prefix_weight * (lse - z_p);", "total_loss += prefix_weight * (lse_aux - z_p);")
        s = _patch(s, "  S_w += w0 * (g_lse - 1.0f);  // the k = 0 softmax coefficient is w0 * g_lse instead of w0",
                   "  S_w = w0 * g_lse + (S_w - w0) / fmaxf(1.0f - pcol, 1e-6f);  // aux terms: softmax over the real vocab")
        s = _patch(s, """    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll""", """    float term1 = (target == copy_col) ? (w0 * g_lse * p) : (S_w * p);  // aux terms do not touch <copy>
    float term2 = 0.0f;

    #pragma unroll""")
    if GATE_OFFSET != 0.0:
        # Start the copy gate open: a constant added to the <copy> column's raw logit before the softcap, applied
        # where the sigmoid row is cached so every later use (lse, mixture, gradients) sees it consistently.
        s = _patch(s, """  float thread_sum = 0.0f;

  #pragma unroll 25""", """  float thread_sum = 0.0f;
  const int copy_col_early = copy_col_ptr[0];

  #pragma unroll 25""")
        s = _patch(s, """        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);""", """        float tmp = CE_LOGIT_TO_F32(result[k]);
        if (idx + k == copy_col_early) tmp += GATE_OFFSET;
        tmp = sigmoid(tmp * inv_C + B_div_C);""")
    if GATE_FLOOR != 0.0 or MTP_COPY:  # (the floored formulas are the original ones at GATE_FLOOR = 0)
        assert not EXT_GATE
        s = _patch(s, """  float pmix = pl * (1.0f + al * as_ * r) + al * pc;""", """  float ae = GATE_FLOOR + (1.0f - GATE_FLOOR) * al;  // floored gate
  float pt = pl * r;                                   // LM probability over the real vocab
  float Bm = 1.0f - ae * (1.0f - as_);
  float pmix = pt * Bm + ae * pc;""")
        s = _patch(s, "  float g_t = -pl * (1.0f + al * as_ * r) * inv;", "  float g_t = -pt * Bm * inv;")
        s = _patch(s, "  float g_c = -al * (pl * as_ * r * r + pc) * inv;",
                   "  float g_c = -al * (pt * r * Bm + (1.0f - GATE_FLOOR) * (pc - pt * (1.0f - as_))) * inv;")
        s = _patch(s, "    d_pc_ptr[blockIdx.x] = w0 * (-al * inv);", "    d_pc_ptr[blockIdx.x] = w0 * (-ae * inv);")
        s = _patch(s, "    d_as_ptr[blockIdx.x] = w0 * (-pl * al * r * inv);", "    d_as_ptr[blockIdx.x] = w0 * (-pt * ae * inv);")
    if RAW_COPY:
        assert GATE_OFFSET == 0.0 and not EXT_GATE
        s = _patch(s, """  float thread_sum = 0.0f;

  #pragma unroll 25""", """  float thread_sum = 0.0f;
  const int copy_col_early = copy_col_ptr[0];

  #pragma unroll 25""")
        s = _patch(s, """        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);""", """        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = (idx + k == copy_col_early) ? (tmp + RAW_C0) * (1.0f / A) : sigmoid(tmp * inv_C + B_div_C);""")
        s = _patch(s, """    float grad_x = (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);""",
                   """    float grad_x = (target == copy_col) ? (1.0f / grad_s) * grad_z
                                        : (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);""")
    if STATS and not EXT_GATE:
        s = _patch(s, """    losses[blockIdx.x] = total_loss;
""", """    losses[blockIdx.x] = total_loss;
    d_al_ptr[blockIdx.x] = al;
""")
    if MTP_COPY:
        assert AUX_NOCOPY and LM_AUX == 0.0 and not EXT_GATE
        s = _patch(s, """  float lse_aux = lse + __logf(fmaxf(1.0f - pcol, 1e-6f));             // log-sum-exp over the real vocabulary
""", """  float lse_aux = lse + __logf(fmaxf(1.0f - pcol, 1e-6f));             // log-sum-exp over the real vocabulary
  // MTP copy mixtures: per-k row scalars as for k = 0 (same gate ae, sink as_, LM-over-real-vocab scale r)
  float gtk[3] = {g_t, 0.0f, 0.0f}, nllk[3] = {nll0, 0.0f, 0.0f}, dpck[3] = {-ae * inv, 0.0f, 0.0f};
  float dask = w0 * (-pt * ae * inv);
  float S_mix = w0 * g_lse, g_c_all = w0 * g_c;
  for (int k = 1; k < n_predict && k < 3; k++) {
    int64_t ti = blockIdx.x + k;
    if (ti < batch_size) {
      int64_t yk = targets[ti];
      if (yk >= 0 && yk < VOCAB_SIZE) {
        float wk = mtp_weights[k];
        float ptk = __expf(A * __half2float(smem[yk]) - lse) * r;
        float pck = p_copy_ptr[k * batch_size + blockIdx.x];
        float ik = 1.0f / (ptk * Bm + ae * pck + 1e-9f);
        float gtk_ = -ptk * Bm * ik;
        float gck_ = -al * (ptk * r * Bm + (1.0f - GATE_FLOOR) * (pck - ptk * (1.0f - as_))) * ik;
        gtk[k] = gtk_;
        nllk[k] = -__logf(ptk * Bm + ae * pck + 1e-9f);
        dpck[k] = -ae * ik;
        dask += wk * (-ptk * ae * ik);
        S_mix += wk * (-(gtk_ + gck_));
        g_c_all += wk * gck_;
      }
    }
  }
""")
        s = _patch(s, "total_loss += weight * (k == 0 ? nll0 : (lse_aux - z_target));", "total_loss += weight * nllk[k];")
        s = _patch(s, """    d_pc_ptr[blockIdx.x] = w0 * (-ae * inv);
    d_as_ptr[blockIdx.x] = w0 * (-pt * ae * inv);
""", """    for (int k = 0; k < n_predict; k++) d_pc_ptr[k * batch_size + blockIdx.x] = mtp_weights[k] * dpck[k];
    d_as_ptr[blockIdx.x] = dask;
""")
        s = _patch(s, "  S_w = w0 * g_lse + (S_w - w0) / fmaxf(1.0f - pcol, 1e-6f);",
                   "  S_w = S_mix + ((ptgt_row >= 0) ? prefix_weight : 0.0f) / fmaxf(1.0f - pcol, 1e-6f);")
        s = _patch(s, "(target == copy_col) ? (w0 * g_lse * p) : (S_w * p);", "(target == copy_col) ? (S_mix * p) : (S_w * p);")
        s = _patch(s, "term2 += (k == 0) ? (-w0 * g_t) : mtp_weights[k];", "term2 += -mtp_weights[k] * gtk[k];")
        s = _patch(s, "      term2 += -w0 * g_c;", "      term2 += -g_c_all;")
    if LM_AUX != 0.0 and AUX_NOCOPY:
        # the LM auxiliary is then one more aux term over the real vocabulary (lse_aux, <copy> untouched)
        s = _patch(s, "(k == 0 ? nll0 :", "(k == 0 ? nll0 + LM_AUX * (lse_aux - z_target) :")
        s = _patch(s, "  S_w = w0 * g_lse + (S_w - w0)", "  S_w += w0 * LM_AUX;\n  S_w = w0 * g_lse + (S_w - w0)")
        s = _patch(s, "term2 += (k == 0) ? (-w0 * g_t) : mtp_weights[k];", "term2 += (k == 0) ? (-w0 * (g_t - LM_AUX)) : mtp_weights[k];")
    elif LM_AUX != 0.0:
        s = _patch(s, "(k == 0 ? nll0 :", "(k == 0 ? nll0 + LM_AUX * (lse - z_target) :")
        s = _patch(s, "  S_w += w0 * (g_lse - 1.0f);", "  S_w += w0 * (g_lse - 1.0f + LM_AUX);")
        s = _patch(s, "term2 += (k == 0) ? (-w0 * g_t) : mtp_weights[k];", "term2 += (k == 0) ? (-w0 * (g_t - LM_AUX)) : mtp_weights[k];")
    return s


def _compile(vocab_size):
    decls = (f"\nconstexpr int VOCAB_SIZE = {vocab_size};\nconstexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n"
             f"constexpr float GATE_OFFSET = {GATE_OFFSET:.6f}f;\nconstexpr float LM_AUX = {LM_AUX:.6f}f;\n"
             f"constexpr float GATE_FLOOR = {GATE_FLOOR:.6f}f;\nconstexpr float RAW_C0 = {RAW_C0:.6f}f;\n")
    kernel = torch.cuda._compile_kernel(decls + cplm_source(), "cplm_ce_fwd_bwd_kernel", compute_capability="90",
                                        cuda_include_dirs=CUDA_INCLUDE_DIRS, nvcc_options=["-lineinfo", "--use_fast_math"])
    kernel.set_shared_memory_config(vocab_size * 2)
    return kernel


CPLM_CE_KERNELS = {v: _compile(v) for v in (CE_KERNEL_VOCAB_SIZE, *ALL_CANDIDATE_COUNTS)}


@torch.library.custom_op("nanogpt::cplm_ce_fwd_bwd", mutates_args={"losses", "grad_input", "d_pc", "d_as", "d_al"})
def cplm_ce_fwd_bwd(
    logits: torch.Tensor, targets: torch.Tensor, mtp_weights: torch.Tensor, prefix_targets: torch.Tensor,
    prefix_weight: torch.Tensor, p_copy: torch.Tensor, a_sink: torch.Tensor, copy_col: torch.Tensor,
    lam: torch.Tensor, alpha: torch.Tensor, losses: torch.Tensor, grad_input: torch.Tensor, d_pc: torch.Tensor,
    d_as: torch.Tensor, d_al: torch.Tensor, n_rows: int, n_predict: int, A: float, B: float, C: float, grad_s: float,
) -> None:
    """ce_fwd_bwd + the copy-sink mixture, with the kernel of width logits.shape[1]."""
    V = logits.shape[1]
    CPLM_CE_KERNELS[V](
        (n_rows, 1, 1), (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s, p_copy, a_sink, d_pc, d_as, copy_col, lam, alpha, d_al),
        shared_mem=V * 2,
    )


def _stats(stats, d_al, d_pc, p_copy):
    if stats is None or EXT_GATE:
        return
    al = d_al.clamp_min(1e-12)
    pc0, dpc0 = (p_copy[0], d_pc[0]) if p_copy.dim() == 2 else (p_copy, d_pc)
    stats[0:4].copy_(torch.stack([al.mean(), al.log().mean(), (al > 1e-2).float().mean(), (-pc0 * dpc0).mean()]))


def _run(logits, targets, mtp_weights, prefix_targets, prefix_weight, p_copy, a_sink, copy_col, lam, alpha, grad_s):
    n_rows, n_cols = logits.shape
    losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    d_pc = torch.zeros(p_copy.shape, dtype=torch.float32, device=logits.device)  # [N] or [n_predict, N] (MTP_COPY)
    d_as, d_al = torch.empty_like(losses), torch.empty_like(losses)
    grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)
    cplm_ce_fwd_bwd(logits.contiguous(), targets.contiguous(), mtp_weights.contiguous(), prefix_targets.contiguous(),
                    prefix_weight.reshape(1).to(torch.float32).contiguous(), p_copy.float().contiguous(),
                    a_sink.float().contiguous(), copy_col.to(torch.int32).reshape(1).contiguous(),
                    lam.float().reshape(1).contiguous(), alpha.float().contiguous(), losses, grad_input, d_pc, d_as, d_al,
                    n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
    return losses, grad_input, d_pc, d_as, d_al


class FusedCPLMCrossEntropy(torch.autograd.Function):
    """FusedSoftcappedCrossEntropy with the k = 0 term replaced by the copy-sink mixture NLL."""
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, prefix_targets, prefix_weight, p_copy, a_sink, copy_col, lam, alpha,
                lm_head_weight, w_f8_col, w_f8_row, x_s, w_s, grad_s, stats=None):
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, w_f8_col.T.contiguous().T, x_s, w_s)
        losses, grad_input, d_pc, d_as, d_al = _run(logits, targets, mtp_weights, prefix_targets, prefix_weight,
                                                    p_copy, a_sink, copy_col, lam, alpha, grad_s)
        _stats(stats, d_al, d_pc, p_copy)
        ctx.save_for_backward(x_f8, w_f8_row, grad_input, d_pc, d_as, d_al)
        ctx.scales = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8_row, grad_input, d_pc, d_as, d_al = ctx.saved_tensors
        grad_x, grad_w = _ce_backward_gemms(grad_input, w_f8_row, x_f8, *ctx.scales)
        # logit grads assume d(loss.sum())/d(loss_i) = 1 (folded in forward); the copy branch uses grad_output
        return (grad_x, None, None, None, None, d_pc * grad_output, d_as * grad_output, None, None,
                d_al * grad_output if EXT_GATE else None, grad_w, None, None, None, None, None, None)


class SampledCPLMCrossEntropy(torch.autograd.Function):
    """SampledSoftcappedCrossEntropy with the copy-sink mixture; <copy> is forced into the candidate set and
    copy_col is its position there (vocab_pos[COPY_TOKEN_ID])."""
    @staticmethod
    def forward(ctx, x, mtp_weights, target_pos, prefix_pos, prefix_weight, p_copy, a_sink, copy_col, lam, alpha,
                lm_head_weight, rows, rows_t, vocab_pos, x_s, w_s, grad_s, stats=None):
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, rows.T, x_s, w_s)
        losses, grad_input, d_pc, d_as, d_al = _run(logits, target_pos, mtp_weights, prefix_pos, prefix_weight,
                                                    p_copy, a_sink, copy_col, lam, alpha, grad_s)
        _stats(stats, d_al, d_pc, p_copy)
        ctx.save_for_backward(x_f8, rows_t, grad_input, vocab_pos, d_pc, d_as, d_al)
        ctx.scales = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, rows_t, grad_input, vocab_pos, d_pc, d_as, d_al = ctx.saved_tensors
        grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)
        grad_w = torch.empty((x_f8.shape[1], vocab_pos.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w)
        return (grad_x, None, None, None, None, d_pc * grad_output, d_as * grad_output, None, None,
                d_al * grad_output if EXT_GATE else None, grad_w, None, None, None, None, None, None, None)

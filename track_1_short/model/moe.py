"""Mixture of experts in the MLP slots of MOE_LAYERS: configuration, parameters and fp8 caches, and the layer itself.

The MLP of each MOE_LAYERS layer is replaced by MOE_EXPERTS routed relu^2 experts of MOE_HIDDEN: top-1 on a softmax
router (fp32 parameter, read in bf16), the output scaled by the chosen probability, Switch load-balancing loss of weight
MOE_AUX. Dropless: tokens sorted by expert into 128-row tiles, every GEMM one grouped FP8 launch
(perf/kernels/moe_mlp.py RoutedFP8MoE; validation runs the same experts in bf16 through torch._grouped_mm).

The experts are one ANVIL bank, moe_bank, [layer][c_fc, c_proj][expert] of [MOE_HIDDEN, 768], whose cascade runs at
Frobenius margin MOE_ANVIL_MARGIN: an expert that sees one or two tokens has a rank-1 gradient, which normalizes to
sigma_max = 1/margin, and a value at the cascade's own divergence edge is pushed over it by the bf16 rounding (inf lane
power, NaN weights). Record #360's six-map cascade diverges from sigma 0.953 and would need a margin of 1.15; the
four-map one (optim/anvil.py) is bounded up to sigma = 1.022, so 1.1 (sigma_max = 0.909) keeps it inside the basin.
Each rank quantizes the expert matrices it owns right after their update and all-gathers the e4m3 rows and the scales
(half the bf16 gather's bytes); the column-major cache is a batched byte transpose of the gathered rows, and the bf16
replica is gathered only before its readers (the terminal decontraction norms and the validation:
AnvilAndAdam.land_bf16_replicas).

MoEMixin holds the model-side pieces of GPT: the bank, router and fp8 buffers (init_moe), the cache refreshes
(quantize_moe_full / gather_moe_f8 / build_moe_cols; GPT.quantize_moe_local owns the exact-scale predicate) and the layer
(_moe). Parameters and buffers are attributes of GPT itself, so the optimizer labels (moe_bank, moe_router) and the
address census see them like any other.

Provenance: ours (not in record #360 or #92).
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from track_1_short.perf.kernels import moe_mlp
from track_1_short.perf.kernels.mlp import quantize_mlp_weights_dual, quantize_weights_dual_ntiles
from track_1_short.perf.kernels.moe_mlp import RoutedFP8MoE, transpose_copy_batched

MOE_LAYERS = (1, 3, 5, 6, 8)   # the MoE layers; the dense slots of mlp_bank are derived from them
MOE_EXPERTS = 8
MOE_HIDDEN = 1024
# Weight of the load-balancing loss. Without expert parallelism (every GPU computes all the experts for its own tokens,
# dropless) an imbalance costs no time: the loss is only a regularizer.
MOE_AUX = 0.01
MOE_TOPK = 1   # experts per token: top-1, the only routing the kernels are used with
# Frobenius margin of moe_bank's cascade (the dense banks run FROBENIUS_MARGIN = 1.05; see the header).
MOE_ANVIL_MARGIN = 1.1
MOE_INDEX = {layer: k for k, layer in enumerate(MOE_LAYERS)}
assert MOE_HIDDEN % 256 == 0, "the FP8 MoE kernels tile MOE_HIDDEN by 256"
assert len(MOE_LAYERS) * 2 * MOE_EXPERTS % 8 == 0, "moe_bank's matrices must shard evenly over the 8 GPUs"
assert not (moe_mlp.GRAD_E4M3 or moe_mlp.DW_PROMOTE or moe_mlp.DX_PROMOTE), \
    "[moe] this trainer runs the experts' gradients in e5m2 without fp32 promotion: unset MOE_GRAD_FP8 / MOE_*_PROMOTE"


class MoEMixin:
    """The MoE parts of GPT. Needs on the host class: world_size, device, and the buffers _mlp_x_scale /
    _mlp_grad_scale of the dense MLP."""

    def init_moe(self, model_dim: int, rank: int, grad_scale: float, bound: float):
        """moe_bank, moe_router and the experts' fp8 state. Called right after mlp_bank is drawn, so the init draws
        keep one fixed order: mlp_bank, then each MoE layer's c_fc, then the router. `bound` is mlp_bank's
        uniform bound, `grad_scale` the static e5m2 gradient scale (FP8_GRAD_SCALE)."""
        # MoE experts: [layer][c_fc, c_proj][expert] of [MOE_HIDDEN, dim], so a layer's c_fc (and c_proj) over its
        # experts is one contiguous [E, h, dim] block; initialized as mlp_bank (c_fc uniform, c_proj zero).
        n, E = len(MOE_LAYERS), MOE_EXPERTS
        self.moe_bank = nn.Parameter(torch.empty(n * 2 * E, MOE_HIDDEN, model_dim))
        self.moe_bank.reshape = (n * 2 * E, MOE_HIDDEN, model_dim)
        # c_proj takes twice the ANVIL lr, as in mlp_bank (_build_param_cfg reads this: the layout is not interleaved)
        self.moe_bank.c_proj_matrices = frozenset(k * 2 * E + E + e for k in range(n) for e in range(E))
        with torch.no_grad():
            for k in range(n):
                self.moe_bank[k * 2 * E:k * 2 * E + E].uniform_(-bound, bound)
                self.moe_bank[k * 2 * E + E:(k + 1) * 2 * E].zero_()
        # Router: small random init, so the top-1 choice starts spread instead of all on one expert.
        self.moe_router = nn.Parameter(0.01 * torch.randn(n, E, model_dim))
        L, h, d, dev = n * 2 * E, MOE_HIDDEN, model_dim, self.device
        nt = quantize_weights_dual_ntiles(h, d)
        f32 = torch.float32
        for name, buf in (
                # e4m3 caches in moe_bank's slot order: row-major [L, h, d] and column-major (the transpose(1, 2)
                # view of a contiguous [L, d, h]), one per-tensor scale per matrix and its per-tile amaxes
                ("_moe_f8_row", torch.empty(L, h, d, dtype=torch.float8_e4m3fn, device=dev)),
                ("_moe_f8_col", torch.empty(L, d, h, dtype=torch.float8_e4m3fn, device=dev).transpose(1, 2)),
                ("_moe_scales", torch.ones(L, dtype=f32, device=dev)),
                ("_moe_partial_amax", torch.zeros(L, nt, dtype=f32, device=dev)),
                # one delayed post (e4m3) and dpre (e5m2) scale per MoE layer, as the dense MLP's
                ("_moe_post_scales", torch.ones(n, dtype=f32, device=dev)),
                ("_moe_post_amax", torch.zeros(n, 1, dtype=f32, device=dev)),
                ("_moe_dpre_scales", torch.full((n,), grad_scale, dtype=f32, device=dev)),
                ("_moe_dpre_amax", torch.zeros(n, 1, dtype=f32, device=dev)),
                ("_moe_g_amax", torch.zeros(n, 1, dtype=f32, device=dev)),   # written by the e5m2 gather, unread
                # the backward writes the experts' dW here, slot by slot (moe_mlp.py dw_into_bank): StepGraphs
                # hands it to ANVIL as moe_bank.grad, so autograd never stacks a bank-sized gradient
                ("moe_bank_grad", torch.zeros(L, h, d, dtype=torch.bfloat16, device=dev)),
                ("moe_grad_accumulate", torch.zeros(1, dtype=torch.int32, device=dev))):   # 0: every backward overwrites
            self.register_buffer(name, buf, persistent=False)
        # The matrices this rank owns, [lo, hi): the optimizer's shard of moe_bank (rank r updates [r*c, (r+1)*c)).
        c = L // self.world_size
        self._moe_lo, self._moe_hi = rank * c, (rank + 1) * c

    # ---- the experts' fp8 weight caches --------------------------------------------------------------------
    # Every rank holds all of them; each rank writes the matrices it owns, [_moe_lo, _moe_hi), right after their ANVIL
    # update (GPT.quantize_moe_local, from the optimizer's publish step), all-gathers the e4m3 rows and the scales
    # (gather_moe_f8, waited by DeferredGathers.flush), then rebuilds the column-major cache from the gathered rows
    # (build_moe_cols). Exact scales for the first FP8_EXACT_SCALE_CALLS calls by the MLP's own host counter, read before
    # the step's quantize_mlp_fp8 bumps it, so the experts and the dense MLP agree on the predicate.

    def _quantize_moe(self, lo: int, hi: int, exact: bool):
        bank = self.moe_bank.data[lo:hi]
        if exact:   # max |w| as max(amax, -amin): no |w| temporary
            flat = bank.reshape(hi - lo, -1)
            amax = torch.maximum(flat.amax(dim=1), -flat.amin(dim=1))
            self._moe_scales[lo:hi] = (amax.clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max).float()
        quantize_mlp_weights_dual(bank, self._moe_scales[lo:hi], self._moe_partial_amax[lo:hi],
                                  row=self._moe_f8_row[lo:hi], col_t=self._moe_f8_col[lo:hi].transpose(1, 2),
                                  update_scales=not exact)

    def quantize_moe_full(self):
        """The whole bank, exact scales: at init and after the post-warmup reset (every rank has the full bf16 bank)."""
        with torch.no_grad():
            self._quantize_moe(0, self._moe_scales.numel(), exact=True)

    def gather_moe_f8(self):
        """Async all-gather of the owners' e4m3 rows (as uint8) and scales, in place: [(label, future)]."""
        lo, hi = self._moe_lo, self._moe_hi
        rows = self._moe_f8_row.view(torch.uint8)
        return [("moe_bank", dist.all_gather_into_tensor(rows, rows[lo:hi], async_op=True).get_future()),
                ("moe_scales", dist.all_gather_into_tensor(self._moe_scales, self._moe_scales[lo:hi],
                                                           async_op=True).get_future())]

    def build_moe_cols(self):
        """After the row gather landed: the column-major cache of every matrix, one batched byte transpose."""
        with torch.no_grad():
            transpose_copy_batched(self._moe_f8_row.view(torch.uint8), self._moe_f8_col.transpose(1, 2).view(torch.uint8))

    def _moe(self, layer: int, normed, moe_weights, post):
        """MoE layer `layer` on the normed input: (routed output, Switch load-balancing loss). Training: the fused FP8
        op, with `post` (the folded post-lambda) inside its per-token gate. Validation (post None): the same experts
        in bf16, tokens sorted by expert, one grouped GEMM per projection, the output gated by the probability."""
        E = MOE_EXPERTS
        shape = normed.shape
        xin = normed.view(-1, shape[-1])
        T = xin.size(0)
        k = MOE_INDEX[layer]
        w_fc, w_proj, router_w = moe_weights[k]                           # [E, h, d] x2, [E, d] bf16
        if self.training:
            up, down = slice(2 * k * E, (2 * k + 1) * E), slice((2 * k + 1) * E, (2 * k + 2) * E)
            d = w_fc.size(-1)
            out, probs, counts = RoutedFP8MoE.apply(
                xin, router_w, post, w_fc, w_proj,
                self._moe_f8_row[up].view(-1, d), self._moe_f8_col[up],
                self._moe_f8_row[down].view(-1, d), self._moe_f8_col[down],
                self._moe_scales[up], self._moe_scales[down], self._mlp_x_scale, self._mlp_grad_scale,
                self._moe_g_amax[k], self._moe_post_scales[k:k + 1], self._moe_post_amax[k],
                self._moe_dpre_scales[k:k + 1], self._moe_dpre_amax[k], self.moe_bank_grad, k,
                self.moe_grad_accumulate, MOE_TOPK)
        else:
            probs = torch.softmax(F.linear(xin, router_w).float(), dim=-1)
            gate, expert = probs.max(dim=-1)
            counts = (expert[:, None] == torch.arange(E, device=xin.device)).sum(0, dtype=torch.int32)
            offs = counts.cumsum(0, dtype=torch.int32)
            perm = torch.argsort(expert.to(torch.int32), stable=True)
            xs = xin.index_select(0, perm)
            hidden = torch.relu(torch._grouped_mm(xs, w_fc.transpose(1, 2), offs=offs)).square()
            ys = torch._grouped_mm(hidden, w_proj, offs=offs)
            y = torch.empty_like(ys).index_copy_(0, perm, ys)
            out = y * gate[:, None].to(y.dtype)
        aux = E * (counts.float() / T * probs.mean(0)).sum()
        return out.view(shape), aux

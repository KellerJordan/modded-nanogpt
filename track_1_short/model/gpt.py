"""The GPT model: embeddings, 11 transformer blocks with MUDD skip connections, and the loss."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exact_match import CELLS as RET_CELLS

from track_1_short.model.attention import AttnArgs, CausalSelfAttention, Yarn
from track_1_short.model.layers import CastedLinearT, next_multiple_of_n, norm
from track_1_short.ngram_table import NGRAM_SIGN_POOL_ROWS
from track_1_short.perf.kernels.cross_entropy import (
    FusedSoftcappedCrossEntropy,
    lm_head_inverse_scale,
    quantize_lm_head_dual,
)
from track_1_short.perf.kernels.fp8_attention_quant import PackedQKVFP8Cache
from track_1_short.perf.kernels.mlp import (
    FusedLinearReLUSquareFunction,
    quantize_dual_layout_fused,
    quantize_mlp_weights_dual,
    quantize_weights_dual_ntiles,
)
from track_1_short.perf.kernels.ngram_embed import ngram_embedding
from track_1_short.perf.kernels.sampled_cross_entropy import SampledSoftcappedCrossEntropy
from track_1_short.perf.kernels.value_embed import value_embed_lookup
from track_1_short.perf.residual_fusion import rms_norm_with_head, scale, scale_add
from track_1_short.sampled_softmax import SampledLoss

# Layer topology (11 layers). Depth cut from record #360 (ANVIL2): layer 7 is removed whole -- its
# residual scaling and x0 injection stay, so cache[7] still exists -- and layers 4 and 9 run
# their MLP only. Layer 6 has had no attention since @YouJiacheng; it adds a gated skip from layer 3.
NUM_LAYERS = 11
NO_ATTN_LAYERS = (4, 6, 7, 9)
NO_MLP_LAYERS = (7,)
ATTN_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in NO_ATTN_LAYERS)  # (0, 1, 2, 3, 5, 8, 10)
# Long sliding window (with a partial key offset) on these layers, short on the other attention layers.
LONG_WINDOW_LAYERS = (3, 10)
PAIRED_HEAD_LAYERS = (0, 2, 5)

# Mixed head widths, from record #360. The long-window layers keep full-width d_qk = 128 query/key
# heads; every other attention layer runs d_qk = 64 with a fully rotating rotary. The HALF_V_LAYERS
# also halve the value/output head (d_v = 64); record #360 lists 1, 4, 7, 8, of which 4 and 7 have
# no attention. The banks keep full-width (128) heads: a narrow layer reads the leading 64 of each
# head's Q/K/V rows and of each head's O columns, and the rest never get a gradient.
WIDE_QK_LAYERS = LONG_WINDOW_LAYERS
HALF_V_LAYERS = (1, 8)
NARROW_QK_LAYERS = tuple(i for i in ATTN_LAYERS if i not in WIDE_QK_LAYERS + HALF_V_LAYERS)  # (0, 2, 5)
NARROW_HEAD_DIM = 64  # the narrow query/key heads' width, and the halved value/output heads'
# Softmax scales at the start of training (YaRN grows them with the window): the narrow layers' from
# record #360; the wide layers' is record #360's Yarn default.
NARROW_ATTN_SCALE = 0.13
WIDE_ATTN_SCALE = 0.085
# Attention layers grouped by packed QKV weight shape. Each group has one fp8 weight cache, refreshed
# in one kernel launch, so the attention banks store the groups back to back in this order (below).
ATTN_WIDTH_GROUPS = {"narrow": NARROW_QK_LAYERS, "half_v": HALF_V_LAYERS, "wide": WIDE_QK_LAYERS}
# Attention-bank slot order: slot j of qk_bank / vo_bank belongs to layer ATTN_BANK_ORDER[j], so each
# width group is a contiguous run of slots, and the d_qk = 64 groups (narrow, half_v) lead.
ATTN_BANK_ORDER = sum(ATTN_WIDTH_GROUPS.values(), ())  # (0, 2, 5, 1, 8, 3, 10)
NUM_QK64_SLOTS = len(NARROW_QK_LAYERS) + len(HALF_V_LAYERS)
assert sorted(ATTN_BANK_ORDER) == list(ATTN_LAYERS), "every attention layer needs exactly one width group"
assert ATTN_BANK_ORDER[NUM_QK64_SLOTS:] == WIDE_QK_LAYERS
# Per-head output gates (from the MUDD gates) on these attention layers.
ATTN_GATE_LAYERS = (3, 10)
# Gated XSA on these attention layers, its per-head strength from the pre MUDD gate.
XSA_LAYERS = (1, 3)
# The residual stream after these layers is kept for later skips: layer 6 re-adds cache[3], and the
# last layer and the post-loop MUDD mix read cache[7].
CACHE_LAYERS = (3, 7)
# Token value embeddings are added to V on these layers, one embedding plane each.
VALUE_EMBED_LAYERS = (1, 2, 8, 10)
# ...each gated by a learned per-head gate, except the last layer, whose gate comes from MUDD.
VALUE_EMBED_GATE_LAYERS = (1, 2, 8)
# The gate reads this many leading channels of both the normed attention input and the value embedding.
VALUE_EMBED_GATE_CHANNELS = 6
# Residual injection sites, from record #360. x0 (the normed input embedding) is added back into the
# residual stream on X0_INJECT_LAYERS, and the hashed n-gram embedding on BIGRAM_INJECT_LAYERS (layer 0's
# before the loop, so it is part of cache[0]); each site has its own per-token MUDD gate lane. The last
# layer injects both through its own MUDD coefficients (mu[10], mu[11]) instead; layer 6 injects neither.
X0_INJECT_LAYERS = (0, 1, 2, 4, 5, 7)
BIGRAM_INJECT_LAYERS = (0, 1, 4, 5, 9)
assert not {6, 10} & set(X0_INJECT_LAYERS + BIGRAM_INJECT_LAYERS)
# MUDD gates (init_mudd_gate): the pre gate, computed from x0, serves the layers before POST_GATE_LAYER;
# the post gate, computed at the start of POST_GATE_LAYER, serves it and the layers after.
POST_GATE_LAYER = 4
PRE_GATE_X0_LAYERS = tuple(i for i in X0_INJECT_LAYERS if i < POST_GATE_LAYER)             # (0, 1, 2)
PRE_GATE_BIGRAM_LAYERS = tuple(i for i in BIGRAM_INJECT_LAYERS if i < POST_GATE_LAYER)     # (0, 1)
POST_GATE_X0_LAYERS = tuple(i for i in X0_INJECT_LAYERS if i >= POST_GATE_LAYER)           # (4, 5, 7)
POST_GATE_BIGRAM_LAYERS = tuple(i for i in BIGRAM_INJECT_LAYERS if i >= POST_GATE_LAYER)   # (4, 5, 9)
PRE_GATE_ATTN_GATE_LAYERS = tuple(i for i in ATTN_GATE_LAYERS if i < POST_GATE_LAYER)      # (3,)
POST_GATE_ATTN_GATE_LAYERS = tuple(i for i in ATTN_GATE_LAYERS if i >= POST_GATE_LAYER)    # (10,)
assert all(i < POST_GATE_LAYER for i in XSA_LAYERS), "the XSA strengths come from the pre gate"
# Layer 8 runs a second MLP in parallel, from MLP bank slot 11 (the slot that was sharding padding).
PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, 11
# The post-loop MUDD mix splits model_dim into this many channel groups, each with its own coefficient delta.
MUDD_GROUPS = 12
# MUDD coefficients at the start of the last layer (init_mudd lists them).
LAST_LAYER_MUDD_COEFS = 14
# MUDD gate lane widths (init_mudd_gate): one lane per head for the XSA strengths and the attention
# gates, one lane per injection site for x0 / the n-gram embedding, one for the layer-6 skip.
MUDD_GATE_HEAD_LANES = 6
MUDD_GATE_SCALE = 0.1  # the gates' output scale at init; biases are stored pre-divided by it
# MLP bank: 12 slots of (c_fc, c_proj), 24 matrices for even sharding over 8 GPUs. Slot i is layer i's
# MLP, slot 11 is PARALLEL_MLP_SLOT, slot 7 is dead (NO_MLP_LAYERS) but keeps the bank even.
NUM_MLP_SLOTS = 12
# MLP hidden size, cut from 4 * 768 = 3072 in record #360.
MLP_HIDDEN_DIM = 2816

# Static FP8 scales of the attention projection's input (e4m3, saturates entries beyond |8|) and of
# its incoming gradient, both from record #360.
FP8_ATTN_X_SCALE = 8.0 / 448.0
FP8_ATTN_GRAD_SCALE = 1.0 / 448.0

# Validation computes the loss over slabs of this many rows (record #360), so its [rows, vocab] fp32
# logits (6.6 GB per slab) stay small next to the rank's 16 GB n-gram shard.
EVAL_CE_SLAB_ROWS = 32768

# FP8 MLP scales (see perf/kernels/mlp.py), from record #360.
FP8_MLP_X_SCALE = 2 ** -4      # static: post-RMS-norm rows have max|x| <= sqrt(768) = 27.7 < 448 * 2^-4
FP8_GRAD_SCALE = 2 ** -6       # static e5m2 scale of the MLP's incoming gradient
FP8_POST_HEADROOM = 1.03       # delayed relu(pre)^2 scale = last step's amax * this / 448
FP8_DPRE_HEADROOM = 1.25       # delayed dpre scale = last step's amax * this / 57344
# The first calls of quantize_mlp_fp8 (the first steps after init or the post-warmup reset) set the
# weight scales exactly; later calls use the one-step-delayed amax the quantize kernel records.
FP8_EXACT_SCALE_CALLS = 16

# Fused triton kernel: relu(x @ W1.T)^2 @ W2.T
# https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    prefix_weight: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int  # longest attention segment in a training batch
    # Training only: the candidate set when this step's loss is a sampled softmax; None = full softmax.
    sampled_loss: SampledLoss | None = None

class GPT(nn.Module):
    """Training runs every projection in fp8 (the attention QKV, the MLP, the lm_head loss); validation
    runs the bf16 path under no_grad."""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int,
                 *, ngram_dim: int, world_size: int, device: torch.device):
        super().__init__()
        assert num_layers == NUM_LAYERS
        self.world_size = world_size
        self.device = device
        self.ngram_dim = ngram_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # there are only 50257 unique GPT-2 tokens; extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977, originates from Karpathy's experiments.
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        # Prefix-token lookup table for prefix token prediction. Allocated here, but filled
        # after the clock starts (see "start the clock") so the build is charged to training
        # time. -1 means "no valid prefix" == term disabled, which is what warmup runs with.
        self.register_buffer("prefix_table", torch.full((self.vocab_size,), -1, dtype=torch.int64), persistent=False)

        # The n-gram table rows the current update cycle (or eval batch) reads, pulled from their owning
        # ranks (see track_1_short/ngram_table.py). Sized for the largest forward, two rows per token,
        # which also holds a training cycle (NgramTable asserts it).
        self.register_buffer("ngram_cache", torch.zeros(2 * max_seq_len, ngram_dim, dtype=torch.bfloat16, device=device),
                             persistent=False)

        # Canonical token mask for the validation softmax, one bit per (prev, cur) pair.
        # Allocated all-zero == no masking.
        self.register_buffer("canon_mask", torch.zeros(self.vocab_size, self.vocab_size // 8, dtype=torch.uint8), persistent=False)

        # fp8 scales of the training loss (the /8 on grad_s is an exact binade shift, record #360).
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, x_s=100/448, w_s=2.0/448, grad_s=(0.75 / 8) / 448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        # fp8 copies of lm_head for the training loss, in both layouts its GEMMs read: row-major for the
        # backward, column-major (strides (1, model_dim)) for the forward. Refreshed by quantize_mlp_fp8.
        self.register_buffer("_lm_head_f8_row", torch.empty(model_dim, self.vocab_size, dtype=torch.float8_e4m3fn, device=device), persistent=False)
        self.register_buffer("_lm_head_f8_col", torch.empty_strided((model_dim, self.vocab_size), (1, model_dim), dtype=torch.float8_e4m3fn, device=device), persistent=False)
        self.register_buffer("_lm_head_inv_w_s", lm_head_inverse_scale(self.lm_head.w_s, device), persistent=False)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            # tie embed and lm_head at init
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.init_attn(model_dim, head_dim, num_heads, max_seq_len)
        self.init_mlp(model_dim)
        self.init_misc(model_dim, num_layers)
        self.init_mudd(num_layers, model_dim)
        self.init_mudd_gate(model_dim)
        # Exact-match retrieval (track_1_short/retrieval.py): per cell a scale of the candidates' mean embedding and an
        # embedding of its own, and a scale per injection site.
        self.ret_next_scale = nn.Parameter(torch.full((RET_CELLS,), 4.0))
        self.ret_bucket_embed = nn.Parameter(torch.zeros(RET_CELLS, model_dim))
        self.ret_site_scale_in = nn.Parameter(torch.tensor(0.1))
        self.ret_site_scale_mid = nn.Parameter(torch.tensor(8.0))
        self.ret_site_scale_out = nn.Parameter(torch.tensor(8.0))

        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def init_attn(self, model_dim, head_dim, num_heads, max_seq_len):
        # One attention module per attention layer, at the layer's head widths (no learned params --
        # weights come from qk_bank/vo_bank). The patched FA3 takes unequal query/key and value widths
        # only as (64, 128) / (128, 64).
        assert head_dim == 2 * NARROW_HEAD_DIM
        self.attn = nn.ModuleDict({
            str(layer): CausalSelfAttention(
                num_heads, head_dim, qk_dim=self.attn_qk_dim(layer), v_dim=self.attn_v_dim(layer),
                val_max_seq_len=max_seq_len, paired=layer in PAIRED_HEAD_LAYERS,
            )
            for layer in ATTN_LAYERS
        })
        # Rotary tables: one per (query/key width, pairing) in use.
        self.yarn = Yarn(NARROW_HEAD_DIM, max_seq_len, attn_scale=NARROW_ATTN_SCALE, device=self.device)
        self.yarn_paired_head = Yarn(
            NARROW_HEAD_DIM, max_seq_len, paired=True, attn_scale=NARROW_ATTN_SCALE, device=self.device,
        )
        self.yarn_wide = Yarn(head_dim, max_seq_len, attn_scale=WIDE_ATTN_SCALE, device=self.device)
        assert not set(PAIRED_HEAD_LAYERS) & set(WIDE_QK_LAYERS), "no paired rotary table at full width"

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        # One [vocab, model_dim] plane per VALUE_EMBED_LAYERS entry. Every rank holds the whole table, but
        # during training only the rows the current update cycle reads are kept current in it; it is
        # made whole again before every validation (perf/value_embed_pull.py).
        num_ve = len(VALUE_EMBED_LAYERS)
        self.value_embeds = nn.Parameter(0.01 * torch.randn(num_ve * self.vocab_size, model_dim, dtype=torch.bfloat16))

        # value embedding gate weights, one per VALUE_EMBED_GATE_LAYERS entry
        self.ve_gate_bank = nn.Parameter(torch.zeros(len(VALUE_EMBED_GATE_LAYERS), num_heads, 12))

        # Parameter banks for sharded optimization, by @chrisjmccormick. Only ATTN_LAYERS own
        # attention weights; bank slot j belongs to layer ATTN_BANK_ORDER[j]. Rows are full width
        # (head_dim per head) on every layer; the narrower layers read a leading part of each head.
        num_slots = len(ATTN_BANK_ORDER)
        hdim = num_heads * head_dim

        # QK bank: per-head-pair ANVIL groups for Q, K weights. Each pair of adjacent heads gets its
        # own independent whitening: a slot's 2 * num_heads heads (Q heads, then K heads) are
        # num_heads groups of two full-width heads.
        qk_groups_per_slot = num_heads
        num_qk_groups = num_slots * qk_groups_per_slot  # 42
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=self.world_size)  # 48
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, 2 * head_dim, model_dim))
        self.qk_bank.reshape = (num_qk_padded, 2 * head_dim, model_dim)

        # VO bank: per-layer ANVIL groups for V and O weights; slot j's V is matrix 2j, its O 2j + 1.
        # V is stored [out = hdim, in = model_dim] and O [out = model_dim, in = hdim], both nn.Linear
        # layout (record #360); the two shapes coincide because hdim == model_dim.
        assert hdim == model_dim
        num_vo_real = 2 * num_slots  # 14
        num_vo_padded = next_multiple_of_n(num_vo_real, n=self.world_size)  # 16
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, model_dim))
        self.vo_bank.reshape = (num_vo_padded, hdim, model_dim)

        # improved init scale by @YouJiacheng and @srashedll. Every live row is drawn, including the
        # rows the narrower layers never read (record #360 does the same).
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()

        # FP8 packed [Q; K; V] weight caches, one per width group (perf/kernels/fp8_attention_quant.py).
        self.attn_fp8 = nn.ModuleDict({
            group: PackedQKVFP8Cache(
                len(layers),
                rows=num_heads * (2 * self.attn_qk_dim(layers[0]) + self.attn_v_dim(layers[0])),
                cols=model_dim, device=self.device,
            )
            for group, layers in ATTN_WIDTH_GROUPS.items()
        })
        self.register_buffer("_attn_x_scale", torch.tensor([FP8_ATTN_X_SCALE], device=self.device), persistent=False)
        self.register_buffer("_attn_grad_scale", torch.tensor([FP8_ATTN_GRAD_SCALE], device=self.device), persistent=False)

    def attn_qk_dim(self, layer: int) -> int:
        return self.head_dim if layer in WIDE_QK_LAYERS else NARROW_HEAD_DIM

    def attn_v_dim(self, layer: int) -> int:
        return NARROW_HEAD_DIM if layer in HALF_V_LAYERS else self.head_dim

    def init_mlp(self, model_dim):
        # MLP bank: stores c_fc and c_proj for all NUM_MLP_SLOTS slots.
        self.mlp_hdim = MLP_HIDDEN_DIM
        self.mlp_bank = nn.Parameter(torch.empty(NUM_MLP_SLOTS, 2, self.mlp_hdim, model_dim))  # (12, 2, 2816, 768)
        self.mlp_bank.reshape = (2 * NUM_MLP_SLOTS, self.mlp_hdim, model_dim)  # Shape for sharding: (24, 2816, 768)
        # The optimizer leaves these matrices untouched: c_fc and c_proj of each NO_MLP_LAYERS slot.
        self.mlp_bank.frozen_matrices = frozenset(2 * layer + j for layer in NO_MLP_LAYERS for j in (0, 1))

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

        self.init_mlp_fp8(model_dim)

    def init_mlp_fp8(self, model_dim):
        """FP8 MLP state, one entry per bank slot; refreshed by quantize_mlp_fp8 after every optimizer step.

        None of it is in the state_dict: rearm_fp8_bootstrap() restarts it after the post-warmup reset.
        """
        n, h, d, dev = NUM_MLP_SLOTS, self.mlp_hdim, model_dim, self.device
        f32, e4m3 = torch.float32, torch.float8_e4m3fn
        col_major = lambda: torch.empty(n, d, h, dtype=e4m3, device=dev).transpose(1, 2)  # (n, h, d), strides (h*d, 1, h)
        num_tiles = quantize_weights_dual_ntiles(h, d)
        buffers = {
            # Weight caches. Up projection: row-major for the forward kernel, column-major for dx.
            # Down projection: column-major for the forward _scaled_mm, row-major for the dpre kernel.
            "_mlp_up_f8": torch.empty(n, h, d, dtype=e4m3, device=dev),
            "_mlp_up_f8_col": col_major(),
            "_mlp_down_f8": col_major(),
            "_mlp_down_f8_row": torch.empty(n, h, d, dtype=e4m3, device=dev),
            # Weight scales, and the per-tile amax each quantize records for the next call's scales.
            "_mlp_up_scales": torch.ones(n, dtype=f32, device=dev),
            "_mlp_down_scales": torch.ones(n, dtype=f32, device=dev),
            "_mlp_up_partial_amax": torch.zeros(n, num_tiles, dtype=f32, device=dev),
            "_mlp_down_partial_amax": torch.zeros(n, num_tiles, dtype=f32, device=dev),
            # Static activation / gradient scales, and the GEMM dequant scales derived from them.
            "_mlp_x_scale": torch.tensor(FP8_MLP_X_SCALE, dtype=f32, device=dev),
            "_mlp_grad_scale": torch.tensor(FP8_GRAD_SCALE, dtype=f32, device=dev),
            "_mlp_up_dequant": torch.ones(n, dtype=f32, device=dev),   # up_scales * x_scale
            "_mlp_dq_bwd": torch.ones(n, dtype=f32, device=dev),       # down_scales * grad_scale
            # Delayed scales of relu(pre)^2 and dpre; the kernels atomic-max into slot 0 of each amax row.
            "_mlp_post_scales": torch.ones(n, dtype=f32, device=dev),
            "_mlp_post_amax": torch.zeros(n, 1, dtype=f32, device=dev),
            "_mlp_dpre_scales": torch.full((n,), FP8_GRAD_SCALE, dtype=f32, device=dev),
            "_mlp_dpre_amax": torch.zeros(n, 1, dtype=f32, device=dev),
            # Each slot's residual post-lambda, and the two down-projection dequant scales it is folded into.
            "_mlp_fold_p": torch.ones(n, dtype=f32, device=dev),
            "_mlp_down_scales_folded": torch.ones(n, dtype=f32, device=dev),
            "_mlp_dq_bwd_folded": torch.ones(n, dtype=f32, device=dev),
        }
        for name, buf in buffers.items():
            self.register_buffer(name, buf, persistent=False)
        # Host counter toward FP8_EXACT_SCALE_CALLS (a replay of the captured refresh bumps it too,
        # perf/cuda_graphs/fp8_refresh_graphs.py).
        self.mlp_quantize_calls = 0

    def init_misc(self, model_dim, num_layers):
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        # The n-gram table itself is sharded across ranks and lives in NgramTable; the model keeps its
        # +-1 sign pool (identical on every rank: main() broadcasts it).
        ngram_sign_pool = torch.randn(NGRAM_SIGN_POOL_ROWS, self.ngram_dim).sign().to(torch.bfloat16)
        self.register_buffer('ngram_sign_pool', ngram_sign_pool)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))

        pad = (-num_layers * 2 - 2) % self.world_size
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

        Start of last layer produces LAST_LAYER_MUDD_COEFS (14) coefficients:
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
        max_num_coef = LAST_LAYER_MUDD_COEFS

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
        # Post-loop only: per-channel-group coefficient deltas, read off the same 64-dim hidden.
        # Zero-init, so step 0 is the plain per-token mix.
        self.mudd_w2g = nn.Parameter(torch.zeros(max_num_coef, MUDD_GROUPS, mudd_dim))

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def forward_mudd_grouped(self, x, id, num_coef):
        """forward_mudd's coefficients plus (B, T, num_coef, MUDD_GROUPS) per-channel-group deltas."""
        h = F.gelu(F.linear(x, self.mudd_w1[id]))
        y = (F.linear(h, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        deltas = torch.einsum("btd,kgd->btkg", h, self.mudd_w2g[:num_coef]) * self._mudd_scale
        return y.split(1, dim=-1), deltas

    def init_mudd_gate(self, model_dim: int):
        self._mudd_gate_scale = nn.Parameter(torch.tensor(MUDD_GATE_SCALE))
        mudd_gate_dim = 64
        H = MUDD_GATE_HEAD_LANES
        assert self.num_heads == H
        # Gate lane layouts, in the order unpack_pre_mudd_gate / unpack_post_mudd_gate read them:
        # pre:  xsa[1,3] 12 + attn[3] 6 + x0[0,1,2] 3 + bigram[0,1] 2 = 23
        # post: attn[10] 6 + x0[4,5,7] 3 + bigram[4,5,9] 3 + skip 1 = 13
        pre_attn_start = H * len(XSA_LAYERS)
        pre_x0_start = pre_attn_start + H * len(PRE_GATE_ATTN_GATE_LAYERS)
        pre_bigram_start = pre_x0_start + len(PRE_GATE_X0_LAYERS)
        self._mudd_gate_pre_num_coef = pre_bigram_start + len(PRE_GATE_BIGRAM_LAYERS)
        post_x0_start = H * len(POST_GATE_ATTN_GATE_LAYERS)
        post_bigram_start = post_x0_start + len(POST_GATE_X0_LAYERS)
        post_skip_lane = post_bigram_start + len(POST_GATE_BIGRAM_LAYERS)
        self._mudd_gate_post_num_coef = post_skip_lane + 1
        max_num_coef = max(self._mudd_gate_pre_num_coef, self._mudd_gate_post_num_coef)
        self.mudd_gate_w1 = nn.Parameter(torch.empty(2, mudd_gate_dim, model_dim))
        self.mudd_gate_w2 = nn.Parameter(torch.zeros(2, max_num_coef, mudd_gate_dim))
        for j in range(2):
            nn.init.kaiming_uniform_(self.mudd_gate_w1.data[j], a=math.sqrt(5))

        # Bias init in the pre-scaled domain (effective = bias * MUDD_GATE_SCALE); XSA and x0 lanes start at 0.
        bs_init = torch.zeros(2, max_num_coef)
        attn_gate_bias = 0.25 / MUDD_GATE_SCALE
        bigram_gate_bias = 0.05 / MUDD_GATE_SCALE
        skip_gate_bias = 0.5 / MUDD_GATE_SCALE
        bs_init[0, pre_attn_start:pre_x0_start].fill_(attn_gate_bias)
        bs_init[0, pre_bigram_start:self._mudd_gate_pre_num_coef].fill_(bigram_gate_bias)
        bs_init[1, 0:post_x0_start].fill_(attn_gate_bias)
        bs_init[1, post_bigram_start:post_skip_lane].fill_(bigram_gate_bias)
        bs_init[1, post_skip_lane].fill_(skip_gate_bias)
        self.mudd_gate_b2 = nn.Parameter(bs_init)

    def forward_mudd_gate(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_gate_w1[id]))
        return (F.linear(x, self.mudd_gate_w2[id, :num_coef]) + self.mudd_gate_b2[id, :num_coef]) * self._mudd_gate_scale.type_as(x)

    @staticmethod
    def _unpack_lanes(gate, start, layers, gates, width=1):
        """gates[layer] = the next `width`-lane slice of `gate` for each of `layers`; returns the next free lane."""
        for k, layer in enumerate(layers):
            lo = start + k * width
            gates[layer] = gate[..., lo:lo + width]
        return start + len(layers) * width

    def unpack_pre_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates, bigram_gates):
        lane = self._unpack_lanes(gate, 0, XSA_LAYERS, xsa_alphas, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane == self._mudd_gate_pre_num_coef

    def unpack_post_mudd_gate(self, gate, attn_gates, x0_gates, bigram_gates):
        """Unpacks the post gate; returns the layer-6 skip gate."""
        lane = self._unpack_lanes(gate, 0, POST_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, POST_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, POST_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane + 1 == self._mudd_gate_post_num_coef
        return gate[..., lane:lane + 1]

    def _inject(self, i, x, x0, x0_bigram, x0_gates, bigram_gates):
        """Layer i's gated x0 and n-gram injections into the residual stream (layer 0's n-gram is pre-loop)."""
        if i in X0_INJECT_LAYERS:
            x = x + x0 * x0_gates[i]
        if i in BIGRAM_INJECT_LAYERS and i != 0:
            x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + x0_bigram * bigram_gates[i]
        return x

    def _mlp(self, x_normed, c_fc, c_proj, fp8_args):
        """relu(x @ c_fc.T)^2 @ c_proj; in FP8 (training) when `fp8_args` holds this slot's _mlp_fp8_args,
        else bf16 (validation)."""
        if fp8_args is None:
            return ReLUSqrdMLP(x_normed, c_fc, c_proj)
        return ReLUSqrdMLP(x_normed, c_fc, c_proj, *fp8_args)

    def _attn_weights(self):
        """Per attention layer: (qk_w, v_w, o_w) from its bank slot, cut to the layer's head widths.

        qk_w is [2 * num_heads * qk_dim, dim] (Q heads, then K heads), v_w is [num_heads * v_dim, dim]
        and o_w is [dim, num_heads * v_dim] (nn.Linear layout). Each bank is unbound once, and the
        d_qk = 64 rows of all their slots are cut in one copy: per-layer indexing of a batched view
        would add a select_backward kernel per access (the same thing mlp_bank's unbind avoids).
        """
        H, head_dim, dim = self.num_heads, self.head_dim, self.qk_bank.shape[-1]
        num_slots = len(ATTN_BANK_ORDER)
        qk_heads = self.qk_bank[:self._num_qk_groups].view(num_slots, 2 * H, head_dim, dim)
        qk_full = qk_heads.flatten(1, 2).unbind(0)
        qk_narrow = qk_heads[:NUM_QK64_SLOTS, :, :NARROW_HEAD_DIM].reshape(
            NUM_QK64_SLOTS, 2 * H * NARROW_HEAD_DIM, dim).unbind(0)
        vo = self.vo_bank[:2 * num_slots].unbind(0)
        weights = {}
        for slot, layer in enumerate(ATTN_BANK_ORDER):
            qk_w = qk_full[slot] if layer in WIDE_QK_LAYERS else qk_narrow[slot]
            v_w, o_w = vo[2 * slot], vo[2 * slot + 1]
            v_dim = self.attn_v_dim(layer)
            if v_dim < head_dim:
                # Each head's leading v_dim V rows, and the O columns they feed (record #360).
                v_w = v_w.view(H, head_dim, dim)[:, :v_dim].reshape(H * v_dim, dim)
                o_w = o_w.view(dim, H, head_dim)[:, :, :v_dim].reshape(dim, H * v_dim)
            weights[layer] = (qk_w, v_w, o_w)
        return weights

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor,
                schedule_cfg: ForwardScheduleConfig, ngram_sink: Tensor | None = None,
                value_embed_grad: Tensor | None = None, ret: Tensor | None = None):
        """Per-token loss for one packed varlen batch (B=1, documents separated by `seqlens`).

        bigram_input_seq: [2T] int32 slots in `ngram_cache` of each token's bigram (first T) and
        trigram (last T) row. It carries cache slots, not hashes; the name is the keyword
        evals/hellaswag.py calls with.
        ngram_sink (training only) is NgramTable.grad_sink: the table rows' gradient lands on it.
        value_embed_grad (training only) is the persistent fp16 buffer value_embeds' gradient accumulates
        into (perf/value_embed_pull.py); value_embeds itself never gets a .grad.
        ret: [T, 3] int32 exact-match retrieval rows (track_1_short/retrieval.py), added to the residual
        stream at the input, at layer 7 and before the output head; None (hellaswag) adds nothing.

        Layer topology (11 layers, 0-indexed):
          - attention on ATTN_LAYERS (0, 1, 2, 3, 5, 8, 10); short sliding window except layers 3 and 10
            (long window, partial key offset). Layer 6 adds a gated skip from layer 3 instead of attention;
            layers 4 and 9 run only their MLP; layer 7 only rescales and re-injects x0 / bigram
          - head widths: query/key 128 on the long-window layers 3, 10 and 64 elsewhere; value/output
            64 on layers 1, 8 and 128 elsewhere
          - paired-head attention on layers 0, 2, 5; token value embeddings added to V on 1, 2, 8, 10
          - MUDD gates are computed from x0 (for layers 0-3) and at the start of layer 4 (for layers 4-10);
            the last layer and the post-loop mix use MUDD dense connections over cached layer outputs
        """
        assert input_seq.ndim == 1

        # ---- Schedule and layer topology ----
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        prefix_weight = schedule_cfg.prefix_weight
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        # sliding-window sizes and key shift: the long windows get the partial key offset
        bm_sizes = [ws_long if i in LONG_WINDOW_LAYERS else ws_short for i in range(self.num_layers)]
        key_offset = [i in LONG_WINDOW_LAYERS for i in range(self.num_layers)]

        # FP8 is training-only; validation runs the bf16 path.
        use_fp8 = self.training
        attn_f8_weights = self._attn_fp8_weights() if use_fp8 else None
        mlp_up_f8 = self._mlp_up_f8.unbind(0) if use_fp8 else None

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)
        ve_gates = [None] * self.num_layers
        for layer, gate in zip(VALUE_EMBED_GATE_LAYERS, self.ve_gate_bank.unbind(0)):
            ve_gates[layer] = gate
        attn_gates = [None] * self.num_layers
        xsa_alphas = [None] * self.num_layers
        x0_gates = [None] * self.num_layers
        bigram_gates = [None] * self.num_layers
        attn_weights = self._attn_weights()
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer
        if ret is not None:
            # Exact-match retrieval: the mean embedding of each position's candidates, each weighted by k * its
            # count / the k candidates' count sum and read without a gradient, times its cell's scale, plus its
            # cell's embedding; zero where nothing matched.
            cell, tokens, counts = ret[:, 0].long(), ret[:, 1:] & 0xFFFF, ret[:, 1:] >> 16
            k = (counts > 0).sum(1, keepdim=True)
            weights = (counts * k).float() / counts.sum(1, keepdim=True).clamp_min(1).float()
            mean = sum(self.embed.weight.detach()[tokens[:, i]].float() * weights[:, i, None] for i in range(2)) / k.clamp_min(1)
            ret = ((mean * self.ret_next_scale[cell, None] + self.ret_bucket_embed[cell]) * (cell > 0)[:, None]).bfloat16()
            x = x + self.ret_site_scale_in.type_as(x) * ret

        # Hashed n-gram embedding: each token's bigram row and trigram row, each times its own +-1 sign
        # row, summed. The sign trick compresses several n-grams into a shared row (details in
        # https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy). In plain torch:
        #   rows = ngram_cache[slots] + ngram_sink    # the sink is exact zeros: it only catches the gradient
        #   x0_bigram = rows[:T] * sign_pool[bigram_sign] + rows[T:] * sign_pool[trigram_sign]
        # computed by one opaque kernel so x0_bigram is materialised once (perf/kernels/ngram_embed.py).
        x0_bigram = ngram_embedding(self.ngram_cache, bigram_input_seq, self.ngram_sign_pool, input_seq,
                                    ngram_sink)[None]                                     # (1, seq, ngram_dim)

        # Value embeddings - always computed (not precomputed)
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        # One plane per VALUE_EMBED_LAYERS entry, read at the input tokens. In plain torch:
        #   ve_planes = self.value_embeds.view(len(VALUE_EMBED_LAYERS), self.vocab_size, -1)[:, input_seq]
        # whose backward would build a dense table-sized gradient every step; the lookup's backward adds
        # the rows' gradient into value_embed_grad instead (perf/kernels/value_embed.py).
        ve_planes = value_embed_lookup(self.value_embeds, input_seq, len(VALUE_EMBED_LAYERS), value_embed_grad)
        ve = [None] * self.num_layers
        for layer, plane in zip(VALUE_EMBED_LAYERS, ve_planes):
            ve[layer] = plane

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
        x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + x0_bigram * bigram_gates[0]
        skip_gate_out = None
        post_skip_gate = None

        # cache[k] is the layer-k snapshot used downstream by MUDD.
        # cache[0] = residual stream after bigram injection (input to layer 0).
        cache = {0: x}
        # (norm(cache[7]), its fp8 copy): every attention layer after layer 7 reads this same input,
        # so it is normed, and in fp8 quantized, once and shared. The post-loop mix reads it too.
        late_attn_in = None
        for i in range(self.num_layers):
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]
            mu = None

            if i == POST_GATE_LAYER:
                post_gate = self.forward_mudd_gate(x, id=1, num_coef=self._mudd_gate_post_num_coef)
                post_skip_gate = self.unpack_post_mudd_gate(post_gate, attn_gates, x0_gates, bigram_gates)

            if i == 7 and ret is not None:
                x = x + self.ret_site_scale_mid.type_as(x) * ret[None]

            # process attn. skip on layer 6 @YouJiacheng
            if i == 6:
                assert post_skip_gate is not None
                skip_gate_out = torch.sigmoid(skip_lambda) * post_skip_gate
                x = x + skip_gate_out * cache[3]
            elif i in NO_ATTN_LAYERS:
                # No attention sublayer: keep the residual scaling and the x0 / bigram injections.
                x = self._inject(i, scale(resid_lambdas_attn[i], x), x0, x0_bigram, x0_gates, bigram_gates)
            else:
                ve_gate_head = None  # norm(attn input)[..., :VALUE_EMBED_GATE_CHANNELS], on VALUE_EMBED_GATE_LAYERS
                if late_attn_in is not None:
                    attn_in_normed, attn_x_f8 = late_attn_in
                else:
                    attn_in = cache.get(7, x)
                    if i in VALUE_EMBED_GATE_LAYERS:
                        attn_in_normed, ve_gate_head = rms_norm_with_head(attn_in, VALUE_EMBED_GATE_CHANNELS)
                    else:
                        attn_in_normed = norm(attn_in)
                    attn_x_f8 = None
                    if use_fp8:
                        attn_x_f8 = quantize_dual_layout_fused(attn_in_normed.detach().view(-1, attn_in_normed.size(-1)),
                                                               self._attn_x_scale, fmt=torch.float8_e4m3fn)
                    if 7 in cache:
                        late_attn_in = (attn_in_normed, attn_x_f8)
                qkv_fp8 = None
                if use_fp8:
                    qkv_fp8 = (*attn_f8_weights[i], self._attn_x_scale, self._attn_grad_scale, *attn_x_f8)
                B, T = attn_in_normed.size(0), attn_in_normed.size(1)

                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=LAST_LAYER_MUDD_COEFS)
                    v_mudd = mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x
                    v_mudd = v_mudd.view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        self.num_heads // 2, dim=-1
                    ).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    # gate pattern g(x[:6] + ve[:6]) by @photomz
                    gate_in = torch.cat([ve_gate_head, ve[i][None, ..., :VALUE_EMBED_GATE_CHANNELS]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None

                if i in WIDE_QK_LAYERS:
                    yarn = self.yarn_wide
                elif i in PAIRED_HEAD_LAYERS:
                    yarn = self.yarn_paired_head
                else:
                    yarn = self.yarn
                attn_args = AttnArgs(
                    sa_lambdas=sa_lambdas[i],
                    seqlens=seqlens,
                    bm_size=bm_sizes[i],
                    yarn=yarn,
                    key_offset=key_offset[i],
                    attn_gate_w=attn_gates[i] if i in ATTN_GATE_LAYERS else None,
                    aux_v=aux_v,
                    xsa_alpha=xsa_alphas[i],
                    train_max_seq_len=train_max_seq_len,
                    # The post-lambda rides the output projection, except on the MUDD layer (mu[9] is per-token).
                    o_gain=post_lambdas_attn[i] if mu is None else None,
                )
                qk_w, v_w, o_w = attn_weights[i]
                attn_out = self.attn[str(i)](attn_in_normed, attn_args, qk_w, v_w, o_w, qkv_fp8)

                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0]
                    x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + mu[11] * x0_bigram
                else:
                    x = scale(resid_lambdas_attn[i], x) + attn_out  # attn_out carries post_lambdas_attn[i]
                    x = self._inject(i, x, x0, x0_bigram, x0_gates, bigram_gates)

            # process mlp
            if i in NO_MLP_LAYERS:
                x = scale(resid_lambdas_mlp[i], x)
                if i in CACHE_LAYERS:
                    cache[i] = x
                continue
            mlp_in = norm(x)
            # fp8 only: the post-lambda is folded into the down-projection scale, so the MLP output
            # already carries it. Not on the MUDD layer, whose post-lambda mu[13] is per-token.
            fold_p = post_lambdas_mlp[i] if use_fp8 and mu is None else None
            if use_fp8:
                # One quantize of the input, shared with the parallel MLP.
                x_f8 = quantize_dual_layout_fused(mlp_in.detach().view(-1, mlp_in.size(-1)), self._mlp_x_scale, fmt=torch.float8_e4m3fn)
                fp8_args = self._mlp_fp8_args(i, mlp_up_f8[i], x_f8, fold_p)
            else:
                fp8_args = None
            mlp_out = self._mlp(mlp_in, c_fc, c_proj, fp8_args)
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            elif fold_p is not None:
                x = scale(resid_lambdas_mlp[i], x) + mlp_out
            else:
                x = scale_add(resid_lambdas_mlp[i], x, post_lambdas_mlp[i], mlp_out)
            if i == PARALLEL_MLP_LAYER:
                # Same input and the same post-lambda as the layer's own MLP.
                k = PARALLEL_MLP_SLOT
                parallel_args = self._mlp_fp8_args(k, mlp_up_f8[k], x_f8, fold_p) if use_fp8 else None
                parallel_out = self._mlp(mlp_in, mlp_fcs[k], mlp_projs[k], parallel_args)
                x = x + (parallel_out if fold_p is not None else scale(post_lambdas_mlp[i], parallel_out))

            if i in CACHE_LAYERS:
                cache[i] = x

        # Post-loop MUDD: mix 10 earlier activations back into the residual, each with a per-token
        # coefficient plus a per-channel-group delta. The last two are the final layer's MLP input and
        # its attention input, norm(cache[7]).
        assert late_attn_in is not None, "norm(cache[7]) is not bound at loop exit"
        sources = [
            cache[0], cache[7], cache[9], ve[1][None].to(dtype=x.dtype), cache[3],
            ve[2][None].to(dtype=x.dtype), ve[10][None].to(dtype=x.dtype), ve[8][None].to(dtype=x.dtype),
            mlp_in, late_attn_in[0],
        ]
        mu, deltas = self.forward_mudd_grouped(x, id=1, num_coef=len(sources))
        grouped = lambda t: t.unflatten(-1, (MUDD_GROUPS, -1))  # (B, T, D) -> (B, T, G, D/G)
        mixed = grouped(x)
        for k, src in enumerate(sources):
            mixed = mixed + (mu[k] + deltas[..., k, :]).unsqueeze(-1) * grouped(src)
        x = mixed.flatten(-2)
        if ret is not None:
            x = x + self.ret_site_scale_out.type_as(x) * ret[None]

        return self._loss(norm(x), input_seq, target_seq, mtp_weights, prefix_weight, schedule_cfg.sampled_loss)

    def _loss(self, x, input_seq, target_seq, mtp_weights, prefix_weight, sampled_loss):
        """Per-token loss from the final normed hidden state.

        Training: fused softcapped CE over next-token + multi-token + prefix-token targets, normalized
        over the vocabulary, or over `sampled_loss`'s candidate set early in training (sampled_softmax.py).
        Validation: plain next-token CE over the full vocabulary, with non-canonical tokens masked out
        (see canonical_mask.py); `sampled_loss` is ignored.
        """
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training and sampled_loss is not None:
            # Targets and prefix targets arrive as positions in the candidate set, built on the host.
            n = target_seq.size(0)
            loss_per_token = SampledSoftcappedCrossEntropy.apply(
                x.view(-1, x.size(-1)), mtp_weights, sampled_loss.target_pos[:n], sampled_loss.prefix_pos[:n],
                prefix_weight, self.lm_head.weight, sampled_loss.rows, sampled_loss.rows_t, sampled_loss.vocab_pos,
                self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s,
            )
        elif self.training:
            prefix_target_seq = self.prefix_table[target_seq]
            loss_per_token = FusedSoftcappedCrossEntropy.apply(
                x.view(-1, x.size(-1)), target_seq, mtp_weights, prefix_target_seq, prefix_weight,
                self.lm_head.weight, self._lm_head_f8_col, self._lm_head_f8_row,
                self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s,
            )
        else:
            # Every step is row-local, so slabs of EVAL_CE_SLAB_ROWS rows give the same per-token losses
            # as one pass; only the [rows, vocab] logit block shrinks.
            x = x.view(-1, x.size(-1))
            shifts = torch.arange(8, dtype=torch.uint8, device=x.device)
            slab_losses = []
            for lo in range(0, x.size(0), EVAL_CE_SLAB_ROWS):
                hi = min(lo + EVAL_CE_SLAB_ROWS, x.size(0))
                # The softcap runs in bf16 and upcasts after, as record #360.
                logits = (23 * torch.sigmoid((self.lm_head(x[lo:hi]) + 5) / 7.5)).float()
                # Drop the tokens the tokenizer would never emit after input_seq. -60 is well below
                # the 0..23 the softcap leaves, so a dropped token contributes nothing to the softmax.
                dropped = (self.canon_mask[input_seq[lo:hi], :, None] >> shifts & 1).view(logits.shape).bool()
                logits = logits.masked_fill(dropped, -60.0)
                slab_losses.append(F.cross_entropy(logits, target_seq[lo:hi], reduction="none"))
            loss_per_token = torch.cat(slab_losses)
        return loss_per_token

    # -------------------------------------------------------------------------
    # Setup and run-level hooks main() calls on the uncompiled model.

    def cast_matrix_weights_bf16(self):
        """Matrix weights train in bf16 (lm_head and value_embeds are created bf16); the scalar and
        lambda parameters stay fp32. Call once, before the optimizer is built."""
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                m.weight.data = m.weight.data.bfloat16()
        for param in (self.ve_gate_bank, self.qk_bank, self.vo_bank, self.mlp_bank,
                      self.mudd_w1, self.mudd_w2, self.mudd_w2g, self.mudd_b2,
                      self.mudd_gate_w1, self.mudd_gate_w2, self.mudd_gate_b2):
            param.data = param.data.bfloat16()

    @property
    def yarns(self) -> tuple[Yarn, ...]:
        return (self.yarn, self.yarn_paired_head, self.yarn_wide)

    def limit_yarn_rebuild(self, rows: int):
        """From now on a window change rebuilds only the first `rows` rotary rows (the longest training
        sequence); complete_yarn_tables() fills in the rest before a validation (model/attention.py Yarn)."""
        for yarn in self.yarns:
            yarn.rebuild_rows = rows

    def complete_yarn_tables(self):
        for yarn in self.yarns:
            yarn.ensure_full()

    @property
    def lm_head_f8_col(self) -> Tensor:
        """The column-major fp8 lm_head copy (refreshed with the MLP caches): the sampled-softmax row source."""
        return self._lm_head_f8_col

    # -------------------------------------------------------------------------
    # FP8 weight caches and scales: refreshed after every optimizer step, right before the next
    # forward (perf/deferred_gathers.py, where the refresh waits for the bank gathers it reads),
    # and unpacked per layer for the forward pass.

    def quantize_attn_fp8(self):
        """Refresh each width group's cached FP8 [Q; K; V] weights after the optimizer update.

        A group's layers are consecutive bank slots (ATTN_BANK_ORDER), so its rows are one strided
        view of each bank, cut to the group's widths.
        """
        H, head_dim, dim = self.num_heads, self.head_dim, self.qk_bank.shape[-1]
        num_slots = len(ATTN_BANK_ORDER)
        with torch.no_grad():
            qk_heads = self.qk_bank[:self._num_qk_groups].view(num_slots, 2 * H, head_dim, dim)
            v_heads = self.vo_bank[:2 * num_slots].view(num_slots, 2, H, head_dim, dim)[:, 0]
            start = 0
            for group, layers in ATTN_WIDTH_GROUPS.items():
                slots = slice(start, start + len(layers))
                start += len(layers)
                qk_dim, v_dim = self.attn_qk_dim(layers[0]), self.attn_v_dim(layers[0])
                qk = qk_heads[slots, :, :qk_dim].reshape(len(layers), 2 * H * qk_dim, dim)
                v = v_heads[slots, :, :v_dim].reshape(len(layers), H * v_dim, dim)
                self.attn_fp8[group].refresh(qk, v)

    @staticmethod
    def _refresh_delayed_scale(amax_buf, scale_buf, gain, floor):
        """Set each slot's scale from the amax the kernels recorded since the last call, then clear the amax.

        A slot whose amax is still zero (no training step since init or the reset) keeps its previous
        scale: flooring it instead would turn step 0's post quantize into a binary mask.
        """
        a = torch.nan_to_num(amax_buf.amax(dim=1), nan=0.0, posinf=0.0, neginf=0.0).clamp(max=1e12)
        new = (a * gain).clamp(min=floor)
        scale_buf.copy_(torch.where(a > 0, new, scale_buf))
        amax_buf.zero_()

    def quantize_mlp_fp8(self, refresh_lm: bool):
        """Refresh the FP8 MLP weight caches and scales, and (if `refresh_lm`) the fp8 lm_head copies.

        lm_head only changes on Adam steps, so the caller passes refresh_lm=False after the others.
        """
        with torch.no_grad():
            self._quantize_mlp_weights_and_scales()
            if refresh_lm:
                quantize_lm_head_dual(self.lm_head.weight, self._lm_head_f8_row, self._lm_head_f8_col, self._lm_head_inv_w_s)

    def _quantize_mlp_weights_and_scales(self):
        e4m3_max = torch.finfo(torch.float8_e4m3fn).max
        e5m2_max = torch.finfo(torch.float8_e5m2).max
        up_weights, down_weights = self.mlp_bank[:, 0], self.mlp_bank[:, 1]
        exact = self.mlp_quantize_calls < FP8_EXACT_SCALE_CALLS
        self.mlp_quantize_calls += 1

        # Weight caches. The exact scale is computed in the weights' bf16, as record #360 does.
        if exact:
            self._mlp_up_scales[:] = (up_weights.reshape(NUM_MLP_SLOTS, -1).abs().amax(dim=1).clamp(min=1e-12) / e4m3_max).float()
            self._mlp_down_scales[:] = (down_weights.reshape(NUM_MLP_SLOTS, -1).abs().amax(dim=1).clamp(min=1e-12) / e4m3_max).float()
        quantize_mlp_weights_dual(up_weights, self._mlp_up_scales, self._mlp_up_partial_amax,
                                  row=self._mlp_up_f8, col_t=self._mlp_up_f8_col.transpose(1, 2), update_scales=not exact)
        quantize_mlp_weights_dual(down_weights, self._mlp_down_scales, self._mlp_down_partial_amax,
                                  row=self._mlp_down_f8_row, col_t=self._mlp_down_f8.transpose(1, 2), update_scales=not exact)

        # Scales derived from them. In-place, so every consumer keeps reading the same tensors.
        torch.mul(self._mlp_up_scales, FP8_MLP_X_SCALE, out=self._mlp_up_dequant)
        torch.mul(self._mlp_down_scales, FP8_GRAD_SCALE, out=self._mlp_dq_bwd)
        self._refresh_delayed_scale(self._mlp_post_amax, self._mlp_post_scales, FP8_POST_HEADROOM / e4m3_max, 1e-6)
        self._refresh_delayed_scale(self._mlp_dpre_amax, self._mlp_dpre_scales, FP8_DPRE_HEADROOM / e5m2_max, 1e-12)

        # Post-lambda folded into the down-projection scale: slot k folds layer k's post-lambda
        # (bf16-rounded, as the forward uses it); the
        # parallel slot folds its layer's. The last layer's slot never folds (its post-lambda is
        # mu[13]), so its entry stays 1.
        post_lambdas = self.post_lambdas[:, 1].bfloat16().float()
        self._mlp_fold_p[:NUM_LAYERS - 1].copy_(post_lambdas[:NUM_LAYERS - 1])
        self._mlp_fold_p[PARALLEL_MLP_SLOT].copy_(post_lambdas[PARALLEL_MLP_LAYER])
        torch.mul(self._mlp_down_scales, self._mlp_fold_p, out=self._mlp_down_scales_folded)
        torch.mul(self._mlp_dq_bwd, self._mlp_fold_p, out=self._mlp_dq_bwd_folded)

    def rearm_fp8_bootstrap(self):
        """After the post-warmup weight reset: restart the exact-scale calls and drop the warmup's delayed amaxes."""
        with torch.no_grad():
            self.mlp_quantize_calls = 0
            for amax in (self._mlp_up_partial_amax, self._mlp_down_partial_amax, self._mlp_post_amax, self._mlp_dpre_amax):
                amax.zero_()
            self._mlp_dpre_scales.fill_(FP8_GRAD_SCALE)
            self._mlp_post_scales.fill_(1.0)

    def _attn_fp8_weights(self):
        """Per attention layer: (qkv_f8, qkv_f8_t, qkv_scale) from its width group's cache, for PackedFP8QKV."""
        weights = {}
        for group, layers in ATTN_WIDTH_GROUPS.items():
            cache = self.attn_fp8[group]
            for j, layer in enumerate(layers):
                weights[layer] = (cache.row[j], cache.col[j], cache.scales[j:j+1])
        return weights

    def _mlp_fp8_args(self, slot, up_f8, x_f8, fold_p):
        """The FP8 arguments of FusedLinearReLUSquareFunction for MLP bank `slot`, after (x, c_fc, c_proj).

        `x_f8` is the (row, transposed) e4m3 input. With `fold_p` (the post-lambda folded into the
        down-projection scale), the down projection's forward and dgrad dequant scales are the folded
        ones, and fold_p rides last for its gradient.
        """
        x_row, x_t = x_f8
        if fold_p is not None:
            down_scales, dq_bwd = self._mlp_down_scales_folded, self._mlp_dq_bwd_folded
        else:
            down_scales, dq_bwd = self._mlp_down_scales, self._mlp_dq_bwd
        return (up_f8, self._mlp_up_dequant[slot:slot+1], x_row, self._mlp_down_f8[slot], down_scales[slot],
                self._mlp_post_scales[slot], self._mlp_post_amax[slot], self._mlp_down_f8_row[slot], dq_bwd[slot], x_t,
                self._mlp_x_scale, self._mlp_up_f8_col[slot], self._mlp_up_scales[slot], self._mlp_grad_scale,
                self._mlp_dpre_scales[slot], self._mlp_dpre_amax[slot]) + ((fold_p,) if fold_p is not None else ())

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from switch_bank.utils import _sanitize, _safe_softmax


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, hdim, dim)).bfloat16())
        self.qkvo_w.detach()[3].zero_()  # out zero init
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, lambdas: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1))\
                   .view(B, T, 3 * self.num_heads, self.head_dim)\
                   .chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
        q = _sanitize(q)
        k = _sanitize(k)
        v = _sanitize(v)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                           block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = _sanitize(y)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return _sanitize(y)


class SharedFFNBank(nn.Module):
    def __init__(self, d: int, h: int, E: int, L: int, flags_dim: int,
                 lb_coeff: float = 1e-3, ent_coeff: float = 0.0, expert_entropy_coeff: float = 0.0, k: int = 1,
                 use_adapters: bool = False,
                 ema_alpha_fwd: float = 0.80, ema_alpha_rev: float | None = None,
                 use_forward_ema: bool = True, use_reverse_ema: bool = False,
                 ema_block_size_fwd: int = 128, ema_block_size_rev: int = 128,
                 ema_window_size_fwd: int = -1, ema_window_size_rev: int = 128,
                 ema_layer_stride: int = 1,
                 extra_wandb_logging: bool = True):
        super().__init__()
        self.d, self.h, self.E, self.L = d, h, E, L
        self.flags_dim = flags_dim
        self.lb_coeff = lb_coeff
        self.ent_coeff = ent_coeff
        self.expert_entropy_coeff = expert_entropy_coeff
        self.k = int(k)
        alpha_fwd_val = float(ema_alpha_fwd)
        alpha_rev_val = float(ema_alpha_rev if ema_alpha_rev is not None else ema_alpha_fwd)
        self.ema_alpha_min_fwd = alpha_fwd_val
        self.ema_alpha_max_fwd = alpha_fwd_val
        self.ema_alpha_min_rev = alpha_rev_val
        self.ema_alpha_max_rev = alpha_rev_val
        self.use_adapters = bool(use_adapters)
        self.use_forward_ema = bool(use_forward_ema)
        self.use_reverse_ema = bool(use_reverse_ema)
        self.enable_extra_wandb_logging = bool(extra_wandb_logging)
        self.ema_block_size_fwd = int(ema_block_size_fwd)
        self.ema_block_size_rev = int(ema_block_size_rev)
        self.ema_window_size_fwd = int(ema_window_size_fwd)
        self.ema_window_size_rev = int(ema_window_size_rev)
        self.ema_layer_stride = max(int(ema_layer_stride), 1)
        assert self.ema_layer_stride <= L, "ema_layer_stride must be <= number of layers"
        self._ema_cache_fwd: dict[int, Tensor] | None = None
        self._ema_cache_rev: dict[int, Tensor] | None = None
        if self.use_adapters:
            self.adapter_scale = nn.Parameter(torch.ones(L, E, d).bfloat16())
            self.adapter_bias = nn.Parameter(torch.zeros(L, E, d).bfloat16())
            self.register_buffer("adapter_initialized", torch.zeros(L, E, dtype=torch.bool), persistent=False)
        else:
            self.adapter_scale = None
            self.adapter_bias = None
            self.adapter_initialized = None
        self.register_buffer("pruned_experts", torch.zeros(E, dtype=torch.bool), persistent=False)
        self.W1 = nn.ParameterList([nn.Parameter(init_linear(torch.empty(h, d)).bfloat16()) for _ in range(E)])
        self.W2 = nn.ParameterList([nn.Parameter(torch.zeros(d, h).bfloat16()) for _ in range(E)])
        for w in list(self.W1) + list(self.W2):
            w.wd_mul = 2.0
        feat_multiplier = 1 + int(self.use_forward_ema) + int(self.use_reverse_ema)
        in_dim = feat_multiplier * d + flags_dim
        self.router_w = nn.ParameterList([nn.Parameter(init_linear(torch.empty(E, in_dim)).bfloat16()) for _ in range(L)])
        self.router_b = nn.ParameterList([nn.Parameter(torch.zeros(E).bfloat16()) for _ in range(L)])
        if self.use_forward_ema:
            alpha_fwd = torch.full((L,), alpha_fwd_val, dtype=torch.bfloat16)
            self.register_buffer("ema_alpha", alpha_fwd)
        else:
            self.ema_alpha = None
        if self.use_reverse_ema:
            alpha_rev = torch.full((L,), alpha_rev_val, dtype=torch.bfloat16)
            self.register_buffer("ema_alpha_rev", alpha_rev)
        else:
            self.ema_alpha_rev = None
        self._router_metrics_buffer: list[dict[str, float] | None] | None = None

    @torch.no_grad()
    @torch._dynamo.disable
    def maybe_init_adapters(self, active_mask: torch.Tensor | None):
        if not (self.use_adapters and self.adapter_initialized is not None):
            return
        if active_mask is None:
            init_mask = torch.ones(self.E, dtype=torch.bool, device=self.adapter_initialized.device)
        else:
            init_mask = active_mask.to(device=self.adapter_initialized.device, dtype=torch.bool)
        if self.pruned_experts.any():
            init_mask = init_mask & (~self.pruned_experts.to(device=init_mask.device))
        if not init_mask.any():
            return
        for layer_idx in range(self.L):
            init_flags = self.adapter_initialized[layer_idx]
            to_init = init_mask & (~init_flags)
            if not to_init.any():
                continue
            src_mask = init_mask & init_flags
            if src_mask.any():
                scale_mean = self.adapter_scale[layer_idx, src_mask].mean(dim=0, keepdim=True)
                bias_mean = self.adapter_bias[layer_idx, src_mask].mean(dim=0, keepdim=True)
            else:
                scale_mean = None
                bias_mean = None
            if scale_mean is not None:
                self.adapter_scale[layer_idx, to_init] = scale_mean
                self.adapter_bias[layer_idx, to_init] = bias_mean
            self.adapter_initialized[layer_idx, to_init] = True

    @staticmethod
    def _ema_blockwise(x: Tensor, alpha: Tensor, block_size: int = 128) -> Tensor:
        B, T, D = x.shape
        assert B == 1
        a = _sanitize(alpha.float(), value=0.8).clamp(1e-4, 0.9999)
        one_minus = (1.0 - a)
        assert T % block_size == 0, "Sequence length must be a multiple of 128."
        nb = T // block_size
        x_blk = x.view(1, nb, block_size, D).float()
        ar = torch.arange(block_size, device=x.device, dtype=torch.float32)
        pow_a = a.pow(ar)
        pow_a_p1 = a.pow(ar + 1.0)
        pow_a_inv = a.pow(-ar)
        y = torch.empty_like(x_blk)
        carry = torch.zeros(1, 1, D, device=x.device, dtype=torch.float32)
        for b in range(nb):
            xb = x_blk[:, b]
            u = xb * pow_a_inv.view(1, -1, 1)
            prefix = torch.cumsum(u, dim=1)
            yb = pow_a_p1.view(1, -1, 1) * carry + (one_minus * (pow_a.view(1, -1, 1) * prefix))
            y[:, b] = yb
            carry = yb[:, -1:, :]
        out = y.view(1, T, D).to(dtype=x.dtype)
        return _sanitize(out)

    @staticmethod
    @torch._dynamo.disable
    def _ema_reverse_since_doc_start(x: Tensor, alpha: Tensor, doc_starts: torch.Tensor, window: int = 128, block_size: int = 128) -> Tensor:
        B, T, D = x.shape
        assert B == 1
        doc_starts = doc_starts.to(dtype=torch.bool, device=x.device)
        if not bool(doc_starts[0]):
            doc_starts[0] = True
        doc_bounds = torch.nonzero(doc_starts, as_tuple=True)[0]
        if doc_bounds.numel() == 0 or doc_bounds[0].item() != 0:
            doc_bounds = torch.cat([doc_bounds.new_tensor([0]), doc_bounds])
        doc_bounds = torch.cat([doc_bounds, doc_bounds.new_tensor([T])])
        out = torch.empty_like(x, dtype=torch.float32)
        a = _sanitize(alpha.float(), value=0.8).clamp(1e-4, 0.9999)
        for idx in range(doc_bounds.numel() - 1):
            start = int(doc_bounds[idx].item())
            end = int(doc_bounds[idx + 1].item())
            if end <= start:
                continue
            seg = x[:, start:end, :]
            length = end - start
            limit = min(window, length)
            if limit > 0:
                head = seg[:, :limit, :]
                rev = torch.flip(head, dims=(1,))
                pad = (-limit) % block_size
                if pad:
                    zero_pad = torch.zeros(1, pad, D, device=x.device, dtype=rev.dtype)
                    rev = torch.cat([rev, zero_pad], dim=1)
                ema = SharedFFNBank._ema_blockwise(rev, a, block_size=block_size)
                ema = ema[:, :limit]
                ema = torch.flip(ema, dims=(1,))
                out[:, start:start + limit] = ema
                if length > limit:
                    out[:, start + limit:end] = ema[:, -1:].expand(1, length - limit, D)
            else:
                out[:, start:end] = 0
        return _sanitize(out.to(dtype=x.dtype))

    def forward(self, x_norm: Tensor, layer_idx: int, flags: Tensor, temperature: float,
                logit_cap: float | None, freeze_ema_alpha: bool, use_gumbel: bool,
                lb_multiplier: float = 1.0, active_mask: Tensor | None = None,
                freeze_router_params: bool = False, freeze_adapter_params: bool = False,
                ema_limits_fwd: tuple[float, float] | None = None,
                ema_limits_rev: tuple[float, float] | None = None,
                freeze_ema_alpha_rev: bool = False) -> tuple[Tensor, Tensor]:
        assert x_norm.size(0) == 1
        deterministic_topk = False
        base_mask = torch.ones(self.E, dtype=torch.bool, device=x_norm.device)
        if active_mask is not None:
            base_mask &= active_mask.to(device=x_norm.device, dtype=torch.bool)
        if self.pruned_experts.any():
            base_mask &= (~self.pruned_experts.to(device=x_norm.device))
        if not base_mask.any():
            base_mask = torch.ones(self.E, dtype=torch.bool, device=x_norm.device)
        regular_active_count = int(base_mask.sum().item())
        if (regular_active_count == 1):
            deterministic_topk = True
            active_idx = int(base_mask.nonzero(as_tuple=True)[0].item())
        features: list[Tensor] = [x_norm]
        feat_names: list[str] = ["tok"]
        feat_sizes: list[int] = [self.d]
        alpha_use_rev = None
        alpha_use = None
        group_stride = max(self.ema_layer_stride, 1)
        base_layer = (layer_idx // group_stride) * group_stride
        if (not deterministic_topk) and self.use_forward_ema and self.ema_alpha is not None:
            min_alpha, max_alpha = (ema_limits_fwd if ema_limits_fwd is not None else (self.ema_alpha_min_fwd, self.ema_alpha_max_fwd))
            alpha_raw = self.ema_alpha[base_layer].float()
            alpha_clip = _sanitize(alpha_raw).clamp(min_alpha, max_alpha)
            alpha_use = alpha_clip.detach() if freeze_ema_alpha else alpha_clip
            cache_fwd = self._ema_cache_fwd if isinstance(self._ema_cache_fwd, dict) else None
            if cache_fwd is not None and base_layer in cache_fwd:
                ema_feat = cache_fwd[base_layer]
            else:
                if 0 < self.ema_window_size_fwd < x_norm.size(1):
                    head_len = min(self.ema_window_size_fwd, x_norm.size(1))
                    pad = (-head_len) % self.ema_block_size_fwd
                    head = x_norm[:, :head_len]
                    if pad:
                        head = torch.cat([head, torch.zeros(1, pad, self.d, device=head.device, dtype=head.dtype)], dim=1)
                    ema_head = self._ema_blockwise(head, alpha_use, block_size=self.ema_block_size_fwd)[:, :head_len]
                    ema_feat = x_norm.new_empty(x_norm.shape)
                    ema_feat[:, :head_len] = ema_head
                    ema_feat[:, head_len:] = ema_head[:, -1:].expand(1, x_norm.size(1) - head_len, self.d)
                else:
                    ema_feat = self._ema_blockwise(x_norm, alpha_use, block_size=self.ema_block_size_fwd)
                if cache_fwd is not None:
                    cache_fwd[base_layer] = ema_feat
            features.append(ema_feat)
            feat_names.append("ema_fwd")
            feat_sizes.append(self.d)
        if (not deterministic_topk) and self.use_reverse_ema and self.ema_alpha_rev is not None:
            min_alpha_rev, max_alpha_rev = (ema_limits_rev if ema_limits_rev is not None else (self.ema_alpha_min_rev, self.ema_alpha_max_rev))
            alpha_raw_rev = self.ema_alpha_rev[base_layer].float()
            alpha_clip_rev = _sanitize(alpha_raw_rev).clamp(min_alpha_rev, max_alpha_rev)
            alpha_use_rev = alpha_clip_rev.detach() if freeze_ema_alpha_rev else alpha_clip_rev
            cache_rev = self._ema_cache_rev if isinstance(self._ema_cache_rev, dict) else None
            if cache_rev is not None and base_layer in cache_rev:
                ema_rev = cache_rev[base_layer]
            else:
                doc_starts = (flags[0, :, 1].float() > 0.5)
                ema_rev = self._ema_reverse_since_doc_start(
                    x_norm, alpha_use_rev, doc_starts,
                    window=self.ema_window_size_rev, block_size=self.ema_block_size_rev,
                )
                if cache_rev is not None:
                    cache_rev[base_layer] = ema_rev
            features.append(ema_rev)
            feat_names.append("ema_rev")
            feat_sizes.append(self.d)
        feat_names.append("flags")
        feat_sizes.append(self.flags_dim)
        if deterministic_topk:
            effective_mask = base_mask
            probs = torch.zeros((1, x_norm.size(1), self.E), device=x_norm.device, dtype=x_norm.dtype)
            probs[..., active_idx] = 1.0
            topk_idx = torch.full((1, x_norm.size(1), 1), active_idx, device=x_norm.device, dtype=torch.int64)
            topk_prob = torch.ones_like(topk_idx, dtype=x_norm.dtype)
            max_logit = float("nan")
            imp = torch.zeros(self.E, device=x_norm.device, dtype=x_norm.dtype)
            imp[active_idx] = 1.0
            load = imp.clone()
            k = 1
        else:
            rin = _sanitize(torch.cat([*features, flags], dim=-1))
            logits = F.linear(rin, self.router_w[layer_idx], self.router_b[layer_idx]).float()
            if use_gumbel:
                u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits = logits + g
            logits = logits / max(temperature, 1e-6)
            if logit_cap is not None and logit_cap > 0:
                logits = logits.clamp(-logit_cap, logit_cap)
            logits = _sanitize(logits)
            if freeze_router_params:
                logits = logits.detach()
            effective_mask = None
            if base_mask is not None:
                effective_mask = base_mask.to(device=logits.device, dtype=torch.bool)
            if effective_mask is not None:
                if not effective_mask.any():
                    fallback = (~self.pruned_experts).to(device=logits.device)
                    if not fallback.any():
                        fallback = torch.ones(self.E, dtype=torch.bool, device=logits.device)
                    effective_mask = fallback
                logits = logits.masked_fill(~effective_mask.view(1, 1, -1), float("-inf"))
            probs = _safe_softmax(logits, dim=-1)
            max_logit = logits.max().item() if logits.numel() > 0 else float("nan")

            active_count = int(effective_mask.sum().item()) if effective_mask is not None else self.E
            k = max(1, min(self.k, active_count))
            topk_prob, topk_idx = probs.topk(k, dim=-1)
            if torch.isnan(topk_prob).any():
                probs = _safe_softmax(torch.zeros_like(logits), dim=-1)
                topk_prob, topk_idx = probs.topk(k, dim=-1)

            imp = _sanitize(probs.mean(dim=(0, 1)))
            top1 = topk_idx[..., 0]
            one_hot = F.one_hot(top1.view(-1), num_classes=self.E).float()
            load = _sanitize(one_hot.mean(dim=0))

        def cv2(v: Tensor):
            m = v.mean()
            return v.var(unbiased=False) / (m * m + 1e-6)

        imp_f = imp.float()
        load_f = load.float()
        single_active = (regular_active_count <= 1)

        imp_entropy = (-(imp_f + 1e-6).log().mul(imp_f)).sum()
        load_entropy = (-(load_f + 1e-6).log().mul(load_f)).sum()

        lb_term = (self.lb_coeff * lb_multiplier) * (cv2(imp_f) + cv2(load_f))
        #entropy_term = -self.ent_coeff * (load_entropy + imp_entropy)
        entropy_term = 1/load - 1/(imp+load)

        expert_entropy_term = lb_term.new_zeros(())
        if self.expert_entropy_coeff > 0:
            expert_entropy_term = self.expert_entropy_coeff * load_entropy

        router_aux = lb_term + entropy_term + expert_entropy_term

        if single_active:
            router_aux = router_aux.new_zeros(())

        other_aux = router_aux.new_zeros(())
        aux = router_aux + other_aux
        if freeze_router_params:
            if freeze_adapter_params:
                aux = x_norm.new_zeros(())
            else:
                aux = other_aux

        y = torch.zeros_like(x_norm)
        pruned_flags = self.pruned_experts.to(dtype=torch.bool, device=x_norm.device)
        for e in range(self.E):
            if pruned_flags[e]:
                continue
            union_mask = torch.zeros(topk_idx.size(1), dtype=torch.bool, device=x_norm.device)
            per_rank_masks = []
            for r in range(k):
                mr = (topk_idx[0, :, r] == e)
                per_rank_masks.append(mr)
                union_mask |= mr
            if not union_mask.any():
                continue
            x_e = x_norm[:, union_mask]
            if self.use_adapters:
                scale = self.adapter_scale[layer_idx, e]
                bias = self.adapter_bias[layer_idx, e]
                if freeze_adapter_params:
                    scale = scale.detach()
                    bias = bias.detach()
                x_e = x_e * scale.to(x_e.dtype) + bias.to(x_e.dtype)
            h1 = F.linear(x_e, self.W1[e])
            h1 = F.relu(h1).square()
            out_e = F.linear(h1, self.W2[e])
            idx_union = union_mask.nonzero(as_tuple=True)[0]
            accum = torch.zeros_like(out_e)
            for r in range(k):
                mr = per_rank_masks[r]
                if not mr.any():
                    continue
                rel = torch.nonzero(mr[union_mask], as_tuple=True)[0]
                scales = topk_prob[0, mr, r].unsqueeze(-1)
                accum[:, rel] += scales * out_e[:, rel]
            y[:, idx_union] += accum

        stats: dict[str, Any] = dict(
            imp_cv2=float(cv2(_sanitize(imp_f)).item()),
            load_cv2=float(cv2(_sanitize(load_f)).item()),
            usage_frac=float(((load_f > 0).float().mean()).item()),
            topk_prob_mean=float(_sanitize(topk_prob).mean().item()),
            imp_entropy=float(imp_entropy.item()),
            load_entropy=float(load_entropy.item()),
        )
        if self.enable_extra_wandb_logging:
            if self.use_forward_ema and self.ema_alpha is not None and alpha_use is not None:
                stats["ema_alpha_forward"] = float(alpha_use.float().item())
            if self.use_reverse_ema and self.ema_alpha_rev is not None and alpha_use_rev is not None:
                stats["ema_alpha_reverse"] = float(alpha_use_rev.float().item())
            start_idx = 0
            for name, size in zip(feat_names, feat_sizes):
                end_idx = start_idx + size
                if end_idx > self.router_w[layer_idx].size(1):
                    break
                w_slice = self.router_w[layer_idx][:, start_idx:end_idx].float()
                stats[f"feat_w_{name}"] = float(w_slice.abs().mean().item())
                start_idx = end_idx
        stats["load_vector"] = _sanitize(load_f).detach().float().cpu()
        self._record_router_metrics(layer_idx, stats, max_logit)

        return y, aux

    @torch.no_grad()
    def compile_warm_all_experts(self, d: int, T_warm: int = 128):
        x = torch.randn(1, T_warm, d, device=self.W1[0].device, dtype=torch.bfloat16)
        for e in range(self.E):
            h1 = F.linear(x, self.W1[e]); h1 = F.relu(h1).square(); _ = F.linear(h1, self.W2[e])

    def begin_router_metrics(self):
        self._router_metrics_buffer = [None] * self.L
        self._ema_cache_fwd = {}
        self._ema_cache_rev = {}

    def _record_router_metrics(self, layer_idx: int, stats: dict[str, float], max_logit: float):
        if self._router_metrics_buffer is not None:
            stats = dict(stats)
            stats["max_logit"] = max_logit
            self._router_metrics_buffer[layer_idx] = stats

    def pop_router_metrics(self):
        metrics = self._router_metrics_buffer
        self._router_metrics_buffer = None
        return metrics or []

    @torch.no_grad()
    def prune_inactive_experts(self, activity: torch.Tensor, threshold: float) -> list[int]:
        if activity is None:
            return []
        if activity.numel() != self.E:
            return []
        device = self.pruned_experts.device
        activity = activity.to(device=device, dtype=torch.float32)
        available = (~self.pruned_experts)
        if not available.any():
            return []
        keep = (activity >= threshold) & available
        if not keep.any():
            masked_activity = activity.clone()
            masked_activity[~available] = float("-inf")
            best_idx = int(torch.argmax(masked_activity).item())
            keep[best_idx] = True
        pruned_mask = available & (~keep)
        new_pruned = pruned_mask.nonzero(as_tuple=True)[0].tolist()
        if not new_pruned:
            return []
        self.pruned_experts |= pruned_mask
        for idx in new_pruned:
            self.W1[idx].zero_()
            self.W2[idx].zero_()
        if self.use_adapters:
            mask = self.pruned_experts
            self.adapter_scale[:, mask] = 0
            self.adapter_bias[:, mask] = 0
        return new_pruned


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, skip_attn_layers: set[int],
                 peak_frac: float, temp_boost: float, lb_boost: float, boost_shape: str = "peak"):
        super().__init__()
        self.attn = None if layer_idx in skip_attn_layers else CausalSelfAttention(dim, num_heads, max_seq_len)
        self.layer_idx = layer_idx
        self.total_layers = None
        self.layer_peak_frac = float(peak_frac)
        self.temp_boost = float(temp_boost)
        self.lb_boost = float(lb_boost)
        self.boost_shape = (boost_shape or "peak").lower()

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask,
                lambdas: Tensor, sa_lambdas: Tensor, bank: SharedFFNBank, layer_idx: int,
                flags: Tensor, temperature: float, logit_cap: float | None,
                freeze_ema_alpha: bool, use_gumbel: bool, active_mask: Tensor | None,
                freeze_router_params: bool, freeze_adapter_params: bool,
                ema_limits_fwd: tuple[float, float] | None,
                ema_limits_rev: tuple[float, float] | None,
                freeze_ema_alpha_rev: bool, decay_scale: float) -> tuple[Tensor, Tensor]:
        if self.total_layers is None:
            self.total_layers = getattr(bank, "L", layer_idx + 1)
        layer_frac = layer_idx / max(self.total_layers - 1, 1)
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
        peak = self.layer_peak_frac
        dist = abs(layer_frac - peak)
        denom = peak if layer_frac <= peak else max(1.0 - peak, 1e-6)
        shape_peak = max(0.0, 1.0 - dist / denom)
        if self.boost_shape == "valley":
            shape = 1.0 - shape_peak
        elif self.boost_shape == "linear_start":
            shape = max(0.0, min(1.0, 1.0 - layer_frac))
        elif self.boost_shape == "linear_end":
            shape = max(0.0, min(1.0, layer_frac))
        else:
            shape = shape_peak
        decay_scale = float(decay_scale)
        temp_multiplier = 1.0 + decay_scale * self.temp_boost * shape
        lb_multiplier = 1.0 + decay_scale * self.lb_boost * shape
        y, aux = bank(
            norm(x),
            layer_idx,
            flags,
            temperature * temp_multiplier,
            logit_cap,
            freeze_ema_alpha,
            use_gumbel,
            lb_multiplier,
            active_mask,
            freeze_router_params=freeze_router_params,
            freeze_adapter_params=freeze_adapter_params,
            ema_limits_fwd=ema_limits_fwd,
            ema_limits_rev=ema_limits_rev,
            freeze_ema_alpha_rev=freeze_ema_alpha_rev,
        )
        x = x + y
        return x, aux

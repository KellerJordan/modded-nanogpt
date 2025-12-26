import math
import os
from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from switch_bank.model.components import Block, SharedFFNBank, norm, init_linear
from switch_bank.utils import next_multiple_of_n


def _compute_router_temp(step: int, total_steps: int, t_init: float, t_final: float,
                         power: float, anchor_delta_steps: int, anchor_ratio: float | None,
                         start_step: int) -> float:
    if step <= start_step:
        return t_init
    effective_total = max(total_steps - start_step, 1)
    progress = (step - start_step) / effective_total
    power_use = power
    if anchor_delta_steps > 0 and anchor_ratio is not None and 0 < anchor_ratio < 1:
        anchor_progress = min(max(anchor_delta_steps / effective_total, 1e-6), 0.999999)
        power_use = math.log(anchor_ratio) / math.log(1.0 - anchor_progress)
    return t_final + (t_init - t_final) * (1.0 - progress) ** power_use


@lru_cache(None)
def _second_expert_step(expert_activation_schedule: tuple[tuple[int, int], ...]) -> int:
    if len(expert_activation_schedule) >= 2:
        return max(0, int(expert_activation_schedule[1][0]))
    if len(expert_activation_schedule) == 1:
        return max(0, int(expert_activation_schedule[0][0]))
    return 0


def _normalize_layer_tie_groups(
    tie_groups: tuple[tuple[int, ...], ...] | None,
    num_layers: int,
    skip_attn_layers: set[int],
) -> tuple[int, ...]:
    tie_map = list(range(num_layers))
    if not tie_groups:
        return tuple(tie_map)
    if not isinstance(tie_groups, (list, tuple)):
        raise ValueError("layer_tie_groups must be a sequence of layer index groups")
    tie_groups = tuple(tie_groups)
    if (
        len(tie_groups) == 1
        and isinstance(tie_groups[0], (list, tuple))
        and tie_groups[0]
        and all(isinstance(entry, (list, tuple)) for entry in tie_groups[0])
    ):
        tie_groups = tuple(tie_groups[0])
    used = set()
    for group in tie_groups:
        if not group:
            continue
        group_layers = []
        for idx in group:
            if isinstance(idx, (list, tuple)):
                raise ValueError("layer_tie_groups should be a sequence of layer indices; remove extra nesting")
            idx_i = int(idx)
            if idx_i < 0 or idx_i >= num_layers:
                raise ValueError(f"Layer tie index {idx_i} out of range for num_layers={num_layers}")
            if idx_i in used or idx_i in group_layers:
                raise ValueError(f"Layer {idx_i} appears multiple times in layer_tie_groups")
            if idx_i in skip_attn_layers:
                raise ValueError(f"Layer tie groups cannot include skip-attn layer {idx_i}")
            group_layers.append(idx_i)
        if len(group_layers) < 2:
            continue
        base = group_layers[0]
        for idx_i in group_layers:
            used.add(idx_i)
            tie_map[idx_i] = base
    return tuple(tie_map)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int,
                 skip_attn_layers: set[int], layer_tie_groups: tuple[tuple[int, ...], ...],
                 E: int, h: int, lb_coeff: float, ent_coeff: float, k: int,
                 num_value_embeds: int,
                 tie_lm_head: bool, untie_lm_head_after: int,
                 ema_alpha_fwd: float, ema_alpha_rev: float,
                 router_temp_init: float, router_temp_final: float, router_temp_power: float,
                 router_temp_anchor_delta_steps: int | None, router_temp_anchor_ratio: float | None,
                 router_logit_cap_initial: float, router_logit_cap_final: float, router_logit_cap_delta_steps: int,
                 router_layer_peak_frac: float, router_temp_boost: float, router_lb_boost: float, router_boost_shape: str,
                 use_router_adapters: bool, expert_activation_schedule: tuple[tuple[int, int], ...],
                 router_freeze_frac: float, router_freeze_adapters: bool,
                 ema_block_size_fwd: int, ema_block_size_rev: int,
                 ema_window_size_fwd: int, ema_window_size_rev: int,
                 ema_layer_stride: int,
                 shared_ffn_freeze_frac: float,
                 router_use_gumbel: bool, router_gumbel_schedule: tuple[tuple[int, int], ...],
                 router_block_pos_bins: int, first_doc_tokens_N: int,
                 router_enable_forward_ema: bool, router_enable_reverse_ema: bool,
                 extra_console_logging: bool, extra_wandb_logging: bool,
                 print_fn=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_size_padded = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(self.vocab_size_padded, model_dim)

        assert 0 <= num_value_embeds <= 3
        self.num_value_embeds = int(num_value_embeds)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(self.num_value_embeds)])
        self.enable_forward_ema = bool(router_enable_forward_ema)
        self.enable_reverse_ema = bool(router_enable_reverse_ema)
        self.extra_console_logging = bool(extra_console_logging)
        self.extra_wandb_logging = bool(extra_wandb_logging)

        self.num_layers = num_layers
        self.layer_tie_map = _normalize_layer_tie_groups(layer_tie_groups, num_layers, skip_attn_layers)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, max_seq_len, i, skip_attn_layers,
                  router_layer_peak_frac, router_temp_boost, router_lb_boost, boost_shape=router_boost_shape)
            for i in range(num_layers)
        ])
        for idx, base in enumerate(self.layer_tie_map):
            if idx == base:
                continue
            base_attn = self.blocks[base].attn
            if base_attn is None:
                raise ValueError(f"Layer {base} has no attention but is used as a tie base")
            self.blocks[idx].attn = base_attn
        self.router_temp_boost = float(router_temp_boost)
        self.router_lb_boost = float(router_lb_boost)
        self.tie_lm_head = bool(tie_lm_head)
        self.untie_lm_head_after = int(untie_lm_head_after)
        needs_lm_head = (not self.tie_lm_head) or (self.untie_lm_head_after >= 0)
        self.lm_head = nn.Parameter(init_linear(torch.empty(self.vocab_size_padded, model_dim)).bfloat16()) if needs_lm_head else None
        self._head_tied_runtime = True
        if not self.tie_lm_head:
            self._head_tied_runtime = False

        assert 1 <= ema_layer_stride <= self.num_layers, "ema_layer_stride must be between 1 and num_layers"
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))
        self.router_temp_init = float(router_temp_init)
        self.router_temp_final = float(router_temp_final)
        self.router_temp_power = float(router_temp_power)
        self.router_temp_anchor_delta_steps = (int(router_temp_anchor_delta_steps) if router_temp_anchor_delta_steps is not None else -1)
        self.router_temp_anchor_ratio = (float(router_temp_anchor_ratio) if router_temp_anchor_ratio is not None else None)
        self.router_logit_cap_delta_steps = int(router_logit_cap_delta_steps)
        self.router_logit_cap_initial = float(router_logit_cap_initial)
        self.router_logit_cap_final = float(router_logit_cap_final)
        self.router_use_gumbel = bool(router_use_gumbel)
        schedule: list[tuple[int, int]] = []
        for entry in router_gumbel_schedule:
            if len(entry) < 2:
                continue
            start, end = entry
            schedule.append((max(0, int(start)), int(end)))
        schedule.sort(key=lambda x: x[0])
        self.router_gumbel_schedule: tuple[tuple[int, int], ...] = tuple(schedule)
        self.second_expert_step_const = _second_expert_step(expert_activation_schedule)
        self.router_freeze_frac = float(router_freeze_frac)
        self.router_freeze_adapters = bool(router_freeze_adapters)
        self.shared_ffn_freeze_frac = float(shared_ffn_freeze_frac)
        self._router_frozen_logged = False
        self._ffn_frozen_logged = False

        assert router_block_pos_bins in (4, 8, 16), "router_block_pos_bins must be 4, 8, or 16"
        self.router_block_pos_bins = int(router_block_pos_bins)
        self.first_doc_tokens_N = int(first_doc_tokens_N)
        self.use_router_adapters = bool(use_router_adapters)
        self.num_experts = E
        schedule: list[tuple[int, int]] = []
        for entry in expert_activation_schedule:
            if len(entry) < 2:
                continue
            step_v, count = entry[0], entry[1]
            step_i = max(0, int(step_v))
            count_i = max(1, min(self.num_experts, int(count)))
            schedule.append((step_i, count_i))
        schedule.sort(key=lambda x: x[0])
        self.expert_activation_schedule: list[tuple[int, int]] = []
        for step_v, count in schedule:
            if self.expert_activation_schedule and step_v == self.expert_activation_schedule[-1][0]:
                self.expert_activation_schedule[-1] = (step_v, count)
            else:
                self.expert_activation_schedule.append((step_v, count))
        mask_needed = any(count < self.num_experts for _, count in self.expert_activation_schedule)
        if mask_needed:
            if not self.expert_activation_schedule:
                self.expert_activation_schedule = [(0, self.num_experts)]
            elif self.expert_activation_schedule[0][0] > 0:
                first = self.expert_activation_schedule[0][1]
                self.expert_activation_schedule.insert(0, (0, first))
            last_step, last_count = self.expert_activation_schedule[-1]
            if last_count < self.num_experts:
                self.expert_activation_schedule.append((last_step, self.num_experts))
        self._expert_schedule_requires_mask = mask_needed
        self._full_activation_step = self._compute_full_activation_step()
        self._full_activation_step = self._compute_full_activation_step()

        flags_dim = 3 + self.router_block_pos_bins

        self.bank = SharedFFNBank(
            d=model_dim, h=h, E=E, L=num_layers, flags_dim=flags_dim,
            lb_coeff=lb_coeff, ent_coeff=ent_coeff, k=k,
            use_adapters=use_router_adapters,
            ema_alpha_fwd=ema_alpha_fwd, ema_alpha_rev=ema_alpha_rev,
            use_forward_ema=self.enable_forward_ema, use_reverse_ema=self.enable_reverse_ema,
            ema_block_size_fwd=ema_block_size_fwd, ema_block_size_rev=ema_block_size_rev,
            ema_window_size_fwd=ema_window_size_fwd, ema_window_size_rev=ema_window_size_rev,
            ema_layer_stride=ema_layer_stride,
            extra_wandb_logging=self.extra_wandb_logging,
            adapter_layer_tie_map=self.layer_tie_map,
        )
        self.latest_router_metrics: list[dict[str, float] | None] | None = None
        self.latest_loss_components: tuple[Tensor, Tensor] | None = None
        self._last_active_expert_count: int | None = None
        self._current_base_active: int = self.num_experts
        self._pending_active_count: tuple[int, int] | None = None
        self._print0 = print_fn or (lambda *args, **kwargs: None)

    def _build_flags(self, input_seq: Tensor) -> Tensor:
        T = input_seq.size(0)
        device = input_seq.device
        is_eod_bool = (input_seq == 50256)
        is_eod = is_eod_bool.float().unsqueeze(0).unsqueeze(-1)
        start_flags = torch.zeros(T, dtype=torch.bool, device=device)
        start_flags[0] = True
        if T > 1:
            start_flags[1:] = is_eod_bool[:-1]
        is_after_eod = start_flags.float().unsqueeze(0).unsqueeze(-1)
        idx = torch.arange(T, device=device, dtype=torch.int64)
        start_idx = torch.where(start_flags, idx, torch.zeros_like(idx))
        last_start = torch.cummax(start_idx, dim=0)[0]
        dist_since_start = (idx - last_start).to(torch.int64)
        N = max(self.first_doc_tokens_N, 0)
        is_first_docN = (dist_since_start < N).float().unsqueeze(0).unsqueeze(-1)
        bins = self.router_block_pos_bins
        pos128 = idx % 128
        bin_idx = torch.clamp((pos128 * bins) // 128, max=bins - 1)
        onehot_bins = F.one_hot(bin_idx, num_classes=bins).float().unsqueeze(0)
        flags = torch.cat([is_eod, is_after_eod, is_first_docN, onehot_bins], dim=-1).to(dtype=torch.bfloat16)
        return flags

    def _ema_limits_for_progress(self, progress: float, reverse: bool = False) -> tuple[float, float]:
        if reverse:
            return (float(self.bank.ema_alpha_min_rev), float(self.bank.ema_alpha_max_rev))
        return (float(self.bank.ema_alpha_min_fwd), float(self.bank.ema_alpha_max_fwd))

    @torch._dynamo.disable
    def _active_expert_mask(self, step: int, device: torch.device) -> torch.Tensor | None:
        if not self._expert_schedule_requires_mask or not self.expert_activation_schedule:
            self._current_base_active = self.num_experts
            return None
        current = self.expert_activation_schedule[0]
        for stage in self.expert_activation_schedule:
            if step >= stage[0]:
                current = stage
            else:
                break
        active_count = max(1, min(self.num_experts, int(current[1])))
        self._current_base_active = active_count
        if active_count >= self.num_experts:
            self._expert_schedule_requires_mask = False
            self._mask_for_runtime = None
            return None
        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=device)
        mask[:active_count] = True
        if mask.all():
            self._mask_for_runtime = None
            return None
        self._mask_for_runtime = mask
        return mask

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx

        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)

        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all

        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def compute_router_temp(self, step: int, total_steps: int) -> float:
        return _compute_router_temp(
            step, total_steps, self.router_temp_init, self.router_temp_final,
            self.router_temp_power, self.router_temp_anchor_delta_steps, self.router_temp_anchor_ratio,
            start_step=self.second_expert_step_const)

    def compute_logit_cap(self, step: int) -> float | None:
        start_step = self.second_expert_step_const
        delta = max(int(self.router_logit_cap_delta_steps), 0)
        start = self.router_logit_cap_initial
        end = self.router_logit_cap_final
        if delta <= 0:
            return end if end > 0 else None
        if step < start_step:
            return start if start > 0 else None
        frac = min(max((step - start_step) / max(delta, 1), 0.0), 1.0)
        if start <= 0 and end <= 0:
            return None
        if start <= 0:
            return end
        if end <= 0:
            return max(start * (1.0 - frac), 0.0)
        shaped = frac ** 4.0
        return start * math.exp(math.log(end / start) * shaped)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor,
                step: int, total_steps: int):
        assert input_seq.ndim == 1
        self.bank.begin_router_metrics()
        self.latest_loss_components = None
        head_should_be_tied = self._head_should_be_tied(step)
        if self.lm_head is not None and self._head_tied_runtime and not head_should_be_tied:
            self.lm_head.data.copy_(self.embed.weight.data)
        self._head_tied_runtime = head_should_be_tied

        progress = step / max(total_steps, 1)
        active_mask = self._active_expert_mask(step, input_seq.device)
        self.bank.maybe_init_adapters(active_mask)
        ema_limits_fwd = self._ema_limits_for_progress(progress, reverse=False)
        ema_limits_rev = self._ema_limits_for_progress(progress, reverse=True)
        freeze_router_params = (progress >= self.router_freeze_frac)
        freeze_adapter_params = (freeze_router_params and self.router_freeze_adapters)
        freeze_ffn = (progress >= self.shared_ffn_freeze_frac)
        base_active = getattr(self, "_current_base_active", self.num_experts if active_mask is None else int(active_mask.sum().item()))
        if self.training:
            last_logged = getattr(self, "_last_active_expert_count", None)
            if last_logged is None or base_active > last_logged:
                self._last_active_expert_count = base_active
                self._pending_active_count = (step, base_active)
        if freeze_router_params and not self._router_frozen_logged and self.extra_console_logging:
            self._print0(f"Routers frozen at step {step}", console=True)
            self._router_frozen_logged = True
        if freeze_ffn and not self._ffn_frozen_logged and self.extra_console_logging:
            self._print0(f"Shared FFN frozen at step {step}", console=True)
            self._ffn_frozen_logged = True
        T_cur = self.compute_router_temp(step, total_steps)
        logit_cap = self.compute_logit_cap(step)
        decay_scale = 1.0
        freeze_ema_alpha_fwd = True
        freeze_ema_alpha_rev = True
        use_gumbel_now = False
        if self.router_use_gumbel:
            for start, end in self.router_gumbel_schedule:
                end_eff = total_steps if end < 0 else end
                if start <= step < end_eff:
                    use_gumbel_now = True
                    break

        ve_tables = [value_embed(input_seq) for value_embed in self.value_embeds]
        L = len(self.blocks)
        if self.num_value_embeds == 3:
            ve = [ve_tables[0], ve_tables[1], ve_tables[2]] + [None] * max(L - 6, 0) + [ve_tables[0], ve_tables[1], ve_tables[2]]
        elif self.num_value_embeds == 2:
            ve = [ve_tables[0], ve_tables[1]] + [None] * max(L - 4, 0) + [ve_tables[0], ve_tables[1]]
        elif self.num_value_embeds == 1:
            ve = [ve_tables[0]] + [None] * max(L - 2, 0)
            if L >= 2:
                ve.append(ve_tables[0])
        else:
            ve = [None] * L
        if len(ve) < L:
            ve = ve + [None] * (L - len(ve))
        elif len(ve) > L:
            ve = ve[:L]

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        if L == 28:
            long_ids = {0, 4, 8, 12, 16, 20, 24}
        elif L == 32:
            long_ids = {0, 4, 8, 12, 16, 20, 24, 28}
        else:
            stride = max(L // 4, 1)
            long_ids = set(range(0, L, stride))
        block_masks = [long_bm if i in long_ids else short_bm for i in range(L)]

        x = x0 = norm(self.embed(input_seq)[None])
        flags = self._build_flags(input_seq)

        skip_connections = []
        skip_map = {9: 6, 10: 4, 11: 2}
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L: 3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L: 5 * L].view(-1, 2)

        aux_loss = x.new_zeros(()).float()
        for i in range(L):
            if i in skip_map and skip_map[i] < len(skip_connections):
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            if freeze_ffn:
                with torch.no_grad():
                    y, aux = self.blocks[i](
                        x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i],
                        self.bank, i, flags, float(T_cur),
                        (float(logit_cap) if logit_cap is not None else None),
                        freeze_ema_alpha_fwd, use_gumbel_now, active_mask,
                        freeze_router_params=freeze_router_params,
                        freeze_adapter_params=freeze_adapter_params,
                        ema_limits_fwd=ema_limits_fwd,
                        ema_limits_rev=ema_limits_rev,
                        freeze_ema_alpha_rev=freeze_ema_alpha_rev,
                        decay_scale=decay_scale,
                    )
                y = y.detach()
                aux = aux.detach()
            else:
                y, aux = self.blocks[i](
                    x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i],
                    self.bank, i, flags, float(T_cur),
                    (float(logit_cap) if logit_cap is not None else None),
                    freeze_ema_alpha_fwd, use_gumbel_now, active_mask,
                    freeze_router_params=freeze_router_params,
                    freeze_adapter_params=freeze_adapter_params,
                    ema_limits_fwd=ema_limits_fwd,
                    ema_limits_rev=ema_limits_rev,
                    freeze_ema_alpha_rev=freeze_ema_alpha_rev,
                    decay_scale=decay_scale,
                )
            x = y
            aux_loss = aux_loss + aux
            skip_connections.append(x)

        x = norm(x)
        self.latest_router_metrics = self.bank.pop_router_metrics()
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self._lm_head_weight()).float()
            loss_main = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            self.latest_loss_components = (loss_main.detach(), aux_loss.detach())
            return loss_main, aux_loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self._lm_head_weight()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        self.latest_loss_components = None
        return loss

    def _head_should_be_tied(self, step: int) -> bool:
        if not self.tie_lm_head:
            return False
        if self.untie_lm_head_after >= 0:
            return step < self.untie_lm_head_after
        return True

    def _lm_head_weight(self) -> Tensor:
        if self.lm_head is None or self._head_tied_runtime:
            return self.embed.weight.bfloat16()
        return self.lm_head.bfloat16()

    def _compute_full_activation_step(self) -> int:
        step_full = 0
        for step_v, count in self.expert_activation_schedule:
            if count >= self.num_experts:
                step_full = int(step_v)
                break
        return max(0, step_full)

    def _latest_activation(self, step: int) -> tuple[int, int]:
        last_step = 0
        last_count = 0
        for stage_step, count in self.expert_activation_schedule:
            if step >= stage_step:
                last_step = int(stage_step)
                last_count = int(count)
            else:
                break
        return last_step, last_count

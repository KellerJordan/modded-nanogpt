import copy
import math
import time
import os
from functools import lru_cache
from collections import defaultdict, deque
from typing import List, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.optim import Optimizer

from switch_bank.utils import next_multiple_of_n, rampdown_multiplier
from switch_bank.data import (
    distributed_data_generator,
    summarize_router_metrics,
    summarize_expert_usage,
    summarize_expert_activity,
    router_summary_str,
)
from switch_bank.model.gpt import _compute_router_temp, _second_expert_step
from switch_bank.model.components import CausalSelfAttention

def get_lr(args, step: int):
    freeze_last = max(int(getattr(args, "lr_freeze_last_steps", 0)), 0)
    schedule_step = min(step, max(args.num_iterations - freeze_last, 0))
    x = schedule_step / max(args.num_iterations, 1)
    x = min(max(x, 0.0), 1.0)
    if x < 1 - args.cooldown_frac:
        return 1.0
    cooldown = max(args.cooldown_frac, 1e-8)
    t = (x - (1 - cooldown)) / cooldown
    t = min(max(t, 0.0), 1.0)
    final_mult = float(getattr(args, "lr_final_mult", 0.0))
    return 1.0 - t * (1.0 - final_mult)


@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


def get_window_size_blocks(args, step: int):
    x = step / args.num_iterations
    assert 0 <= x <= 1
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)


def get_router_temp(args, step: int):
    return _compute_router_temp(
        step, args.num_iterations, args.router_temp_init, args.router_temp_final,
        args.router_temp_power, args.router_temp_anchor_delta_steps, args.router_temp_anchor_ratio,
        start_step=_second_expert_step(tuple(args.expert_activation_schedule)))


def get_logit_cap(args, step: int):
    start_step = _second_expert_step(tuple(args.expert_activation_schedule))
    delta = max(int(args.router_logit_cap_delta_steps), 0)
    start = args.router_logit_cap_initial
    end = args.router_logit_cap_final
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

def gumbel_active(args, step: int):
    if not args.router_use_gumbel:
        return False
    schedule = getattr(args, "router_gumbel_schedule", ())
    for start, end in schedule:
        end_eff = args.num_iterations if end < 0 else end
        if start <= step < end_eff:
            return True
    return False


def _update_logit_stats(logit_stats: dict[str, float], max_logit: float, logit_cap: float | None):
    if logit_cap is None or logit_cap <= 0 or math.isnan(logit_cap):
        return
    if math.isnan(max_logit) or math.isinf(max_logit):
        return
    ratio = max_logit / logit_cap if logit_cap > 0 else float("nan")
    if math.isnan(ratio) or math.isinf(ratio):
        return
    ratio = min(max(ratio, 0.0), 1.5)
    logit_stats["count"] += 1.0
    logit_stats["sum_ratio"] += ratio
    if ratio >= 0.98:
        logit_stats["cap_hits"] += 1.0
    if ratio > logit_stats["max_ratio"]:
        logit_stats["max_ratio"] = ratio


def _finalize_logit_stats(logit_stats: dict[str, float]) -> dict[str, float]:
    count = int(logit_stats.get("count", 0.0))
    if count <= 0:
        return {
            "logit_cap_steps": 0.0,
            "logit_cap_hit_rate": float("nan"),
            "logit_cap_ratio_mean": float("nan"),
            "logit_cap_ratio_max": float("nan"),
            "logit_headroom_mean": float("nan"),
            "logit_score": float("nan"),
        }
    mean_ratio = logit_stats.get("sum_ratio", 0.0) / max(count, 1)
    cap_hit_rate = logit_stats.get("cap_hits", 0.0) / max(count, 1)
    ratio_target = 0.85
    ratio_score = 1.0 - abs(mean_ratio - ratio_target) / max(ratio_target, 1e-8)
    ratio_score = min(max(ratio_score, 0.0), 1.0)
    logit_score = 0.7 * (1.0 - cap_hit_rate) + 0.3 * ratio_score
    logit_score = min(max(logit_score, 0.0), 1.0)
    return {
        "logit_cap_steps": float(count),
        "logit_cap_hit_rate": float(cap_hit_rate),
        "logit_cap_ratio_mean": float(mean_ratio),
        "logit_cap_ratio_max": float(logit_stats.get("max_ratio", 0.0)),
        "logit_headroom_mean": float(1.0 - mean_ratio),
        "logit_score": float(logit_score),
    }



def _chunk_by_bytes(tensors: List[torch.Tensor], max_bytes: int) -> List[List[torch.Tensor]]:
    """Chunk tensors into buckets with total size <= max_bytes.
    Each tensor is assumed dense and on CUDA.
    """
    buckets: List[List[torch.Tensor]] = []
    cur: List[torch.Tensor] = []
    cur_bytes = 0

    for t in tensors:
        t_bytes = t.numel() * t.element_size()
        if cur and cur_bytes + t_bytes > max_bytes:
            buckets.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(t)
        cur_bytes += t_bytes

    if cur:
        buckets.append(cur)
    return buckets


class GradAllReducer:
    """Deterministic, bucketed gradient all-reduce that handles per-rank unused params.

    Guarantees that all ranks execute the same collective sequence, even if some params
    have p.grad is None on some ranks.

    Notes:
      - Uses a global-used mask to skip params unused on *all* ranks this step.
      - Uses cached zero buffers for locally-missing grads for globally-used params.
    """

    def __init__(self, model: torch.nn.Module):
        self._id2name = {id(p): n for n, p in model.named_parameters()}
        self._zero_grad_cache: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def allreduce_grads(
        self,
        opt2params: Dict[Optimizer, List[Parameter]],
        *,
        bucket_cap_mb: int = 32,
    ) -> Dict[Optimizer, List[torch.futures.Future]]:
        """All-reduce grads for each optimizer's param list.

        Args:
          opt2params: mapping opt -> params (Parameters may overlap across opts).
          bucket_cap_mb: bucket size in MiB for coalesced all-reduce.

        Returns:
          opt2futures: mapping opt -> list of futures that complete when reduction is done.
        """
        world_size = dist.get_world_size()
        inv_world = 1.0 / float(world_size)
        max_bytes = int(bucket_cap_mb * 1024 * 1024)

        opt2futures: Dict[Optimizer, List[torch.futures.Future]] = {}

        for opt, params in opt2params.items():
            # Deterministic order across ranks.
            params_sorted = sorted(params, key=lambda p: self._id2name.get(id(p), ""))

            # Local "has grad" flags (int32 on CUDA so NCCL can reduce it fast).
            has_grad = torch.empty(len(params_sorted), device="cuda", dtype=torch.int32)
            for i, p in enumerate(params_sorted):
                has_grad[i] = 1 if (p.requires_grad and p.grad is not None) else 0

            # Global-used mask: >0 means at least one rank produced a grad for that param this step.
            dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)

            grads_to_reduce: List[torch.Tensor] = []
            for i, p in enumerate(params_sorted):
                if not p.requires_grad:
                    continue
                if has_grad[i].item() == 0:
                    # Unused everywhere this step; skip entirely (preserves your old semantics).
                    continue

                if p.grad is None:
                    # Locally unused but globally used: supply a cached zero grad buffer.
                    pid = id(p)
                    buf = self._zero_grad_cache.get(pid)
                    if buf is None or buf.shape != p.shape or buf.dtype != p.dtype or buf.device != p.device:
                        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                        self._zero_grad_cache[pid] = buf
                    else:
                        buf.zero_()
                    p.grad = buf

                grads_to_reduce.append(p.grad)

            # Bucketed all-reduce. Use SUM + scale (AVG isn’t always supported on all APIs).
            futures: List[torch.futures.Future] = []
            for bucket in _chunk_by_bytes(grads_to_reduce, max_bytes):
                if not bucket:
                    continue
                fut = _work_to_future(
                    dist.all_reduce_coalesced(bucket, op=dist.ReduceOp.SUM, async_op=True)
                )

                # Scale grads after the SUM completes (foreach is reasonably cheap).
                def _scale_cb(f, bucket=bucket, inv_world=inv_world):
                    torch._foreach_mul_(bucket, inv_world)
                    return f.value()

                futures.append(fut.then(_scale_cb))

            opt2futures[opt] = futures

        return opt2futures

def _work_to_future(work_or_future):
    """Normalize c10d async returns to a Future.

    PyTorch returns either:
      - a Work object (with .get_future()), or
      - a torch._C.Future directly.
    """
    return work_or_future.get_future() if hasattr(work_or_future, "get_future") else work_or_future

def run_training(
    args,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    opt2params: dict,
    train_micro_len: int,
    untie_lm_head_after: int,
    run_id_full: str | None,
    master_process: bool,
    print0,
    code: str,
    wandb_run,
    metrics_csv_writer,
    expert_usage_headers: list[str],
    expert_active_headers: list[str],
    world_size: int,
    rank: int,
    log_param_counts_fn=None,
    start_step: int = 0,
    checkpoint_save_step: int = -1,
    early_stop_step: int | None = None,
    early_stop_val_multiplier: int = 1,
    early_stop_as_final: bool = False,
):
    training_time_ms = 0
    log_dir = getattr(args, "log_dir", "logs")
    approx_step_time_ms_resume = getattr(args, "approx_step_time_ms", None)
    train_loader = distributed_data_generator(
        args.train_files,
        world_size * train_micro_len,
        rank,
        world_size,
        skip_batches=start_step * args.grad_accum_steps,
    )
    print0("Starting training.", console=True)
    dist.barrier()
    t0 = time.perf_counter()
    train_steps = args.num_iterations
    stop_step = train_steps if early_stop_step is None else min(train_steps, early_stop_step)
    last_val_loss: float | None = None
    gumbel_prev_state = gumbel_active(args, start_step - 1) if start_step > 0 else gumbel_active(args, 0)
    logit_cap_decay_logged = False
    lm_head_untie_step = untie_lm_head_after
    lm_head_untied_logged = lm_head_untie_step < 0
    turbo_muon_warmstart_prev: bool | None = None
    turbo_muon_warmstart_start_step: int | None = None
    warmstart_start_frac = float(getattr(args, "turbo_muon_warmstart_smax_start_frac", -1.0))
    if warmstart_start_frac >= 0:
        turbo_muon_warmstart_start_step = int(warmstart_start_frac * train_steps)
        turbo_muon_warmstart_start_step = min(max(turbo_muon_warmstart_start_step, 0), train_steps)

    logit_stats = {"count": 0.0, "sum_ratio": 0.0, "cap_hits": 0.0, "max_ratio": 0.0}

    _grad_allreducer = GradAllReducer(model)

    def run_validation(val_steps_multiplier: float, log_val_loss: bool, extra_log: dict | None = None, log_to_wandb: bool = True):
        nonlocal training_time_ms, t0, last_val_loss
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        print0("Running validation...")
        model.eval()
        prev_k = model.bank.k
        model.bank.k = int(args.topk if args.topk_val is None else max(1, min(args.topk_val, args.num_experts)))
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        base_steps = args.val_tokens // val_batch_size
        val_steps = int(round(base_steps * float(val_steps_multiplier)))
        val_steps = max(val_steps, 1)
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, window_blocks, step, args.num_iterations)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_scalar = float(val_loss.detach().item())
        last_val_loss = val_scalar
        print0(
            f"step:{step}/{train_steps} val_loss:{val_scalar:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
            console=True)
        if wandb_run is not None and log_to_wandb:
            log_payload = {
                "val/step": step,
                "perf/approx_step_time_ms": training_time_ms,
            }
            if log_val_loss:
                log_payload["val/loss"] = val_scalar
            if extra_log:
                log_payload.update(extra_log)
            wandb_run.log(log_payload, step=step)
        model.train()
        model.bank.k = prev_k
        dist.barrier()
        t0 = time.perf_counter()
        return val_scalar

    if start_step == 0 and (early_stop_step is None or early_stop_step > 0):
        step = 0
        window_blocks = get_window_size_blocks(args, step)
        tokens_target = getattr(args, "val_tokens_intermediate", None)
        if tokens_target is None:
            tokens_target = args.val_tokens
        val_steps_multiplier = float(tokens_target) / float(args.val_tokens)
        run_validation(val_steps_multiplier, log_val_loss=True)

    router_clip_base = getattr(args, "router_grad_clip_norm", None)
    router_clip_base = float(router_clip_base) if router_clip_base is not None else None
    if router_clip_base is not None and router_clip_base <= 0:
        router_clip_base = None
    router_params_by_opt: dict[torch.optim.Optimizer, list[nn.Parameter]] = {}
    router_autoclip = bool(getattr(args, "router_autoclip", False))
    autoclip_window = 250
    router_autoclip_state: dict[torch.optim.Optimizer, dict[str, object]] = {}
    needs_router_clip = router_autoclip or (router_clip_base is not None)
    if needs_router_clip:
        for opt in optimizers:
            params = []
            for group in opt.param_groups:
                if group.get("component") == "router":
                    params.extend(group["params"])
            if params:
                router_params_by_opt[opt] = params
                if router_autoclip:
                    initial_clip = router_clip_base if router_clip_base is not None else None
                    router_autoclip_state[opt] = {
                        "history": deque(maxlen=autoclip_window),
                        "clip": (initial_clip if (initial_clip is not None and initial_clip > 0) else None),
                    }

    for step in range(start_step, train_steps + 1):
        final_step = (step == train_steps)
        last_step = (step == stop_step)
        finalize_now = final_step or (early_stop_as_final and last_step)
        turbo_muon_warmstart_now = (
            turbo_muon_warmstart_start_step is not None and step >= turbo_muon_warmstart_start_step
        )
        if turbo_muon_warmstart_prev is None or turbo_muon_warmstart_now != turbo_muon_warmstart_prev:
            for opt in optimizers:
                if hasattr(opt, "set_turbomuon_warmstart_smax"):
                    opt.set_turbomuon_warmstart_smax(turbo_muon_warmstart_now)
            turbo_muon_warmstart_prev = turbo_muon_warmstart_now
        window_blocks = get_window_size_blocks(args, step)
        progress = step / max(train_steps, 1)
        gumbel_now = gumbel_active(args, step)
        if args.enable_extra_logging and gumbel_now != gumbel_prev_state:
            status = "enabled" if gumbel_now else "disabled"
            print0(f"Gumbel router noise {status} at step {step}", console=True)
        gumbel_prev_state = gumbel_now
        current_logit_cap = get_logit_cap(args, step)
        cap_start_step = _second_expert_step(tuple(args.expert_activation_schedule))
        if not logit_cap_decay_logged and step >= cap_start_step:
            cap_str = f"{current_logit_cap:.4f}" if current_logit_cap is not None else "disabled"
            if args.enable_extra_logging:
                print0(f"Router logit cap entered ramp at step {step}: cap={cap_str}", console=True)
            logit_cap_decay_logged = True
        if (not lm_head_untied_logged) and lm_head_untie_step >= 0 and step >= lm_head_untie_step:
            if args.enable_extra_logging:
                print0(f"LM head untied at step {step}", console=True)
            if log_param_counts_fn:
                log_param_counts_fn(model)
            lm_head_untied_logged = True

        if last_step or (step > 0 and args.val_loss_every > 0 and step % args.val_loss_every == 0):
            extra_log = None
            if last_step:
                extra_log = _finalize_logit_stats(logit_stats)
            tokens_target = getattr(args, "val_tokens", None)
            if finalize_now:
                final_tokens = getattr(args, "val_tokens_final", None)
                if final_tokens is not None:
                    tokens_target = final_tokens
            else:
                intermediate_tokens = getattr(args, "val_tokens_intermediate", None)
                if intermediate_tokens is not None:
                    tokens_target = intermediate_tokens
            val_steps_multiplier = float(tokens_target) / float(args.val_tokens)
            run_validation(val_steps_multiplier, log_val_loss=True, extra_log=extra_log)
            if last_step and not finalize_now:
                result = {"val_loss": last_val_loss, "stop_step": step, "aborted": False}
                result.update(_finalize_logit_stats(logit_stats))
                return result

        if last_step and finalize_now:
            should_save = getattr(args, "save_final_checkpoint", getattr(args, "save_checkpoint", False))
            if should_save and getattr(args, "save_final_checkpoint_if_loss_below", False):
                max_loss = float(getattr(args, "save_final_checkpoint_max_loss", float("inf")))
                should_save = last_val_loss is not None and math.isfinite(last_val_loss) and last_val_loss < max_loss
            if master_process and should_save:
                model_to_save = getattr(model, "_orig_mod", model)
                log = dict(step=step, code=code, model=model_to_save.state_dict())
                run_dir = os.path.join(log_dir, run_id_full)
                os.makedirs(run_dir, exist_ok=True)
                torch.save(log, os.path.join(run_dir, f"final_model_step{step:06d}.pt"))
            break

        model.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        micro_main_losses: list[float] = []
        micro_aux_losses: list[float] = []
        router_metric_accum: list[dict[str, float]] = []
        router_layer_metric_sums = [defaultdict(float) for _ in range(model.num_layers)]
        router_layer_metric_counts = [defaultdict(int) for _ in range(model.num_layers)]
        expert_usage_accum: list[torch.Tensor] = []
        expert_active_accum: list[torch.Tensor] = []
        for micro in range(args.grad_accum_steps):
            inputs, targets = next(train_loader)
            outputs = model(inputs, targets, window_blocks, step, args.num_iterations)
            if isinstance(outputs, tuple):
                loss_main, loss_aux = outputs
                loss_main_v = float(loss_main.detach().item())
                loss_aux_v = float(loss_aux.detach().item())
                loss_val = loss_main_v + loss_aux_v
                micro_losses.append(loss_val)
                micro_main_losses.append(loss_main_v)
                micro_aux_losses.append(loss_aux_v)
                loss_total = (loss_main + loss_aux) / args.grad_accum_steps
                loss_total.backward()
                components = (loss_main.detach(), loss_aux.detach())
                main_loss = loss_main_v
                aux_loss = loss_aux_v
            else:
                loss = outputs
                loss_val = float(loss.detach().item())
                micro_losses.append(loss_val)
                (loss / args.grad_accum_steps).backward()
                components = model.latest_loss_components
                main_loss = float(components[0].item()) if components else float("nan")
                aux_loss = float(components[1].item()) if components else float("nan")
                if not math.isnan(main_loss):
                    micro_main_losses.append(main_loss)
                if not math.isnan(aux_loss):
                    micro_aux_losses.append(aux_loss)
            router_summary = summarize_router_metrics(model.latest_router_metrics or [])
            if router_summary:
                router_metric_accum.append(router_summary)
            layer_metrics = model.latest_router_metrics or []
            for layer_idx, stats in enumerate(layer_metrics):
                if not stats:
                    continue
                layer_sum = router_layer_metric_sums[layer_idx]
                layer_count = router_layer_metric_counts[layer_idx]
                for key, value in stats.items():
                    scalar_val = None
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            scalar_val = float(value.item())
                        else:
                            continue
                    elif isinstance(value, (int, float)):
                        scalar_val = float(value)
                    else:
                        continue
                    layer_sum[key] += scalar_val
                    layer_count[key] += 1
            usage = summarize_expert_usage(layer_metrics, args.num_experts)
            if usage is not None:
                expert_usage_accum.append(usage)
            active = summarize_expert_activity(layer_metrics, args.num_experts)
            if active is not None:
                expert_active_accum.append(active)
            if args.enable_extra_logging:
                print0(
                    f"[train step {step} micro {micro + 1}/{args.grad_accum_steps}] "
                    f"loss={loss_val:.6f} main={main_loss:.6f} aux={aux_loss:.6f} "
                    f"{router_summary_str(router_summary, args.router_enable_forward_ema, args.router_enable_reverse_ema)}",
                    console=True)

        avg_loss = sum(micro_losses) / max(len(micro_losses), 1)
        avg_main_loss = sum(micro_main_losses) / max(len(micro_main_losses), 1) if micro_main_losses else float("nan")
        avg_aux_loss = sum(micro_aux_losses) / max(len(micro_aux_losses), 1) if micro_aux_losses else float("nan")
        router_step_summary = summarize_router_metrics(router_metric_accum)
        router_layer_avg: dict[int, dict[str, float]] = {}
        for layer_idx in range(model.num_layers):
            sums = router_layer_metric_sums[layer_idx]
            if not sums:
                continue
            counts = router_layer_metric_counts[layer_idx]
            router_layer_avg[layer_idx] = {key: sums[key] / max(counts[key], 1) for key in sums}
        expert_usage = torch.stack(expert_usage_accum).mean(0) if expert_usage_accum else None
        expert_active = torch.stack(expert_active_accum).mean(0) if expert_active_accum else None
        pending_event = getattr(model, "_pending_active_count", None)
        if pending_event is not None:
            event_step, active_count = pending_event
            if wandb_run is not None:
                wandb_run.log({"router/active_experts": active_count}, step=event_step)
                wandb_run.log({"router/active_total_ffn_dim": active_count * args.ffn_hidden}, step=event_step)
            model._pending_active_count = None

        abort_flag = False
        abort_reason = ""
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            abort_flag = True
            abort_reason = "non-finite avg_loss"
        elif not math.isnan(avg_main_loss) and not math.isinf(avg_main_loss) and math.isnan(avg_main_loss):
            abort_flag = True
            abort_reason = "non-finite main loss"
        else:
            max_logit_val = router_step_summary.get("max_logit", float("nan")) if router_step_summary else float("nan")
            if (not math.isnan(max_logit_val)) and (max_logit_val == 0.0 or math.isinf(max_logit_val)):
                abort_flag = True
                abort_reason = f"router max_logit suspicious ({max_logit_val})"

        abort_tensor = torch.tensor(1 if abort_flag else 0, device="cuda", dtype=torch.int32)
        dist.all_reduce(abort_tensor, op=dist.ReduceOp.SUM)
        if abort_tensor.item() > 0:
            if abort_reason and master_process:
                print0(f"Aborting training at step {step} due to: {abort_reason}", console=True)
            result = {"val_loss": last_val_loss, "stop_step": step, "aborted": True, "abort_reason": abort_reason}
            result.update(_finalize_logit_stats(logit_stats))
            return result

        # Build a stable name for each param once (same across ranks as long as the model is the same).
        if not hasattr(model, "_param_id_to_name"):
            model._param_id_to_name = {id(p): n for n, p in model.named_parameters()}
        #id2name = model._param_id_to_name

        _grad_allreducer.allreduce_grads(opt2params, bucket_cap_mb=32)

        progress = step / max(args.num_iterations, 1)
        router_lr_mult = rampdown_multiplier(progress, args.router_lr_reduce_start_frac, model.router_freeze_frac)
        adapter_lr_mult = router_lr_mult if args.router_freeze_adapters else 1.0
        ffn_lr_mult = rampdown_multiplier(progress, args.shared_ffn_lr_reduce_start_frac, model.shared_ffn_freeze_frac)
        for opt in optimizers:
            for group in opt.param_groups:
                base_lr = group["initial_lr"] * get_lr(args, step)
                component = group.get("component")
                mult = 1.0
                if component == "router":
                    mult = router_lr_mult
                elif component == "adapter":
                    mult = adapter_lr_mult
                elif component == "shared_ffn":
                    mult = ffn_lr_mult
                group["lr"] = base_lr * mult
        # Muon-style momentum warmup for spectral groups.
        target_muon_momentum = float(getattr(args, "muon_momentum", getattr(args, "neomuon_muon_momentum", 0.95)))
        frac = min(step / 300, 1)
        warm_momentum = (1 - frac) * 0.85 + frac * target_muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                if group.get("spectral", True):
                    group["momentum"] = warm_momentum
        for opt in optimizers:
            torch.futures.collect_all(opt2futures[opt]).wait()
            if opt in router_params_by_opt:
                state = router_autoclip_state.get(opt)
                clip_value = router_clip_base
                if state is not None and state.get("clip") is not None:
                    clip_value = float(state["clip"])
                max_norm = clip_value if (clip_value is not None and clip_value > 0) else float("inf")
                total_norm = float(torch.nn.utils.clip_grad_norm_(router_params_by_opt[opt], max_norm))
                if state is not None:
                    history: deque = state["history"]  # type: ignore[assignment]
                    history.append(total_norm)
                    if len(history) >= autoclip_window:
                        hist_tensor = torch.tensor(list(history), device="cpu")
                        new_clip = float(torch.quantile(hist_tensor, 0.10).item())
                        new_clip = max(new_clip, 1e-6)
                        if state.get("clip") is None or not math.isclose(state["clip"], new_clip):
                            state["clip"] = new_clip
                            if args.enable_extra_logging:
                                print0(
                                    f"[router grad clip auto] norm={total_norm:.4f} clip-> {new_clip:.4f}",
                                    console=True,
                                )
                elif clip_value is not None and clip_value > 0 and args.enable_extra_logging and total_norm > clip_value:
                    print0(
                        f"[router grad clip] norm={total_norm:.4f} clip={clip_value:.4f}",
                        console=True,
                    )
            # Feed last activations to Muon for spectral gating.
            if hasattr(opt, "set_last_activation") and bool(getattr(opt, "enable_spectral_gating", False)):
                for group in opt.param_groups:
                    if not group.get("spectral", True):
                        continue
                    for p in group.get("params", []):
                        act = getattr(p, "_neomuon_last_activation", None)
                        if act is not None:
                            opt.set_last_activation(p, act)
            opt.step()
        model.zero_grad(set_to_none=True)
        if args.enable_extra_logging and router_layer_avg:
            metric_keys = ["imp_cv2", "load_cv2", "usage_frac", "topk_prob_mean"]
            if args.router_enable_forward_ema:
                metric_keys.append("ema_alpha_forward")
            if args.router_enable_reverse_ema:
                metric_keys.append("ema_alpha_reverse")
            layer_fragments = []
            for layer_idx in sorted(router_layer_avg):
                stats = router_layer_avg[layer_idx]
                metrics = ", ".join(f"{key}={stats.get(key, float('nan')):.4f}" for key in metric_keys if key in stats)
                layer_fragments.append(f"L{layer_idx}: {metrics}")
            print0("[router layers] " + " | ".join(layer_fragments), console=True)
        print0(
            f"[train step {step}] avg_loss={avg_loss:.6f} main={avg_main_loss:.6f} aux={avg_aux_loss:.6f} "
            f"{router_summary_str(router_step_summary, args.router_enable_forward_ema, args.router_enable_reverse_ema)}",
            console=True)
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if approx_step_time_ms_resume is not None:
            approx_training_time_ms = approx_step_time_ms_resume
            # reset base timer so subsequent steps are correct
            training_time_ms = approx_step_time_ms_resume
            t0 = time.perf_counter()
            approx_step_time_ms_resume = None
        print0(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
            console=True)
        current_logit_cap = get_logit_cap(args, step)
        current_router_temp = get_router_temp(args, step)
        max_logit_val = router_step_summary.get("max_logit", float("nan")) if router_step_summary else float("nan")
        _update_logit_stats(logit_stats, max_logit_val, current_logit_cap)
        active_count_val = None
        if expert_active is not None:
            active_count_val = float(expert_active.mean().item() * args.num_experts)
        if active_count_val is None or active_count_val <= 0:
            active_count_val = float(args.num_experts)
        active_count_val = max(active_count_val, 1.0)
        if wandb_run is not None and (step % max(args.wandb_log_every, 1) == 0):
            if args.enable_extra_wandb_logging:
                # Preserve historical lr keys by mapping to first non-spectral/spectral groups.
                adamw_lr = next(
                    (g["lr"] for g in optimizers[0].param_groups if not g.get("spectral", True)),
                    float("nan"),
                )
                muon_lr = next(
                    (g["lr"] for g in optimizers[0].param_groups if g.get("spectral", True)),
                    float("nan"),
                )
                log_data = {
                    "train/loss": avg_loss,
                    "train/loss_main": avg_main_loss,
                    "train/loss_aux": avg_aux_loss,
                    "perf/approx_step_time_ms": approx_training_time_ms,
                    "train/tokens_seen": float((step + 1) * args.train_seq_len * world_size),
                    "lr/adamw": adamw_lr,
                    "lr/muon": muon_lr,
                    "train/step": step,
                    "router/logit_cap": (current_logit_cap if current_logit_cap is not None else float("nan")),
                    "router/logit_cap_enabled": float(current_logit_cap is not None),
                    "router/temperature": current_router_temp,
                    "router/max_logit": router_step_summary.get("max_logit", float("nan")),
                }
                # feature weight percentages
                feat_keys_all = [k for k in router_step_summary.keys() if k.startswith("feat_w_")]
                feat_keys = [k for k in feat_keys_all if k in ("feat_w_tok", "feat_w_ema_fwd", "feat_w_ema_rev", "feat_w_flags")]
                if not feat_keys:
                    feat_keys = feat_keys_all
                feat_total = sum(router_step_summary[k] for k in feat_keys) if feat_keys else 0.0
                if feat_total and feat_total != 0:
                    for k in feat_keys:
                        pct = 100.0 * router_step_summary[k] / feat_total
                        log_data[f"router/feat_pct/{k.replace('feat_w_', '')}"] = pct
                # normalized CV metrics
                imp_cv2 = router_step_summary.get("imp_cv2", float("nan"))
                load_cv2 = router_step_summary.get("load_cv2", float("nan"))
                log_data["router/imp_cv2_norm"] = imp_cv2 / active_count_val if not math.isnan(imp_cv2) else float("nan")
                log_data["router/load_cv2_norm"] = load_cv2 / active_count_val if not math.isnan(load_cv2) else float("nan")
                # entropy gaps
                expected_entropy = math.log(active_count_val) if active_count_val > 0 else float("nan")
                load_entropy = router_step_summary.get("load_entropy", float("nan"))
                imp_entropy = router_step_summary.get("imp_entropy", float("nan"))
                def _entropy_gap(val):
                    if math.isnan(val) or math.isnan(expected_entropy) or expected_entropy <= 0:
                        return float("nan")
                    return max(0.0, abs(expected_entropy - val) / expected_entropy)
                load_entropy_gap = _entropy_gap(load_entropy)
                imp_entropy_gap = _entropy_gap(imp_entropy)
                log_data["router/load_entropy_gap"] = load_entropy_gap
                log_data["router/imp_entropy_gap"] = imp_entropy_gap
                # usage gap and health score
                usage_frac = router_step_summary.get("usage_frac", float("nan"))
                target_usage = min(1.0, active_count_val / max(args.num_experts, 1))
                usage_gap = abs(usage_frac - target_usage) if not math.isnan(usage_frac) else float("nan")
                weights = {
                    "imp_cv2_norm": 0.8,
                    "load_cv2_norm": 0.9,
                    "load_entropy_gap": 0.5,
                    "imp_entropy_gap": 0.2,
                    "usage_gap": 1.25,
                }
                components_weighted = []
                components_weighted.append(weights["imp_cv2_norm"] * log_data["router/imp_cv2_norm"] if not math.isnan(log_data["router/imp_cv2_norm"]) else float("nan"))
                components_weighted.append(weights["load_cv2_norm"] * log_data["router/load_cv2_norm"] if not math.isnan(log_data["router/load_cv2_norm"]) else float("nan"))
                components_weighted.append(weights["load_entropy_gap"] * load_entropy_gap if not math.isnan(load_entropy_gap) else float("nan"))
                components_weighted.append(weights["imp_entropy_gap"] * imp_entropy_gap if not math.isnan(imp_entropy_gap) else float("nan"))
                components_weighted.append(weights["usage_gap"] * usage_gap if not math.isnan(usage_gap) else float("nan"))
                health_terms = [v for v in components_weighted if not math.isnan(v)]
                if health_terms:
                    health_penalty = sum(health_terms)
                    log_data["router/health_score"] = 1.0 / (1.0 + health_penalty)
                    health_penalty = health_penalty
                    log_data["router/health_penalty"] = health_penalty
                # raw router stats (skip feat_w_*; percents already logged)
                for key, value in router_step_summary.items():
                    if key.startswith("feat_w_"):
                        continue
                    log_data[f"router/{key}"] = value
                for layer_idx, stats in router_layer_avg.items():
                    for key, value in stats.items():
                        log_data[f"router_layer/{layer_idx}/{key}"] = value
                if expert_usage is not None:
                    expert_list = expert_usage.tolist()
                    log_data["router_expert/min_usage"] = float(min(expert_list))
                    log_data["router_expert/max_usage"] = float(max(expert_list))
                    log_data["router_expert/mean_usage"] = float(sum(expert_list) / len(expert_list))
                    for idx, value in enumerate(expert_list):
                        log_data[f"router_expert/e{idx}"] = float(value)
                if expert_active is not None:
                    base_model = getattr(model, "_orig_mod", model)
                    scheduled_active = getattr(base_model, "_current_base_active", None)
                    active_count = args.num_experts
                    if isinstance(scheduled_active, int):
                        active_count = max(1, min(args.num_experts, scheduled_active))
                    inferred_active = expert_active.sum().item()
                    if inferred_active > 0:
                        active_count = max(1, min(active_count, int(round(inferred_active))))
                    source = expert_usage if expert_usage is not None else expert_active
                    active_list = source.tolist()
                    active_slice = active_list[:active_count] if active_list else []
                    if not active_slice:
                        active_slice = [0.0]
                    denom = sum(active_slice)
                    if denom > 0:
                        active_slice = [v / denom for v in active_slice]
                    log_data["router_expert_active/min"] = float(min(active_slice))
                    log_data["router_expert_active/max"] = float(max(active_slice))
                    log_data["router_expert_active/mean"] = float(sum(active_slice) / len(active_slice))
                    per_expert = []
                    for idx in range(args.num_experts):
                        if idx < active_count and idx < len(active_slice):
                            per_expert.append(float(active_slice[idx]))
                        else:
                            per_expert.append(0.0)
                    for idx, value in enumerate(per_expert):
                        log_data[f"router_expert_active/e{idx}"] = value
                wandb_run.log(log_data, step=step)
            else:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/loss_main": avg_main_loss,
                        "train/loss_aux": avg_aux_loss,
                        "perf/approx_step_time_ms": approx_training_time_ms,
                        "train/tokens_seen": float((step + 1) * args.train_seq_len * world_size),
                        "train/step": step,
                        "router/logit_cap": (current_logit_cap if current_logit_cap is not None else float("nan")),
                        "router/temperature": current_router_temp,
                    },
                    step=step,
                )
        if master_process and metrics_csv_writer and (step % max(args.metrics_log_every, 1) == 0):
            expert_usage_list = []
            if expert_usage is not None:
                expert_usage_list = [float(x) for x in expert_usage.tolist()]
            else:
                expert_usage_list = [float("nan")] * len(expert_usage_headers)
            expert_active_list = []
            if expert_active is not None:
                expert_active_list = [float(x) for x in expert_active.tolist()]
            else:
                expert_active_list = [float("nan")] * len(expert_active_headers)
            router_ema_vals: list[float] = []
            if args.router_enable_forward_ema:
                router_ema_vals.append(router_step_summary.get("ema_alpha_forward", float("nan")))
            if args.router_enable_reverse_ema:
                router_ema_vals.append(router_step_summary.get("ema_alpha_reverse", float("nan")))
            row = [
                step,
                avg_loss,
                avg_main_loss,
                avg_aux_loss,
                router_step_summary.get("imp_cv2", float("nan")),
                router_step_summary.get("load_cv2", float("nan")),
                router_step_summary.get("usage_frac", float("nan")),
                router_step_summary.get("topk_prob_mean", float("nan")),
                *router_ema_vals,
                router_step_summary.get("max_logit", float("nan")),
                (current_logit_cap if current_logit_cap is not None else float("nan")),
                current_router_temp,
                int(window_blocks.item()),
                *expert_usage_list,
                *expert_active_list,
            ]
            metrics_csv_writer.writerow(row)
        if master_process and checkpoint_save_step >= 0 and step == checkpoint_save_step:
            run_dir = os.path.join(log_dir, run_id_full)
            os.makedirs(run_dir, exist_ok=True)
            model_to_save = getattr(model, "_orig_mod", model)
            checkpoint_payload = dict(
                step=step,
                code=code,
                model=model_to_save.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
                approx_step_time_ms=approx_training_time_ms,
                meta=dict(
                    model_dim=getattr(args, "model_dim", None),
                    num_layers=getattr(args, "num_layers", None),
                    num_heads=getattr(args, "num_heads", None),
                    num_experts=getattr(args, "num_experts", None),
                    ffn_hidden=getattr(args, "ffn_hidden", None),
                    vocab_size=getattr(args, "vocab_size", None),
                ),
            )
            torch.save(checkpoint_payload, os.path.join(run_dir, f"state_step{step:06d}.pt"))

    result = {"val_loss": last_val_loss, "stop_step": stop_step, "aborted": False}
    result.update(_finalize_logit_stats(logit_stats))
    return result

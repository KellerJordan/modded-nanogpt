import copy
import math
import time
import os
from functools import lru_cache
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

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
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        return (1 - x) / args.cooldown_frac


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
):
    training_time_ms = 0
    train_loader = distributed_data_generator(args.train_files, world_size * train_micro_len, rank, world_size)
    print0("Starting training.", console=True)
    dist.barrier()
    t0 = time.perf_counter()
    train_steps = args.num_iterations
    gumbel_off_logged = False
    logit_cap_decay_logged = False
    lm_head_untie_step = untie_lm_head_after
    lm_head_untied_logged = lm_head_untie_step < 0
    pending_expert_prune = False
    experts_pruned = False
    EXPERT_ACTIVITY_THRESHOLD = 0.015

    for step in range(train_steps + 1):
        last_step = (step == train_steps)
        window_blocks = get_window_size_blocks(args, step)
        progress = step / max(train_steps, 1)
        if args.router_use_gumbel and not gumbel_off_logged and progress >= args.router_gumbel_frac:
            if args.enable_extra_logging:
                print0(f"Gumbel router noise disabled at step {step}", console=True)
            gumbel_off_logged = True
        if not experts_pruned and progress >= args.router_freeze_frac:
            pending_expert_prune = True
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
            dist.barrier()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            print0("Running validation...")
            model.eval()
            prev_k = model.bank.k
            model.bank.k = int(args.topk if args.topk_val is None else max(1, min(args.topk_val, args.num_experts)))
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = (args.val_tokens // val_batch_size) * 7 if last_step else 1
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, window_blocks, step, args.num_iterations)
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
                console=True)
            if wandb_run is not None:
                wandb_run.log({"val/loss": float(val_loss.detach().item()), "val/step": step}, step=step)
            model.train()
            model.bank.k = prev_k
            dist.barrier()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                import os
                os.makedirs(f"logs/{run_id_full}", exist_ok=True)
                torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
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
            loss = model(inputs, targets, window_blocks, step, args.num_iterations)
            loss_val = float(loss.detach().item())
            micro_losses.append(loss_val)
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
            (loss / args.grad_accum_steps).backward()

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
            model._pending_active_count = None
        if pending_expert_prune and not experts_pruned and expert_active is not None:
            activity_tensor = expert_active.to(device=model.bank.pruned_experts.device, dtype=torch.float32).clone()
            dist.all_reduce(activity_tensor, op=dist.ReduceOp.AVG)
            pruned_indices = model.bank.prune_inactive_experts(activity_tensor, EXPERT_ACTIVITY_THRESHOLD)
            if pruned_indices:
                if args.enable_extra_logging:
                    print0(
                        f"Pruned experts (activity<{EXPERT_ACTIVITY_THRESHOLD * 100:.1f}%): {pruned_indices}",
                        console=True,
                    )
                if wandb_run is not None:
                    wandb_run.log({
                        "router/pruned_expert_count": len(pruned_indices),
                        "router/pruned_expert_ids": ",".join(str(idx) for idx in pruned_indices),
                    }, step=step)
            else:
                if args.enable_extra_logging:
                    print0("Router freeze pruning skipped (no experts below threshold).", console=True)
            experts_pruned = True
            pending_expert_prune = False

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
            break

        opt2futures = {
            opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                  for p in params if (p.grad is not None)]
            for opt, params in opt2params.items()
        }

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
        for group in optimizers[-1].param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        for opt in optimizers:
            torch.futures.collect_all(opt2futures[opt]).wait()
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
        print0(
            f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
            console=True)
        current_logit_cap = get_logit_cap(args, step)
        current_router_temp = get_router_temp(args, step)
        if wandb_run is not None and (step % max(args.wandb_log_every, 1) == 0):
            if args.enable_extra_wandb_logging:
                log_data = {
                    "train/loss": avg_loss,
                    "train/loss_main": avg_main_loss,
                    "train/loss_aux": avg_aux_loss,
                    "perf/approx_step_time_ms": approx_training_time_ms,
                    "train/tokens_seen": float((step + 1) * args.train_seq_len * world_size),
                    "lr/adamw": optimizers[0].param_groups[0]["lr"],
                    "lr/muon": optimizers[-1].param_groups[0]["lr"],
                    "train/step": step,
                    "router/logit_cap": (current_logit_cap if current_logit_cap is not None else float("nan")),
                    "router/logit_cap_enabled": float(current_logit_cap is not None),
                    "router/temperature": current_router_temp,
                    "router/max_logit": router_step_summary.get("max_logit", float("nan")),
                }
                for key, value in router_step_summary.items():
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
                    active_list = expert_active.tolist()
                    log_data["router_expert_active/min"] = float(min(active_list))
                    log_data["router_expert_active/max"] = float(max(active_list))
                    log_data["router_expert_active/mean"] = float(sum(active_list) / len(active_list))
                    for idx, value in enumerate(active_list):
                        log_data[f"router_expert_active/e{idx}"] = float(value)
                wandb_run.log(log_data, step=step)
            else:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/loss_main": avg_main_loss,
                        "train/loss_aux": avg_aux_loss,
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

    return experts_pruned

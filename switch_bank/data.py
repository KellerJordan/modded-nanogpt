import torch
from pathlib import Path
from torch import Tensor


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, skip_batches: int = 0):
    files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise RuntimeError(f"No data files match pattern '{filename_pattern}' in {Path.cwd()}")
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    try:
        tokens, pos = _load_data_shard(next(file_iter)), 0
    except StopIteration as exc:
        raise RuntimeError(f"No data files available for pattern '{filename_pattern}'") from exc
    # fast-forward if resuming
    while skip_batches > 0:
        if pos + batch_size + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                raise RuntimeError(f"Ran out of data while skipping batches for '{filename_pattern}'")
        pos += batch_size
        skip_batches -= 1

    while True:
        if pos + batch_size + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                return
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets


def summarize_router_metrics(metrics: list[dict[str, float] | None]) -> dict[str, float]:
    summary: dict[str, float] = {}
    counts: dict[str, int] = {}
    for layer_stats in metrics or []:
        if not layer_stats:
            continue
        for key, value in layer_stats.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = float(value.item())
                else:
                    continue
            elif not isinstance(value, (int, float)):
                continue
            summary[key] = summary.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    for key, total in list(summary.items()):
        summary[key] = total / max(counts.get(key, 1), 1)
    return summary


def summarize_expert_usage(metrics: list[dict[str, float] | None], num_experts: int) -> Tensor | None:
    accum: Tensor | None = None
    count = 0
    for layer_stats in metrics or []:
        if not layer_stats:
            continue
        load_vec = layer_stats.get("load_vector")
        if load_vec is None:
            continue
        load_vec = load_vec.to(torch.float32)
        if load_vec.numel() != num_experts:
            continue
        if accum is None:
            accum = load_vec.clone()
        else:
            accum += load_vec
        count += 1
    if accum is None or count == 0:
        return None
    return (accum / count).cpu()


def summarize_expert_activity(metrics: list[dict[str, float] | None], num_experts: int) -> Tensor | None:
    accum: Tensor | None = None
    count = 0
    for layer_stats in metrics or []:
        if not layer_stats:
            continue
        load_vec = layer_stats.get("load_vector")
        if load_vec is None:
            continue
        load_vec = load_vec.to(torch.float32)
        if load_vec.numel() != num_experts:
            continue
        active = (load_vec > 0).to(torch.float32)
        if accum is None:
            accum = active.clone()
        else:
            accum += active
        count += 1
    if accum is None or count == 0:
        return None
    return (accum / count).cpu()


def router_summary_str(summary: dict[str, float], enable_forward_ema: bool, enable_reverse_ema: bool) -> str:
    if not summary:
        return "router=NA"
    fragments = []
    extra_keys: list[str] = []
    if enable_forward_ema:
        extra_keys.append("ema_alpha_forward")
    if enable_reverse_ema:
        extra_keys.append("ema_alpha_reverse")
    keys = ("imp_cv2", "load_cv2", "usage_frac", "topk_prob_mean", *extra_keys, "max_logit")
    for key in keys:
        val = summary.get(key, float("nan"))
        fragments.append(f"{key}={val:.4f}")
    return " ".join(fragments)

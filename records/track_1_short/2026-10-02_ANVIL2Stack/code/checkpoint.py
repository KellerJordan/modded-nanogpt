import hashlib
from pathlib import Path

import torch
import torch.distributed as dist


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root):
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and not path.is_symlink()
        and "logs" not in path.relative_to(root).parts
    }


def save_checkpoint(model, step, training_time_ms, val_loss, val_history, sources):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    root = Path("logs")
    root.mkdir(exist_ok=True)
    entries = [None] * world_size
    entry = None
    for writer in range(0, world_size, 2):
        if writer <= rank < writer + 2:
            filename = f"rank_{rank:02d}.pt"
            tensors = dict(model.state_dict())
            for key, value in model.named_buffers():
                if key.rsplit(".", 1)[-1] in {"canon_mask", "prefix_table", "factor1", "factor2"}:
                    tensors[key] = value
            for key, module in model.named_modules():
                if hasattr(module, "angular_freq") and hasattr(module, "attn_scale"):
                    tensors[key + ".angular_freq"] = module.angular_freq
                    tensors[key + ".attn_scale"] = torch.as_tensor(module.attn_scale, dtype=torch.float64)
            state = {
                key: value.detach().to(device='cpu', copy=True).contiguous()
                for key, value in tensors.items()
            }
            torch.save(state, root / filename)
            del state
            entry = {"rank": int(rank), "file": filename, "sha256": sha256(root / filename)}
        dist.barrier()
    dist.all_gather_object(entries, entry)
    if rank == 0:
        manifest = {
            "step": int(step),
            "train_time_ms": float(training_time_ms),
            "val_loss": float(val_loss),
            "val_tokens": 10485760,
            "val_history": val_history,
            "world_size": int(world_size),
            "source_sha256": sources,
            "shards": entries,
        }
        torch.save(manifest, root / "final_state.pt")
    dist.barrier()

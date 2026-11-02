"""Interleaved full train_gpt.py baseline/candidate runs on an existing 8×H100 machine.

This is the real record trainer (FP8, FA3, ANVIL, sparse comms, CUDA graphs and final validation),
not the reduced CPU fixture. It fails explicitly without the required hardware.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import uuid

import torch


def final_metrics(text):
    rows = re.findall(r"step:(\d+)/(\d+)\s+val_loss:(\S+)\s+train_time:(\S+)ms", text)
    if not rows:
        raise ValueError("missing final training/validation metrics")
    step, total, loss, milliseconds = rows[-1]
    step, total, loss, milliseconds = int(step), int(total), float(loss), float(milliseconds)
    if step != total or total <= 0:
        raise ValueError("training ended before the final validation")
    if not math.isfinite(loss) or not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError("non-finite loss or invalid training time")
    return dict(steps=total, val_loss=loss, train_time_ms=milliseconds)


def require_record_hardware():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("real record validation requires an existing machine with 8 CUDA H100 GPUs")
    names = [torch.cuda.get_device_name(i) for i in range(8)]
    if any("H100" not in name for name in names):
        raise RuntimeError(f"expected 8 H100 GPUs, got {names}")
    if torch.__version__.split("+")[0] not in ("2.10", "2.10.0"):
        raise RuntimeError(f"the certified record uses torch 2.10, got {torch.__version__}")
    return names


def checkout_revisions(paths):
    if len(set(paths.values())) != 2:
        raise ValueError("baseline and candidate must be separate checkouts")
    revisions = {}
    for name, path in paths.items():
        if not (path / "train_gpt.py").is_file():
            raise ValueError(f"{name} checkout has no train_gpt.py")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True)
        if dirty.strip():
            raise ValueError(f"{name} has tracked changes; commit them before benchmarking")
        revisions[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if len(set(revisions.values())) != 2:
        raise ValueError("baseline and candidate must use different commits")
    return revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="separate baseline checkout")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate checkout")
    parser.add_argument("--data-path", type=Path, required=True, help="directory containing data/fineweb10B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=387)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two paired runs are required")
    paths = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    revisions = checkout_revisions(paths)
    hardware = require_record_hardware()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda,
                  cache_policy="fresh compiler caches per run; OS page cache is not flushed",
                  runs=[])
    report["revisions"] = revisions
    try:
        for repeat in range(args.repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for name in order:
                key = f"{repeat}-{name}-{uuid.uuid4().hex[:8]}"
                cache = output / "caches" / key
                env = dict(os.environ, TRAIN_SEED=str(args.seed+repeat), DATA_PATH=str(args.data_path.resolve()),
                           OMP_NUM_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                           TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"))
                env.pop("NUM_SCHEDULED_ITERATIONS", None)  # never shorten training to manufacture a speedup
                log = output / f"{key}.log"
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "train_gpt.py"]
                print(f"Running {name}, seed={args.seed+repeat}; full log: {log}", flush=True)
                with log.open("w") as handle:
                    subprocess.run(command, cwd=paths[name], env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   check=True, timeout=3600)
                metrics = final_metrics(log.read_text())
                report["runs"].append(dict(variant=name, seed=args.seed+repeat, log=str(log), **metrics))
                (output / "results.json").write_text(json.dumps(report, indent=2))
        if len({r["steps"] for r in report["runs"]}) != 1:
            raise RuntimeError("baseline and candidate used different training step counts")
        report["summary"] = {}
        for name in paths:
            rows = [r for r in report["runs"] if r["variant"] == name]
            report["summary"][name] = dict(mean_ms=statistics.mean(r["train_time_ms"] for r in rows),
                                           stdev_ms=statistics.stdev(r["train_time_ms"] for r in rows),
                                           mean_val_loss=statistics.mean(r["val_loss"] for r in rows))
        report["measured_speedup"] = report["summary"]["baseline"]["mean_ms"] / report["summary"]["candidate"]["mean_ms"]
        report["claim"] = "measurements only; assess variance and loss before accepting a record"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import torch


TARGET_LOSS = 2.92
VAL_LOSS_RE = re.compile(r"val_loss:([0-9.]+)")
TRAIN_TIME_RE = re.compile(r"train_time:([0-9.]+)ms")


def _parse_log(path: Path) -> tuple[float, float] | None:
    last_loss = None
    last_time_ms = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "val_loss:" not in line:
                continue
            loss_match = VAL_LOSS_RE.search(line)
            time_match = TRAIN_TIME_RE.search(line)
            if loss_match:
                last_loss = float(loss_match.group(1))
            if time_match:
                last_time_ms = float(time_match.group(1))
    if last_loss is None or last_time_ms is None:
        return None
    return last_loss, last_time_ms


def _p_value_less(values: list[float], mu: float) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    data = torch.tensor(values, dtype=torch.float64)
    mean = data.mean().item()
    std = data.std(unbiased=True).item()
    if std == 0.0:
        return 0.0 if mean < mu else 1.0
    t_stat = (mean - mu) / (std / math.sqrt(n))
    dist = torch.distributions.StudentT(df=n - 1)
    return float(dist.cdf(torch.tensor(t_stat)))


def main() -> int:
    root = Path(__file__).resolve().parent
    logs = sorted(root.glob("*.txt"))
    if not logs:
        print("No .txt logs found.")
        return 1

    losses: list[float] = []
    times_ms: list[float] = []
    skipped = 0
    for path in logs:
        parsed = _parse_log(path)
        if parsed is None:
            skipped += 1
            continue
        loss, time_ms = parsed
        losses.append(loss)
        times_ms.append(time_ms)

    if not losses:
        print("No usable logs found (missing val_loss/train_time).")
        return 1

    mean_loss = float(torch.tensor(losses, dtype=torch.float64).mean().item())
    mean_time_ms = float(torch.tensor(times_ms, dtype=torch.float64).mean().item())
    mean_time_min = mean_time_ms / 1000.0 / 60.0
    p_value = _p_value_less(losses, TARGET_LOSS)

    print(f"runs: {len(losses)} (skipped {skipped})")
    print(f"mean_loss: {mean_loss:.6f}")
    print(f"mean_time_ms: {mean_time_ms:.0f}")
    print(f"mean_time_min: {mean_time_min:.4f}")
    if math.isnan(p_value):
        print("p_value: nan (need at least 2 runs)")
    else:
        print(f"p_value(loss <= {TARGET_LOSS}): {p_value:.6g}")

    summary_path = root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["runs", len(losses)])
        writer.writerow(["skipped", skipped])
        writer.writerow(["mean_loss", f"{mean_loss:.6f}"])
        writer.writerow(["mean_time_ms", f"{mean_time_ms:.0f}"])
        writer.writerow(["mean_time_min", f"{mean_time_min:.6f}"])
        if math.isnan(p_value):
            p_value_str = "nan"
        else:
            p_value_str = f"{p_value:.6g}"
        writer.writerow([f"p_value_loss_le_{TARGET_LOSS}", p_value_str])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

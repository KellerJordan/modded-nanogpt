#!/usr/bin/env python3
"""Recompute loss and paired timing statistics from preserved run logs."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path

from scipy import stats


FINAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P=step) val_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time>\d+)ms",
    re.MULTILINE,
)
TIMING_NAME_RE = re.compile(
    r"^block(?P<block>\d+)-(?P<position>\d+)-"
    r"(?P<arm>baseline|candidate)\.full\.txt$"
)


def final_result(path: Path) -> tuple[float, int]:
    matches = list(FINAL_RE.finditer(path.read_text()))
    if not matches:
        raise ValueError(f"no terminal validation result in {path}")
    match = matches[-1]
    return float(match["loss"]), int(match["time"])


def summarize_losses(name: str, losses: list[float]) -> None:
    if len(losses) < 2:
        raise ValueError(f"{name} requires at least two outcomes")
    result = stats.ttest_1samp(losses, 3.28, alternative="less")
    print(
        f"{name}: n={len(losses)} mean={statistics.mean(losses):.8f} "
        f"sd={statistics.stdev(losses):.8f} min={min(losses):.4f} "
        f"max={max(losses):.4f} one-sided-p={result.pvalue:.8g}"
    )


def analyze_accuracy(directory: Path) -> None:
    files = sorted(directory.glob("*-candidate.full.txt"))
    if not files:
        raise ValueError(f"no accuracy logs found in {directory}")
    results = [final_result(path) for path in files]
    summarize_losses("accuracy gate", [loss for loss, _ in results])
    print(
        f"accuracy runtime: mean={statistics.mean(time for _, time in results):.3f}ms "
        f"sd={statistics.stdev(time for _, time in results):.3f}ms"
    )


def analyze_timing(directory: Path) -> None:
    blocks: dict[int, dict[str, list[tuple[float, int]]]] = {}
    for path in sorted(directory.glob("block*.full.txt")):
        match = TIMING_NAME_RE.match(path.name)
        if not match:
            continue
        block = int(match["block"])
        arm = match["arm"]
        blocks.setdefault(block, {"baseline": [], "candidate": []})[arm].append(
            final_result(path)
        )
    if not blocks:
        raise ValueError(f"no timing logs found in {directory}")

    block_log_ratios: list[float] = []
    all_results: dict[str, list[tuple[float, int]]] = {"baseline": [], "candidate": []}
    for block, arms in sorted(blocks.items()):
        if len(arms["baseline"]) != 2 or len(arms["candidate"]) != 2:
            raise ValueError(f"block {block} does not contain two outcomes per arm")
        for arm in all_results:
            all_results[arm].extend(arms[arm])
        baseline_logs = [math.log(time) for _, time in arms["baseline"]]
        candidate_logs = [math.log(time) for _, time in arms["candidate"]]
        log_ratio = statistics.mean(candidate_logs) - statistics.mean(baseline_logs)
        block_log_ratios.append(log_ratio)
        print(
            f"block {block:02d}: ratio={math.exp(log_ratio):.8f} "
            f"baseline={statistics.mean(time for _, time in arms['baseline']):.3f}ms "
            f"candidate={statistics.mean(time for _, time in arms['candidate']):.3f}ms"
        )

    mean_log_ratio = statistics.mean(block_log_ratios)
    paired_test = stats.ttest_1samp(block_log_ratios, 0.0, alternative="less")
    sem = stats.sem(block_log_ratios)
    radius = stats.t.ppf(0.975, len(block_log_ratios) - 1) * sem
    ratio = math.exp(mean_log_ratio)
    ratio_low = math.exp(mean_log_ratio - radius)
    ratio_high = math.exp(mean_log_ratio + radius)
    print(
        f"paired timing: blocks={len(block_log_ratios)} geometric-ratio={ratio:.8f} "
        f"speedup={(1.0 - ratio) * 100:.4f}% 95%-CI=[{ratio_low:.8f}, {ratio_high:.8f}] "
        f"one-sided-p={paired_test.pvalue:.8g}"
    )
    for arm, results in all_results.items():
        times = [time for _, time in results]
        print(
            f"{arm} runtime: n={len(times)} mean={statistics.mean(times):.3f}ms "
            f"sd={statistics.stdev(times):.3f}ms"
        )
    summarize_losses(
        "timing candidate loss", [loss for loss, _ in all_results["candidate"]]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--timing", type=Path)
    args = parser.parse_args()
    if not args.accuracy and not args.timing:
        parser.error("provide --accuracy and/or --timing")
    if args.accuracy:
        analyze_accuracy(args.accuracy)
    if args.timing:
        analyze_timing(args.timing)


if __name__ == "__main__":
    main()

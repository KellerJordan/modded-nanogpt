#!/usr/bin/env python3
"""Plot every frozen H100 run and each configuration's five-run mean."""

import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
INPUT = HERE / "H100X8_RUNS.csv"
OUTPUT = HERE / "H100X8_TIME_VS_LOSS.svg"
CONFIGS = {
    "baseline": ("Baseline", "#4C78A8", "o"),
    "weighted_r8": ("Weighted, recovery 8", "#F58518", "s"),
    "topological_r8": ("Topological, recovery 8", "#54A24B", "D"),
    "weighted_r6": ("Weighted, recovery 6 (rejected)", "#9D9DA1", "^"),
}


def main() -> None:
    grouped = {name: [] for name in CONFIGS}
    with INPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["configuration"]].append(
                (int(row["run"]), float(row["time_seconds"]), float(row["final_loss"]))
            )

    mpl.rcParams.update({"font.size": 10, "svg.hashsalt": "layer-dropout-h100x8"})
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax.axhline(3.28, color="#B22222", linestyle="--", linewidth=1.2, label="Loss ceiling (3.28)")

    for name, (label, color, marker) in CONFIGS.items():
        rows = grouped[name]
        if not rows:
            continue
        assert len(rows) == 5, f"{name} must contain exactly five official runs"
        times = [row[1] for row in rows]
        losses = [row[2] for row in rows]
        ax.scatter(times, losses, color=color, marker=marker, s=48, alpha=0.82, label=label)
        for run, time_s, loss in rows:
            ax.annotate(str(run), (time_s, loss), xytext=(5, 4), textcoords="offset points", fontsize=7, color=color)
        mean_time = statistics.mean(times)
        mean_loss = statistics.mean(losses)
        ax.errorbar(
            mean_time,
            mean_loss,
            xerr=statistics.stdev(times),
            yerr=statistics.stdev(losses),
            fmt="X",
            markersize=11,
            markeredgecolor="black",
            markeredgewidth=0.8,
            color=color,
            capsize=4,
            linewidth=1.4,
            label="Five-run mean ± 1 sample SD" if name == "baseline" else None,
        )

    ax.set_title("8×H100 layer dropout: baseline, weighted, and topological")
    ax.set_xlabel("Program-reported training time (s; compile/warmup excluded)")
    ax.set_ylabel("Final validation loss")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.savefig(OUTPUT, metadata={"Date": None})


if __name__ == "__main__":
    main()

"""Plot Parallax vs. vanilla-attention val-loss trajectories on the Track-3 benchmark.

Produces two comparable figures in parallax/imgs/:
  1. parallax_all[.zoom].png   -- Parallax attention under every optimizer choice
  2. attention_all[.zoom].png  -- vanilla (softmax) attention under the SAME optimizers

Each optimizer keeps a fixed color across both figures so they overlay 1:1.

Curve data:
  - Parallax curves come from parallax/results/<variant>/seed*.txt  (seeds averaged).
  - Vanilla curves come from the matched official Track-3 record log(s) under
    records/track_3_optimization/results/ (all seed files in the record dir averaged,
    same convention as that benchmark's make_figures.py).

Each curve is drawn up to its "steps to val<3.28" crossing (seed-mean), and the crossing
step is shown in the legend -- matching the benchmark figure convention.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  TextArea, VPacker)

REPO_ROOT = Path(__file__).resolve().parent.parent
PARALLAX_RESULTS = REPO_ROOT / "parallax" / "results"
VANILLA_RESULTS = REPO_ROOT / "records" / "track_3_optimization" / "results"
OUT_DIR = REPO_ROOT / "parallax" / "imgs"

STEP_PATTERN = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)")
TARGET = 3.28

# One entry per optimizer choice. parallax_cross / vanilla_cross are the documented
# seed-mean "steps to val<3.28" (see parallax/README.md). vanilla_log is a path under
# VANILLA_RESULTS; if it points into a sub-directory, every *.txt in that dir is averaged
# (matching records/track_3_optimization/make_figures.py).
VARIANTS = [
    dict(key="soaph",       label="SOAP-H (#27)",            parallax_dir="rec27_soaph",
         parallax_cross=2880, vanilla_log="20260518_soaph/SOAPH_run1.txt",                     vanilla_cross=3125),
    dict(key="dynmuon",     label="DynMuon (#28)",           parallax_dir="rec28_dynmuon",
         parallax_cross=2975, vanilla_log="20260519_dynmuon/50172610-d038-4f90-9a12-b9a0853f035d.txt", vanilla_cross=3175),
    dict(key="contra_soft", label="Contra-Soft-Muon (#20)",  parallax_dir="rec20_contra_soft",
         parallax_cross=3000, vanilla_log="20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt", vanilla_cross=3030),
    dict(key="aurora",      label="Aurora (#17)",            parallax_dir="rec17_aurora",
         parallax_cross=3025, vanilla_log="20260505_aurora/298f02bc-dbb4-4661-9ad8-f6429d532873.txt", vanilla_cross=3175),
    dict(key="sinksoap",    label="SinkSOAP (#26)",          parallax_dir="rec26_sinksoap",
         parallax_cross=3025, vanilla_log="20260514_sinksoap/d0155dd0-f77d-48a9-8eb4-453f894b9476.txt", vanilla_cross=3090),
    dict(key="trustlight",  label="trustlight (#16)",        parallax_dir="rec16_trustlight",
         parallax_cross=3052, vanilla_log="20260506_trustlight/fake_log_from_seed0.txt",       vanilla_cross=3125),
    dict(key="soap_mlp",    label="SOAP-MLP (#14)",          parallax_dir="rec14_soap_mlp",
         parallax_cross=3100, vanilla_log="20260504_contra_muon_mlp_soapish/0248394b-0d6c-4133-9ff7-e7ff2763cdd9.txt", vanilla_cross=3150),
    dict(key="split_cd",    label="split-cooldown (#24)",    parallax_dir="rec24_split_cd",
         parallax_cross=3125, vanilla_log="20260509_contra_muon_split_cooldown/c1af0bd1-6999-44d1-a618-3d1234ea32f0.txt", vanilla_cross=3175),
]

COLORS = plt.colormaps["tab10"].colors  # 10 distinct, one per variant (<=10)


def parse_logfile(path):
    """Return list of (steps, losses) runs in a log file (a step:0 line starts a new run)."""
    runs, steps, losses = [], [], []
    with open(path) as f:
        for line in f:
            m = STEP_PATTERN.search(line)
            if not m:
                continue
            step, loss = int(m.group(1)), float(m.group(3))
            if step == 0 and steps:
                runs.append((steps, losses))
                steps, losses = [], []
            steps.append(step)
            losses.append(loss)
    if steps:
        runs.append((steps, losses))
    return runs


def average_runs(runs):
    """Return (steps, means, stds) across seeds; std is 0 where only one seed covers a step."""
    by_step = defaultdict(list)
    for steps, losses in runs:
        for s, l in zip(steps, losses):
            by_step[s].append(l)
    steps = sorted(by_step)
    means, stds = [], []
    for s in steps:
        vals = by_step[s]
        m = sum(vals) / len(vals)
        means.append(m)
        stds.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
    return steps, means, stds


def logfile_paths(base, logfile):
    """Mirror the Track-3 make_figures rule: a file directly under results/ is used as-is;
    a file inside a sub-directory globs every *.txt in that sub-directory (seed average)."""
    path = base / logfile
    return [path] if path.parent == base else sorted(path.parent.glob("*.txt"))


def load_curve(base, logfile, cross):
    runs = []
    for p in logfile_paths(base, logfile):
        runs.extend(parse_logfile(p))
    if not runs:
        raise RuntimeError(f"No loss curve found for {base / logfile}")
    steps, means, stds = average_runs(runs)  # full trajectory, to the end of the record
    return list(steps), list(means), list(stds)


def curves_for(kind):
    """Yield (color_index, variant, steps, means, stds, cross) for every optimizer."""
    for i, v in enumerate(VARIANTS):
        if kind == "parallax":
            steps, means, stds = load_curve(PARALLAX_RESULTS, f"{v['parallax_dir']}/seed0.txt", v["parallax_cross"])
            cross = v["parallax_cross"]
        else:
            steps, means, stds = load_curve(VANILLA_RESULTS, v["vanilla_log"], v["vanilla_cross"])
            cross = v["vanilla_cross"]
        yield i, v, steps, means, stds, cross


def shared_zoom_ylim(zoom, margin=0.01):
    """y-range covering the mean±std bands of both kinds inside the zoom window."""
    los, his = [], []
    for kind in ("parallax", "vanilla"):
        for _, _, steps, means, stds, cross in curves_for(kind):
            if cross > zoom[1]:
                continue
            for s, m, sd in zip(steps, means, stds):
                if zoom[0] <= s <= zoom[1]:
                    los.append(m - sd)
                    his.append(m + sd)
    return min(los) - margin, max(his) + margin


def build_legend(ax, title, rows):
    """Floating-window legend with bold step counts and a bold-green improvement %.

    rows: list of (color, label, cross, suffix, pct) where pct is the % step reduction
    vs. softmax attention (None to omit, e.g. on the softmax panel itself).
    """
    fs = 7.5

    def ta(s, **kw):
        return TextArea(s, textprops={"fontsize": fs, **kw})

    row_boxes = []
    for color, label, cross, suffix, pct in rows:
        handle = DrawingArea(22, 10, 0, 0)
        handle.add_artist(Line2D([1, 11, 21], [5, 5, 5], marker="o", markevery=[1],
                                 markersize=3, linewidth=2, color=color))
        parts = [ta(f"{label} → "), ta(str(cross), fontweight="bold")]
        if suffix:
            parts.append(ta(suffix))
        if pct is not None:
            parts.append(ta(f"  {pct:+.1f}%", fontweight="bold", color="green"))
        text = HPacker(children=parts, align="baseline", pad=0, sep=0)
        row_boxes.append(HPacker(children=[handle, text], align="center", pad=0, sep=5))

    body = VPacker(children=[ta(title, fontsize=8), *row_boxes], align="left", pad=0, sep=3)
    box = AnchoredOffsetbox(loc="upper right", child=body, frameon=True, borderpad=0.8)
    box.patch.set(facecolor="white", edgecolor="0.4", linewidth=1.0, alpha=1.0)
    box.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
    box.patch.set_path_effects([patheffects.withSimplePatchShadow(offset=(2, -2), alpha=0.3)])
    box.set_zorder(20)
    ax.add_artist(box)


def make_figure(kind, out_name, title, zoom=None, ylim=None):
    """kind in {'parallax','vanilla'}; zoom = (min_step, max_step) or None for full view."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    legend_rows = []
    for i, v, steps, means, stds, cross in curves_for(kind):
        if zoom and cross > zoom[1]:
            continue  # crossing past the zoom window -> skip for legibility
        n_seeds = max(1, len(logfile_paths(
            PARALLAX_RESULTS if kind == "parallax" else VANILLA_RESULTS,
            f"{v['parallax_dir']}/seed0.txt" if kind == "parallax" else v["vanilla_log"])))
        suffix = f" (n={n_seeds})" if any(sd > 0 for sd in stds) else ""
        # % step reduction of Parallax vs. softmax attention (shown on the Parallax panel).
        pct = (v["vanilla_cross"] - cross) / v["vanilla_cross"] * 100 if kind == "parallax" else None
        legend_rows.append((COLORS[i], v["label"], cross, suffix, pct))
        ax.plot(steps, means, marker="o", markersize=3, linewidth=2, color=COLORS[i])
        if any(sd > 0 for sd in stds):  # ±1 std band for multi-seed curves
            lo = [m - sd for m, sd in zip(means, stds)]
            hi = [m + sd for m, sd in zip(means, stds)]
            ax.fill_between(steps, lo, hi, color=COLORS[i], alpha=0.2, linewidth=0)

    ax.axhline(TARGET, color="gray", linestyle="--", linewidth=1.3)
    label_x = zoom[1] if zoom else 3600
    ax.annotate("target=3.28", xy=(label_x, TARGET), xytext=(-8, 6),
                textcoords="offset points", color="gray", fontsize=9, ha="right")
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_xlabel("Training steps @ 0.5M bsz", fontsize=11)
    ax.set_ylabel("Validation loss", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    if zoom:
        ax.set_xlim(*zoom)
        if ylim:
            ax.set_ylim(*ylim)
    else:
        ax.set_xlim(0, 3600)
        ax.set_ylim(3.2, 3.85)

    legend_title = ("optimizer → steps to <3.28  (% vs. softmax)"
                    if kind == "parallax" else "optimizer → steps to <3.28")
    build_legend(ax, legend_title, legend_rows)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / out_name}")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    make_figure("parallax", "parallax_all.png",
                "Modded-NanoGPT Parallax — all optimizers")
    make_figure("vanilla", "attention_all.png",
                "Modded-NanoGPT Softmax Attention — same optimizers")
    # Zoomed crossing region (where curves meet the 3.28 target); shared y-axis so the
    # two panels are directly comparable.
    zoom = (2800, 3360)
    ylim = (3.25, 3.34)
    make_figure("parallax", "parallax_all_zoom.png",
                "Modded-NanoGPT - Parallax - crossing region", zoom=zoom, ylim=ylim)
    make_figure("vanilla", "attention_all_zoom.png",
                "Modded-NanoGPT - Softmax Attention - crossing region", zoom=zoom, ylim=ylim)


if __name__ == "__main__":
    main()

"""Draws stratified_softmax.png: the idea, the schedule and the measured step times.

    pip install matplotlib
    python records/track_1_short/2026-11-15_StratifiedSoftmaxTail/figure.py

The schedule and set sizes are the constants of track_1_short/stratified_softmax.py and
sampled_softmax.py; the step times are read from the run logs in this_pr/ and baseline/.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
RECORD, THIS_PR = "#b5b4ae", "#2a78d6"            # the two versions
EXACT, TAIL, LEFT_OUT = "#1baf7a", "#6250d6", "#e6e5e0"  # the kinds of output ids
VOCAB, RECORD_WIDTH, EXACT_IDS, TAIL_IDS = 50304, 24576, 6144, 2048
STEPS = [0, 681, 965, 1107, 1194]                  # where the candidate count changes; the run's end
RECORD_COLUMNS = [10240, 14336, RECORD_WIDTH, VOCAB]
PR_COLUMNS = [10240, 14336, EXACT_IDS + TAIL_IDS, VOCAB]
STEP_LINE = re.compile(r"^step:(\d+)/1194 train_time:(\d+)ms", re.M)


def window_ms_per_step(folder):
    """Mean over a pool's runs of the time per step in each logged window: (window end steps, ms)."""
    runs = []
    for path in sorted((ROOT / folder).glob("*.txt")):
        marks = sorted({int(s): int(t) for s, t in STEP_LINE.findall(path.read_text(errors="replace"))}.items())
        runs.append([(s1, (t1 - t0) / (s1 - s0)) for (s0, t0), (s1, t1) in zip([(0, 0)] + marks, marks)])
    ends = [end for end, _ in runs[0]]
    ends = [end for end in ends if end % 25 == 0]  # the 25-step windows; the last two marks are shorter
    return ends, [sum(run[i][1] for run in runs) / len(runs) for i in range(len(ends))], len(runs)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heading(ax, title, subtitle):
    for text, lift, kwargs in [(title, 30, dict(fontsize=12.5, fontweight="bold", color=INK)),
                               (subtitle, 12, dict(fontsize=9.5, color=INK2))]:
        ax.annotate(text, (0, 1), (0, lift), xycoords="axes fraction", textcoords="offset points", va="bottom", **kwargs)


fig = plt.figure(figsize=(10, 12.4), facecolor=SURFACE)
grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.12, 1.0], hspace=0.5, left=0.075, right=0.965, top=0.865, bottom=0.05)
fig.text(0.075, 0.965, "Stratified sampled softmax on the widest steps", fontsize=16, fontweight="bold", color=INK)
fig.text(0.075, 0.931, "Record #92 trains steps 965-1106 on 24,576 output ids per GPU. This PR trains them on 8,192:\n"
         "6,144 exact ids plus 2,048 random ones that are weighted to stand for all the ids left out.",
         fontsize=10, color=INK2, linespacing=1.5)

# 1. The schedule.
ax = fig.add_subplot(grid[0])
style(ax)
heading(ax, "1. Output ids in the training loss, by step",
        "Columns of the lm_head each row is scored against. Everything outside steps 965-1106 is the record's.")
ax.step(STEPS, RECORD_COLUMNS + RECORD_COLUMNS[-1:], where="post", color=RECORD, linewidth=5, solid_capstyle="butt", label="Record #92")
ax.step(STEPS, PR_COLUMNS + PR_COLUMNS[-1:], where="post", color=THIS_PR, linewidth=2, label="This PR")
ax.fill_between([965, 1107], PR_COLUMNS[2], RECORD_WIDTH, color=THIS_PR, alpha=0.10, linewidth=0)
for x, y, text, dy in [(340, 10240, "10,240", 1700), (823, 14336, "14,336", 1700), (1036, RECORD_WIDTH, "24,576  record", 1700),
                       (1036, PR_COLUMNS[2], "8,192  this PR", -4300), (1150, VOCAB, "50,304", -4300)]:
    ax.text(x, y + dy, text, ha="center", fontsize=9.5, color=INK)
ax.text(1150, VOCAB - 8600, "full softmax", ha="center", fontsize=8.5, color=INK2)
ax.text(1036, 15900, "3 x fewer\ncolumns", ha="center", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0, 1194)
ax.set_ylim(0, 56000)
ax.set_xticks(STEPS)
ax.set_yticks(range(0, 50001, 10000), [f"{v // 1000}K" if v else "0" for v in range(0, 50001, 10000)])
ax.set_xlabel("training step", fontsize=9, color=INK2)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)

# 2. How a pruned step builds its sets.
ax = fig.add_subplot(grid[1])
ax.set_facecolor(SURFACE)
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
heading(ax, "2. One pruned step on one GPU",
        "The model runs once over the GPU's 49,152 rows. Only the loss is split, into 3 chunks, each with its own set of ids.")
for k in range(3):
    ax.add_patch(Rectangle((k * 33.6, 40.5), 32.8, 5.5, color="#dbe7f6", linewidth=0))
    ax.text(k * 33.6 + 16.4, 43.25, f"chunk {k + 1}:  16,384 rows", ha="center", va="center", fontsize=9.5, color=INK)
ax.add_patch(FancyArrowPatch((50, 40.0), (50, 34.6), arrowstyle="-|>", mutation_scale=12, color=INK2, linewidth=1))
ax.text(52, 37.2, "each chunk picks 8,192 of the 50,304 output ids", fontsize=9.5, color=INK, va="center")
scale = 100 / VOCAB
exact_w, tail_w = EXACT_IDS * scale, TAIL_IDS * scale
ax.add_patch(Rectangle((0, 28), exact_w - 0.25, 6, color=EXACT, linewidth=0))
ax.add_patch(Rectangle((exact_w, 28), tail_w - 0.25, 6, color=TAIL, linewidth=0))
ax.add_patch(Rectangle((exact_w + tail_w, 28), 100 - exact_w - tail_w, 6, color=LEFT_OUT, linewidth=0))
ax.text((exact_w + tail_w + 100) / 2, 31, "42,112 ids left out of this step", ha="center", va="center", fontsize=9.5, color=INK2)
rows = [(exact_w / 2, EXACT, 21.0, "6,144 exact ids, weight 1",
         "every target in the chunk (at most about 5,300), filled up with this GPU's most frequent targets so far"),
        (exact_w + tail_w / 2, TAIL, 12.5, "2,048 random ids, weight 21.6",
         "drawn uniformly from the other 44,160 ids, so each one stands for 44,160 / 2,048 = 21.6 of them")]
for x, color, y, title, detail in rows:
    ax.plot([x, x, 19.2], [28, y, y], color=INK2, linewidth=0.8)
    ax.add_patch(Rectangle((19.6, y - 1.1), 2.2, 2.2, color=color, linewidth=0))
    ax.text(23, y + 0.2, title, fontsize=10, fontweight="bold", color=INK, va="center")
    ax.text(23, y - 3.4, detail, fontsize=9, color=INK2, va="center")
ax.text(0, 2.6, r"normalizer  $\approx$  $\sum_{\mathrm{exact}} e^{z}$  +  21.6 $\times$ $\sum_{\mathrm{random}} e^{z}$",
        fontsize=12, color=INK, va="center")
ax.text(47, 2.6, "an unbiased estimate of the sum over all 50,304 ids,\nso rare ids are still pushed down with the right strength on average",
        fontsize=9, color=INK2, va="center")

# 3. The measured step times.
ax = fig.add_subplot(grid[2])
style(ax)
ends, record_ms, record_runs = window_ms_per_step("baseline")
_, pr_ms, pr_runs = window_ms_per_step("this_pr")
heading(ax, "3. Measured time per step on 8xH100",
        f"Mean of {pr_runs} runs of this PR and {record_runs} runs of record #92, interleaved on one node. "
        "25-step windows, startup window omitted.")
ax.axvspan(965, 1107, color=THIS_PR, alpha=0.07, linewidth=0)
ax.plot(ends[1:], record_ms[1:], color=RECORD, linewidth=5, solid_capstyle="round", label="Record #92")
ax.plot(ends[1:], pr_ms[1:], color=THIS_PR, linewidth=2, label="This PR")
inside = [i for i, end in enumerate(ends) if 990 <= end <= 1100]
record_in, pr_in = (sum(series[i] for i in inside) / len(inside) for series in (record_ms, pr_ms))
ax.annotate(f"{record_in:.1f} ms", (1045, record_in), (1045, record_in + 5.5), ha="center", fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.annotate(f"{pr_in:.1f} ms", (1045, pr_in), (1045, pr_in - 9), ha="center", fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.text(20, 4.5, "Whole run: 39.702 s against 40.235 s, 0.533 s faster (1.32%), at +0.62 millinats of validation loss",
        fontsize=9.5, color=INK)
ax.set_xlim(0, 1194)
ax.set_ylim(0, 58)
ax.set_xticks(STEPS)
ax.set_yticks(range(0, 51, 10))
ax.set_ylabel("ms per step", fontsize=9, color=INK2)
ax.set_xlabel("training step", fontsize=9, color=INK2)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)

fig.savefig(ROOT / "stratified_softmax.png", dpi=160, facecolor=SURFACE)
print("wrote", ROOT / "stratified_softmax.png", f"pruned-window mean: record {record_in:.2f} ms, this PR {pr_in:.2f} ms")

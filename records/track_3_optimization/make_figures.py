import re
import math
import colorsys
import hashlib
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


# Extract results
logfiles = {
    # key: number in README results history
    # value: label or (label, color)
    6: ('Muon', '#ffa500'),
    # 2: ('AdamW', '#1f77b4'),
    5: ('MuonH', '#2ca02c'),
    4: ('AdamH', '#9467bd'),
    7: ('Muon²', '#e377c2'),
    8: ('NorMuonH', '#32CD32'),
    9: 'NorMuon with update-clamping strategy',
    10: ('NorMuon', '#7e1e56'),
    11: 'Nor-Contra-Muon with update-clamping strategy',
}
readme_rows = {}
row_pattern = re.compile(
    r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|.*?\|\s*\[log\]\((results/[^)]+)\)'
)
with open('README.md', 'r') as f:
    for line in f:
        m = row_pattern.search(line)
        if m:
            number = int(m.group(1))
            steps_to_target = int(m.group(2))
            evidence = m.group(3).strip()
            logfile = m.group(4).removeprefix('results/').removesuffix('.txt')
            readme_rows[number] = (steps_to_target, evidence, logfile)
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')


def color_from_title(title):
    digest = hashlib.md5(title.encode('utf-8')).hexdigest()
    hue = int(digest[:8], 16) / 0xffffffff
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.8)
    return f'#{round(255 * r):02x}{round(255 * g):02x}{round(255 * b):02x}'


def format_evidence(evidence, steps_to_target):
    evidence = re.sub(r'[✓✔✅Ⓧ]', '', evidence).strip()
    return re.sub(r'\((n=\d+)\)', f'in {steps_to_target} steps (\\1)', evidence)


def add_legend(ax, legend_entries):
    rows = []
    for label, evidence, color in legend_entries:
        handle = DrawingArea(26, 18, 0, 0)
        handle.add_artist(Line2D(
            [3, 13, 23],
            [9, 9, 9],
            marker='o',
            markevery=[1],
            markersize=3.5,
            linewidth=2.2,
            color=color,
        ))
        label_size = 7 if len(label) > 25 else 9
        text = VPacker(
            children=[
                TextArea(label, textprops={'fontsize': label_size}),
                TextArea(evidence, textprops={'fontsize': 7}),
            ],
            align='left',
            pad=0,
            sep=1,
        )
        rows.append(HPacker(
            children=[handle, text],
            align='center',
            pad=0,
            sep=5,
        ))
    legend = AnchoredOffsetbox(
        loc='upper right',
        child=VPacker(children=rows, align='left', pad=0, sep=3),
        frameon=True,
        pad=0.15,
        borderpad=0.75,
    )
    legend.patch.set_alpha(plt.rcParams['legend.framealpha'])
    legend.patch.set_edgecolor(plt.rcParams['legend.edgecolor'])
    legend.patch.set_linewidth(1.0)
    if plt.rcParams['legend.fancybox']:
        legend.patch.set_boxstyle('round,pad=0.15,rounding_size=0.2')
    ax.add_artist(legend)


def get_logfile_paths(logfile):
    path = Path(f'results/{logfile}.txt')
    if path.parent == Path('results'):
        return [path]
    return sorted(path.parent.glob('*.txt'))


def parse_logfile(path):
    runs = []
    steps, losses = [], []
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                if step == 0 and steps:
                    runs.append((steps, losses))
                    steps, losses = [], []
                steps.append(step)
                losses.append(loss)
    if steps:
        runs.append((steps, losses))
    return runs


def average_runs(runs):
    losses_by_step = defaultdict(list)
    for steps, losses in runs:
        for step, loss in zip(steps, losses):
            losses_by_step[step].append(loss)
    steps = sorted(losses_by_step)
    losses = [
        sum(losses_by_step[step]) / len(losses_by_step[step])
        for step in steps
    ]
    return steps, losses


max_step = 0
results = {}
for number, entry in logfiles.items():
    if isinstance(entry, tuple):
        label, color = entry
    else:
        label = entry
        color = color_from_title(label)
    if number not in readme_rows:
        raise RuntimeError(f'No results-history row found in README for #{number}')
    steps_to_target, evidence, logfile = readme_rows[number]
    runs = []
    for path in get_logfile_paths(logfile):
        runs.extend(parse_logfile(path))
    if not runs:
        raise RuntimeError(f'No loss curve found for results/{logfile}.txt')
    steps, losses = average_runs(runs)
    kept_points = [
        (step, loss)
        for step, loss in zip(steps, losses)
        if step <= steps_to_target
    ]
    if not kept_points:
        raise RuntimeError(f'No loss curve points found at or before step {steps_to_target} for results/{logfile}.txt')
    steps, losses = zip(*kept_points)

    max_step = max(max_step, max(steps))
    results[number] = (label, steps_to_target, evidence, steps, losses, color)


# Generate figure
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
legend_entries = []
for label, steps_to_target, evidence, steps, losses, color in results.values():
    legend_entries.append((label, f'→ {format_evidence(evidence, steps_to_target)}', color))
    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=3.5,
        linewidth=2.2,
        color=color,
    )
ax.axhline(3.28, color='gray', linestyle='--', linewidth=1.5)
ax.annotate(
    'target=3.28',
    xy=(0, 3.28),
    xytext=(8, 6),
    textcoords='offset points',
    color='gray',
    fontsize=9,
)
ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/01', pad=11, fontsize=11)
ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
add_legend(ax, legend_entries)
ax.set_xlim(0, math.ceil(max_step / 1000) * 1000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout()
fig.savefig('figure.png', bbox_inches='tight')


# Generate zoomed-in figure
zoom_min_step = 3000
zoom_max_step = 3500
zoom_results = [
    result for result in results.values()
    if result[1] < zoom_max_step
]
zoom_losses = [
    loss
    for _, _, _, steps, losses, _ in zoom_results
    for step, loss in zip(steps, losses)
    if zoom_min_step <= step <= zoom_max_step
]
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
legend_entries = []
for label, steps_to_target, evidence, steps, losses, color in zoom_results:
    legend_entries.append((label, f'→ {format_evidence(evidence, steps_to_target)}', color))
    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=3.5,
        linewidth=2.2,
        color=color,
    )
ax.axhline(3.28, color='gray', linestyle='--', linewidth=1.5)
ax.annotate(
    'target=3.28',
    xy=(zoom_min_step, 3.28),
    xytext=(8, 6),
    textcoords='offset points',
    color='gray',
    fontsize=9,
)
ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/01', pad=11, fontsize=11)
ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
add_legend(ax, legend_entries)
ax.set_xlim(zoom_min_step, zoom_max_step)
if zoom_losses:
    zoom_margin = 0.01
    ax.set_ylim(min(zoom_losses) - zoom_margin, max(zoom_losses) + zoom_margin)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout()
fig.savefig('zoomed_figure.png', bbox_inches='tight')

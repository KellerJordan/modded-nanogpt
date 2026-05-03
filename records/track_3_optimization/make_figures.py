import re
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


# Extract results
logfiles = {
    # key: number in README results history
    # value: (label, color)
    6: ('Muon', '#ffa500'),
    2: ('AdamW', '#1f77b4'),
    5: ('MuonH', '#2ca02c'),
    4: ('AdamH', '#9467bd'),
    7: ('Muon²', '#e377c2'),
    8: ('NorMuonH', '#32CD32'),
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
for number, (label, color) in logfiles.items():
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
    results[number] = (f'{label} ({steps_to_target} steps)', steps_to_target, evidence, steps, losses, color)


# Generate figure
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
for label, _, _, steps, losses, color in results.values():
    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=3.5,
        linewidth=2.2,
        label=label,
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
ax.legend(frameon=True)
ax.set_xlim(0, math.ceil(max_step / 1000) * 1000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout()
out = 'figure.png'
fig.savefig(out, bbox_inches='tight')
print(out)


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
for label, _, evidence, steps, losses, color in zoom_results:
    evidence = re.sub(r'[✓✔✅]', '', evidence).strip()
    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=3.5,
        linewidth=2.2,
        label=f'{label}\n→ {evidence}',
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
ax.legend(frameon=True)
ax.set_xlim(zoom_min_step, zoom_max_step)
if zoom_losses:
    zoom_margin = 0.01
    ax.set_ylim(min(zoom_losses) - zoom_margin, max(zoom_losses) + zoom_margin)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout()
out = 'figure_zoom.png'
fig.savefig(out, bbox_inches='tight')
print(out)

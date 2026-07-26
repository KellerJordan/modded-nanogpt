import re
import math
import random
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


parser = argparse.ArgumentParser()
parser.add_argument(
    '--colors',
    choices=('pyplot', 'extended'),
    default='pyplot',
    help='Color cycle to use: pyplot default, or the previous shuffled extended palette.',
)
args = parser.parse_args()

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
extended_color_cycle = [
    *plt.colormaps['tab20'].colors,
    *plt.colormaps['tab20b'].colors,
    *plt.colormaps['tab20c'].colors,
]
random.Random(46).shuffle(extended_color_cycle)
pyplot_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = pyplot_color_cycle if args.colors == 'pyplot' else extended_color_cycle


def get_results(logfiles):
    max_step = 0
    max_date = ''
    results = {}
    for i, (number, label) in enumerate(logfiles.items()):
        # label = f"#{number}: {label}"
        color = color_cycle[i % len(color_cycle)]
        if number not in readme_rows:
            raise RuntimeError(f'No results-history row found in README for #{number}')
        steps_to_target, evidence, date, logfile = readme_rows[number]
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
        max_date = max(max_date, date)
        results[number] = (label, steps_to_target, evidence, steps, losses, color)
    return results, max_step, max_date


def format_evidence(evidence, steps_to_target):
    evidence = re.sub(r'[✓✔✅Ⓧ]', '', evidence).strip()
    return re.sub(r'\((n=\d+)\)', f'in {steps_to_target} steps (\\1)', evidence)


def add_legend(ax, legend_entries):
    if not legend_entries:
        return None
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
        label_lines = label.splitlines()
        title_size = 6.5 if any(len(line) > 25 for line in label_lines) else 8
        label_texts = [
            TextArea(line, textprops={'fontsize': title_size})
            for line in label_lines
        ]
        text = VPacker(
            children=[
                *label_texts,
                TextArea(evidence, textprops={'fontsize': 6.5}),
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
        loc='upper left',
        child=VPacker(children=rows, align='left', pad=0, sep=3),
        frameon=True,
        pad=0.15,
        borderpad=0.75,
        bbox_to_anchor=(1.02, 1),
        bbox_transform=ax.transAxes,
    )
    legend.patch.set_alpha(plt.rcParams['legend.framealpha'])
    legend.patch.set_edgecolor(plt.rcParams['legend.edgecolor'])
    legend.patch.set_linewidth(1.0)
    if plt.rcParams['legend.fancybox']:
        legend.patch.set_boxstyle('round,pad=0.15,rounding_size=0.2')
    legend.set_in_layout(False)
    ax.add_artist(legend)
    return legend


def shift_legend_left_half_width(ax, legend, anchor_x=0.95, anchor_y=1.01, width_fraction=0.9):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    legend_width = legend.get_window_extent(renderer=renderer).width
    axes_width = ax.get_window_extent(renderer=renderer).width
    legend.set_bbox_to_anchor(
        (anchor_x - width_fraction * legend_width / axes_width, anchor_y),
        transform=ax.transAxes,
    )


def plot_results(ax, plot_results, target_label_x, title_date):
    legend_entries = []
    for label, steps_to_target, evidence, steps, losses, color in plot_results:
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
        xy=(target_label_x, 3.28),
        xytext=(8, 6),
        textcoords='offset points',
        color='gray',
        fontsize=9,
    )
    ax.set_title(f'Modded-NanoGPT Optimization Benchmark as of {title_date}', pad=11, fontsize=11)
    ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
    ax.set_ylabel('Validation loss', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)
    return add_legend(ax, legend_entries)


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


readme_rows = {}
log_link_pattern = re.compile(r'\[log\]\((results/[^)]+)\)')
with open('README.md', 'r') as f:
    for line in f:
        if not (line.startswith('| ') and len(line) > 2 and line[2].isdigit()):
            continue
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) != 7:
            raise RuntimeError(f'Expected 7 columns in README results row: {line}')
        number, steps, evidence, _description, date, log_cell, _authorship = cells
        m = log_link_pattern.search(log_cell)
        if not m:
            raise RuntimeError(f'No log link found in README results row: {line}')
        steps_m = re.match(r'\d+', steps)
        if not steps_m:
            raise RuntimeError(f'No step count found in README results row: {line}')
        logfile = m.group(1).removeprefix('results/').removesuffix('.txt')
        readme_rows[int(number)] = (int(steps_m.group()), evidence, date, logfile)
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')

for suffix in ["wr", "best"]:
    # key: number in README results history
    # value: label
    # Include results with lowest step count and a few baselines
    if suffix == "wr":
        logfiles = {
            1: 'Muon (bad hparams)',
            3: 'Muon (less bad hparams)',
            5: 'Muon-Hyperball',
            9: 'NorMuon-UpdateClampMin',
            11: '#9 + ContraMuon',
            13: '#8 + MuLoCo',
            14: '#11 + SOAPMuon on MLP',
            16: '#14 + SOAPMuon on attn proj w/ trust gate',
            20: '#16 + power law lr sched + SoftMuon',
        }
    elif suffix == "best":
        logfiles = {
            12: 'Muon-W',
            4: 'Adam-Hyperball',
            9: '#9: NorMuon-UpdateClampMin',
            11: '#11: #9 + ContraMuon',
            16: '#11 + SOAPMuon',
            17: '#11 + Aurora',
            15: 'NewtonMuon-W',
            19: 'KLSOAP-Hyperball',
            21: 'Shampoo-W',
            22: 'Ortho-W',
            23: 'Muon-RowNormControl',
        }
    else:
        assert False

    results, max_step, title_date = get_results(logfiles)

    # Generate figure
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
    legend = plot_results(ax, results.values(), 0, title_date)
    ax.set_xlim(0, 3800)
    ax.set_ylim(3.2, 3.85)
    fig.tight_layout()
    shift_legend_left_half_width(ax, legend)
    fig.savefig(f'img/figure_{suffix}.png', bbox_inches='tight', bbox_extra_artists=[legend])

    # Generate zoomed-in figure
    zoom_min_step = 2800
    zoom_max_step = {"wr": 3650, "best": 3400}[suffix]
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
    legend = plot_results(ax, zoom_results, zoom_min_step, title_date)
    ax.set_xlim(zoom_min_step, zoom_max_step)
    if zoom_losses:
        zoom_margin = 0.01
        ax.set_ylim(min(zoom_losses) - zoom_margin, max(zoom_losses) + zoom_margin)
    fig.tight_layout()
    shift_legend_left_half_width(ax, legend)
    fig.savefig(f'img/zoomed_figure_{suffix}.png', bbox_inches='tight', bbox_extra_artists=[legend])

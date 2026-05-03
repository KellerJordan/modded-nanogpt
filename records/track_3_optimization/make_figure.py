import re
import math
import matplotlib.pyplot as plt


# Extract results
logfiles = {
    # key: number in README results history
    # value: (label, color)
    6: ('Muon (best, 3375 steps)', '#ffa500'),
    2: ('AdamW (best, 5625 steps)', '#1f77b4'),
    5: ('MuonH (best, 3325 steps)', '#2ca02c'),
    4: ('AdamH (best, 4875 steps)', '#9467bd'),
    7: ('Muon² (best, 3325 steps)', '#e377c2'),
    8: ('NorMuonH (best, 3250 steps)', '#32CD32'),
}
readme_rows = {}
row_pattern = re.compile(
    r'^\|\s*(\d+)\s*\|.*?\|\s*\[log\]\((results/[^)]+)\)'
)
with open('README.md', 'r') as f:
    for line in f:
        m = row_pattern.search(line)
        if m:
            number = int(m.group(1))
            logfile = m.group(2).removeprefix('results/').removesuffix('.txt')
            readme_rows[number] = logfile
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')
max_step = 0
results = {}
for number, (label, color) in logfiles.items():
    if number not in readme_rows:
        raise RuntimeError(f'No results-history row found in README for #{number}')
    logfile = readme_rows[number]
    steps, losses = [], []
    path = f'results/{logfile}.txt'
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                if step == 0:
                    # there may be multiple runs in the logfile, take the last one
                    steps, losses = [], []
                steps.append(step)
                losses.append(loss)
    if not steps:
        raise RuntimeError(f'No loss curve found in {path}')

    max_step = max(max_step, max(steps))
    results[label] = (steps, losses, color)


# Generate figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=180)
for label, (steps, losses, color) in results.items():
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
ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/01', pad=12, fontsize=12)
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
from pathlib import Path
import re
import matplotlib.pyplot as plt

runs = {
    'Muon (best, 3500 steps)': ('311d7833-8dfc-43ea-a55c-fd313a11c4a8.txt', '#d04a1f'),
    'AdamW (best, 5625 steps)': ('a63a68d1-24aa-4a22-af9a-224e43209ea4.txt', '#1f77b4'),
}
out = Path('figure.png')
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=180)

for label, (logfile, color) in runs.items():
    steps, losses = [], []
    path = Path('results') / logfile
    with path.open() as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    if not steps:
        raise RuntimeError(f'No loss curve found in {path}')

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

ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/04/26', pad=12, fontsize=12)
ax.set_xlabel('Training step', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(frameon=True)
ax.set_xlim(left=0, right=6000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
print(out)


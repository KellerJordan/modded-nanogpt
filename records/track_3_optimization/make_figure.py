import re
import matplotlib.pyplot as plt

runs = {
    'Muon (best, 3500 steps)': ('311d7833-8dfc-43ea-a55c-fd313a11c4a8', '#d04a1f'),
    'AdamW (best, 5625 steps)': ('a63a68d1-24aa-4a22-af9a-224e43209ea4', '#1f77b4'),
    'MuonH (best, 3325 steps)': ('20260430_muonh/9319c798-6643-464a-b407-b05468e468f5', '#2ca02c'),
    'AdamH (best, 4875 steps)': ('20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac', '#9467bd'),
}
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')
out = 'figure.png'

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=180)

for label, (logfile, color) in runs.items():
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

ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/04/30', pad=12, fontsize=12)
ax.set_xlabel('Training step (0.5M bsz)', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(frameon=True)
ax.set_xlim(0, 6000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
print(out)

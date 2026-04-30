import re
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

REFS = {
    "AdamW (5625 steps)":      ("a63a68d1-24aa-4a22-af9a-224e43209ea4", "#1f77b4"),
    "Muon (3500 steps)":       ("311d7833-8dfc-43ea-a55c-fd313a11c4a8", "#d04a1f"),
}
SKYLIGHT_RUNS = [
    "23bb1400-4864-4f25-a5af-12443ea613bf",  # primary (best margin: 3.27661)
    "b09d60d9-44be-4631-b9ad-949e7db7a8a2",  # 3.27993
    "8bab61b6-9744-44f6-bb2a-34567ee7fe31",  # 3.27782
    "d005264f-7481-4f85-968e-d51938d214dd",  # 3.27776
    "b27ace07-9738-4d91-a50d-6ea920ab87c8",  # 3.27799
    "e792d794-4205-41d0-9c4d-25863fa5d88e",  # 3.27895
]
SKYLIGHT_COLOR = "#2ca02c"
OUT = os.path.join(HERE, "figure.png")
pattern = re.compile(r"step:(\d+)/\d+\s+val_loss:([0-9.]+)")


def load(uuid):
    path = os.path.join(RESULTS, f"{uuid}.txt")
    s, v = [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                s.append(int(m.group(1)))
                v.append(float(m.group(2)))
    return np.array(s), np.array(v)


plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=180)

# Reference baselines
for label, (uuid, color) in REFS.items():
    s, v = load(uuid)
    ax.plot(s, v, marker='o', markersize=3.0, linewidth=2.0, label=label, color=color, alpha=0.9)

# Skylight-001: aggregate of 6 seeds (mean + min/max band)
all_v = []
for uuid in SKYLIGHT_RUNS:
    s, v = load(uuid)
    all_v.append(v)
all_v = np.stack(all_v)
sky_mean = all_v.mean(axis=0)
sky_min = all_v.min(axis=0)
sky_max = all_v.max(axis=0)
ax.fill_between(s, sky_min, sky_max, color=SKYLIGHT_COLOR, alpha=0.2,
                label="Skylight-001 (n=6 seeds, min–max)")
ax.plot(s, sky_mean, marker='o', markersize=3.2, linewidth=2.4,
        label="Skylight-001 (n=6 mean, 3250 steps)", color=SKYLIGHT_COLOR)

ax.axhline(3.28, color='gray', linestyle='--', linewidth=1.5)
ax.annotate('target=3.28', xy=(0, 3.28), xytext=(8, 6), textcoords='offset points',
            color='gray', fontsize=9)
ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/04/29',
             pad=12, fontsize=12)
ax.set_xlabel('Training step', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(frameon=True, fontsize=9, loc='upper right')
ax.set_xlim(0, 6000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
print(OUT)

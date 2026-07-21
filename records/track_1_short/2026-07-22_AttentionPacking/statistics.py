import re
import math
from pathlib import Path

from scipy.stats import ttest_1samp


ROOT = Path(__file__).parent
FINAL_RE = re.compile(
    r"step:(?P<steps>\d+)/\1 val_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time_ms>\d+)ms"
)
SEED_RE = re.compile(r"seed=(?P<seed>\d+) steps=1410")


def mean(values):
    return sum(values) / len(values)


def sample_stdev(values):
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def final_metrics(path):
    text = path.read_text()
    matches = list(FINAL_RE.finditer(text))
    assert len(matches) == 1, path
    match = matches[0]
    return int(match["steps"]), float(match["loss"]), int(match["time_ms"]) / 1000


baseline_paths = sorted((ROOT / "baseline_h200").glob("*.txt"))
baseline = [
    (path.name, *final_metrics(path))
    for path in baseline_paths
]
assert baseline
assert {row[1] for row in baseline} == {1390}
assert all(
    path.read_text().count("NVIDIA H200") >= 8
    for path in baseline_paths
)

candidate = []
candidate_paths = sorted((ROOT / "candidate_h200").glob("*.txt"))
for path in candidate_paths:
    text = path.read_text()
    seed_match = SEED_RE.search(text)
    assert seed_match, path
    steps, loss, time_s = final_metrics(path)
    candidate.append((int(seed_match["seed"]), steps, loss, time_s))
candidate.sort()
assert [row[0] for row in candidate] == list(range(12))
assert {row[1] for row in candidate} == {1410}
assert all(
    path.read_text().count("NVIDIA H200") >= 8
    for path in candidate_paths
)

losses = [row[2] for row in candidate]
times = [row[3] for row in candidate]
baseline_losses = [row[2] for row in baseline]
baseline_times = [row[3] for row in baseline]
test = ttest_1samp(losses, 3.28, alternative="less")
mean_time = mean(times)
baseline_mean_time = mean(baseline_times)

for name, steps, loss, time_s in baseline:
    print(f"baseline {name}: steps={steps} loss={loss:.4f} time={time_s:.3f}s")
for seed, steps, loss, time_s in candidate:
    print(f"seed={seed}: steps={steps} loss={loss:.4f} time={time_s:.3f}s")
print(f"baseline mean loss: {mean(baseline_losses):.9f}")
print(f"baseline mean time: {baseline_mean_time:.6f}s")
print(f"candidate mean loss: {mean(losses):.9f}")
print(f"candidate loss stddev: {sample_stdev(losses):.9f}")
print(f"one-sided t: {test.statistic:.9f}")
print(f"one-sided p: {test.pvalue:.9f}")
print(f"candidate mean time: {mean_time:.6f}s")
print(f"candidate time stddev: {sample_stdev(times):.6f}s")
print(f"time delta: {mean_time - baseline_mean_time:.6f}s")
print(f"relative time delta: {(mean_time / baseline_mean_time - 1) * 100:.4f}%")

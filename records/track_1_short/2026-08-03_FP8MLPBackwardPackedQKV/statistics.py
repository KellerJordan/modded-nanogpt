import math
import re
from pathlib import Path

from scipy.stats import ttest_1samp


ROOT = Path(__file__).parent
EXPECTED_SEEDS = [2, 4, 42, 1337]
EXPECTED_STEPS = 1315
FINAL_RE = re.compile(
    r"step:(?P<steps>\d+)/\1 val_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time_ms>\d+)ms"
)
RUN_RE = re.compile(
    r"attention heads=\d+ value_dim=\d+ qk_dim=\d+ "
    r"seed=(?P<seed>\d+) steps=(?P<steps>\d+)"
)


def mean(values):
    return sum(values) / len(values)


def sample_stdev(values):
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def parse_run(path):
    text = path.read_text()
    run_matches = list(RUN_RE.finditer(text))
    final_matches = list(FINAL_RE.finditer(text))
    assert len(run_matches) == 1, path
    assert len(final_matches) == 1, path
    assert text.count("NVIDIA H100") >= 8, path

    run = run_matches[0]
    final = final_matches[0]
    seed = int(run["seed"])
    run_steps = int(run["steps"])
    final_steps = int(final["steps"])
    assert run_steps == final_steps == EXPECTED_STEPS, path
    return seed, final_steps, float(final["loss"]), int(final["time_ms"]) / 1000


paths = sorted((ROOT / "candidate_h100").glob("*.txt"))
runs = sorted(parse_run(path) for path in paths)
assert [run[0] for run in runs] == EXPECTED_SEEDS

losses = [run[2] for run in runs]
times = [run[3] for run in runs]
test = ttest_1samp(losses, 3.28, alternative="less")

for seed, steps, loss, time_s in runs:
    print(f"seed={seed}: steps={steps} loss={loss:.4f} time={time_s:.3f}s")
print(f"mean loss: {mean(losses):.9f}")
print(f"loss stddev: {sample_stdev(losses):.9f}")
print(f"one-sided t: {test.statistic:.9f}")
print(f"one-sided p: {test.pvalue:.9f}")
print(f"mean time: {mean(times):.6f}s")
print(f"time stddev: {sample_stdev(times):.6f}s")

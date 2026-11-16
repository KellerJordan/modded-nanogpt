"""Statistics of a run pool: this PR's logs in this_pr/, the record's same-node logs in baseline/.

    pip install scipy
    python records/track_1_short/2026-11-15_StratifiedSoftmaxTail/pool_statistics.py

Every log in the two folders counts; none is dropped. Prints, per pool, the mean and sample std of the
final validation loss and of the training time; for the candidate the one-sided p-value that its mean
loss is <= 3.28 (rule 2 needs p < 0.01); and the time difference to the baseline with a one-sided Welch
test that the candidate is faster (rule 4).
"""
import math
import re
from pathlib import Path

from scipy.stats import ttest_1samp, ttest_ind

ROOT = Path(__file__).parent
GATE = 3.28
EXPECTED_STEPS = 1194
FINAL_RE = re.compile(r"step:(?P<steps>\d+)/(?P=steps) val_loss:(?P<loss>[0-9.]+) train_time:(?P<time_ms>\d+)ms")
STRATIFIED_ON = re.compile(r"stratified softmax: steps \d+-\d+ run ")  # the printed line, not the source


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def parse_run(path, stratified):
    text = path.read_text(errors="replace")
    finals = list(FINAL_RE.finditer(text))
    assert len(finals) == 1, f"{path}: expected one final validation line, found {len(finals)}"
    final = finals[0]
    assert int(final["steps"]) == EXPECTED_STEPS, f"{path}: {final['steps']} steps"
    assert text.count("H100") >= 8, f"{path}: not an 8xH100 log"
    assert bool(STRATIFIED_ON.search(text)) == stratified, f"{path}: wrong pool (stratified softmax on/off)"
    return float(final["loss"]), int(final["time_ms"]) / 1000


def pool(name, stratified):
    paths = sorted((ROOT / name).glob("*.txt"))
    assert len(paths) >= 2, f"{name}/: at least two run logs are needed, found {len(paths)}"
    runs = [parse_run(path, stratified) for path in paths]
    for path, (loss, seconds) in zip(paths, runs):
        print(f"  {name}/{path.name}: val_loss={loss:.4f} train_time={seconds:.3f}s")
    losses, times = [run[0] for run in runs], [run[1] for run in runs]
    print(f"{name}: runs={len(runs)} val_loss={mean(losses):.6f} +/- {sample_std(losses):.6f} "
          f"train_time={mean(times):.3f} +/- {sample_std(times):.3f} s")
    return losses, times


candidate_losses, candidate_times = pool("this_pr", stratified=True)
baseline_losses, baseline_times = pool("baseline", stratified=False)

gate = ttest_1samp(candidate_losses, GATE, alternative="less")
faster = ttest_ind(candidate_times, baseline_times, equal_var=False, alternative="less")
saved = mean(baseline_times) - mean(candidate_times)
print(f"rule 2: candidate mean val_loss <= {GATE}: t={gate.statistic:.3f} one-sided p={gate.pvalue:.3e} "
      f"({'PASS' if gate.pvalue < 0.01 else 'NOT YET'}: needs p < 0.01)")
print(f"rule 4: candidate faster than the same-node baseline by {saved:.3f} s "
      f"({100 * saved / mean(baseline_times):.2f}%), one-sided Welch p={faster.pvalue:.3e}")
print(f"loss change vs baseline: {1000 * (mean(candidate_losses) - mean(baseline_losses)):+.2f} millinats")

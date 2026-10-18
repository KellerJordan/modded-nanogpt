"""Recompute this record's statistics from the run logs in this folder.

    python summarize.py

Reads the final "step:N/N val_loss:X train_time:Yms" line of every log in certification/, no_barrier/,
baseline_master/ and baseline_379/. Prints per-folder mean ± sd, the README's one-sided t-test of the
certification runs' mean val loss against 3.28, and Welch tests of train time against the baselines.
"""
import glob
import os
import re
import statistics as st

import scipy.stats

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = re.compile(r"step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms")


def load(folder):
    runs = []
    for path in sorted(glob.glob(os.path.join(HERE, folder, "*.txt"))):
        last = None
        with open(path) as f:
            for line in f:
                m = FINAL.search(line)
                if m and m.group(1) == m.group(2):
                    last = m
        if last:
            runs.append((float(last.group(3)), int(last.group(4)) / 1000))
    return runs


def main():
    arms = {k: load(k) for k in ("certification", "no_barrier", "baseline_master", "baseline_379")}
    for k, runs in arms.items():
        loss, t = [r[0] for r in runs], [r[1] for r in runs]
        sd = (lambda x: st.stdev(x) if len(x) > 1 else float("nan"))
        print(f"{k:16s} n={len(runs):2d}  train_time {st.mean(t):.3f} ± {sd(t):.3f} s   val_loss {st.mean(loss):.5f} ± {sd(loss):.5f}")
    loss = [r[0] for r in arms["certification"]]
    p = scipy.stats.ttest_1samp(loss, 3.28, alternative="less").pvalue
    print(f"\ncertification: one-sided t-test of mean val_loss < 3.28: p = {p:.2e} (n = {len(loss)})")
    t = [r[1] for r in arms["certification"]]
    for k in ("baseline_master", "baseline_379"):
        b = [r[1] for r in arms[k]]
        w = scipy.stats.ttest_ind(t, b, equal_var=False)
        print(f"train_time vs {k}: {st.mean(t) - st.mean(b):+.3f} s, Welch p = {w.pvalue:.1e}")


if __name__ == "__main__":
    main()

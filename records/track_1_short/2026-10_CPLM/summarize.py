"""Summarize record runs: final val loss and train time per log, mean/std, and the one-sided t-test vs 3.28.

Usage: python summarize.py logs/*.txt   (the logs/<run_id>.txt files train_gpt.py writes)
"""
import re, sys, statistics

import scipy.stats

losses, times = [], []
for path in sys.argv[1:]:
    final = None
    for line in open(path):
        m = re.match(r"step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", line)
        if m and m[1] == m[2]:
            final = (float(m[3]), int(m[4]) / 1000)
    if final is None:
        print(f"{path}: no final val line (incomplete run?)"); continue
    losses.append(final[0]); times.append(final[1])
    print(f"{path}: val_loss {final[0]:.4f}  train_time {final[1]:.3f} s")
if len(losses) > 1:
    print(f"n={len(losses)}  val_loss mean {statistics.mean(losses):.4f} (std {statistics.stdev(losses):.4f})"
          f"  p={scipy.stats.ttest_1samp(losses, 3.28, alternative='less').pvalue:.4f}")
    print(f"train_time mean {statistics.mean(times):.3f} s (std {statistics.stdev(times):.3f})")

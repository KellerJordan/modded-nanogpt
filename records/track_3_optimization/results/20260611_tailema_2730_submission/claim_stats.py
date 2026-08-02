#!/usr/bin/env python3
"""Track-3 significance stats over a set of run logs.

Usage: claim_stats.py <stdout-or-log files...>

Parses baseline `step:S/... val_loss:V` lines and `readout step:S alpha:A
val_loss:V` sweep lines, then for every (step, alpha) cell present in ALL
files reports mean, (3.28-mean)*sqrt(n), and the n required for a claim
((3.28-mean)*sqrt(n) >= 0.004) at the current mean.
"""
import math
import re
import sys

REQ = 0.004
files = sys.argv[1:]
assert files, __doc__

per_file = []
for path in files:
    cells = {}
    for line in open(path):
        m = re.match(r"step:(\d+)/\d+ val_loss:([\d.]+)", line)
        if m:
            cells[(int(m.group(1)), None)] = float(m.group(2))
        m = re.match(r"readout step:(\d+) alpha:([\d.]+) val_loss:([\d.]+)", line)
        if m:
            cells[(int(m.group(1)), float(m.group(2)))] = float(m.group(3))
    per_file.append(cells)

common = set(per_file[0])
for c in per_file[1:]:
    common &= set(c)

n = len(per_file)
print(f"n = {n} runs; significance requires (3.28-mean)*sqrt(n) >= {REQ}")
print(f"{'step':>5} {'alpha':>6} {'mean':>9} {'sig':>8} {'pass':>5} {'n_req':>6}")
for step, alpha in sorted(common, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
    vals = [c[(step, alpha)] for c in per_file]
    mean = sum(vals) / n
    margin = 3.28 - mean
    sig = margin * math.sqrt(n)
    n_req = math.ceil((REQ / margin) ** 2) if margin > 0 else float("inf")
    a = "base" if alpha is None else f"{alpha:g}"
    print(f"{step:>5} {a:>6} {mean:.6f} {sig:+.5f} {'PASS' if sig >= REQ else '----':>5} {n_req:>6}")

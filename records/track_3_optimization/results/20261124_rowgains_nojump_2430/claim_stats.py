"""Check this submission's claim from its logfiles (CPU only, standard library only).

For every logfile (``*.txt``) in this folder: check that the bytes before the line of 100 ``=`` are
the submitted trainer plus the one newline its ``print`` adds, read the seed and start time from the
header, and read ``val_ema_loss`` at every validation, refusing a malformed, non-finite, repeated or
out-of-order validation line. Every seed was run exactly once; a repeated seed is an error.

Then, for the pooled 16 seeds and for each pre-registered batch of 8, print the mean loss and the
Track 3 statistic ``(3.28 - mean) * sqrt(n)`` at every validation step from 2,400 to 2,450, and the
README's pairwise statistic against the published results listed in ``PUBLISHED``. Finally check
every part of the claim and exit 1 if any check fails.

Usage, from this folder::

    python3 claim_stats.py                     # print the tables; exit 1 if any check fails
    python3 claim_stats.py --summary out.tsv   # also write the per-run table
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_gpt_rowgains_nojump.py"
SEPARATOR = "=" * 100

CLAIM_STEP = 2430
STEPS = list(range(2400, 2455, 5))
# Seeds were fixed before any run (two pre-registered batches; see README).
BATCHES = {
    "batch A (seeds 5001-5008)": set(range(5001, 5009)),
    "batch B (seeds 6001-6008)": set(range(6001, 6009)),
}

# Published means used for the README's pairwise statistic, with their sources:
#: #395: summary.tsv of PR #395, the 39 rows with counted == "yes", column "val_ema_loss_2450".
#: #398: README of PR #398, table "mean val_ema (n = 16)", row 2635.
#: #46:  as quoted in PR #395's claim_stats.py (summary of record #46, column "step 2690").
PUBLISHED = (
    ("#395 (open)", 2450, 3.27888, 39),
    ("#398 (open)", 2635, 3.27888, 16),
    ("#46 (accepted)", 2690, 3.27832875, 8),
)
LOSS_PER_STEP = 0.0045 / 100  # README: 0.0045 val loss per 100 steps

VAL_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:(\S+) val_ema_loss:(\S+) ")


def parse_log(path: Path, code: str) -> tuple[int, str, dict[int, float]]:
    text = path.read_text()
    head, sep, body = text.partition("\n" + SEPARATOR + "\n")
    if not sep:
        raise ValueError(f"{path.name}: no separator line")
    if head + "\n" != code + "\n" and head != code:
        raise ValueError(f"{path.name}: embedded code differs from {TRAINER.name}")
    seed_m = re.search(r"^Using seed=(\d+)$", body, re.M)
    time_m = re.search(r"^Run UTC timestamp=(\S+)$", body, re.M)
    if not seed_m or not time_m:
        raise ValueError(f"{path.name}: missing seed or timestamp")
    vals: dict[int, float] = {}
    last = -1
    for line in body.splitlines():
        if not line.startswith("step:") or " val_loss:" not in line:
            continue
        m = VAL_RE.match(line)
        if not m:
            raise ValueError(f"{path.name}: malformed validation line: {line!r}")
        step, ema = int(m.group(1)), float(m.group(4))
        if not math.isfinite(ema):
            raise ValueError(f"{path.name}: non-finite val_ema_loss at step {step}")
        if step <= last:
            raise ValueError(f"{path.name}: repeated or out-of-order validation at step {step}")
        vals[step], last = ema, step
    return int(seed_m.group(1)), time_m.group(1), vals


def stats(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else float("nan")
    return mean, sd, (3.28 - mean) * math.sqrt(n)


def pairwise(mean: float, n: int, step: int, ref_steps: int, ref_mean: float, ref_n: int) -> float:
    return ((ref_mean - mean) + (ref_steps - step) * LOSS_PER_STEP) / math.sqrt(1 / n + 1 / ref_n)


def table(name: str, runs: dict[int, dict[int, float]]) -> int | None:
    print(f"\n== {name}: n={len(runs)}")
    header = "step   mean     sd       stat     " + " ".join(f"{p[0]:>15}" for p in PUBLISHED)
    print(header)
    first = None
    for s in STEPS:
        if not all(s in v for v in runs.values()):
            continue
        mean, sd, stat = stats([v[s] for v in runs.values()])
        ok = stat >= 0.004
        if ok and first is None:
            first = s
        pw = " ".join(f"{pairwise(mean, len(runs), s, rs, rm, rn):15.5f}" for _, rs, rm, rn in PUBLISHED)
        print(f"{s}  {mean:.5f}  {sd:.5f}  {stat:.5f}{' *' if ok else '  '} {pw}")
    print(f"earliest step with (3.28 - mean) * sqrt(n) >= 0.004: {first}")
    return first


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--summary", type=Path, help="write the per-run table to this TSV file")
    args = parser.parse_args()

    code = TRAINER.read_text()
    runs: dict[int, dict[int, float]] = {}
    rows = []
    for path in sorted(HERE.glob("*.txt")):
        seed, started, vals = parse_log(path, code)
        if seed in runs:
            raise ValueError(f"seed {seed} appears more than once")
        runs[seed] = vals
        rows.append((path.name, seed, started, max(vals), vals))

    failures = []
    expected = set().union(*BATCHES.values())
    if set(runs) != expected:
        failures.append(f"seeds {sorted(runs)} differ from the pre-registered {sorted(expected)}")

    claim = table("all pre-registered seeds (the claim)", runs)
    for name, seeds in BATCHES.items():
        table(name, {s: runs[s] for s in sorted(seeds) if s in runs})

    if claim != CLAIM_STEP:
        failures.append(f"earliest passing step over all seeds is {claim}, not {CLAIM_STEP}")
    mean, _, _ = stats([v[CLAIM_STEP] for v in runs.values()])
    print(f"\nAt the claimed step {CLAIM_STEP} (n={len(runs)}): mean {mean:.5f}")
    for name, rs, rm, rn in PUBLISHED:
        pw = pairwise(mean, len(runs), CLAIM_STEP, rs, rm, rn)
        print(f"  pairwise vs {name:15} ({rs} steps, mean {rm:.5f}, n={rn}): {pw:.5f} "
              f"{'>= 0.004' if pw >= 0.004 else '< 0.004 (not pairwise significant)'}")

    if args.summary:
        with open(args.summary, "w") as f:
            f.write("log\tseed\tbatch\tstarted_utc\tlast_step\t" + "\t".join(f"val_ema_loss_{s}" for s in STEPS) + "\n")
            for name, seed, started, last, vals in rows:
                batch = "A" if seed in BATCHES["batch A (seeds 5001-5008)"] else "B"
                f.write(f"{name}\t{seed}\t{batch}\t{started}\t{last}\t"
                        + "\t".join(f"{vals[s]:.5f}" if s in vals else "" for s in STEPS) + "\n")

    if failures:
        print("\nCLAIM CHECK FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nclaim check passed: {CLAIM_STEP} steps, n={len(runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

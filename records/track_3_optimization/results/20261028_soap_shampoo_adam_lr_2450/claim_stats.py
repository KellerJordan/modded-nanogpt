"""Check this submission's claim from its logfiles (CPU only, standard library only).

For every logfile (``*.txt``) in this folder: check that the bytes before the line of 100 ``=`` are the
submitted trainer plus the one newline its ``print`` adds, read the seed and start time from the
header, and read ``val_ema_loss`` at every validation, refusing a malformed, non-finite, repeated
or out-of-order validation line. A seed run more than once counts once, by its earliest run.

Then, for each pool of seeds, print the mean loss and upstream's statistic
``(3.28 - mean) * sqrt(n)`` at steps 2,440 to 2,455 (a step is used only when every seed in the
pool logged it) and the pairwise statistic against the published results of #46, #384 and #388,
and check every part of the claim. The pools come from the logs; which seeds chose the trainer
and which formed the fixed batch, and the published means, are written below with their sources.

Usage, from this folder::

    python3 claim_stats.py                     # print the tables; exit 1 if any check fails
    python3 claim_stats.py --summary out.tsv   # also write the per-run table
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_gpt_soap_shampoo_adam_lr_2450.py"
STEPS = (2440, 2445, 2450, 2455)
TARGET = 3.28
BAR = 0.004
#: Loss per step upstream uses to compare results at different step counts (0.0045 per 100 steps).
LOSS_PER_STEP = 0.000045
#: The six runs on whose results this trainer was chosen (README, "Every run").
SELECTION_SEEDS = frozenset(
    {1882862862, 1010139989, 731620423, 2106907334, 523501693, 812330076}
)
#: The 8-seed batch fixed before it ran (README, "Every run").
FIXED_BATCH_SEEDS = frozenset(
    {
        1027527099,
        2130788512,
        1692227747,
        149068086,
        1624839566,
        275970201,
        1212769246,
        1645686566,
    }
)
#: Published results: (name, steps, mean val_ema_loss there, seeds). Each mean is the average of
#: the per-seed table its write-up publishes:
#: #46: records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/README.md, seeds 0-7,
#:      column "step 2690".
#: #384: summary.tsv of PR #384 (results/20261009_spectral_annealing_2540/), seeds 0-13, column
#:      "step2540".
#: #388: README.md of PR #388 (results/20261015_soap_shampoo_damping/), confirmation seeds 49-64,
#:      Tail-EMA table, column "step 2615".
PUBLISHED = (
    ("#46 (accepted)", 2690, 3.27832875, 8),
    ("#384 (open)", 2540, 3.27878571, 14),
    ("#388 (open)", 2615, 3.278969375, 16),
)
#: What this submission claims, checked by check_claim().
CLAIM_STEP = 2450
EXPECTED_LOGS = 50
EXPECTED_SEEDS = 39
#: Each pool's first passing step, as the README's first table states it; every step of
#: STEPS up to it must be logged by every seed in the pool.
EXPECTED_FIRST_PASS = {
    "all seeds": 2445,
    "without the 6 choosing runs": 2450,
    "fixed 8-seed batch": 2455,
}

SEPARATOR = b"\n" + b"=" * 100 + b"\n"
NUMBER = r"([0-9]+\.[0-9]+)"
VALIDATION = re.compile(
    rf"step:(\d+)/(\d+) val_loss:{NUMBER} val_ema_loss:{NUMBER}(?: .*)?"
)


class LogError(ValueError):
    """A logfile that cannot be read as evidence."""


@dataclass(frozen=True)
class Run:
    """One training run, read from its logfile."""

    log: str
    seed: int
    started: str
    code_matches: bool
    ema: dict[int, float]

    @property
    def last_step(self) -> int:
        return max(self.ema)


def read_run(path: Path, trainer: bytes) -> Run:
    """Read one logfile; raise LogError on anything that is not clean evidence."""
    data = path.read_bytes()
    cut = data.find(SEPARATOR)
    if cut < 0:
        raise LogError(f"{path.name}: no line of 100 '=' after the code")
    code_matches = data[: cut + 1] == trainer + b"\n"
    seeds: list[int] = []
    starts: list[str] = []
    ema: dict[int, float] = {}
    for line in data[cut + len(SEPARATOR) :].decode().split("\n"):
        if line.startswith("Using seed="):
            seeds.append(int(line.split("=", 1)[1]))
        elif line.startswith("Run UTC timestamp="):
            starts.append(line.split("=", 1)[1])
        elif line.startswith("step:") and "val_" in line:
            match = VALIDATION.fullmatch(line)
            if match is None:
                raise LogError(f"{path.name}: malformed validation line {line!r}")
            step, value = int(match.group(1)), float(match.group(4))
            if not math.isfinite(value):
                raise LogError(f"{path.name}: non-finite loss at step {step}")
            if ema and step <= max(ema):
                raise LogError(f"{path.name}: step {step} repeated or out of order")
            ema[step] = value
    if len(seeds) != 1 or len(starts) != 1 or not ema:
        raise LogError(
            f"{path.name}: needs one seed line, one start line and validations"
        )
    return Run(path.name, seeds[0], starts[0], code_matches, ema)


def read_runs(folder: Path = HERE, trainer: Path = TRAINER) -> list[Run]:
    code = trainer.read_bytes()
    runs = sorted(
        (read_run(p, code) for p in folder.glob("*.txt")),
        key=lambda r: r.started,
    )
    keys = [(r.seed, r.started) for r in runs]
    if len(set(keys)) != len(keys):
        raise LogError("two logfiles share a seed and a start time")
    return runs


def one_per_seed(runs: list[Run]) -> list[Run]:
    """The earliest run of each seed; ``runs`` must be sorted by start time."""
    first: dict[int, Run] = {}
    for run in runs:
        first.setdefault(run.seed, run)
    return list(first.values())


@dataclass(frozen=True)
class StepResult:
    step: int
    n: int
    mean: float
    statistic: float

    @property
    def passes(self) -> bool:
        return self.statistic >= BAR


def at_step(pool: list[Run], step: int) -> StepResult | None:
    """The pool's result at ``step``, or None when the pool is empty or a seed did not log it."""
    if not pool or any(step not in run.ema for run in pool):
        return None
    mean = sum(run.ema[step] for run in pool) / len(pool)
    return StepResult(step, len(pool), mean, (TARGET - mean) * math.sqrt(len(pool)))


def first_pass(pool: list[Run]) -> StepResult | None:
    """The earliest step every seed in the pool logged at which the pool meets the rule."""
    if not pool:
        return None
    for step in sorted(set.intersection(*(set(run.ema) for run in pool))):
        result = at_step(pool, step)
        if result is not None and result.passes:
            return result
    return None


def pairwise(new: StepResult, steps: int, mean: float, n: int) -> float:
    """Upstream's pairwise statistic of ``new`` against a published result."""
    gap = (mean - new.mean) + (steps - new.step) * LOSS_PER_STEP
    return gap / math.sqrt(1 / n + 1 / new.n)


def pools(runs: list[Run]) -> dict[str, list[Run]]:
    seeds = one_per_seed(runs)
    return {
        "all seeds": seeds,
        "without the 6 choosing runs": [
            r for r in seeds if r.seed not in SELECTION_SEEDS
        ],
        "fixed 8-seed batch": [r for r in seeds if r.seed in FIXED_BATCH_SEEDS],
    }


def check_claim(runs: list[Run]) -> list[str]:
    """Every part of the claim that does not hold, in words; empty when the claim holds."""
    failures = [
        f"{r.log}: its code is not the submitted trainer"
        for r in runs
        if not r.code_matches
    ]
    seeds = {r.seed for r in runs}
    if len(runs) != EXPECTED_LOGS or len(seeds) != EXPECTED_SEEDS:
        failures.append(
            f"expected {EXPECTED_LOGS} logs on {EXPECTED_SEEDS} seeds, found {len(runs)} on {len(seeds)}"
        )
    for name, wanted in (
        ("choosing", SELECTION_SEEDS),
        ("fixed-batch", FIXED_BATCH_SEEDS),
    ):
        if not wanted <= seeds:
            failures.append(
                f"{name} seeds missing from the logs: {sorted(wanted - seeds)}"
            )
    p = pools(runs)
    for name, expected in EXPECTED_FIRST_PASS.items():
        missing = [s for s in STEPS if s <= expected and at_step(p[name], s) is None]
        if missing:
            failures.append(f"{name}: steps {missing} are not logged by every seed")
        found = first_pass(p[name])
        if found is None or found.step != expected:
            got = found.step if found else "none"
            failures.append(f"{name}: first passing step is {got}, not {expected}")
    for name in ("all seeds", "without the 6 choosing runs"):
        result = at_step(p[name], CLAIM_STEP)
        if result is None or not result.passes:
            failures.append(f"{name}: does not meet the rule at {CLAIM_STEP}")
    claim = at_step(p["all seeds"], CLAIM_STEP)
    if claim is not None:
        for name, steps, mean, n in PUBLISHED:
            if pairwise(claim, steps, mean, n) < BAR:
                failures.append(f"pairwise against {name}: below {BAR}")
    return failures


def write_summary(runs: list[Run], path: Path) -> None:
    counted = {id(r) for r in one_per_seed(runs)}
    header = [
        "log",
        "seed",
        "started_utc",
        "last_step",
        "counted",
        "choosing_run",
        "fixed_batch",
    ]
    header += [f"val_ema_loss_{s}" for s in STEPS]
    rows = ["\t".join(header)]
    for run in runs:
        cells = [
            run.log,
            str(run.seed),
            run.started,
            str(run.last_step),
            "yes" if id(run) in counted else "no (repeat of an earlier seed)",
            "yes" if run.seed in SELECTION_SEEDS else "no",
            "yes" if run.seed in FIXED_BATCH_SEEDS else "no",
        ]
        cells += [f"{run.ema[s]:.5f}" if s in run.ema else "-" for s in STEPS]
        rows.append("\t".join(cells))
    path.write_text("\n".join(rows) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--summary", type=Path, help="also write the per-run table here"
    )
    args = parser.parse_args(argv)
    runs = read_runs()
    print(f"{len(runs)} logfiles, {len(one_per_seed(runs))} distinct seeds")
    for name, pool in pools(runs).items():
        cells = []
        for step in STEPS:
            result = at_step(pool, step)
            cells.append(
                f"{step}: not logged by every seed"
                if result is None
                else f"{step}: {result.mean:.5f} {result.statistic:.5f}{' pass' if result.passes else ''}"
            )
        found = first_pass(pool)
        print(
            f"{name} (n={len(pool)}): "
            + " | ".join(cells)
            + f" | first passing step: {found.step if found else 'none'}"
        )
    claim = at_step(pools(runs)["all seeds"], CLAIM_STEP)
    if claim is not None:
        for name, steps, mean, n in PUBLISHED:
            print(
                f"pairwise against {name}, {steps} steps: {pairwise(claim, steps, mean, n):.4f}"
            )
    if args.summary:
        write_summary(runs, args.summary)
    failures = check_claim(runs)
    for failure in failures:
        print("FAILED:", failure)
    print(
        "claim holds"
        if not failures
        else f"claim does not hold ({len(failures)} checks failed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

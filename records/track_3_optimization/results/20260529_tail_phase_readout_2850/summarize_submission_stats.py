#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path


VAL_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)")
EXTRA_EVAL_RE = re.compile(
    r"(xewa_eval|tail_vel_eval)\s+step:(\d+)/(\d+)\s+gamma:([-+0-9.eE]+)\s+val_loss:([0-9.]+)"
)
CONFIG_GAMMA_RE = re.compile(r"Using (tail_ema_gamma|tail_vel_gamma)=([-+0-9.eE]+)")
TAIL_VEL_FILENAME_GAMMA_RE = re.compile(r"_vg([-+]?[0-9]+(?:p[0-9]+)?)_")
XEWA_FILENAME_GAMMA_RE = re.compile(r"_g([-+]?[0-9]+(?:p[0-9]+)?)_md")

EVAL_KIND_TO_PREFIX = {
    "xewa": "xewa_eval",
    "tail_vel": "tail_vel_eval",
}
EVAL_KIND_TO_CONFIG_KEY = {
    "xewa": "tail_ema_gamma",
    "tail_vel": "tail_vel_gamma",
}


def same_float(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def parse_filename_gamma(path: Path, eval_kind: str) -> float | None:
    if eval_kind == "tail_vel":
        match = TAIL_VEL_FILENAME_GAMMA_RE.search(path.name)
    elif eval_kind == "xewa":
        match = XEWA_FILENAME_GAMMA_RE.search(path.name)
    else:
        match = None
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def parse_loss(path: Path, step: int | None, eval_kind: str, gamma: float | None) -> tuple[int, float]:
    matches: list[tuple[int, float]] = []
    extra_matches: list[tuple[int, float]] = []
    config_gamma = parse_filename_gamma(path, eval_kind)
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        config_match = CONFIG_GAMMA_RE.search(line)
        if (
            config_match is not None
            and eval_kind != "standard"
            and config_match.group(1) == EVAL_KIND_TO_CONFIG_KEY[eval_kind]
        ):
            config_gamma = float(config_match.group(2))

        match = VAL_RE.search(line)
        if match is not None:
            current_step = int(match.group(1))
            total_step = int(match.group(2))
            loss = float(match.group(3))
            if step is None:
                if current_step == total_step:
                    matches.append((current_step, loss))
            elif current_step == step:
                matches.append((current_step, loss))

        extra_match = EXTRA_EVAL_RE.search(line)
        if eval_kind != "standard" and gamma is not None and extra_match is not None:
            current_prefix = extra_match.group(1)
            current_step = int(extra_match.group(2))
            total_step = int(extra_match.group(3))
            current_gamma = float(extra_match.group(4))
            loss = float(extra_match.group(5))
            if current_prefix != EVAL_KIND_TO_PREFIX[eval_kind] or not same_float(current_gamma, gamma):
                continue
            if step is None:
                if current_step == total_step:
                    extra_matches.append((current_step, loss))
            elif current_step == step:
                extra_matches.append((current_step, loss))

    if eval_kind != "standard":
        if gamma is None:
            raise ValueError("--gamma is required when --eval-kind is not standard")
        if extra_matches:
            return extra_matches[-1]
        if config_gamma is not None and same_float(config_gamma, gamma) and matches:
            return matches[-1]
        requested = "final" if step is None else str(step)
        raise ValueError(f"{path}: no {eval_kind} validation loss found for step {requested} and gamma {gamma:g}")

    if not matches:
        requested = "final" if step is None else str(step)
        raise ValueError(f"{path}: no validation loss found for step {requested}")
    return matches[-1]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--step", type=int, default=2985)
    parser.add_argument("--eval-kind", choices=["standard", "xewa", "tail_vel"], default="standard")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--target", type=float, default=3.28)
    parser.add_argument("--sigma", type=float, default=0.0013)
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        step, loss = parse_loss(path, args.step, args.eval_kind, args.gamma)
        rows.append((path, step, loss))

    losses = [loss for _, _, loss in rows]
    mean = sum(losses) / len(losses)
    if len(losses) > 1:
        variance = sum((loss - mean) ** 2 for loss in losses) / (len(losses) - 1)
        std = math.sqrt(variance)
        stderr = std / math.sqrt(len(losses))
        t_stat = (mean - args.target) / stderr if stderr > 0 else float("-inf")
    else:
        std = float("nan")
        stderr = float("nan")
        t_stat = float("nan")

    precision = (args.target - mean) * math.sqrt(len(losses))
    z_stat = precision / args.sigma
    z_pvalue = normal_cdf(-z_stat)
    passes_track3 = precision >= 0.004

    if len(losses) < 2:
        pvalue_text = "needs_at_least_2_runs"
    else:
        try:
            import scipy.stats

            pvalue = scipy.stats.ttest_1samp(losses, args.target, alternative="less").pvalue
            pvalue_text = f"{pvalue:.6g}"
        except Exception as exc:
            pvalue_text = f"unavailable ({exc}); normal_approx={normal_cdf(t_stat):.6g}"

    print("logs:")
    for path, step, loss in rows:
        print(f"  {path} step={step} val_loss={loss:.5f}")
    print(f"n={len(losses)}")
    print(f"mean={mean:.6f}")
    print(f"std={std:.6f}")
    print(f"stderr={stderr:.6f}")
    print(f"target={args.target:.6f}")
    print(f"track3_sigma={args.sigma:.6f}")
    print(f"track3_precision=(target-mean)*sqrt(n)={precision:.6f}")
    print(f"track3_required_precision=0.004000")
    print(f"track3_z={z_stat:.6f}")
    print(f"track3_z_pvalue_less={z_pvalue:.6g}")
    print(f"passes_track3_precision={passes_track3}")
    print(f"t_stat={t_stat:.6f}")
    print(f"ttest_pvalue_less={pvalue_text}")


if __name__ == "__main__":
    main()

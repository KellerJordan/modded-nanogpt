#!/usr/bin/env python3
"""Summarize a sweep directory: per-run val_loss + train_time, mean ± sd, t-stats."""
import argparse, glob, math, re, statistics, sys
from pathlib import Path

FINAL = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+train_time:(\d+)ms")


def collect(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/*.txt")):
        txt = Path(f).read_text()
        m = FINAL.findall(txt)
        if m:
            rows.append((float(m[-1][2]), int(m[-1][3]), Path(f).name))
    return rows


def t_lower_p(t, df):
    """One-sided P(T <= t) via regularized incomplete beta (no scipy)."""
    x = df / (df + t * t)
    a, b = df / 2, 0.5

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-7, 1e-30
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1, 1 - qab * x / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1 / d; h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1 / d; del_ = d * c; h *= del_
            if abs(del_ - 1.0) < EPS: break
        return h

    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        I = bt * betacf(a, b, x) / a
    else:
        I = 1 - bt * betacf(b, a, 1 - x) / b
    upper_tail = I / 2  # P(T >= |t|)
    return upper_tail if t < 0 else 1 - upper_tail  # P(T <= t_obs)


def welch(a, b):
    """Welch's t-test of mean(a) < mean(b). Returns (t, df, p_one_sided)."""
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = t_lower_p(t, df)
    return t, df, p


def summarize(label, rows):
    vls = [r[0] for r in rows]
    tts = [r[1] for r in rows]
    n = len(vls)
    if n == 0:
        print(f"{label:24s} n=0"); return None, None
    m_v, s_v = statistics.mean(vls), statistics.stdev(vls) if n > 1 else 0
    m_t, s_t = statistics.mean(tts), statistics.stdev(tts) if n > 1 else 0
    t = (m_v - 3.28) / (s_v / math.sqrt(n)) if s_v > 0 and n > 1 else float("inf")
    p = t_lower_p(t, n - 1) if n > 1 else 1.0
    print(f"{label:24s} n={n:2d}  val_loss={m_v:.4f}±{s_v:.4f}  "
          f"t={t:+.3f} p(mean<3.28)={p:.4g}  "
          f"time={m_t/1000:.2f}±{s_t/1000:.2f}s")
    return vls, tts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", default="logs/runpod_2026-05-07/pr")
    ap.add_argument("--master", default="logs/runpod_2026-05-07/master")
    args = ap.parse_args()

    pr_rows = collect(args.pr)
    ms_rows = collect(args.master)
    pr_v, pr_t = summarize("xsa-gated-layers PR", pr_rows)
    ms_v, ms_t = summarize("master baseline", ms_rows)

    if pr_v and ms_v and len(pr_t) > 1 and len(ms_t) > 1:
        t, df, p = welch(pr_t, ms_t)
        delta = (statistics.mean(ms_t) - statistics.mean(pr_t)) / 1000
        print(f"\nWelch on wall-time (PR < master?): t={t:+.3f}  df={df:.1f}  "
              f"p(PR<master)={p:.4g}  delta_mean={delta:+.3f}s")


if __name__ == "__main__":
    main()

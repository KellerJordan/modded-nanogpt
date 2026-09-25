# Eight-H100 submission results

Generated 2026-09-25T02:57:37.853648+00:00. Tables include all available records; planned and completed counts distinguish partial results.

Candidate source: `c5be2fbf86c918e91b9e283b4044595bb74a350f`, based on PR #360 `c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`. All reported runs use one eight-H100 SXM 80GB node, NV18 topology, driver 580.126.09 and torch 2.10.0+cu128. No candidate or control run is discarded.

| Phase / implementation | Completed / planned | Training seconds (mean ± SD) | Validation loss (mean ± SD) |
|---|---:|---:|---:|
| pilot / baseline | 2 / 2 | 40.609500 ± 0.334462 | 3.27560000 ± 0.00042426 |
| pilot / candidate | 2 / 2 | 39.882255 ± 0.274922 | 3.27855992 ± 0.00201968 |
| certify / baseline | 6 / 6 | 40.341833 ± 0.183616 | 3.27900000 ± 0.00473667 |
| certify / candidate | 12 / 12 | 39.582968 ± 0.265670 | 3.27870470 ± 0.00127645 |
| short_control / short12 | 4 / 4 | 39.798500 ± 0.026839 | 3.28122500 ± 0.00180254 |
| short_control / short24 | 4 / 4 | 39.389000 ± 0.035223 | 3.28112500 ± 0.00040311 |
| accepted / accepted | 2 / 2 | 69.386500 ± 0.031820 | 3.27645000 ± 0.00007071 |

Baseline losses are printed by the untouched source to four decimal places. Candidate diagnostics also preserve full precision. SD is the sample standard deviation. Pilot timing is reported separately; matching-source pilot candidate losses are included in the final quality test.

## Fixed-cohort comparison

Certification cohort complete: **True**. Required quality criterion established: **True**.

Candidate mean saving: 0.758866 seconds (1.881%). Mean loss difference: -0.00029530. The original one-second aspiration is not met by these observed means.

Quality test: n=14 matching-source candidate runs including pilots, mean 3.27868402, SD 0.00130200, one-sided p=0.00114243 against 3.28. One-sided 99% upper confidence bound: 3.27960626. An incomplete cohort is never marked certified.

An approximate Welch 95% confidence interval for the mean time saving is 0.528827 to 0.988905 seconds. This describes run variability on this node and does not cover between-machine variation.

Baseline losses are unusually variable in this cohort. Every result, including the baseline loss of 3.2879 and candidate loss above 3.282, is retained. The small baseline sample does not establish a quality improvement or equivalence. The candidate passes the threshold test independently of the baseline comparison.

For completeness, pooling baseline quality across pilots and certification gives n=8, mean loss 3.27815000, SD 0.00430448. Timing above still uses the prespecified fixed cohort.

## Ordinary shorter-training controls

These use unmodified PR360: short12 has 1182 total updates; short24 has 1170; the candidate and original control have 1194. Comparisons below describe observed means and do not by themselves establish equality or superiority.

- short12: candidate minus control = -0.215532s, -0.00252030 loss.
- short24: candidate minus control = +0.193968s, -0.00242030 loss.

Candidate loss is outside the observed control mean-loss range [3.27900000, 3.28122500]. These controls measure two ordinary shortening choices; they do not identify an exact equal-loss training budget or prove superiority over every possible step count.

## Accepted-master comparison

Accepted master `bc3a0c2d640d0d73dedaef87eae26148d2e32afb` ran unmodified at its default 1290 updates. Across 2 completed runs its mean time is 69.386500 seconds. Candidate saving relative to this separate comparison is 29.803532 seconds (42.953%). Most of that improvement belongs to PR #360; the incremental claim for this work remains the comparison against PR #360 above. These two accepted-master runs are a timing reference, not an independent loss certification.

## Calibration diagnostics

Over 12 certification candidates, rank 0 calibration work averaged 119.326 ms (charged to training). The selected rule shortened between 15 and 18 of 21 eligible heads. Per-run logs retain windows, all-rank maximum Q/K/V errors and the two observations. Rank 0 measured calibration duration is not a separate estimate of its net distributed critical-path cost.

## Individual outcomes

| Phase / run | Status | Updates | Seconds | Loss |
|---|---|---:|---:|---:|
| pilot/00_baseline | complete | 1194 | 40.846000 | 3.27530000 |
| pilot/01_candidate | complete | 1194 | 40.076654 | 3.27713180 |
| pilot/02_candidate | complete | 1194 | 39.687856 | 3.27998805 |
| pilot/03_baseline | complete | 1194 | 40.373000 | 3.27590000 |
| certify/00_baseline | complete | 1194 | 40.177000 | 3.27700000 |
| certify/01_candidate | complete | 1194 | 39.375698 | 3.27748895 |
| certify/02_candidate | complete | 1194 | 39.933017 | 3.27890587 |
| certify/03_candidate | complete | 1194 | 39.405707 | 3.27989745 |
| certify/04_baseline | complete | 1194 | 40.546000 | 3.27560000 |
| certify/05_candidate | complete | 1194 | 39.362062 | 3.27840352 |
| certify/06_candidate | complete | 1194 | 39.873407 | 3.27762365 |
| certify/07_candidate | complete | 1194 | 39.814987 | 3.27850485 |
| certify/08_baseline | complete | 1194 | 40.454000 | 3.27660000 |
| certify/09_baseline | complete | 1194 | 40.519000 | 3.27610000 |
| certify/10_candidate | complete | 1194 | 39.843213 | 3.27800870 |
| certify/11_candidate | complete | 1194 | 39.940388 | 3.27807593 |
| certify/12_candidate | complete | 1194 | 39.363754 | 3.28213644 |
| certify/13_baseline | complete | 1194 | 40.148000 | 3.28080000 |
| certify/14_candidate | complete | 1194 | 39.347598 | 3.27817059 |
| certify/15_candidate | complete | 1194 | 39.390636 | 3.27922273 |
| certify/16_candidate | complete | 1194 | 39.345148 | 3.27801776 |
| certify/17_baseline | complete | 1194 | 40.207000 | 3.28790000 |
| short_control/00_short12 | complete | 1182 | 39.766000 | 3.28150000 |
| short_control/01_short24 | complete | 1170 | 39.365000 | 3.28090000 |
| short_control/02_short24 | complete | 1170 | 39.409000 | 3.28110000 |
| short_control/03_short12 | complete | 1182 | 39.787000 | 3.27860000 |
| short_control/04_short12 | complete | 1182 | 39.822000 | 3.28230000 |
| short_control/05_short24 | complete | 1170 | 39.354000 | 3.28170000 |
| short_control/06_short24 | complete | 1170 | 39.428000 | 3.28080000 |
| short_control/07_short12 | complete | 1182 | 39.819000 | 3.28250000 |
| accepted/00_accepted | complete | 1290 | 69.409000 | 3.27640000 |
| accepted/01_accepted | complete | 1290 | 69.364000 | 3.27650000 |

## Reproduction and timing policy

Compilation caches were retained across successive processes, with the same policy for both arms; compilation and graph warmup are outside the inherited training clock. Pilot times are kept separate from the fixed comparison. This differs from the cold-cache protocol reported in PR #360, so published times from another node are not used to estimate this contribution.

All complete console and source-containing training logs accompany these records. The final archive manifest hashes every artifact; source hashes must match the frozen checkout before release. Earlier one-H200 serial-replica experiments are excluded from these eight-GPU statistics.

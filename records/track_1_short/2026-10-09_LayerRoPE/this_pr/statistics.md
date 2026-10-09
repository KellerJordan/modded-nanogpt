# Certification pool statistics

alphagpu14, 8x H100 80GB HBM3 (driver 580.126.09); one record-#92 run, then two runs of this PR, nine times;
every run counted. Times are the trainer's own train_time at the final validation (as in record #92).

```
candidate: n=18 (nan 0) steps=[1177] | val 3.27628 +- 0.00120 (min 3.2740 max 3.2785) one-sided p(mean<3.28)=1.17e-10 | time 40.200 +- 0.022 s | training section 39.645 +- 0.014 s
baseline : n=9 (nan 0) steps=[1194] | val 3.27584 +- 0.00152 (min 3.2737 max 3.2788) one-sided p(mean<3.28)=1.78e-05 | time 40.562 +- 0.016 s | training section 40.003 +- 0.019 s
delta val +0.43 millinats (Welch p=0.467) | delta time -0.362 s (-0.89%, Welch p=6.12e-24) | delta section -0.358 s (p=1.9e-15)
```

# Track 3 2850 Tail Phase Readout

This branch adds a frozen 2850-step Track 3 submission candidate under:

```text
records/track_3_optimization/results/20260529_tail_phase_readout_2850/
```

It includes:

- `train_gpt_tail_phase_readout_2850.py`
- `run.sh`
- `summarize_submission_stats.py`
- full logs for seeds 0 through 12

Result:

```text
n=13
mean=3.278562
precision=(3.28 - mean) * sqrt(13)=0.005186
required=0.004000
passes_track3_precision=True
```

The method is the PR311-style 2850 TrailDelta/final-readout recipe plus a
fixed BroadDelta pulse on `muon_other` and a normalized orthogonal phase readout
on `muon_other`, followed by the step-2400 final readout.

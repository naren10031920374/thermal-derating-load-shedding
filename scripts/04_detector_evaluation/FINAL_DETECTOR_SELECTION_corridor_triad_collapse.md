# Final Early-Warning Detector Selection — corridor_triad_collapse

Of the 5 lead-time models trained in `step41_early_warning_corridor_triad_collapse.py`,
**N=30s is selected as the production detector** going forward (stage 06/07 work should
load this model, not re-evaluate all 5).

## Selected model

| | |
|---|---|
| Lead time | **30 seconds** |
| Model file | `model_outputs\unified_controller_10bus_derating\41_early_warning_corridor_triad_collapse\early_warning_N30s.pkl` |
| Architecture | sklearn Pipeline: StandardScaler → MLPClassifier(hidden_layer_sizes=(128,64)) |
| Input features | 83 strict-precursor features (voltage / GEI / source-power excluded) |
| Held out during training | Bus F (entire bus, not row-level) |
| **Operating threshold** | **0.1** |
| Recall at this threshold, zero false alarms | **96%** |
| AUC (held-out Bus F) | 1.0 |
| Average precision (PR curve) | 0.9999 |

## Why N=30s over the other 4

All 5 lead times scored a tied AUC of 1.0, so AUC alone doesn't distinguish them. The
tie-breaker is real operating performance — recall at the threshold that gives zero false
alarms, from `step44_threshold_sweep_corridor_triad_collapse.py`:

| Lead time | Threshold | Recall @ 0 false alarms | Average precision |
|---|---|---|---|
| 5s | 0.01 | 57% | 0.9924 |
| 15s | 0.002 | 84% | 0.9989 |
| **30s** | **0.1** | **96%** | **0.9999** |
| 60s | 0.0001 | 50%* | 0.9999 |
| 90s | 0.0001 | 86% | 1.0000 |

\* N=60s's 50% recall is a calibration artifact (its probability outputs only range up to
0.0155), not a weaker model — its average precision (0.9999) is on par with N=30s. See
`46_precision_recall_curves_corridor_triad_collapse` for the full curves.

N=30s gives the best realistic detection rate (96%) with zero false alarms, and 30
seconds is also a plausible amount of time to actually act on in a closed-loop system —
short enough to be usable, long enough to give real warning. N=90s has a marginally
higher average-precision score (1.0000 vs 0.9999) but 10 points lower recall at its
zero-false-alarm threshold (86% vs 96%), so it's not the better practical choice here.

## What this means for stages 06/07

- **Stage 05 lead-time test** (`test_shed_lead_time_corridor_triad_collapse.m`, running
  now): once it reports the latest safe shed-trigger time, compare that number against
  30 seconds specifically — that comparison is what determines whether N=30s's warning
  is fast enough to trigger the validated 0.80 shed fraction in time.
- **Stage 07 closed-loop validation**: load `early_warning_N30s.pkl`, apply threshold
  **0.1** to its output, and trigger the shed actuator on crossing — do not re-run the
  4-way threshold/lead-time comparison, that question is now closed.
- The other 4 trained models (`N5s.pkl`, `N15s.pkl`, `N60s.pkl`, `N90s.pkl`) are kept
  as-is for reference/comparison but are not the production path.

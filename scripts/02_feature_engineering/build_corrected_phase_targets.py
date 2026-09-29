"""
BUILD CORRECTED PHASE TARGETS FOR NN TRAINING
================================================================================
Per Apoorv's explanation: PI's raw phase command during a bus collapse is not
a good imitation target. PI pushes the affected converter to +/-90 deg trying
to save a bus that cannot be saved, which increases temperature further and
is exactly the behavior the NN is meant to improve on. Training directly
against Phase_<CONV>_cmd_deg during a collapse would teach the network to
reproduce that failure mode, not fix it.

This script builds a corrected target using a two-tier rule, not three:

    corrected_phase(t, conv) = 0                                  if EITHER
                                connected bus is currently collapsed
                              = raw_phase(t, conv) * derate_factor(t, conv)
                                otherwise

Why two tiers and not three: derate_factor is already a continuous, physically
computed signal (the model's own thermal_derater logic, ramping 1.0 -> 0.0
as junction temperature climbs 125 C -> 175 C), so it already IS the
graduated "stressed but not collapsing" behavior. Multiplying by it directly
means:
  - Healthy, fully-loaded converter (derate_factor = 1.0): corrected target
    equals PI's original phase exactly. A converter that genuinely needs
    85 deg still gets 85 deg, nothing is throttled that should not be.
  - Thermally stressed, not collapsing: target shrinks smoothly as
    derate_factor shrinks, using a signal the model already computes rather
    than an invented threshold.
  - Actively collapsing (either connected bus): target is forced to exactly
    0, overriding both PI and derate_factor, regardless of what they say.

COLLAPSE DETECTION: a bus counts as collapsed when its voltage drops below
COLLAPSE_VOLTAGE_V (100 V, well under the ~750-795 V normal operating range
seen throughout this project's data) AND stays there for at least
MIN_CONSECUTIVE_SAMPLES (50 samples = 0.5 s at the 10 ms sample rate). The
debounce exists to filter a brief noisy dip from a genuine collapse. Once a
run of consecutive below-threshold samples is confirmed long enough, the
ENTIRE run is marked collapsed from its true first sample, not just from
sample 50 onward, so the correction is not lagged behind the real onset.

A converter is "affected" if either of its two connected buses (ring
adjacency, e.g. AB touches A and B, AL touches A and L) is collapsed at that
timestamp.

Output: a separate CSV keyed by time, with corrected target columns and the
underlying collapse/affected flags (kept for inspection and for evaluating
model performance specifically during collapse windows later, not just
averaged across the whole run). This does NOT modify the feature parquet
from step33 v3. Join on 'time' when building the training set.

Run:
    python build_corrected_phase_targets.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
             / "thermal_derating_v7_ALfix_GEIfix_5000s.csv")
OUTPUT_CSV = (PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
              / "corrected_phase_targets_5000s.csv")

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

CONV_TO_BUSES = {
    "AB": ("A","B"), "BC": ("B","C"), "CD": ("C","D"), "DE": ("D","E"),
    "EF": ("E","F"), "FG": ("F","G"), "GH": ("G","H"), "HK": ("H","K"),
    "KL": ("K","L"), "AL": ("A","L"),
}

COLLAPSE_VOLTAGE_V = 100.0        # well under the ~750-795 V normal range
MIN_CONSECUTIVE_SAMPLES = 50      # 0.5 s at 10 ms sampling, filters noise


def detect_collapse(voltage: pd.Series, threshold: float, min_run: int) -> pd.Series:
    """
    True for every sample in a run of consecutive below-threshold samples,
    but only if that run is at least min_run samples long. Short dips
    (noise) are left False. The full qualifying run is marked True from its
    real first sample, not delayed by the debounce window.
    """
    below = voltage < threshold
    run_id = (below != below.shift(fill_value=False)).cumsum()
    run_length = below.groupby(run_id).transform("size")
    return below & (run_length >= min_run)


def main():
    print("=" * 70)
    print("BUILD CORRECTED PHASE TARGETS")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")
    raw = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(raw):,} rows.")

    required = (["time"]
                + [f"V_Bus_{b}" for b in BUSES]
                + [f"Phase_{c}_cmd_deg" for c in CONV]
                + [f"DAB_{c}_Derate_Factor" for c in CONV])
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns:\n{missing}")

    # --------------------------------------------------------------------
    # 1. Per-bus collapse detection
    # --------------------------------------------------------------------
    print(f"\nDetecting bus collapse (voltage < {COLLAPSE_VOLTAGE_V:.0f} V for "
          f">= {MIN_CONSECUTIVE_SAMPLES} consecutive samples, "
          f"{MIN_CONSECUTIVE_SAMPLES * 0.01:.2f} s) ...")
    collapsed = pd.DataFrame(index=raw.index)
    for b in BUSES:
        collapsed[b] = detect_collapse(raw[f"V_Bus_{b}"], COLLAPSE_VOLTAGE_V,
                                        MIN_CONSECUTIVE_SAMPLES)

    print("\n--- Per-bus collapse summary ---")
    for b in BUSES:
        n_rows = collapsed[b].sum()
        pct = 100 * n_rows / len(raw)
        if n_rows > 0:
            t_first = raw.loc[collapsed[b], "time"].iloc[0]
            t_last = raw.loc[collapsed[b], "time"].iloc[-1]
            print(f"  Bus {b}: {n_rows:7,d} rows ({pct:5.2f}%), "
                  f"t={t_first:.2f}s to t={t_last:.2f}s")
        else:
            print(f"  Bus {b}: never collapsed in this run")

    # --------------------------------------------------------------------
    # 2. Per-converter affected flag
    # --------------------------------------------------------------------
    affected = pd.DataFrame(index=raw.index)
    for c in CONV:
        b1, b2 = CONV_TO_BUSES[c]
        affected[c] = collapsed[b1] | collapsed[b2]

    print("\n--- Per-converter affected summary ---")
    for c in CONV:
        n_rows = affected[c].sum()
        pct = 100 * n_rows / len(raw)
        print(f"  {c:4s}: {n_rows:7,d} rows ({pct:5.2f}%) will be zeroed")

    # --------------------------------------------------------------------
    # 3. Build the corrected target
    # --------------------------------------------------------------------
    print("\nBuilding corrected targets ...")
    out = pd.DataFrame(index=raw.index)
    out["time"] = raw["time"]

    n_zeroed_total = 0
    n_derate_scaled_total = 0
    for c in CONV:
        raw_phase = raw[f"Phase_{c}_cmd_deg"]
        derate = raw[f"DAB_{c}_Derate_Factor"]
        corrected = raw_phase * derate
        corrected = corrected.where(~affected[c], other=0.0)
        out[f"target_Phase_{c}_corrected"] = corrected
        out[f"affected_{c}"] = affected[c]

        n_zeroed_total += affected[c].sum()
        n_derate_scaled_total += ((derate < 0.999) & ~affected[c]).sum()

    for b in BUSES:
        out[f"collapsed_{b}"] = collapsed[b]

    # --------------------------------------------------------------------
    # 4. Sanity check: corrected magnitude can only shrink, never grow,
    # relative to the raw phase, since it is either multiplied by a factor
    # in [0,1] or set to exactly 0. If this fails, something is wrong with
    # the logic above, not with the data.
    # --------------------------------------------------------------------
    print("\nRunning sanity check: |corrected| <= |raw| for every converter, every row ...")
    violations = 0
    for c in CONV:
        raw_abs = raw[f"Phase_{c}_cmd_deg"].abs()
        corr_abs = out[f"target_Phase_{c}_corrected"].abs()
        bad = (corr_abs > raw_abs + 1e-6).sum()
        violations += bad
        if bad > 0:
            print(f"  WARNING: {c} has {bad} rows where |corrected| > |raw|, "
                  f"this should be impossible, check derate_factor is in [0,1].")
    if violations == 0:
        print("Passed: corrected phase magnitude never exceeds the raw PI command, "
              "for any converter, any row.")
    else:
        raise RuntimeError(
            f"{violations} sanity check violations found. Do not use this output "
            "for training until this is resolved."
        )

    # --------------------------------------------------------------------
    # 5. Write output
    # --------------------------------------------------------------------
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")

    print(f"\n--- OVERALL SUMMARY ---")
    total_rows = len(raw) * len(CONV)
    print(f"Across all {len(CONV)} converters and {len(raw):,} rows "
          f"({total_rows:,} converter-timesteps total):")
    print(f"  Zeroed (collapse override):     {n_zeroed_total:,} "
          f"({100*n_zeroed_total/total_rows:.3f}%)")
    print(f"  Derate-scaled, not zeroed:      {n_derate_scaled_total:,} "
          f"({100*n_derate_scaled_total/total_rows:.3f}%)")
    print(f"  Unchanged (healthy, factor=1.0): "
          f"{total_rows - n_zeroed_total - n_derate_scaled_total:,} "
          f"({100*(total_rows - n_zeroed_total - n_derate_scaled_total)/total_rows:.3f}%)")

    print("\nDone. Next step: join this file with the step33 v3 features parquet "
          "on 'time', use target_Phase_<CONV>_corrected as the training target "
          "instead of target_Phase_<CONV>, and keep the affected_<CONV> / "
          "collapsed_<BUS> columns for reporting MAE separately during collapse "
          "windows versus normal operation.")


if __name__ == "__main__":
    main()

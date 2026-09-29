"""
BUILD CORRECTED PHASE TARGETS FOR NN TRAINING (corridor_triad_collapse)
================================================================================
Same logic as build_corrected_phase_targets.py, applied to the new
corridor_triad_collapse scenario instead of the baseline.

Per Apoorv's explanation: PI's raw phase command during a bus collapse is not
a good imitation target. PI pushes the affected converter to +/-90 deg trying
to save a bus that cannot be saved, which increases temperature further and
is exactly the behavior the NN is meant to improve on. Training directly
against Phase_<CONV>_cmd_deg during a collapse would teach the network to
reproduce that failure mode, not fix it.

Two-tier rule:
    corrected_phase(t, conv) = 0                                  if EITHER
                                connected bus is currently collapsed
                              = raw_phase(t, conv) * derate_factor(t, conv)
                                otherwise

COLLAPSE DETECTION: identical thresholds to the baseline and to this
scenario's own generation-time validation (generate_corridor_triad_collapse.m):
voltage < 100V for >= 50 consecutive samples (0.5s at 10ms sampling).

SCENARIO-SPECIFIC NOTE: unlike the baseline (Bus F then Bus E, 53s apart),
this scenario has E and F collapsing ~0.1s apart (t=1149.1s and 1149.2s) -
essentially simultaneous, not staggered. G never collapses. That matters for
whatever downstream training/holdout design gets built on top of this file
(e.g. step41-style early-warning experiments choose a holdout bus assuming
staggered onsets) - flagging here so it isn't missed later, not something
this script itself needs to handle.

Output: a separate CSV keyed by time, with corrected target columns and the
underlying collapse/affected flags. Does NOT modify the step33 features
parquet. Join on 'time' when building the training set.

Run:
    python build_corrected_phase_targets_corridor_triad_collapse.py
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
             / "corridor_triad_collapse" / "scenario_corridor_triad_collapse_5000s.csv")
OUTPUT_CSV = (PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
              / "corrected_phase_targets_corridor_triad_collapse_5000s.csv")

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
    print("BUILD CORRECTED PHASE TARGETS (corridor_triad_collapse)")
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
    # 4. Sanity check: corrected magnitude can only shrink, never grow.
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

    print("\nDone. Next step: join this file with the step33 corridor_triad_collapse "
          "features parquet on 'time', use target_Phase_<CONV>_corrected as the "
          "training target instead of target_Phase_<CONV>, and keep the "
          "affected_<CONV> / collapsed_<BUS> columns for reporting MAE separately "
          "during collapse windows versus normal operation.")


if __name__ == "__main__":
    main()
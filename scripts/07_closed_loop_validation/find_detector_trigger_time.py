"""
FIND_DETECTOR_TRIGGER_TIME.PY  (corrected, real pipeline)
================================================================================
Pass 1 of the closed-loop shedding validation:

    Run simulation normally
          |
    Feed live signals into detector, PER BUS         <- THIS SCRIPT
          |
    Record exact trigger time, PER BUS                <- THIS SCRIPT
          |
    Automatically start shedding on whichever buses    (Pass 2, MATLAB)
    the detector flagged
          |
    See whether ALL 10 buses survive                   (Pass 2, MATLAB)

CORRECTED FROM THE PREVIOUS VERSION, after reading step34, step44, and
step46 directly:
  - The real detector is NOT anything in step33/step34. step34 trains a
    PHASE REGRESSION model (target_Phase_<CONV>_corrected), a different,
    already-flagged-unsafe model. The actual collapse detector is step41's
    saved pipeline, evaluated (not retrained) by step44/45/46.
  - Model path is model_outputs/unified_controller_10bus_derating/
    41_early_warning_shifted_label/early_warning_N60s.pkl, not a made-up
    path under a "models" folder.
  - There is NO separate scaler file. step44/45/46 all call
    pipeline.predict_proba(X) directly on raw feature values, the scaler
    is bundled inside the saved pipeline. Do not call a separate
    scaler.transform() here either.
  - The feature builder is NOT a step33 function. It is
    get_local_feature_map(bus_idx), copied VERBATIM from step44/step46,
    which maps canonical feature names to each bus's actual column names
    in the wide feature table (features_df joined with corrected targets
    on 'time'). This is what makes one model work for every bus.
    DIRECT_SIGNAL_KEYS and STRICT_ADDITIONAL_KEYS are also copied
    verbatim, they define the strict-precursor 83-feature set.
  - Ground-truth collapse per bus does not need to be recomputed from raw
    voltage. collapsed_<BUS> already exists in corrected_targets_csv,
    computed with the same debounce logic, reused directly.

WHY THIS DOESN'T NEED A LIVE SIMULINK<->PYTHON BRIDGE:
The detector is causal, and before any bus's shed command fires, that
bus's ShedFrac = 1 (no shedding), so the trajectory up to that point is
identical to the existing feature/target data already generated. Scoring
the detector against that data, timestep by timestep, per bus, is
mathematically identical to a true real-time coupled run for the purpose
of finding each bus's trigger time.

SUBSAMPLE_STRIDE = 5 is kept identical to step44/45/46 (every 5th row,
50ms resolution instead of 10ms), since that is the data distribution the
threshold (5e-6) was calibrated against. Scoring at a different stride is
still causal but changes the resolution the threshold was tuned for, not
a decision to make silently.

Run:
    python find_detector_trigger_time.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Paths, must match step33 through step46
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
STEP41_DIR = MODEL_ROOT / "41_early_warning_shifted_label"
MODEL_PATH = STEP41_DIR / "early_warning_N60s.pkl"   # the N=60s winner per step45/46

# Confirmed in step46's HIGHLIGHT_POINTS: N=60s, threshold ~5e-6,
# recall 0.454, precision 0.951 on the Bus F holdout.
DECISION_THRESHOLD = 5e-6

# Persistence requirement: a single noisy sample above threshold is not a
# real trigger, same principle as the collapse debounce itself. Found
# necessary after the first run of this script: without this, buses that
# never collapse fired within the first ~150s (feature transients near
# simulation start), and Bus E/F also fired at t~125-133s, ~3500s before
# their real onset, an unusable "lead time" that is really just noise,
# not genuine precursor signal. Require the probability to stay at or
# above threshold for MIN_CONSECUTIVE_TRIGGER_SAMPLES in a row (at the
# stride-5 / 50ms resolution below) before it counts.
MIN_CONSECUTIVE_TRIGGER_SAMPLES = 10   # 10 samples x 50ms = 0.5s sustained

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

SUBSAMPLE_STRIDE = 5   # identical to step44/45/46, matches the threshold's calibration

OUTPUT_JSON = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
               / "closed_loop_trigger" / "detector_trigger_times_all_buses.json")


# ----------------------------------------------------------------------
# Verified verbatim against step44_threshold_sweep_original_models.py /
# step46_precision_recall_curves.py
# ----------------------------------------------------------------------
def get_local_feature_map(bus_idx: int) -> dict:
    bus = BUSES[bus_idx]
    conv_next = CONV[bus_idx]
    conv_prev = CONV[(bus_idx - 1) % 10]
    nb_next = BUSES[(bus_idx + 1) % 10]
    nb_prev = BUSES[(bus_idx - 1) % 10]
    m = {}
    with_roll = ["Voltage", "Source_Power", "GEI", "Commanded_Load"]
    no_roll = ["Temp"]
    for sig in with_roll + no_roll:
        base_col = f"Bus_{bus}_{sig}"
        m[f"own_{sig}"] = base_col
        for lag in (1, 2, 3):
            m[f"own_{sig}_lag{lag}"] = f"{base_col}_lag{lag}"
        m[f"own_d_{sig}"] = f"d_{base_col}"
        if sig in with_roll:
            m[f"own_{sig}_roll_mean_5"] = f"{base_col}_roll_mean_5"
            m[f"own_{sig}_roll_std_5"] = f"{base_col}_roll_std_5"
    imb_with_lag = {
        "GEI_Error": f"GEI_Error_{bus}", "Power_Imbalance": f"Power_Imbalance_{bus}",
        "Voltage_Imbalance": f"Voltage_Imbalance_{bus}", "Load_Imbalance": f"Load_Imbalance_{bus}",
    }
    for name, col in imb_with_lag.items():
        m[f"own_{name}"] = col
        for lag in (1, 2, 3):
            m[f"own_{name}_lag{lag}"] = f"{col}_lag{lag}"
    imb_no_lag = {
        "Abs_GEI_Error": f"Abs_GEI_Error_{bus}", "Abs_Voltage_Imbalance": f"Abs_Voltage_Imbalance_{bus}",
        "Abs_Power_Imbalance": f"Abs_Power_Imbalance_{bus}", "Temp_Imbalance": f"Bus_{bus}_Temp_Imbalance",
        "Abs_Temp_Imbalance": f"Abs_Bus_{bus}_Temp_Imbalance", "Abs_Load_Imbalance": f"Abs_Load_Imbalance_{bus}",
        "Voltage_Error": f"Voltage_Error_{bus}", "Abs_Voltage_Error": f"Abs_Voltage_Error_{bus}",
    }
    for name, col in imb_no_lag.items():
        m[f"own_{name}"] = col
    for role, conv in [("conv_prev", conv_prev), ("conv_next", conv_next)]:
        for sig in ["Heat_Sink_Temp", "Derate_Factor", "Junction_Temp"]:
            base_col = f"DAB_{conv}_{sig}"
            m[f"{role}_{sig}"] = base_col
            for lag in (1, 2, 3):
                m[f"{role}_{sig}_lag{lag}"] = f"{base_col}_lag{lag}"
            m[f"{role}_{sig}_roll_mean_5"] = f"{base_col}_roll_mean_5"
            m[f"{role}_{sig}_roll_std_5"] = f"{base_col}_roll_std_5"
            m[f"{role}_d_{sig}"] = f"d_{base_col}"
        imb_map = {
            "Temp_Imbalance": f"DAB_{conv}_Temp_Imbalance", "Abs_Temp_Imbalance": f"Abs_DAB_{conv}_Temp_Imbalance",
            "Derate_Imbalance": f"DAB_{conv}_Derate_Imbalance", "Abs_Derate_Imbalance": f"Abs_DAB_{conv}_Derate_Imbalance",
            "Junction_Imbalance": f"DAB_{conv}_Junction_Imbalance", "Abs_Junction_Imbalance": f"Abs_DAB_{conv}_Junction_Imbalance",
            "Junction_HeatSink_Diff": f"DAB_{conv}_Junction_HeatSink_Diff",
        }
        for name, col in imb_map.items():
            m[f"{role}_{name}"] = col
    for role, nb in [("neighbor_prev", nb_prev), ("neighbor_next", nb_next)]:
        for sig in ["Voltage", "Source_Power", "GEI", "Temp"]:
            m[f"{role}_{sig}"] = f"Bus_{nb}_{sig}"
    return m


DIRECT_SIGNAL_KEYS = [
    "own_Voltage", "own_Voltage_lag1", "own_Voltage_lag2", "own_Voltage_lag3",
    "own_Voltage_roll_mean_5", "own_Voltage_roll_std_5", "own_d_Voltage",
    "own_Voltage_Error", "own_Abs_Voltage_Error",
    "own_Voltage_Imbalance", "own_Voltage_Imbalance_lag1",
    "own_Voltage_Imbalance_lag2", "own_Voltage_Imbalance_lag3", "own_Abs_Voltage_Imbalance",
    "own_GEI", "own_GEI_lag1", "own_GEI_lag2", "own_GEI_lag3",
    "own_GEI_roll_mean_5", "own_GEI_roll_std_5", "own_d_GEI",
    "own_GEI_Error", "own_GEI_Error_lag1", "own_GEI_Error_lag2",
    "own_GEI_Error_lag3", "own_Abs_GEI_Error",
]
STRICT_ADDITIONAL_KEYS = [
    "own_Source_Power", "own_Source_Power_lag1", "own_Source_Power_lag2",
    "own_Source_Power_lag3", "own_Source_Power_roll_mean_5", "own_Source_Power_roll_std_5",
    "own_d_Source_Power", "own_Power_Imbalance", "own_Power_Imbalance_lag1",
    "own_Power_Imbalance_lag2", "own_Power_Imbalance_lag3", "own_Abs_Power_Imbalance",
]


def load_corrected_targets():
    df = pd.read_csv(CORRECTED_TARGETS_CSV)
    bool_cols = [c for c in df.columns if c.startswith("affected_") or c.startswith("collapsed_")]
    for c in bool_cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df[c] = df[c].astype(bool)
    return df


def first_sustained_crossing(proba: np.ndarray, threshold: float, min_run: int,
                              search_mask: np.ndarray) -> int | None:
    """Returns the index of the FIRST sample that begins a run of at least
    min_run consecutive samples with proba >= threshold, restricted to
    positions where search_mask is True. Same debounce principle as
    detect_collapse in build_corrected_phase_targets.py: a single noisy
    sample crossing the threshold is not treated as a real trigger, only
    a sustained run is. Returns None if no qualifying run exists."""
    above = (proba >= threshold) & search_mask
    if not above.any():
        return None
    run_id = (above != np.concatenate(([False], above[:-1]))).cumsum()
    run_id_of_above = np.where(above, run_id, -1)
    # length of each run
    unique_runs, counts = np.unique(run_id_of_above[run_id_of_above >= 0], return_counts=True)
    qualifying_runs = unique_runs[counts >= min_run]
    if len(qualifying_runs) == 0:
        return None
    first_qualifying_run = qualifying_runs.min()
    idx = np.argmax(run_id_of_above == first_qualifying_run)
    return int(idx)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FIND DETECTOR TRIGGER TIME, ALL BUSES")
    print("(real step41 N=60s pipeline, threshold 5e-6, per step44-46)")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n"
            "Run step41_early_warning_shifted_label.py first if this is missing."
        )
    if not CORRECTED_TARGETS_CSV.exists():
        raise FileNotFoundError(
            f"Corrected targets not found:\n{CORRECTED_TARGETS_CSV}\n"
            "Run build_corrected_phase_targets.py first."
        )

    print(f"\nLoading model: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    print(f"Loaded features: {len(features_df):,} rows x {features_df.shape[1]} cols")

    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    print(f"After join + stride-{SUBSAMPLE_STRIDE} subsample: {len(wide):,} rows")

    canonical_cols = list(get_local_feature_map(0).keys())
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    print(f"Strict-precursor feature count: {len(strict_cols)}")

    t_all = wide["time"].to_numpy()
    results = {}

    for bus_idx, bus in enumerate(BUSES):
        print(f"\n{'-'*70}")
        print(f"Bus {bus}")
        print(f"{'-'*70}")

        fmap = get_local_feature_map(bus_idx)
        actual_cols = [fmap[c] for c in strict_cols]
        missing = [c for c in actual_cols if c not in wide.columns]
        if missing:
            raise ValueError(f"Bus {bus}: missing columns in wide table: {missing[:5]}"
                              f"{'...' if len(missing) > 5 else ''}")

        X = wide[actual_cols].to_numpy(dtype=np.float64)
        proba = pipeline.predict_proba(X)[:, 1]   # no separate scaler call, bundled in pipeline

        collapsed_col = f"collapsed_{bus}"
        if collapsed_col not in wide.columns:
            raise ValueError(f"{collapsed_col} not found, check build_corrected_phase_targets.py output.")
        collapsed = wide[collapsed_col].to_numpy().astype(bool)

        if collapsed.any():
            onset = float(t_all[np.argmax(collapsed)])
            search_mask = t_all < onset
        else:
            onset = None
            # Bus never collapses in ground truth. Still score the full run:
            # a crossing here is a false positive worth knowing about.
            search_mask = np.ones_like(t_all, dtype=bool)

        crossing = (proba >= DECISION_THRESHOLD) & search_mask
        max_proba = float(proba[search_mask].max()) if search_mask.any() else float("nan")

        trigger_idx = first_sustained_crossing(proba, DECISION_THRESHOLD,
                                                MIN_CONSECUTIVE_TRIGGER_SAMPLES, search_mask)

        if trigger_idx is not None:
            trigger_time = float(t_all[trigger_idx])
            lead_time = (onset - trigger_time) if onset is not None else None
            print(f"  Ground-truth onset: {onset if onset is not None else 'never collapses'}")
            print(f"  Max probability (before onset): {max_proba:.3e}")
            print(f"  TRIGGER (sustained {MIN_CONSECUTIVE_TRIGGER_SAMPLES}+ samples) at t={trigger_time:.2f}s"
                  + (f", lead time {lead_time:.2f}s" if lead_time is not None else " (false positive, bus never collapses)"))
            if crossing.any() and not (trigger_idx == np.argmax(crossing)):
                first_blip_t = float(t_all[np.argmax(crossing)])
                print(f"  (note: first single-sample blip was earlier, at t={first_blip_t:.2f}s, "
                      f"discarded as noise, not sustained)")
        else:
            trigger_time = None
            lead_time = None
            print(f"  Ground-truth onset: {onset if onset is not None else 'never collapses'}")
            print(f"  Max probability: {max_proba:.3e}")
            if crossing.any():
                print("  No SUSTAINED trigger, only isolated single-sample blips, discarded as noise."
                      + (" MISS: bus collapses but detector never gave a sustained warning." if onset is not None else " Correctly quiet in the sustained sense."))
            else:
                print("  No trigger at all."
                      + (" MISS: bus collapses but detector never crossed threshold." if onset is not None else " Correctly quiet, bus never collapses."))

        results[bus] = {
            "collapse_onset": onset,
            "trigger_time": trigger_time,
            "lead_time": lead_time,
            "max_probability": max_proba if not np.isnan(max_proba) else None,
        }

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Bus':4s} {'GT onset':>10s} {'Trigger':>10s} {'Lead time':>10s} {'Note':<25s}")
    for bus in BUSES:
        r = results[bus]
        onset_s = f"{r['collapse_onset']:.1f}" if r['collapse_onset'] is not None else "never"
        trig_s = f"{r['trigger_time']:.1f}" if r['trigger_time'] is not None else "none"
        lead_s = f"{r['lead_time']:.1f}" if r['lead_time'] is not None else "-"
        if r['collapse_onset'] is not None and r['trigger_time'] is None:
            note = "MISS (no warning)"
        elif r['collapse_onset'] is None and r['trigger_time'] is not None:
            note = "false positive"
        elif r['collapse_onset'] is not None and r['trigger_time'] is not None:
            note = "caught"
        else:
            note = "correctly quiet"
        print(f"{bus:4s} {onset_s:>10s} {trig_s:>10s} {lead_s:>10s} {note:<25s}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "buses": results,
            "threshold": DECISION_THRESHOLD,
            "model_path": str(MODEL_PATH),
            "subsample_stride": SUBSAMPLE_STRIDE,
        }, f, indent=2)
    print(f"\nWrote: {OUTPUT_JSON}")
    print("\nNext: run run_closed_loop_verification.m, it reads this file directly")
    print("and wires a shed command for every bus that has a trigger_time.")


if __name__ == "__main__":
    main()

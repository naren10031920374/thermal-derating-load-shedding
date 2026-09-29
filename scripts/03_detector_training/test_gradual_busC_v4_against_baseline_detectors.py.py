"""
TEST: baseline-trained early-warning detectors (N=5/15/30/60/90s) against the
NEW gradual_busC_v4 collapse scenario.

This is a pure generalization test -- NOTHING is retrained here. The 5
MLPClassifier pipelines already sitting in
    model_outputs\\unified_controller_10bus_derating\\41_early_warning_shifted_label\\
        early_warning_N{5,15,30,60,90}s.pkl
were trained ONLY on the baseline scenario (Bus F/E sudden-spike cascade).
This script feeds them the brand-new gradual_busC_v4 run (Bus C ramps up over
800-2800s and genuinely collapses at t=2094.47s -- wait, ramp is 800-2800s and
collapse happens at 2094.47s, i.e. mid-ramp, before the ramp even finishes)
and asks: does ANY of these 5 lead-time models produce a usable early-warning
signal on a scenario they have never seen, built from a completely different
collapse mechanism (slow thermal ramp vs. sudden spike)?

WHY THIS MATTERS (per the project's own prior finding): pooled/AUC-style
metrics can look fine while a detector is actually just memorizing one
training bus. The only honest test is: (a) does probability rise on the bus
that actually collapses, before it collapses, and (b) does the same threshold
that catches that rise also fire false alarms on buses that never collapse.
Both are checked here, exactly the way find_detector_trigger_time.py checked
them for the baseline scenario's Bus F holdout.

STRICT-PRECURSOR FEATURE SET: reproduced verbatim from
step41_early_warning_shifted_label.py's get_local_feature_map() /
DIRECT_SIGNAL_KEYS / STRICT_ADDITIONAL_KEYS. This MUST match exactly, in the
same column order, or the pretrained pipelines will silently score garbage
(sklearn Pipelines index features positionally, not by name).

Run (on the Windows machine, inside the venv, AFTER running
step33_build_gradual_busC_v4_features.py):
    python test_gradual_busC_v4_against_baseline_detectors.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "gradual_busC_v4_features"
FEATURE_PARQUET = FEATURE_DIR / "gradual_busC_v4_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "gradual_busC_v4_5000s_features.csv"

RAW_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_gradual_busC"
           / "thermal_derating_gradual_busC_v4_5000s.csv")

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
DETECTOR_DIR = MODEL_ROOT / "41_early_warning_shifted_label"

OUTPUT_DIR = MODEL_ROOT / "gradual_busC_v4_cross_scenario_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]
LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]

COLLAPSE_BUS = "C"
COLLAPSE_V_THRESHOLD = 100.0
COLLAPSE_MIN_RUN_SECONDS = 0.5   # matches the 50-sample @ 10ms debounce used everywhere else

# Threshold sweep -- deliberately wide, because the calibration artifact noted
# in this project (extreme class imbalance compresses probabilities toward 0)
# means the useful operating point could be anywhere from ~1e-8 to ~0.5.
CANDIDATE_THRESHOLDS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
SUSTAIN_SECONDS = 0.5   # an "alarm" must stay above threshold for this long to count,
                         # not just a single noisy row -- same spirit as the collapse debounce


# ----------------------------------------------------------------------
# EXACT COPY of the strict-precursor feature map from step41 (order matters:
# the pretrained pipelines expect this exact column set, this exact order)
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


def get_onsets(t: np.ndarray, collapsed: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[False], collapsed[:-1]])
    onset_mask = collapsed & ~prev
    return t[onset_mask]


def compute_collapsed_mask(v: np.ndarray, t: np.ndarray, threshold=COLLAPSE_V_THRESHOLD,
                            min_run_seconds=COLLAPSE_MIN_RUN_SECONDS) -> np.ndarray:
    """Same debounce rule as the MATLAB detect_collapse_summary: below
    threshold for a sustained run, converted from sample-count to seconds so
    it doesn't silently break if row spacing ever differs from 10ms."""
    below = v < threshold
    if len(t) > 1:
        dt = np.median(np.diff(t))
    else:
        dt = 0.01
    min_run = max(1, int(round(min_run_seconds / dt)))
    collapsed = np.zeros(len(v), dtype=bool)
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_run:
                collapsed[i:] = True   # once genuinely collapsed, stays collapsed (matches
                                        # the "sustained" framing used throughout this project)
                break
            i = j
        else:
            i += 1
    return collapsed


def sustained_trigger_time(t: np.ndarray, proba: np.ndarray, threshold: float,
                            sustain_seconds=SUSTAIN_SECONDS):
    """First time `proba` stays >= threshold for at least sustain_seconds.
    Returns None if it never does."""
    above = proba >= threshold
    if not above.any():
        return None
    if len(t) > 1:
        dt = np.median(np.diff(t))
    else:
        dt = 0.01
    min_run = max(1, int(round(sustain_seconds / dt)))
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if (j - i) >= min_run:
                return float(t[i])
            i = j
        else:
            i += 1
    return None


def main():
    print("=" * 70)
    print("CROSS-SCENARIO TEST: baseline N=5/15/30/60/90s detectors")
    print("vs. the NEW gradual_busC_v4 collapse (Bus C, t=2094.47s)")
    print("=" * 70)

    if not FEATURE_PARQUET.exists() and not FEATURE_CSV.exists():
        raise FileNotFoundError(
            "Run step33_build_gradual_busC_v4_features.py first -- no features found at:\n"
            f"{FEATURE_PARQUET}\n{FEATURE_CSV}"
        )
    wide = pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists() else pd.read_csv(FEATURE_CSV)
    wide = wide.sort_values("time").reset_index(drop=True)
    print(f"Loaded features: {len(wide):,} rows x {wide.shape[1]} cols")

    raw = pd.read_csv(RAW_CSV, usecols=["time"] + [f"V_Bus_{b}" for b in BUSES])
    raw = raw.sort_values("time").reset_index(drop=True)

    # Ground truth collapse mask per bus, computed directly from raw voltage
    # (same rule the MATLAB script itself used) -- this is what "trigger
    # before/after collapse" is measured against, independent of the
    # detectors entirely.
    collapsed_by_bus = {}
    onset_by_bus = {}
    for b in BUSES:
        v = raw[f"V_Bus_{b}"].to_numpy()
        t = raw["time"].to_numpy()
        collapsed = compute_collapsed_mask(v, t)
        collapsed_by_bus[b] = collapsed
        onsets = get_onsets(t, collapsed)
        onset_by_bus[b] = float(onsets[0]) if len(onsets) else None
    print("\nGround-truth collapse onsets (from raw voltage, independent of any detector):")
    for b in BUSES:
        onset = onset_by_bus[b]
        print(f"  Bus {b}: {'collapses at t=' + format(onset, '.2f') + 's' if onset is not None else 'never collapses'}")
    if onset_by_bus[COLLAPSE_BUS] is None:
        raise RuntimeError(f"Expected Bus {COLLAPSE_BUS} to collapse in this scenario, but it did not "
                            f"per this script's own voltage check -- stop and investigate before trusting "
                            f"anything below.")
    print(f"\n(Reference: the MATLAB run's own health check reported Bus C collapsing at t=2094.47s;"
          f"\n this script's independent recomputation gives t={onset_by_bus[COLLAPSE_BUS]:.2f}s"
          f" -- should match closely, small differences only from row alignment.)")

    canonical_cols = list(get_local_feature_map(0).keys())
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    print(f"\nUsing strict-precursor feature set: {len(strict_cols)} features (must match training exactly)")

    # Build each bus's strict-precursor feature matrix once (rows in `wide`'s time order)
    t_all = wide["time"].to_numpy()
    bus_matrices = {}
    for bus_idx, bus in enumerate(BUSES):
        fmap = get_local_feature_map(bus_idx)
        actual_cols = [fmap[c] for c in strict_cols]
        missing = [c for c in actual_cols if c not in wide.columns]
        if missing:
            raise RuntimeError(f"Bus {bus}: missing expected feature columns in gradual_busC_v4 "
                                f"features table (feature recipe mismatch?): {missing[:10]}")
        bus_matrices[bus] = wide[actual_cols].to_numpy(dtype=np.float64)

    all_results = {}

    for n_seconds in LEAD_TIMES_SECONDS:
        model_path = DETECTOR_DIR / f"early_warning_N{n_seconds}s.pkl"
        print("\n" + "-" * 70)
        print(f"N = {n_seconds}s   ({model_path.name})")
        print("-" * 70)
        if not model_path.exists():
            print(f"  MISSING: {model_path} -- skipping this N.")
            continue
        pipeline = joblib.load(model_path)

        proba_by_bus = {}
        for bus in BUSES:
            X = bus_matrices[bus]
            proba = pipeline.predict_proba(X)[:, 1]
            proba_by_bus[bus] = proba

        onset_c = onset_by_bus[COLLAPSE_BUS]
        proba_c = proba_by_bus[COLLAPSE_BUS]
        pre_onset_mask = t_all < onset_c
        max_proba_pre_onset = float(proba_c[pre_onset_mask].max()) if pre_onset_mask.any() else float("nan")
        max_proba_overall = float(proba_c.max())
        print(f"  Bus C (the bus that actually collapses): "
              f"max probability BEFORE onset = {max_proba_pre_onset:.3e}, "
              f"max probability anywhere = {max_proba_overall:.3e}")

        threshold_rows = []
        for thr in CANDIDATE_THRESHOLDS:
            trig_c = sustained_trigger_time(t_all, proba_c, thr)
            lead_time = (onset_c - trig_c) if (trig_c is not None and trig_c < onset_c) else None
            false_positive_buses = []
            for bus in BUSES:
                if bus == COLLAPSE_BUS:
                    continue
                trig_b = sustained_trigger_time(t_all, proba_by_bus[bus], thr)
                if trig_b is not None:
                    false_positive_buses.append(bus)
            threshold_rows.append({
                "threshold": thr,
                "bus_c_trigger_time": trig_c,
                "bus_c_lead_time_before_collapse": lead_time,
                "n_false_positive_buses": len(false_positive_buses),
                "false_positive_buses": false_positive_buses,
            })
            lead_str = f"{lead_time:.2f}s lead" if lead_time is not None else (
                "triggered AFTER collapse" if trig_c is not None else "never triggers")
            print(f"    threshold={thr:.0e}: Bus C -> {lead_str}"
                  f"  |  false alarms on {len(false_positive_buses)}/9 other buses "
                  f"{false_positive_buses if false_positive_buses else ''}")

        all_results[n_seconds] = {
            "max_proba_pre_onset_bus_c": max_proba_pre_onset,
            "max_proba_overall_bus_c": max_proba_overall,
            "onset_bus_c": onset_c,
            "threshold_sweep": threshold_rows,
        }

    # --------------------------------------------------------------------
    # Summary: which N actually gives usable early warning with acceptable
    # false-positive behavior, for THIS scenario.
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY -- best usable operating point per N (real lead time AND zero false alarms)")
    print("=" * 70)
    summary_rows = []
    for n_seconds, res in all_results.items():
        best = None
        for row in res["threshold_sweep"]:
            if row["bus_c_lead_time_before_collapse"] is not None and row["n_false_positive_buses"] == 0:
                if best is None or row["bus_c_lead_time_before_collapse"] > best["bus_c_lead_time_before_collapse"]:
                    best = row
        if best:
            print(f"  N={n_seconds:>2}s: BEST clean operating point -> threshold={best['threshold']:.0e}, "
                  f"lead time={best['bus_c_lead_time_before_collapse']:.2f}s, 0 false alarms")
        else:
            print(f"  N={n_seconds:>2}s: NO threshold gives real lead time with zero false alarms on this scenario.")
        summary_rows.append({"n_seconds": n_seconds, "best_clean_operating_point": best})

    out_path = OUTPUT_DIR / "cross_scenario_results.json"
    out_path.write_text(json.dumps({
        "scenario": "gradual_busC_v4",
        "collapse_bus": COLLAPSE_BUS,
        "collapse_onset": onset_by_bus[COLLAPSE_BUS],
        "detectors_source": "trained ONLY on baseline scenario (Bus F/E cascade) -- this is a "
                             "true cross-scenario generalization test, nothing retrained here.",
        "results_by_n": all_results,
        "summary": summary_rows,
    }, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    print("\nRead this the same way the baseline evaluation was read: a lead time alone is not")
    print("enough -- only trust an N if it ALSO has 0 false alarms on the 9 buses that never")
    print("collapse in this run. If every N shows either no signal or false alarms everywhere,")
    print("that itself is the finding: this collapse mechanism (slow thermal ramp) may need its")
    print("own freshly-trained detector rather than reusing the baseline's sudden-spike detectors.")

if __name__ == "__main__":
    main()
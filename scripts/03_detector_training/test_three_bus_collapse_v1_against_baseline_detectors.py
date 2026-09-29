"""
TEST: baseline-trained early-warning detectors (N=5/15/30/60/90s) against the
NEW three_bus_collapse_v1 regional collapse scenario (formerly
"corridor_triad_collapse").

This is a pure generalization test -- NOTHING is retrained here. The 5
MLPClassifier pipelines already sitting in
    model_outputs\\unified_controller_10bus_derating\\41_early_warning_shifted_label\\
        early_warning_N{5,15,30,60,90}s.pkl
were trained ONLY on the baseline scenario (Bus F/E sudden-spike cascade).
This script feeds them the brand-new three_bus_collapse_v1 run -- Bus G
collapses at t=1081.96s, Bus H at t=1082.05s, Bus K at t=1108.70s, all three
within 26.74s of each other (a genuine regional collapse per the MATLAB
script's own co-collapse window check) -- and asks: does ANY of these 5
lead-time models produce a usable early-warning signal on THREE buses at
once, in a scenario they have never seen, built from yet another different
collapse mechanism (correlated multi-bus preheat + synchronized surge +
boundary starvation, vs. baseline's sudden spike and gradual_busC_v4's
single-bus slow ramp)?

DIFFERENCE FROM THE gradual_busC_v4 VERSION OF THIS TEST: there, only one bus
(C) ever collapsed, so there was one "does it detect the real collapse" check
and 9 "does it false-alarm on a bus that never collapses" checks. Here there
are THREE buses that genuinely collapse (G, H, K), so this script reports a
per-target-bus lead time/trigger AND checks false alarms only against the
remaining 7 buses that never collapse in this run (A, B, C, D, E, F, L).

WHY THIS MATTERS (per the project's own prior finding): pooled/AUC-style
metrics can look fine while a detector is actually just memorizing one
training bus. The only honest test is: (a) does probability rise on each bus
that actually collapses, before it collapses, and (b) does the same
threshold that catches that rise also fire false alarms on buses that never
collapse. Both are checked here, exactly the way find_detector_trigger_time.py
checked them for the baseline scenario's Bus F holdout, and the same way
test_gradual_busC_v4_against_baseline_detectors.py checked them for Bus C.

STRICT-PRECURSOR FEATURE SET: reproduced verbatim from
step41_early_warning_shifted_label.py's get_local_feature_map() /
DIRECT_SIGNAL_KEYS / STRICT_ADDITIONAL_KEYS. This MUST match exactly, in the
same column order, or the pretrained pipelines will silently score garbage
(sklearn Pipelines index features positionally, not by name).

Run (on the Windows machine, inside the venv, AFTER running
step33_build_three_bus_collapse_v1_features.py):
    python test_three_bus_collapse_v1_against_baseline_detectors.py
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
FEATURE_DIR = PROJECT_ROOT / "data_d" / "three_bus_collapse_v1_features"
FEATURE_PARQUET = FEATURE_DIR / "three_bus_collapse_v1_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "three_bus_collapse_v1_5000s_features.csv"

RAW_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_three_bus_collapse"
           / "thermal_derating_three_bus_collapse_v1_5000s.csv")

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
DETECTOR_DIR = MODEL_ROOT / "41_early_warning_shifted_label"

OUTPUT_DIR = MODEL_ROOT / "three_bus_collapse_v1_cross_scenario_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]
LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]

# The three buses this scenario was designed to collapse together, plus the
# two immediate boundary buses -- everyone else (A, B, C, D, E) is ordinary
# background load and should never collapse either.
COLLAPSE_BUSES = ["G", "H", "K"]
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
    print("vs. the NEW three_bus_collapse_v1 regional collapse (G, H, K)")
    print("=" * 70)

    if not FEATURE_PARQUET.exists() and not FEATURE_CSV.exists():
        raise FileNotFoundError(
            "Run step33_build_three_bus_collapse_v1_features.py first -- no features found at:\n"
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

    missing_collapse = [b for b in COLLAPSE_BUSES if onset_by_bus[b] is None]
    if missing_collapse:
        raise RuntimeError(f"Expected buses {COLLAPSE_BUSES} to collapse in this scenario, but "
                            f"{missing_collapse} did not per this script's own voltage check -- "
                            f"stop and investigate before trusting anything below.")
    unexpected_collapse = [b for b in BUSES if b not in COLLAPSE_BUSES and onset_by_bus[b] is not None]
    if unexpected_collapse:
        print(f"\nNOTE: bus(es) {unexpected_collapse} also collapsed, outside the intended "
              f"target trio {COLLAPSE_BUSES} -- treating these as additional true collapses, "
              f"not false alarms, in the analysis below.")

    print(f"\n(Reference: the MATLAB run's own health check + co-collapse window verdict reported "
          f"G/H/K collapsing at t=1081.96/1082.05/1108.70s, all within 26.74s of each other;"
          f"\n this script's independent recomputation gives:"
          + "".join(f" {b}=t{onset_by_bus[b]:.2f}s" for b in COLLAPSE_BUSES)
          + " -- should match closely, small differences only from row alignment.)")

    # False-alarm buses: everyone that never actually collapses in this run
    # (normally the 7 background/boundary buses A, B, C, D, E, F, L).
    never_collapse_buses = [b for b in BUSES if onset_by_bus[b] is None]
    print(f"\nBuses that never collapse in this run (used for false-alarm checks): {never_collapse_buses}")

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
            raise RuntimeError(f"Bus {bus}: missing expected feature columns in three_bus_collapse_v1 "
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

        # Per-target-bus lead time / max probability
        per_target = {}
        for tb in COLLAPSE_BUSES:
            onset_tb = onset_by_bus[tb]
            proba_tb = proba_by_bus[tb]
            pre_onset_mask = t_all < onset_tb
            max_proba_pre_onset = float(proba_tb[pre_onset_mask].max()) if pre_onset_mask.any() else float("nan")
            max_proba_overall = float(proba_tb.max())
            print(f"  Bus {tb} (collapses at t={onset_tb:.2f}s): "
                  f"max probability BEFORE onset = {max_proba_pre_onset:.3e}, "
                  f"max probability anywhere = {max_proba_overall:.3e}")
            per_target[tb] = {
                "onset": onset_tb,
                "max_proba_pre_onset": max_proba_pre_onset,
                "max_proba_overall": max_proba_overall,
            }

        threshold_rows = []
        for thr in CANDIDATE_THRESHOLDS:
            per_target_trigger = {}
            for tb in COLLAPSE_BUSES:
                onset_tb = onset_by_bus[tb]
                trig_tb = sustained_trigger_time(t_all, proba_by_bus[tb], thr)
                lead_time = (onset_tb - trig_tb) if (trig_tb is not None and trig_tb < onset_tb) else None
                per_target_trigger[tb] = {"trigger_time": trig_tb, "lead_time": lead_time}
            false_positive_buses = []
            for bus in never_collapse_buses:
                trig_b = sustained_trigger_time(t_all, proba_by_bus[bus], thr)
                if trig_b is not None:
                    false_positive_buses.append(bus)
            threshold_rows.append({
                "threshold": thr,
                "per_target_bus": per_target_trigger,
                "n_false_positive_buses": len(false_positive_buses),
                "false_positive_buses": false_positive_buses,
            })
            lead_strs = []
            for tb in COLLAPSE_BUSES:
                lt = per_target_trigger[tb]["lead_time"]
                tt = per_target_trigger[tb]["trigger_time"]
                if lt is not None:
                    lead_strs.append(f"{tb}:{lt:.1f}s lead")
                elif tt is not None:
                    lead_strs.append(f"{tb}:AFTER collapse")
                else:
                    lead_strs.append(f"{tb}:never triggers")
            print(f"    threshold={thr:.0e}: " + ", ".join(lead_strs)
                  + f"  |  false alarms on {len(false_positive_buses)}/{len(never_collapse_buses)} "
                  f"other buses {false_positive_buses if false_positive_buses else ''}")

        all_results[n_seconds] = {
            "per_target_bus_summary": per_target,
            "threshold_sweep": threshold_rows,
        }

    # --------------------------------------------------------------------
    # Summary: which N actually gives usable early warning, for ALL THREE
    # target buses simultaneously, with acceptable false-positive behavior.
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY -- best usable operating point per N")
    print("(must give real lead time on ALL of G, H, K AND zero false alarms elsewhere)")
    print("=" * 70)
    summary_rows = []
    for n_seconds, res in all_results.items():
        best = None
        for row in res["threshold_sweep"]:
            all_have_lead = all(row["per_target_bus"][tb]["lead_time"] is not None for tb in COLLAPSE_BUSES)
            if all_have_lead and row["n_false_positive_buses"] == 0:
                min_lead = min(row["per_target_bus"][tb]["lead_time"] for tb in COLLAPSE_BUSES)
                if best is None or min_lead > best["min_lead_time"]:
                    best = {
                        "threshold": row["threshold"],
                        "min_lead_time": min_lead,
                        "per_target_bus": row["per_target_bus"],
                    }
        if best:
            lead_detail = ", ".join(f"{tb}={best['per_target_bus'][tb]['lead_time']:.1f}s"
                                     for tb in COLLAPSE_BUSES)
            print(f"  N={n_seconds:>2}s: BEST clean operating point -> threshold={best['threshold']:.0e}, "
                  f"min lead time={best['min_lead_time']:.2f}s ({lead_detail}), 0 false alarms")
        else:
            print(f"  N={n_seconds:>2}s: NO threshold gives real lead time on ALL of G/H/K with zero "
                  f"false alarms on this scenario.")
        summary_rows.append({"n_seconds": n_seconds, "best_clean_operating_point": best})

    out_path = OUTPUT_DIR / "cross_scenario_results.json"
    out_path.write_text(json.dumps({
        "scenario": "three_bus_collapse_v1",
        "collapse_buses": COLLAPSE_BUSES,
        "onset_by_collapse_bus": {b: onset_by_bus[b] for b in COLLAPSE_BUSES},
        "never_collapse_buses": never_collapse_buses,
        "detectors_source": "trained ONLY on baseline scenario (Bus F/E cascade) -- this is a "
                             "true cross-scenario generalization test, nothing retrained here.",
        "results_by_n": all_results,
        "summary": summary_rows,
    }, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    print("\nRead this the same way the gradual_busC_v4 cross-scenario evaluation was read: a lead")
    print("time alone is not enough -- only trust an N if it ALSO gives real lead time on ALL THREE")
    print("target buses (not just one of them) AND has 0 false alarms on the buses that never")
    print("collapse in this run. If no N clears that bar, that itself is the finding: a genuinely")
    print("correlated multi-bus regional collapse may need its own freshly-trained detector rather")
    print("than reusing the baseline's single-bus sudden-spike detectors.")

if __name__ == "__main__":
    main()
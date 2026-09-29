"""
STEP 40: FINE ONSET-BUCKET CHECK + t=4200 LOAD-EVENT ALIGNMENT
================================================================================
Follows directly from step39's plots. The strict-precursor model's holdout
recall (0.659) turned out to be ENTIRELY inside the 100s+ bucket, recall was
exactly 0 in every bucket under 100s. That means the model detects Bus F's
collapse late, not early, and the open question is WHY: is the confidence
rise around t~3950-4300 tracking F's own original collapse at t~3643 slowly
building evidence, or is it actually reacting to the grid's SECOND load event
at t=4200 (grid_derating_dataset_5000s_v7_2.m: event(t>=4200) = -(10000+600k)),
which is a different physical trigger that happens to land inside F's still-
collapsed window?

This script does three things:

  1. Re-runs the onset-recall breakdown with much finer buckets under 600s
     (0-5, 5-30, 30-100, 100-250, 250-400, 400-600, 600+), for all three
     saved/retrained models side by side, so the shape of the rise is visible
     precisely instead of hidden inside one wide "100-inf" bucket.

  2. Plots Bus F's predicted probability zoomed to its collapse window with an
     explicit vertical marker at t=4200 (the load event) alongside the actual
     debounced onset at t~3643. If the confidence curve's steepest rise sits
     right at or just after t=4200 rather than shortly after the true onset,
     that is direct visual evidence for the load-event-reaction hypothesis
     over the slow-precursor hypothesis.

  3. Prints the actual predicted probability value in a narrow window
     straddling t=4200, plus the probability's rate of change there vs. at
     the true onset, as a numeric complement to the plot.

Does NOT retrain anything. Loads the strict-precursor pickle saved by step39
(strict_precursor_collapse_detector_retrained.pkl) and the two models saved
earlier (step36 full, step38 raw ceiling). If any is missing, that part of
the comparison is skipped, not fatal.

Run:
    python step40_onset_bucket_and_event_check.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths, must match step33/36/38/39
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
RAW_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
           / "thermal_derating_v7_ALfix_GEIfix_5000s.csv")
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
FULL_MODEL_PATH = MODEL_ROOT / "36_bus_symmetric_detector" / "bus_symmetric_collapse_detector.pkl"
RAW_CEILING_MODEL_PATH = MODEL_ROOT / "38_raw_signal_detector" / "raw_signal_collapse_detector.pkl"
STRICT_MODEL_PATH = MODEL_ROOT / "39_presentation_plots" / "strict_precursor_collapse_detector_retrained.pkl"

OUTPUT_DIR = MODEL_ROOT / "40_onset_bucket_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5
COLLAPSE_VOLTAGE_V = 100.0

# The grid-wide load event from grid_derating_dataset_5000s_v7_2.m:
#   event(t >= 3000 & t < 4200) = 20000 + 1200*k     <- the event driving the collapse
#   event(t >= 4200)            = -(10000 + 600*k)   <- the SECOND event, direction reverses
LOAD_EVENT_T_ORIGINAL = 3000.0
LOAD_EVENT_T_SECOND = 4200.0

# Fine buckets under 600s, coarse catch-all after. This is the whole point of
# this script: step39 only had one bucket ("100-inf") covering everything
# past 100s, which is exactly where the interesting structure was hiding.
ONSET_BUCKETS = [(0, 5), (5, 30), (30, 100), (100, 250), (250, 400), (400, 600), (600, np.inf)]

N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE = 20, 0.20, 10, 42


# ----------------------------------------------------------------------
# Shared utilities, identical to step36/37/38/39
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


RAW_FEATURE_KEYS = (
    ["own_Voltage", "own_Voltage_lag1", "own_Voltage_lag2", "own_Voltage_lag3",
     "own_Voltage_roll_mean_5", "own_Voltage_roll_std_5", "own_d_Voltage"]
    + ["own_GEI", "own_GEI_lag1", "own_GEI_lag2", "own_GEI_lag3",
       "own_GEI_roll_mean_5", "own_GEI_roll_std_5", "own_d_GEI"]
    + ["own_Commanded_Load", "own_Commanded_Load_lag1", "own_Commanded_Load_lag2",
       "own_Commanded_Load_lag3", "own_Commanded_Load_roll_mean_5",
       "own_Commanded_Load_roll_std_5", "own_d_Commanded_Load"]
    + [f"{role}_Junction_Temp{suffix}" for role in ("conv_prev", "conv_next")
       for suffix in ("", "_lag1", "_lag2", "_lag3", "_roll_mean_5", "_roll_std_5")]
    + [f"{role}_d_Junction_Temp" for role in ("conv_prev", "conv_next")]
    + [f"{role}_Heat_Sink_Temp{suffix}" for role in ("conv_prev", "conv_next")
       for suffix in ("", "_lag1", "_lag2", "_lag3", "_roll_mean_5", "_roll_std_5")]
    + [f"{role}_d_Heat_Sink_Temp" for role in ("conv_prev", "conv_next")]
    + ["conv_prev_Junction_HeatSink_Diff", "conv_next_Junction_HeatSink_Diff"]
)

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


def blocked_purged_split(n_rows, n_blocks, test_frac, purge_rows, random_state):
    edges = np.linspace(0, n_rows, n_blocks + 1).astype(int)
    blocks = [(edges[i], edges[i+1]) for i in range(n_blocks)]
    rng = np.random.default_rng(random_state)
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)
    n_test_blocks = max(1, int(round(n_blocks * test_frac)))
    test_block_set = set(block_ids[:n_test_blocks].tolist())
    split = np.full(n_rows, "purge", dtype=object)
    for bi, (start, end) in enumerate(blocks):
        p_start = start + purge_rows if bi > 0 else start
        p_end   = end - purge_rows if bi < n_blocks - 1 else end
        if p_end <= p_start:
            continue
        split[p_start:p_end] = "test" if bi in test_block_set else "train"
    return split


def load_corrected_targets():
    df = pd.read_csv(CORRECTED_TARGETS_CSV)
    bool_cols = [c for c in df.columns if c.startswith("affected_") or c.startswith("collapsed_")]
    for c in bool_cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df[c] = df[c].astype(bool)
    return df


def build_bus_frame(bus_idx, wide, canonical_cols):
    fmap = get_local_feature_map(bus_idx)
    bus = BUSES[bus_idx]
    actual_cols = [fmap[c] for c in canonical_cols]
    sub = wide[actual_cols].copy()
    sub.columns = canonical_cols
    sub["time"] = wide["time"].to_numpy()
    sub["y_collapsed"] = wide[f"collapsed_{bus}"].to_numpy().astype(int)
    return sub


def onset_recall_fine(sub_holdout: pd.DataFrame, y_proba: np.ndarray, threshold: float = 0.5):
    t = sub_holdout["time"].to_numpy()
    y_true = sub_holdout["y_collapsed"].to_numpy().astype(bool)
    y_pred = (y_proba >= threshold).astype(int).astype(bool)
    since_onset = np.full(len(t), np.nan)
    in_run, onset_t = False, None
    for i in range(len(t)):
        if y_true[i] and not in_run:
            in_run, onset_t = True, t[i]
        elif not y_true[i]:
            in_run, onset_t = False, None
        if in_run:
            since_onset[i] = t[i] - onset_t
    rows = []
    for lo, hi in ONSET_BUCKETS:
        mask = (since_onset >= lo) & (since_onset < hi)
        n = int(mask.sum())
        rec = float(y_pred[mask].mean()) if n > 0 else np.nan
        hi_str = "inf" if hi == np.inf else f"{hi:.0f}"
        rows.append({"window": f"{lo:.0f}-{hi_str}s", "n_rows": n, "recall": rec})
    return rows, onset_t if False else None  # onset_t not meaningful post-loop, ignore


def get_true_onset_time(sub_holdout: pd.DataFrame) -> float:
    y_true = sub_holdout.sort_values("time")["y_collapsed"].to_numpy().astype(bool)
    t = sub_holdout.sort_values("time")["time"].to_numpy()
    if not y_true.any():
        return np.nan
    return float(t[np.argmax(y_true)])


def main():
    print("=" * 70)
    print("STEP 40: FINE ONSET-BUCKET CHECK + t=4200 LOAD-EVENT ALIGNMENT")
    print("=" * 70)

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)

    split = blocked_purged_split(len(wide), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide["_time_split"] = split
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    print(f"Working table: {len(wide):,} rows")

    canonical_cols = list(get_local_feature_map(0).keys())
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]

    bus_idx_F = BUSES.index(HOLDOUT_BUS)
    sub_full = build_bus_frame(bus_idx_F, wide, canonical_cols)
    sub_raw = build_bus_frame(bus_idx_F, wide, RAW_FEATURE_KEYS) if RAW_FEATURE_KEYS else None
    sub_strict = sub_full[strict_cols + ["time", "y_collapsed"]].copy()

    true_onset = get_true_onset_time(sub_full)
    print(f"\nBus F true debounced collapse onset: t={true_onset:.2f}s")
    print(f"Second load-schedule event (grid_derating_dataset_5000s_v7_2.m): t={LOAD_EVENT_T_SECOND:.0f}s")
    print(f"Gap between true onset and second load event: "
          f"{LOAD_EVENT_T_SECOND - true_onset:.1f}s")

    models_to_check = {}
    if FULL_MODEL_PATH.exists():
        models_to_check["Full (121 features)"] = (joblib.load(FULL_MODEL_PATH), canonical_cols, sub_full)
    else:
        print(f"SKIPPING full model, not found: {FULL_MODEL_PATH}")
    if RAW_CEILING_MODEL_PATH.exists() and sub_raw is not None:
        models_to_check["Raw ceiling (51 features)"] = (joblib.load(RAW_CEILING_MODEL_PATH), RAW_FEATURE_KEYS, sub_raw)
    else:
        print(f"SKIPPING raw ceiling model, not found: {RAW_CEILING_MODEL_PATH}")
    if STRICT_MODEL_PATH.exists():
        models_to_check["Strict precursor (83 features)"] = (joblib.load(STRICT_MODEL_PATH), strict_cols, sub_strict)
    else:
        print(f"SKIPPING strict precursor model, not found: {STRICT_MODEL_PATH}")

    if not models_to_check:
        raise FileNotFoundError("No saved models found, cannot proceed. Check the paths at the top of this script.")

    # --------------------------------------------------------------------
    # 1. Fine onset-bucket recall, all available models, side by side
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINE ONSET-BUCKET RECALL, BUS F (threshold=0.5)")
    print("=" * 70)
    all_bucket_rows = {}
    all_proba = {}
    for name, (pipeline, cols, sub) in models_to_check.items():
        sub_sorted = sub.sort_values("time").reset_index(drop=True)
        X = sub_sorted[cols].to_numpy(dtype=np.float64)
        proba = pipeline.predict_proba(X)[:, 1]
        all_proba[name] = (sub_sorted, proba)
        rows, _ = onset_recall_fine(sub_sorted, proba)
        all_bucket_rows[name] = rows
        print(f"\n{name}:")
        print(f"  {'window':>10s} {'n_rows':>8s} {'recall':>8s}")
        for r in rows:
            rec_str = "n/a" if np.isnan(r["recall"]) else f"{r['recall']:.3f}"
            print(f"  {r['window']:>10s} {r['n_rows']:>8d} {rec_str:>8s}")

    fig, ax = plt.subplots(figsize=(13, 6))
    labels = [r["window"] for r in next(iter(all_bucket_rows.values()))]
    x = np.arange(len(labels))
    width = 0.8 / max(len(all_bucket_rows), 1)
    colors = ["#85B7EB", "#F0997B", "#1D9E75"]
    for i, (name, rows) in enumerate(all_bucket_rows.items()):
        vals = [0 if np.isnan(r["recall"]) else r["recall"] for r in rows]
        ax.bar(x + (i - (len(all_bucket_rows) - 1) / 2) * width, vals, width,
               label=name, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Bus F recall by fine time-since-onset window, all models\n"
                 f"(true onset t={true_onset:.0f}s, second load event at t={LOAD_EVENT_T_SECOND:.0f}s "
                 f"is {LOAD_EVENT_T_SECOND - true_onset:.0f}s after onset, in the 400-600s bucket)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out1 = OUTPUT_DIR / "fine_onset_bucket_recall_comparison.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nWrote: {out1}")

    # --------------------------------------------------------------------
    # 2. Strict-precursor probability zoomed to F's collapse, t=4200 marked
    # --------------------------------------------------------------------
    if "Strict precursor (83 features)" in all_proba:
        sub_sorted, proba = all_proba["Strict precursor (83 features)"]
        t = sub_sorted["time"].to_numpy()
        y_true = sub_sorted["y_collapsed"].to_numpy().astype(bool)
        t0 = max(t[0], true_onset - 100)
        t1 = min(t[-1], true_onset + 800)
        mask = (t >= t0) & (t <= t1)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(t[mask], 0, 1, where=y_true[mask], color="tab:blue", alpha=0.15,
                        step="pre", label="actual collapsed")
        ax.plot(t[mask], proba[mask], color="tab:orange", linewidth=1.5, label="predicted probability")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(true_onset, color="tab:green", linestyle="--", linewidth=1.3,
                   label=f"true collapse onset (t={true_onset:.0f}s)")
        ax.axvline(LOAD_EVENT_T_SECOND, color="tab:red", linestyle="--", linewidth=1.3,
                   label=f"2nd load event (t={LOAD_EVENT_T_SECOND:.0f}s)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("P(collapsed)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Strict-precursor model, Bus F: does the confidence rise track true onset or the 2nd load event?")
        ax.legend(loc="center right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out2 = OUTPUT_DIR / "strict_precursor_F_zoomed_with_event_markers.png"
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"Wrote: {out2}")

        # --------------------------------------------------------------------
        # 3. Numeric check: probability level and local slope at each marker
        # --------------------------------------------------------------------
        def local_slope_and_level(center_t, window=15):
            m = (t >= center_t - window) & (t <= center_t + window)
            if m.sum() < 2:
                return np.nan, np.nan
            tt, pp = t[m], proba[m]
            level = float(np.interp(center_t, tt, pp))
            slope = float(np.polyfit(tt, pp, 1)[0])  # probability change per second
            return level, slope

        onset_level, onset_slope = local_slope_and_level(true_onset)
        event_level, event_slope = local_slope_and_level(LOAD_EVENT_T_SECOND)
        print("\n--- Numeric check: probability level and local slope (prob/sec) ---")
        print(f"  At true onset   (t={true_onset:.0f}s): level={onset_level:.4f}  slope={onset_slope:.6f}/s")
        print(f"  At 2nd event    (t={LOAD_EVENT_T_SECOND:.0f}s): level={event_level:.4f}  slope={event_slope:.6f}/s")
        if not np.isnan(event_slope) and not np.isnan(onset_slope):
            if abs(event_slope) > abs(onset_slope) * 1.5:
                print("\n  Slope is markedly steeper at the 2nd load event than at true onset. "
                      "This SUPPORTS the load-event-reaction hypothesis: the model's confidence is "
                      "rising fastest near the second scheduled disturbance, not near the original "
                      "collapse instant.")
            elif abs(onset_slope) > abs(event_slope) * 1.5:
                print("\n  Slope is markedly steeper near true onset than at the 2nd load event. "
                      "This SUPPORTS the slow-precursor hypothesis over the load-event-reaction one.")
            else:
                print("\n  Slopes are comparable at both markers, not conclusive from this check alone. "
                      "Look at the plot directly and consider shifting the label to test this properly.")

    print("\nDone. Review:")
    print(f"  {out1}")
    if "Strict precursor (83 features)" in all_proba:
        print(f"  {out2}")


if __name__ == "__main__":
    main()

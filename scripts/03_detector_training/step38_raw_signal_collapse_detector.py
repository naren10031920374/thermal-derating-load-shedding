"""
STEP 38: RAW-SIGNAL COLLAPSE DETECTOR (voltage, GEI, load, Tj, Th, Tj-Th, rates)
================================================================================
Same bus-symmetric, single-shared-model, Bus-F-completely-held-out methodology
as bus_symmetric_collapse_detector.py. The only change is WHICH canonical
features go in: instead of the full 121-feature engineered set, this uses
only the raw physical signals requested directly: own voltage, own GEI, own
commanded load, and the two adjacent converters' junction temp, heat sink
temp, and Tj-Th differential, each with lag1-3, rolling mean/std (trend), and
first difference (rate) where those variants exist in the canonical map.

THIS IS A CEILING TEST, NOT A PRECURSOR TEST. own_Voltage and own_GEI are
same-instant symptoms of the label (collapsed_<BUS> is literally defined as
own_Voltage < 100V for 0.5s; GEI crashes in the same instant because a
near-zero-volt bus delivers near-zero power). Expect this to score very high,
comparable to the earlier full-feature result. That is not evidence of early
or transferable pattern recognition, it mostly confirms the reshaping and
holdout mechanics still work with this smaller feature set. The useful
by-products are (a) the direct comparison of "5 raw signal families" vs "121
engineered features" vs "83 strict-precursor features" all on the identical
Bus-F holdout, and (b) the time-since-onset recall breakdown below, which
starts answering "how early" without a second experiment.

TIME-SINCE-ONSET BREAKDOWN: for Bus F's real collapse window, this buckets
recall by how many seconds have elapsed since the debounced onset (0-5s,
5-30s, 30-100s, 100s+). If recall is near 1.0 in every bucket including 0-5s,
detection is immediate, this model is not catching anything before the
voltage threshold trips, consistent with it being a same-instant reader. If
recall is low in the 0-5s bucket and climbs afterward, that is a hint the
model is lagging the debounce itself, worth knowing before building an
actual early-warning (shifted-label) version next.

Run:
    python step38_raw_signal_collapse_detector.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, roc_auc_score, confusion_matrix)

# ----------------------------------------------------------------------
# Config, paths/split identical to bus_symmetric_collapse_detector.py
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

OUTPUT_DIR = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
              / "38_raw_signal_detector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5     # match bus_symmetric_collapse_detector.py for a fair comparison

N_BLOCKS     = 20
TEST_FRAC    = 0.20
PURGE_ROWS   = 10
RANDOM_STATE = 42

HIDDEN_LAYER_SIZES = (128, 64)
MLP_KWARGS = dict(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    activation="relu",
    solver="adam",
    alpha=0.01,
    batch_size=256,
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=False,
    n_iter_no_change=25,
    tol=1e-4,
    random_state=RANDOM_STATE,
    verbose=True,
)

# Buckets for the time-since-onset recall breakdown, in seconds.
ONSET_BUCKETS = [(0, 5), (5, 30), (30, 100), (100, np.inf)]

# Threshold sweep for the precision/recall tradeoff (see step37 finding:
# precision pinned at 1.000 with strict features means 0.5 is too conservative
# for a ~2-3% positive class; this checks whether that also holds here).
THRESHOLD_SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


# ----------------------------------------------------------------------
# Canonical local+neighbor feature map, IDENTICAL to bus_symmetric_collapse_detector.py
# (kept in full so this script runs standalone with no import dependency)
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
        "GEI_Error": f"GEI_Error_{bus}",
        "Power_Imbalance": f"Power_Imbalance_{bus}",
        "Voltage_Imbalance": f"Voltage_Imbalance_{bus}",
        "Load_Imbalance": f"Load_Imbalance_{bus}",
    }
    for name, col in imb_with_lag.items():
        m[f"own_{name}"] = col
        for lag in (1, 2, 3):
            m[f"own_{name}_lag{lag}"] = f"{col}_lag{lag}"

    imb_no_lag = {
        "Abs_GEI_Error": f"Abs_GEI_Error_{bus}",
        "Abs_Voltage_Imbalance": f"Abs_Voltage_Imbalance_{bus}",
        "Abs_Power_Imbalance": f"Abs_Power_Imbalance_{bus}",
        "Temp_Imbalance": f"Bus_{bus}_Temp_Imbalance",
        "Abs_Temp_Imbalance": f"Abs_Bus_{bus}_Temp_Imbalance",
        "Abs_Load_Imbalance": f"Abs_Load_Imbalance_{bus}",
        "Voltage_Error": f"Voltage_Error_{bus}",
        "Abs_Voltage_Error": f"Abs_Voltage_Error_{bus}",
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
            "Temp_Imbalance": f"DAB_{conv}_Temp_Imbalance",
            "Abs_Temp_Imbalance": f"Abs_DAB_{conv}_Temp_Imbalance",
            "Derate_Imbalance": f"DAB_{conv}_Derate_Imbalance",
            "Abs_Derate_Imbalance": f"Abs_DAB_{conv}_Derate_Imbalance",
            "Junction_Imbalance": f"DAB_{conv}_Junction_Imbalance",
            "Abs_Junction_Imbalance": f"Abs_DAB_{conv}_Junction_Imbalance",
            "Junction_HeatSink_Diff": f"DAB_{conv}_Junction_HeatSink_Diff",
        }
        for name, col in imb_map.items():
            m[f"{role}_{name}"] = col

    for role, nb in [("neighbor_prev", nb_prev), ("neighbor_next", nb_next)]:
        for sig in ["Voltage", "Source_Power", "GEI", "Temp"]:
            m[f"{role}_{sig}"] = f"Bus_{nb}_{sig}"

    return m


# ----------------------------------------------------------------------
# The raw-signal subset. Every key below must exist in get_local_feature_map's
# output for every bus_idx, checked at runtime, not assumed.
# ----------------------------------------------------------------------
RAW_FEATURE_KEYS = (
    # own voltage: value, trend, rate (same-instant symptom, included on request)
    ["own_Voltage", "own_Voltage_lag1", "own_Voltage_lag2", "own_Voltage_lag3",
     "own_Voltage_roll_mean_5", "own_Voltage_roll_std_5", "own_d_Voltage"]
    # own GEI: value, trend, rate (same-instant symptom, included on request)
    + ["own_GEI", "own_GEI_lag1", "own_GEI_lag2", "own_GEI_lag3",
       "own_GEI_roll_mean_5", "own_GEI_roll_std_5", "own_d_GEI"]
    # own commanded load: value, trend, rate (the one genuinely causal driver)
    + ["own_Commanded_Load", "own_Commanded_Load_lag1", "own_Commanded_Load_lag2",
       "own_Commanded_Load_lag3", "own_Commanded_Load_roll_mean_5",
       "own_Commanded_Load_roll_std_5", "own_d_Commanded_Load"]
    # adjacent converters' junction temp: value, trend, rate, both directions
    + [f"{role}_Junction_Temp{suffix}"
       for role in ("conv_prev", "conv_next")
       for suffix in ("", "_lag1", "_lag2", "_lag3", "_roll_mean_5", "_roll_std_5")]
    + [f"{role}_d_Junction_Temp" for role in ("conv_prev", "conv_next")]
    # adjacent converters' heat sink temp: value, trend, rate, both directions
    + [f"{role}_Heat_Sink_Temp{suffix}"
       for role in ("conv_prev", "conv_next")
       for suffix in ("", "_lag1", "_lag2", "_lag3", "_roll_mean_5", "_roll_std_5")]
    + [f"{role}_d_Heat_Sink_Temp" for role in ("conv_prev", "conv_next")]
    # Tj-Th differential, both directions (proportional to instantaneous heat flow)
    + ["conv_prev_Junction_HeatSink_Diff", "conv_next_Junction_HeatSink_Diff"]
)


def load_corrected_targets():
    df = pd.read_csv(CORRECTED_TARGETS_CSV)
    bool_cols = [c for c in df.columns if c.startswith("affected_") or c.startswith("collapsed_")]
    for c in bool_cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df[c] = df[c].astype(bool)
    return df


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


def evaluate(name, y_true, y_proba, threshold=0.5):
    if len(y_true) == 0:
        print(f"\n{name}: no rows, skipping.")
        return None
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    if y_true.sum() == 0:
        print(f"\n{name}: accuracy={acc:.4f} (no positive examples, precision/recall n/a)")
        return {"accuracy": acc, "precision": None, "recall": None, "f1": None, "auc": None}
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"\n{name} (threshold={threshold}):")
    print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  "
          f"auc={auc if auc is None else round(auc,4)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def onset_recall_breakdown(sub_holdout: pd.DataFrame, y_proba: np.ndarray, threshold: float = 0.5):
    """
    sub_holdout must have 'time' and 'y_collapsed' columns in the same row order
    as y_proba, for the held-out bus only. Buckets recall by seconds elapsed
    since the debounced collapse onset (the first row where y_collapsed flips
    True after a run of False), restarting the clock each time the bus recovers
    and collapses again.
    """
    t = sub_holdout["time"].to_numpy()
    y_true = sub_holdout["y_collapsed"].to_numpy().astype(bool)
    y_pred = (y_proba >= threshold).astype(int).astype(bool)

    # seconds since the most recent onset, for every row currently collapsed
    since_onset = np.full(len(t), np.nan)
    in_run = False
    onset_t = None
    for i in range(len(t)):
        if y_true[i] and not in_run:
            in_run = True
            onset_t = t[i]
        elif not y_true[i]:
            in_run = False
            onset_t = None
        if in_run:
            since_onset[i] = t[i] - onset_t

    print("\n--- Recall by time-since-onset (held-out bus, real collapse rows only) ---")
    print(f"{'window':>14s} {'n_rows':>8s} {'recall':>8s}")
    rows = []
    for lo, hi in ONSET_BUCKETS:
        mask = (since_onset >= lo) & (since_onset < hi)
        n = mask.sum()
        if n == 0:
            print(f"{lo:>5.0f}-{hi if hi != np.inf else 'inf':<5}s {n:>8d}      n/a")
            continue
        rec = y_pred[mask].mean()   # all rows here are true positives by construction
        rows.append({"window_start_s": lo, "window_end_s": None if hi == np.inf else hi,
                      "n_rows": int(n), "recall": float(rec)})
        hi_str = "inf" if hi == np.inf else f"{hi:.0f}"
        print(f"{lo:>5.0f}-{hi_str:<5}s {n:>8d}   {rec:6.3f}")
    return rows


def main():
    print("=" * 70)
    print("STEP 38: RAW-SIGNAL COLLAPSE DETECTOR")
    print("=" * 70)
    print(f"Feature families: own voltage, own GEI, own commanded load, "
          f"adjacent Tj, adjacent Th, adjacent Tj-Th diff")
    print("NOTE: voltage and GEI are same-instant symptoms of the label, this is a "
          "ceiling test, not a precursor test. See step37 (bus_symmetric_precursor_only_test.py) "
          "for the precursor-only comparison.")

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    print(f"\nJoined wide table: {len(wide):,} rows")

    time_split = blocked_purged_split(len(wide), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide["_time_split"] = time_split

    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
        print(f"Subsampled every {SUBSAMPLE_STRIDE} rows: {len(wide):,} timesteps remain")

    print("\nReshaping to bus-symmetric long format, raw-signal columns only ...")
    long_frames = []
    for bus_idx, bus in enumerate(BUSES):
        fmap = get_local_feature_map(bus_idx)
        missing_keys = [k for k in RAW_FEATURE_KEYS if k not in fmap]
        if missing_keys:
            raise ValueError(f"RAW_FEATURE_KEYS references keys not produced by "
                              f"get_local_feature_map: {missing_keys}")
        actual_cols = [fmap[k] for k in RAW_FEATURE_KEYS]
        missing = [c for c in actual_cols if c not in wide.columns]
        if missing:
            raise ValueError(f"Bus {bus}: missing expected columns in the feature file:\n{missing[:10]}")
        sub = wide[actual_cols].copy()
        sub.columns = RAW_FEATURE_KEYS
        sub["bus"] = bus
        sub["time"] = wide["time"].to_numpy()
        sub["split"] = wide["_time_split"].to_numpy()
        sub["y_collapsed"] = wide[f"collapsed_{bus}"].to_numpy().astype(int)
        long_frames.append(sub)

    long_df = pd.concat(long_frames, ignore_index=True)
    print(f"Total long-format rows: {len(long_df):,}  |  raw feature count: {len(RAW_FEATURE_KEYS)} "
          f"(vs 121 full engineered, 83 strict-precursor)")

    print(f"\n--- Holding out Bus {HOLDOUT_BUS} COMPLETELY from training ---")
    train_pool = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "train")]
    test_pool_seen = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "test")]
    holdout_pool = long_df[long_df["bus"] == HOLDOUT_BUS].sort_values("time")

    X_train = train_pool[RAW_FEATURE_KEYS].to_numpy(dtype=np.float64)
    y_train = train_pool["y_collapsed"].to_numpy(dtype=int)
    X_test_seen = test_pool_seen[RAW_FEATURE_KEYS].to_numpy(dtype=np.float64)
    y_test_seen = test_pool_seen["y_collapsed"].to_numpy(dtype=int)
    X_holdout = holdout_pool[RAW_FEATURE_KEYS].to_numpy(dtype=np.float64)
    y_holdout = holdout_pool["y_collapsed"].to_numpy(dtype=int)

    print(f"Train rows: {len(train_pool):,}  |  Test-seen rows: {len(test_pool_seen):,}  |  "
          f"Holdout ({HOLDOUT_BUS}) rows: {len(holdout_pool):,}")
    n_pos_train = y_train.sum()
    if n_pos_train == 0:
        raise RuntimeError("No positive examples in the training pool, cannot proceed.")
    print(f"Positive examples in training pool: {n_pos_train:,} ({100*n_pos_train/len(train_pool):.2f}%)")

    pipeline = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MLP_KWARGS)),
    ])

    print("\nTraining shared model on raw signals only (never sees any Bus F data) ...")
    pipeline.fit(X_train, y_train)
    mlp = pipeline.named_steps["mlp"]
    print(f"\nStopped after {mlp.n_iter_} iterations, final training loss={mlp.loss_:.6f}")

    proba_train = pipeline.predict_proba(X_train)[:, 1]
    proba_test_seen = pipeline.predict_proba(X_test_seen)[:, 1]
    proba_holdout = pipeline.predict_proba(X_holdout)[:, 1]

    print("\n===== EVALUATION AT DEFAULT THRESHOLD (0.5) =====")
    result_train = evaluate("Training set (fit quality check)", y_train, proba_train)
    result_seen = evaluate("Held-out TIME blocks, buses model DID train on", y_test_seen, proba_test_seen)
    result_holdout = evaluate(f"Bus {HOLDOUT_BUS}, NEVER trained on", y_holdout, proba_holdout)

    print("\n===== THRESHOLD SWEEP (held-out bus) =====")
    print("Checking whether precision is pinned at 1.000 the way it was for the "
          "strict-precursor model (step37), which would mean 0.5 is too conservative "
          "for this class balance rather than the model being weak.")
    sweep_rows = []
    for thr in THRESHOLD_SWEEP:
        r = evaluate(f"Bus {HOLDOUT_BUS} @ threshold={thr}", y_holdout, proba_holdout, threshold=thr)
        if r is not None:
            r["threshold"] = thr
            sweep_rows.append(r)

    print("\n===== TIME-SINCE-ONSET RECALL BREAKDOWN (held-out bus, threshold=0.5) =====")
    onset_rows = onset_recall_breakdown(holdout_pool, proba_holdout, threshold=0.5)

    model_path = OUTPUT_DIR / "raw_signal_collapse_detector.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nWrote: {model_path}")

    meta = {
        "task": "raw-signal (voltage, GEI, load, Tj, Th, Tj-Th) same-instant collapse detection, "
                "bus-symmetric shared model, Bus F fully held out",
        "holdout_bus": HOLDOUT_BUS,
        "raw_feature_keys": RAW_FEATURE_KEYS,
        "n_raw_features": len(RAW_FEATURE_KEYS),
        "subsample_stride": SUBSAMPLE_STRIDE,
        "architecture": {"hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
                          **{k: v for k, v in MLP_KWARGS.items() if k != "hidden_layer_sizes"}},
        "results_default_threshold": {
            "train": result_train, "test_seen_buses": result_seen,
            f"holdout_bus_{HOLDOUT_BUS}": result_holdout,
        },
        "threshold_sweep_holdout": sweep_rows,
        "onset_recall_breakdown_holdout": onset_rows,
        "caveat": (
            "own_Voltage and own_GEI are same-instant symptoms of the label "
            "(collapsed_<BUS> is defined from voltage; GEI shares voltage's collapse "
            "instant via P=VI). High recall here is expected and is not evidence of "
            "early or transferable pattern recognition. Compare against "
            "bus_symmetric_precursor_only_test.py's strict-precursor result (83 features, "
            "no own voltage/GEI/power) for the actual precursor question."
        ),
    }
    meta_path = OUTPUT_DIR / "raw_signal_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. Compare these numbers against precursor_only_comparison.json's "
          "results_full and results_strict_precursor for the same Bus-F holdout.")


if __name__ == "__main__":
    main()

"""
POOLED MULTI-SCENARIO EARLY-WARNING DETECTOR (leave-one-scenario-out, N sweep)
================================================================================
This is the next step after three separate scenarios were each built end-to-
end (dataset -> features -> cross-scenario detector test -> closed-loop
verification): baseline (sudden Bus F/E spike cascade), gradual_busC_v4
(slow single-bus Bus C ramp), three_bus_collapse_v1 (correlated regional
Bus G/H/K collapse). Every detector used so far was trained ONLY on baseline
and then tested against the other two without retraining. This script
retrains from scratch, POOLING all three scenarios together, following
Atharva's stated plan: "create a model that learns and memorizes all the
scenarios ... trains efficiently to avoid collapse for all the scenarios."

DESIGN DECISIONS (all confirmed explicitly before writing this script):
  1. Retrain ALL 5 lead times: N = 5, 15, 30, 60, 90 seconds. Not just N=60s.
  2. Evaluation = leave-one-scenario-out, 3 folds. For each fold, train on
     the POOLED rows from 2 scenarios and test on the 3rd scenario, which the
     model has never seen in any form. Repeat for all 3 choices of held-out
     scenario. This is the direct multi-scenario analog of step41's
     "hold out the whole bus, not a row split" rule: a precursor window is a
     short, spatially/temporally contiguous thing, so any split finer than
     "hold out a whole unit" either leaks the answer into training or starves
     it out of testing. Here the "whole unit" is an entire scenario. This was
     picked explicitly over "train on all 3, evaluate on all 3 (in-sample)"
     to avoid the exact false-positive-hiding trap already documented
     elsewhere in this project: in-sample numbers can look perfect while the
     model has just memorized which scenario it's looking at.
  3. Labels: recomputed independently and IDENTICALLY for all three
     scenarios (baseline included), using the same voltage-threshold +
     debounce rule used everywhere else in this project
     (compute_collapsed_mask, verbatim from the cross-scenario test scripts).
     Baseline's original separate corrected_phase_targets_5000s.csv is
     deliberately NOT used here, even though it exists and even though
     step41 used it -- mixing a hand-corrected label source for one scenario
     with an independently-recomputed one for the other two would mean two
     different labeling methods feeding one pooled model. One method, all
     three scenarios, is safer.
  4. Resolution: SUBSAMPLE_STRIDE=5 (50ms/row, 1-in-5 rows), same as
     step41's original baseline-only training run, applied identically to
     all 3 scenarios. UPDATE: the first version of this script tried full
     10ms resolution (no subsampling) instead, as an explicit choice to keep
     every row. That version crashed with a MemoryError -- a single
     scenario's 10-bus-stacked feature matrix at full resolution needed
     ~2.93GB in one contiguous block, and that's before even pooling two
     scenarios together (which would need roughly double that again, plus
     another same-sized copy during the StandardScaler step). Reverted to
     step41's proven 1-in-5 subsampling to keep this actually runnable.

STRICT-PRECURSOR FEATURE SET: reproduced verbatim from
step41_early_warning_shifted_label.py's get_local_feature_map() /
DIRECT_SIGNAL_KEYS / STRICT_ADDITIONAL_KEYS (voltage, GEI, source power
excluded, same as every detector in this project). MLP_KWARGS is also
reproduced verbatim from step41 so the pooled models stay comparable to the
single-scenario ones.

OUTPUT: a NEW directory, `pooled_multi_scenario_detector`, distinct from
`41_early_warning_shifted_label` -- the existing 5 baseline-only .pkl files
there are NOT touched or overwritten by this script.

Run (on the Windows machine, inside the venv, AFTER step33_build_
gradual_busC_v4_features.py and step33_build_three_bus_collapse_v1_features.py
have both been run at least once, and the original baseline features already
exist from step33_build_10bus_derating_features_v3.py):
    python train_pooled_multi_scenario_detector.py
"""
from __future__ import annotations
import gc
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
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)

SCENARIO_PATHS = {
    "baseline": dict(
        parquet=PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
                 / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet",
        csv=PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
                 / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv",
    ),
    "gradual_busC_v4": dict(
        parquet=PROJECT_ROOT / "data_d" / "gradual_busC_v4_features"
                 / "gradual_busC_v4_5000s_features.parquet",
        csv=PROJECT_ROOT / "data_d" / "gradual_busC_v4_features"
                 / "gradual_busC_v4_5000s_features.csv",
    ),
    "three_bus_collapse_v1": dict(
        parquet=PROJECT_ROOT / "data_d" / "three_bus_collapse_v1_features"
                 / "three_bus_collapse_v1_5000s_features.parquet",
        csv=PROJECT_ROOT / "data_d" / "three_bus_collapse_v1_features"
                 / "three_bus_collapse_v1_5000s_features.csv",
    ),
}
SCENARIO_NAMES = list(SCENARIO_PATHS.keys())

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
OUTPUT_DIR = MODEL_ROOT / "pooled_multi_scenario_detector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "pooled_multi_scenario_loso_results.json"

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
CONV = ["AB", "BC", "CD", "DE", "EF", "FG", "GH", "HK", "KL", "AL"]

LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]

# UPDATED after the first run: training on every row (no subsampling) tried to
# hold a single scenario's 10-bus-stacked feature matrix (~4.7M rows x 83 cols
# = ~2.93GB) in memory at once and failed with a MemoryError -- and that's
# before even pooling two scenarios together, which would roughly double it
# again. Reverted to the same 1-in-5 row subsampling step41 already used for
# baseline's original single-scenario training (SUBSAMPLE_STRIDE=5, 50ms/row),
# applied identically to all 3 scenarios here for consistency. This cuts the
# in-memory footprint ~5x, back down to a size already proven to work.
SUBSAMPLE_STRIDE = 5
DT_SECONDS = 0.01 * SUBSAMPLE_STRIDE  # 50ms per row after subsampling

COLLAPSE_V_THRESHOLD = 100.0
COLLAPSE_MIN_RUN_SECONDS = 0.5  # matches the 50-sample @ 10ms debounce used everywhere else

HIDDEN_LAYER_SIZES = (128, 64)
RANDOM_STATE = 42
MLP_KWARGS = dict(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation="relu", solver="adam",
    alpha=0.01, batch_size=256, learning_rate="constant", learning_rate_init=0.001,
    max_iter=1000, early_stopping=False, n_iter_no_change=25, tol=1e-4,
    random_state=RANDOM_STATE, verbose=False,
)


# ----------------------------------------------------------------------
# EXACT COPY of the strict-precursor feature map from step41 (order matters:
# any pretrained pipeline expects this exact column set, this exact order)
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

CANONICAL_COLS = list(get_local_feature_map(0).keys())
STRICT_COLS = [c for c in CANONICAL_COLS if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]


# ----------------------------------------------------------------------
# EXACT COPY of the collapse-mask / onset / lookahead-label logic used
# throughout this project (compute_collapsed_mask + get_onsets verbatim from
# the cross-scenario test scripts; build_lookahead_label verbatim from
# step41). Applied UNIFORMLY to all three scenarios, including baseline --
# this is the one deliberate change vs. step41, which used baseline's
# separate hand-corrected label file instead.
# ----------------------------------------------------------------------
def compute_collapsed_mask(v: np.ndarray, t: np.ndarray, threshold=COLLAPSE_V_THRESHOLD,
                            min_run_seconds=COLLAPSE_MIN_RUN_SECONDS) -> np.ndarray:
    below = v < threshold
    if len(t) > 1:
        dt = np.median(np.diff(t))
    else:
        dt = 0.01
    min_run = max(1, int(round(min_run_seconds / dt)))
    collapsed = np.zeros(len(v), dtype=bool)
    i, n = 0, len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_run:
                collapsed[i:] = True
                break
            i = j
        else:
            i += 1
    return collapsed


def get_onsets(t: np.ndarray, collapsed: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[False], collapsed[:-1]])
    onset_mask = collapsed & ~prev
    return t[onset_mask]


def build_lookahead_label(t: np.ndarray, collapsed: np.ndarray, n_seconds: float):
    onsets = get_onsets(t, collapsed)
    keep_mask = ~collapsed
    if len(onsets) == 0:
        return np.zeros(len(t), dtype=int), keep_mask
    idx = np.searchsorted(onsets, t, side="right")
    has_future_onset = idx < len(onsets)
    time_to_next = np.full(len(t), np.inf)
    time_to_next[has_future_onset] = onsets[idx[has_future_onset]] - t[has_future_onset]
    label = ((time_to_next > 0) & (time_to_next <= n_seconds)).astype(int)
    return label, keep_mask


def evaluate(name, y_true, y_proba, threshold=0.5):
    if len(y_true) == 0 or y_true.sum() == 0:
        print(f"    {name}: no positive examples in this slice, skipping metrics.")
        return None
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"    {name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} "
          f"f1={f1:.4f} auc={auc if auc is None else round(auc, 4)}  TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ----------------------------------------------------------------------
# Per-scenario loading: read the wide feature table ONCE, build the strict-
# precursor feature matrix for every bus and the ground-truth collapse mask
# for every bus, then drop the wide table from memory. This work does not
# depend on N, so it happens exactly once per scenario regardless of how
# many lead times are swept.
# ----------------------------------------------------------------------
def load_scenario(name: str, paths: dict) -> dict:
    parquet, csv = paths["parquet"], paths["csv"]
    if parquet.exists():
        wide = pd.read_parquet(parquet)
    elif csv.exists():
        wide = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"No feature table found for scenario '{name}':\n{parquet}\n{csv}\n"
            f"Run the matching step33_build_*_features.py script first."
        )
    wide = wide.sort_values("time").reset_index(drop=True)
    n_before_subsample = len(wide)
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    t = wide["time"].to_numpy()
    print(f"  [{name}] loaded {n_before_subsample:,} rows, subsampled (1-in-{SUBSAMPLE_STRIDE}) "
          f"to {len(wide):,} rows x {wide.shape[1]} cols ({DT_SECONDS*1000:.0f}ms per row)")

    collapsed_by_bus = {}
    onset_by_bus = {}
    for b in BUSES:
        v = wide[f"Bus_{b}_Voltage"].to_numpy()
        collapsed = compute_collapsed_mask(v, t)
        collapsed_by_bus[b] = collapsed
        onsets = get_onsets(t, collapsed)
        onset_by_bus[b] = float(onsets[0]) if len(onsets) else None
    onset_str = ", ".join(
        f"{b}={onset_by_bus[b]:.2f}s" if onset_by_bus[b] is not None else f"{b}=never"
        for b in BUSES
    )
    print(f"  [{name}] independently recomputed collapse onsets: {onset_str}")

    bus_feature_matrix = {}
    for bus_idx, bus in enumerate(BUSES):
        fmap = get_local_feature_map(bus_idx)
        actual_cols = [fmap[c] for c in STRICT_COLS]
        missing = [c for c in actual_cols if c not in wide.columns]
        if missing:
            raise RuntimeError(f"[{name}] bus {bus}: missing expected feature columns "
                                f"(feature recipe mismatch?): {missing[:10]}")
        bus_feature_matrix[bus] = wide[actual_cols].to_numpy(dtype=np.float64)

    n_rows = len(wide)
    del wide
    gc.collect()

    return {
        "name": name,
        "t": t,
        "collapsed_by_bus": collapsed_by_bus,
        "onset_by_bus": onset_by_bus,
        "bus_feature_matrix": bus_feature_matrix,
        "n_rows": n_rows,
    }


def build_scenario_xy(cache: dict, n_seconds: float):
    """Stack all 10 buses into one (X, y) pair for this scenario at this N,
    dropping rows that are currently collapsed (same rule as step41)."""
    t = cache["t"]
    X_parts, y_parts = [], []
    for bus in BUSES:
        collapsed = cache["collapsed_by_bus"][bus]
        label, keep_mask = build_lookahead_label(t, collapsed, n_seconds)
        X_parts.append(cache["bus_feature_matrix"][bus][keep_mask])
        y_parts.append(label[keep_mask])
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def save_results(all_results):
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))


def main():
    print("=" * 70)
    print("POOLED MULTI-SCENARIO EARLY-WARNING DETECTOR")
    print("Leave-one-scenario-out, N in", LEAD_TIMES_SECONDS)
    print("=" * 70)
    print("Loading all 3 scenarios' feature tables (this is the expensive, one-time part)...\n")

    caches = {}
    for name in SCENARIO_NAMES:
        caches[name] = load_scenario(name, SCENARIO_PATHS[name])
    print(f"\nUsing strict-precursor feature set: {len(STRICT_COLS)} features "
          f"(voltage, GEI, source power excluded)\n")

    all_results = {
        "task": "pooled multi-scenario early-warning detector, leave-one-scenario-out",
        "scenarios": SCENARIO_NAMES,
        "lead_times_seconds": LEAD_TIMES_SECONDS,
        "n_strict_features": len(STRICT_COLS),
        "labels_note": "Collapse labels recomputed independently and identically for ALL THREE "
                        "scenarios (including baseline) via compute_collapsed_mask on each "
                        "scenario's own Bus_<X>_Voltage column -- baseline's separate "
                        "corrected_phase_targets_5000s.csv was deliberately NOT used here.",
        "resolution_note": f"Subsampled 1-in-{SUBSAMPLE_STRIDE} rows ({DT_SECONDS*1000:.0f}ms/row) "
                            f"for all 3 scenarios, matching step41's original baseline-only "
                            f"training resolution. (An earlier version of this script used full "
                            f"10ms resolution with no subsampling and crashed with a MemoryError.)",
        "onsets_by_scenario": {name: caches[name]["onset_by_bus"] for name in SCENARIO_NAMES},
        "folds": [],
    }

    for n_seconds in LEAD_TIMES_SECONDS:
        print("=" * 70)
        print(f"LEAD TIME N = {n_seconds}s")
        print("=" * 70)

        scenario_xy = {}
        for name in SCENARIO_NAMES:
            X, y = build_scenario_xy(caches[name], n_seconds)
            n_pos = int(y.sum())
            print(f"  [{name}] N={n_seconds}s: {len(y):,} rows (after dropping currently-"
                  f"collapsed rows), positives={n_pos:,}")
            scenario_xy[name] = (X, y, n_pos)

        for holdout in SCENARIO_NAMES:
            train_names = [s for s in SCENARIO_NAMES if s != holdout]
            print("-" * 70)
            print(f"N={n_seconds}s | holdout scenario = {holdout}  "
                  f"(train on {train_names[0]} + {train_names[1]}, pooled)")
            print("-" * 70)

            X_train = np.concatenate([scenario_xy[s][0] for s in train_names], axis=0)
            y_train = np.concatenate([scenario_xy[s][1] for s in train_names], axis=0)
            X_test, y_test, n_pos_test = scenario_xy[holdout]
            n_pos_train = int(y_train.sum())

            print(f"  Train rows: {len(y_train):,} (positives: {n_pos_train:,})  |  "
                  f"Test rows (holdout={holdout}): {len(y_test):,} (positives: {n_pos_test:,})")

            fold_record = {
                "n_seconds": n_seconds, "holdout_scenario": holdout,
                "train_scenarios": train_names,
                "n_train_rows": len(y_train), "n_pos_train": n_pos_train,
                "n_test_rows": len(y_test), "n_pos_test": n_pos_test,
                "train_metrics": None, "test_metrics": None, "skipped": False,
            }

            if n_pos_train == 0 or n_pos_test == 0:
                print(f"  SKIPPED: no positive examples in train or test at N={n_seconds}s, "
                      f"holdout={holdout}.")
                fold_record["skipped"] = True
                all_results["folds"].append(fold_record)
                save_results(all_results)
                del X_train, y_train
                gc.collect()
                continue

            pipeline = Pipeline([("x_scaler", StandardScaler()), ("mlp", MLPClassifier(**MLP_KWARGS))])
            pipeline.fit(X_train, y_train)
            mlp = pipeline.named_steps["mlp"]
            print(f"  Trained: {mlp.n_iter_} iterations, final loss {mlp.loss_:.6f}")

            proba_train = pipeline.predict_proba(X_train)[:, 1]
            proba_test = pipeline.predict_proba(X_test)[:, 1]

            r_train = evaluate("train (fit quality only, not generalization)", y_train, proba_train)
            r_test = evaluate(f"test ({holdout}, unseen scenario -- the metric that matters)",
                               y_test, proba_test)

            fold_record["train_metrics"] = r_train
            fold_record["test_metrics"] = r_test
            all_results["folds"].append(fold_record)

            model_path = OUTPUT_DIR / f"pooled_N{n_seconds}s_holdout_{holdout}.pkl"
            joblib.dump(pipeline, model_path)
            print(f"  Wrote: {model_path}")

            save_results(all_results)
            del X_train, y_train, X_test, y_test, proba_train, proba_test, pipeline
            gc.collect()
            print()

        del scenario_xy
        gc.collect()

    # ------------------------------------------------------------------
    # Summary: per-N average test performance across the 3 held-out folds
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY -- per-N test performance, averaged across the 3 leave-one-")
    print("scenario-out folds (each fold's test scenario was never seen in training)")
    print("=" * 70)
    summary = []
    for n_seconds in LEAD_TIMES_SECONDS:
        rows = [f for f in all_results["folds"] if f["n_seconds"] == n_seconds and not f["skipped"]
                and f["test_metrics"] is not None]
        if not rows:
            print(f"  N={n_seconds:>2}s: no usable folds (all skipped -- no positive examples).")
            summary.append({"n_seconds": n_seconds, "n_usable_folds": 0})
            continue
        avg_recall = float(np.mean([r["test_metrics"]["recall"] for r in rows]))
        avg_precision = float(np.mean([r["test_metrics"]["precision"] for r in rows]))
        avg_f1 = float(np.mean([r["test_metrics"]["f1"] for r in rows]))
        aucs = [r["test_metrics"]["auc"] for r in rows if r["test_metrics"]["auc"] is not None]
        avg_auc = float(np.mean(aucs)) if aucs else None
        per_fold = ", ".join(
            f"{r['holdout_scenario']}: recall={r['test_metrics']['recall']:.3f}"
            for r in rows
        )
        print(f"  N={n_seconds:>2}s: avg recall={avg_recall:.4f}, avg precision={avg_precision:.4f}, "
              f"avg f1={avg_f1:.4f}, avg auc={avg_auc if avg_auc is None else round(avg_auc, 4)}  "
              f"[{per_fold}]")
        summary.append({
            "n_seconds": n_seconds, "n_usable_folds": len(rows),
            "avg_recall": avg_recall, "avg_precision": avg_precision,
            "avg_f1": avg_f1, "avg_auc": avg_auc,
        })

    all_results["summary_by_n"] = summary
    save_results(all_results)
    print(f"\nWrote: {RESULTS_PATH}")
    print(f"Models written to: {OUTPUT_DIR}")
    print("\nRead this the same way every other cross-scenario result in this project has been")
    print("read: a high average recall is not enough on its own -- check that recall is reasonably")
    print("consistent across all 3 per-fold holdouts (not one scenario carrying the average while")
    print("another sits near zero), since that would mean the pooled model still only really works")
    print("on scenarios that resemble two-thirds of its training mix.")


if __name__ == "__main__":
    main()
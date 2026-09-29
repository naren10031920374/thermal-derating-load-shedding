"""
STEP 44: THRESHOLD SWEEP ON THE ORIGINAL step41 MODELS (NO RETRAINING)
================================================================================
step42 (oversampling) and step43 (undersampling) both made things WORSE at
N=60s/90s, the two lead times step41 already ranked well (AUC 0.993, 0.965).
That is the tell: those two models already separate precursor from non-
precursor rows correctly, they just output very small, poorly calibrated
probabilities, the classic symptom of severe class imbalance affecting
calibration without affecting ranking. The direct fix for THAT symptom is
lowering the decision threshold, nothing about training needs to change.
We skipped straight to changing the training data in step42/43 without first
trying the cheapest possible experiment: sweep the threshold on the original,
untouched step41 models.

This script does exactly that and nothing else. It loads step41's five saved
pickles, re-scores the Bus F holdout set (same 83 strict-precursor features,
same labels), and reports precision/recall/F1 across a threshold sweep. No
retraining, no resampling, no architecture change. If this recovers real
recall at N=60s/90s where step42/43 could not, that confirms the imbalance
was a calibration problem, not an information problem, and resampling was
solving the wrong issue for those two lead times.

Requires step41_early_warning_shifted_label.py to have been run already,
with its .pkl files present in:
  model_outputs/unified_controller_10bus_derating/41_early_warning_shifted_label/

Run:
    python step44_threshold_sweep_original_models.py
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ----------------------------------------------------------------------
# Paths, must match step33 through step43
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
OUTPUT_DIR = MODEL_ROOT / "44_threshold_sweep_original"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5
LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]
# Finer and lower than step41/42/43's sweep, since the interesting behavior
# for a well-ranked but poorly-calibrated model often sits well under 0.01.
THRESHOLD_SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]


# ----------------------------------------------------------------------
# Shared utilities (identical to step36 through step43)
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


def build_holdout_frame(bus_idx, wide, canonical_cols, n_seconds):
    fmap = get_local_feature_map(bus_idx)
    bus = BUSES[bus_idx]
    actual_cols = [fmap[c] for c in canonical_cols]
    sub = wide[actual_cols].copy()
    sub.columns = canonical_cols
    sub["time"] = wide["time"].to_numpy()
    collapsed = wide[f"collapsed_{bus}"].to_numpy().astype(bool)
    t = wide["time"].to_numpy()
    label, keep_mask = build_lookahead_label(t, collapsed, n_seconds)
    sub["y_early"] = label
    sub["keep"] = keep_mask
    return sub[sub["keep"]].sort_values("time").reset_index(drop=True)


def sweep_metrics(y_true, y_proba, thresholds):
    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({"threshold": thr, "precision": prec, "recall": rec, "f1": f1,
                      "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    return rows


def main():
    print("=" * 70)
    print("STEP 44: THRESHOLD SWEEP ON step41's ORIGINAL (UNRESAMPLED) MODELS")
    print("=" * 70)
    print("No retraining. Loading step41's saved pickles and re-scoring Bus F")
    print("across a much finer, lower threshold range than step41 originally used.\n")

    missing = [n for n in LEAD_TIMES_SECONDS if not (STEP41_DIR / f"early_warning_N{n}s.pkl").exists()]
    if missing:
        print(f"WARNING: step41 pickles missing for N={missing}, those will be skipped.")

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)

    canonical_cols = list(get_local_feature_map(0).keys())
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    bus_idx_F = BUSES.index(HOLDOUT_BUS)

    sweep_results = []
    saved_probas = {}

    for n_seconds in LEAD_TIMES_SECONDS:
        model_path = STEP41_DIR / f"early_warning_N{n_seconds}s.pkl"
        if not model_path.exists():
            continue
        print("-" * 70)
        print(f"LEAD TIME N = {n_seconds}s")
        print("-" * 70)

        pipeline = joblib.load(model_path)
        holdout_pool = build_holdout_frame(bus_idx_F, wide, strict_cols, n_seconds)
        X_holdout = holdout_pool[strict_cols].to_numpy(dtype=np.float64)
        y_holdout = holdout_pool["y_early"].to_numpy(dtype=int)

        proba_holdout = pipeline.predict_proba(X_holdout)[:, 1]
        print(f"  Holdout positives: {int(y_holdout.sum())}  |  "
              f"probability range: [{proba_holdout.min():.6f}, {proba_holdout.max():.6f}]")
        try:
            auc = roc_auc_score(y_holdout, proba_holdout)
        except ValueError:
            auc = None
        print(f"  AUC (should match step41 exactly): {auc if auc is None else round(auc,4)}")

        thr_rows = sweep_metrics(y_holdout, proba_holdout, THRESHOLD_SWEEP)
        print(f"  {'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'TP':>6s} {'FP':>6s}")
        for r in thr_rows:
            print(f"  {r['threshold']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['tp']:>6d} {r['fp']:>6d}")

        sweep_results.append({"n_seconds": n_seconds, "auc_holdout": auc,
                               "n_pos_holdout": int(y_holdout.sum()),
                               "threshold_sweep_holdout": thr_rows})
        saved_probas[n_seconds] = (holdout_pool, proba_holdout)
        print()

    if not sweep_results:
        raise RuntimeError("No step41 pickles found, run step41 first.")

    # --------------------------------------------------------------------
    # Plot: recall vs threshold, fine-grained, one line per N
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sweep_results)))
    for r, color in zip(sweep_results, colors):
        thrs = [t["threshold"] for t in r["threshold_sweep_holdout"]]
        recs = [t["recall"] for t in r["threshold_sweep_holdout"]]
        ax.plot(thrs, recs, "o-", color=color, linewidth=2, markersize=5,
                label=f"N={r['n_seconds']}s (AUC={r['auc_holdout']:.3f})" if r["auc_holdout"] else f"N={r['n_seconds']}s")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("decision threshold (log scale, reading right to left = more permissive)")
    ax.set_ylabel("recall (Bus F holdout)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("step41's ORIGINAL models, no resampling, threshold swept much lower\n"
                 "(same models as step41, only the reading changes)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out1 = OUTPUT_DIR / "original_recall_vs_threshold_log.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out1}")

    meta = {
        "task": "threshold sweep on step41's original unresampled models, Bus F holdout",
        "note": "No retraining. Same models as step41, evaluated at a finer, lower threshold range.",
        "lead_times_seconds": LEAD_TIMES_SECONDS,
        "threshold_sweep": THRESHOLD_SWEEP,
        "sweep_results": sweep_results,
    }
    meta_path = OUTPUT_DIR / "threshold_sweep_original_results.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. Read original_recall_vs_threshold_log.png. If N=60s/90s now show recall")
    print("rising to something meaningful at very low thresholds (the printed probability range")
    print("tells you how low you'll need to go), that confirms the original model already had")
    print("usable signal, it was purely a calibration/threshold issue, not a training-data issue.")
    print("If recall still stays near zero even at the lowest thresholds tried, the AUC advantage")
    print("really was too thin to extract a working detector from, and step42/43's instinct to")
    print("touch training was reasonable, it just didn't have enough data to work with either.")


if __name__ == "__main__":
    main()

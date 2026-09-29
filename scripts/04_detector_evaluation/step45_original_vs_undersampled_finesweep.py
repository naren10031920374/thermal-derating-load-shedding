"""
STEP 45: ORIGINAL vs UNDERSAMPLED, FINE LOW-THRESHOLD COMPARISON (N=60/90 ONLY)
================================================================================
step44 found the original (step41) models' probabilities are compressed to an
extreme degree: max output on the entire Bus F holdout was 0.000018 at N=60s,
0.000031 at N=30s, meaning no threshold could recover recall there, the
information isn't expressed at any usable scale. N=90s was the exception
(max probability 0.269) BECAUSE it has the most positive training examples
(least severe imbalance), not because 90s is a special lead time.

A side-by-side check at N=90s showed something worth following up: at the
SAME threshold (0.005), step43's undersampled model recovered 275 true
positives (recall 0.153) versus step41's original 31 (recall 0.017), a ~9x
difference, even though step43's raw AUC (0.886) was slightly LOWER than
step41's (0.965). That means undersampling relaxed the probability
compression, made the output easier to threshold, even while very slightly
hurting fine-grained ranking. step43 never swept below 0.005 though, so we
don't know if the same relief happened for N=60s, where step43's AUC dropped
more (0.835 vs step41's 0.993) and every threshold tried showed recall=0.

This script sweeps BOTH model sets (step41 original, step43 undersampled)
down to much lower thresholds (as low as 1e-6), for N=60s and N=90s only,
since those are the only two lead times worth pursuing further, N=5/15/30
had AUC too weak or negative to matter regardless of threshold. No
retraining, both model sets are loaded from disk as-is.

Run:
    python step45_original_vs_undersampled_finesweep.py
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
# Paths, must match step33 through step44
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
STEP41_DIR = MODEL_ROOT / "41_early_warning_shifted_label"
STEP43_DIR = MODEL_ROOT / "43_early_warning_undersampled"
OUTPUT_DIR = MODEL_ROOT / "45_original_vs_undersampled_finesweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5
LEAD_TIMES_SECONDS = [60, 90]   # only the two lead times with usable AUC
MODEL_SOURCES = {
    "original (step41)": (STEP41_DIR, "early_warning_N{n}s.pkl"),
    "undersampled (step43)": (STEP43_DIR, "early_warning_undersampled_N{n}s.pkl"),
}
# Much lower and finer than step44's sweep, since step41's N=60 max output
# was 0.000018, need to go below that to see anything move at all.
THRESHOLD_SWEEP = [0.5, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001,
                    0.00005, 0.00001, 0.000005, 0.000001]


# ----------------------------------------------------------------------
# Shared utilities (identical to step36 through step44)
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
    print("STEP 45: ORIGINAL vs UNDERSAMPLED, FINE LOW-THRESHOLD COMPARISON")
    print("(N=60s and N=90s only, no retraining)")
    print("=" * 70 + "\n")

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

    all_results = {}  # (source_name, n_seconds) -> dict

    for n_seconds in LEAD_TIMES_SECONDS:
        holdout_pool = build_holdout_frame(bus_idx_F, wide, strict_cols, n_seconds)
        X_holdout = holdout_pool[strict_cols].to_numpy(dtype=np.float64)
        y_holdout = holdout_pool["y_early"].to_numpy(dtype=int)

        for source_name, (source_dir, fname_tmpl) in MODEL_SOURCES.items():
            model_path = source_dir / fname_tmpl.format(n=n_seconds)
            print("-" * 70)
            print(f"N={n_seconds}s  |  {source_name}")
            print("-" * 70)
            if not model_path.exists():
                print(f"  SKIPPED: model not found at {model_path}")
                continue

            pipeline = joblib.load(model_path)
            proba_holdout = pipeline.predict_proba(X_holdout)[:, 1]
            print(f"  Holdout positives: {int(y_holdout.sum())}  |  "
                  f"probability range: [{proba_holdout.min():.8f}, {proba_holdout.max():.8f}]")
            try:
                auc = roc_auc_score(y_holdout, proba_holdout)
            except ValueError:
                auc = None
            print(f"  AUC: {auc if auc is None else round(auc,4)}")

            thr_rows = sweep_metrics(y_holdout, proba_holdout, THRESHOLD_SWEEP)
            print(f"  {'threshold':>12s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'TP':>6s} {'FP':>6s}")
            for r in thr_rows:
                print(f"  {r['threshold']:>12.7f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
                      f"{r['f1']:>8.4f} {r['tp']:>6d} {r['fp']:>6d}")
            print()

            all_results[(source_name, n_seconds)] = {
                "n_pos_holdout": int(y_holdout.sum()), "auc_holdout": auc,
                "proba_min": float(proba_holdout.min()), "proba_max": float(proba_holdout.max()),
                "threshold_sweep_holdout": thr_rows,
            }

    # --------------------------------------------------------------------
    # Plot: recall vs threshold, side by side for each N, original vs undersampled
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(LEAD_TIMES_SECONDS), figsize=(7 * len(LEAD_TIMES_SECONDS), 6), sharey=True)
    if len(LEAD_TIMES_SECONDS) == 1:
        axes = [axes]
    colors = {"original (step41)": "#85B7EB", "undersampled (step43)": "#1D9E75"}
    for ax, n_seconds in zip(axes, LEAD_TIMES_SECONDS):
        for source_name in MODEL_SOURCES:
            key = (source_name, n_seconds)
            if key not in all_results:
                continue
            r = all_results[key]
            thrs = [t["threshold"] for t in r["threshold_sweep_holdout"]]
            recs = [t["recall"] for t in r["threshold_sweep_holdout"]]
            ax.plot(thrs, recs, "o-", color=colors[source_name], linewidth=2, markersize=5,
                    label=f"{source_name} (AUC={r['auc_holdout']:.3f})" if r["auc_holdout"] else source_name)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("decision threshold (log scale)")
        ax.set_title(f"N={n_seconds}s")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("recall (Bus F holdout)")
    fig.suptitle("Original vs undersampled: does undersampling actually improve usable calibration?", y=1.02)
    fig.tight_layout()
    out1 = OUTPUT_DIR / "original_vs_undersampled_recall_vs_threshold.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out1}")

    meta = {
        "task": "fine low-threshold comparison, step41 original vs step43 undersampled, N=60s/90s only",
        "note": "No retraining, both model sets loaded as-is from disk.",
        "threshold_sweep": THRESHOLD_SWEEP,
        "results": {f"{k[0]}__N{k[1]}s": v for k, v in all_results.items()},
    }
    meta_path = OUTPUT_DIR / "comparison_results.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. Read original_vs_undersampled_recall_vs_threshold.png. For N=60s specifically:")
    print("if the undersampled (green) line rises to real recall at a threshold the original (blue)")
    print("line never reaches even at 1e-6, that confirms undersampling fixed N=60s's calibration the")
    print("same way it appears to have helped N=90s. If both lines stay near zero for N=60s even down")
    print("to 1e-6, that lead time's signal is too thin for any resampling or threshold choice to")
    print("recover, and it should be reported as such rather than tuned further.")


if __name__ == "__main__":
    main()

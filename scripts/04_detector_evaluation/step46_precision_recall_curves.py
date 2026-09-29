"""
STEP 46: PRECISION-RECALL CURVES FOR N=60s AND N=90s (THE WINNING MODELS)
================================================================================
step45 established that step41's ORIGINAL models (no resampling) are the
better detector at both lead times worth keeping, once read at the correct,
very small threshold. Recall-vs-threshold on its own was misleading because
it hides how many false alarms each point costs. This script builds the
actual precision-recall curve, which shows that tradeoff directly, and marks
the two operating points already identified as reasonable starting choices:

  N=60s, threshold ~5e-6: recall 0.454, precision 0.951, 28 false alarms
  N=90s, threshold ~5e-5: recall 0.229, precision 1.000, 0 false alarms

No retraining. Loads step41's saved pickles only.

HONEST CAVEAT, worth keeping on the figure or saying alongside it: these
operating points were chosen after seeing how the model performs on the one
holdout bus that exists. With only Bus F available as a generalization
check, there is no second independent collapse to confirm the exact
threshold transfers. This is a starting point for tuning once more collapse
scenarios exist (see next-steps item 3), not a validated, finished operating
point.

Run:
    python step46_precision_recall_curves.py
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
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

# ----------------------------------------------------------------------
# Paths, must match step33 through step45
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
OUTPUT_DIR = MODEL_ROOT / "46_precision_recall_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5
LEAD_TIMES_SECONDS = [60, 90]

# The operating points identified in step45, marked directly on each curve.
HIGHLIGHT_POINTS = {
    60: {"threshold": 5e-6, "label": "5e-6: recall 0.45, precision 0.95"},
    90: {"threshold": 5e-5, "label": "5e-5: recall 0.23, precision 1.00"},
}


# ----------------------------------------------------------------------
# Shared utilities (identical to step36 through step45)
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


def nearest_point_on_curve(precisions, recalls, thresholds, target_threshold):
    """precision_recall_curve returns thresholds of length n-1 vs precision/recall
    of length n. Find the curve point whose threshold is closest to target."""
    idx = int(np.argmin(np.abs(thresholds - target_threshold)))
    return precisions[idx], recalls[idx], thresholds[idx]


def main():
    print("=" * 70)
    print("STEP 46: PRECISION-RECALL CURVES, N=60s AND N=90s")
    print("(step41 original models, the winners per step45, no retraining)")
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

    fig, axes = plt.subplots(1, len(LEAD_TIMES_SECONDS), figsize=(7.5 * len(LEAD_TIMES_SECONDS), 6.5))
    if len(LEAD_TIMES_SECONDS) == 1:
        axes = [axes]

    summary = {}

    for ax, n_seconds in zip(axes, LEAD_TIMES_SECONDS):
        model_path = STEP41_DIR / f"early_warning_N{n_seconds}s.pkl"
        print("-" * 70)
        print(f"N={n_seconds}s")
        print("-" * 70)
        if not model_path.exists():
            print(f"  SKIPPED: model not found at {model_path}")
            continue

        pipeline = joblib.load(model_path)
        holdout_pool = build_holdout_frame(bus_idx_F, wide, strict_cols, n_seconds)
        X_holdout = holdout_pool[strict_cols].to_numpy(dtype=np.float64)
        y_holdout = holdout_pool["y_early"].to_numpy(dtype=int)
        proba = pipeline.predict_proba(X_holdout)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_holdout, proba)
        ap = average_precision_score(y_holdout, proba)
        try:
            auc = roc_auc_score(y_holdout, proba)
        except ValueError:
            auc = None
        print(f"  Holdout positives: {int(y_holdout.sum())}  |  AUC={auc:.4f}  |  "
              f"Average precision (area under PR curve)={ap:.4f}")

        # curve
        ax.plot(recalls, precisions, color="#1D9E75", linewidth=2, label="precision-recall curve")
        ax.fill_between(recalls, 0, precisions, color="#1D9E75", alpha=0.08)

        # baseline: precision of a random classifier = positive class prevalence
        prevalence = y_holdout.mean()
        ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1,
                   label=f"random baseline (prevalence={prevalence:.4f})")

        # highlighted operating point
        hp = HIGHLIGHT_POINTS.get(n_seconds)
        if hp is not None:
            p_at, r_at, thr_at = nearest_point_on_curve(precisions, recalls, thresholds, hp["threshold"])
            ax.scatter([r_at], [p_at], color="#D85A30", s=90, zorder=5, edgecolor="white", linewidth=1.2)
            ax.annotate(hp["label"], xy=(r_at, p_at), xytext=(r_at + 0.05, p_at - 0.08),
                        fontsize=9, color="#D85A30",
                        arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1))
            print(f"  Marked operating point: threshold~{thr_at:.2e}, "
                  f"recall={r_at:.4f}, precision={p_at:.4f}")
            summary[n_seconds] = {
                "auc": auc, "average_precision": ap, "prevalence": float(prevalence),
                "marked_threshold": float(thr_at), "marked_recall": float(r_at),
                "marked_precision": float(p_at),
            }

        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"N={n_seconds}s  (AUC={auc:.3f}, AP={ap:.3f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        print()

    fig.suptitle("Precision-recall tradeoff, Bus F held out, strict-precursor features\n"
                 "(honest caveat: operating points chosen on the only holdout bus available, "
                 "starting points for tuning, not validated)", fontsize=11, y=1.03)
    fig.tight_layout()
    out1 = OUTPUT_DIR / "precision_recall_curves_N60_N90.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out1}")

    meta = {
        "task": "precision-recall curves, step41 original models, N=60s/90s, Bus F holdout",
        "caveat": "Operating points chosen after seeing performance on the only holdout bus "
                   "available. Starting points for tuning once more collapse scenarios exist, "
                   "not a validated finished operating point.",
        "results": summary,
    }
    meta_path = OUTPUT_DIR / "precision_recall_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. This is the figure to use in the presentation instead of the recall-vs-threshold")
    print("chart from step45, it shows the actual cost (in precision) of each recall level, which")
    print("is the honest way to present a detector operating in a heavily imbalanced regime.")


if __name__ == "__main__":
    main()

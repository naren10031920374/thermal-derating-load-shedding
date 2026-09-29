"""
STEP 42: EARLY-WARNING EXPERIMENT, CORRECTED (OVERSAMPLING + THRESHOLD SWEEP)
================================================================================
step41 found AUC well above 0.5 at N=60s (0.993) and N=90s (0.965), meaning the
strict-precursor features DO rank true precursor moments above random ones, at
those lead times. But recall at the default 0.5 threshold was 0.000 everywhere,
because the positive class was 0.01% to 0.2% of training rows and
MLPClassifier.fit() has no sample_weight/class_weight argument, so nothing in
training pushed back against the loss-minimizing shortcut of predicting
"negative" almost everywhere. This script fixes that at the training step,
not just at the decision threshold, and then also sweeps the threshold so the
two fixes aren't confused with each other.

FIX 1: OVERSAMPLE THE POSITIVE CLASS BEFORE TRAINING.
MLPClassifier cannot take per-sample weights, so the only way to make training
"care" about the minority class is to change what it sees: positive rows are
resampled with replacement so they make up TARGET_POS_FRACTION of the training
set (10% by default), rather than their natural ~0.01-0.2%. This does not add
new information (it's the same ~100-1800 real positive rows, repeated), it
only changes the loss weighting during training. Still worth being honest that
it cannot manufacture signal that isn't there, it can only stop training from
ignoring the signal that is.

FIX 2: SWEEP THE DECISION THRESHOLD, DON'T READ ONLY AT 0.5.
For each N, evaluate at multiple thresholds (0.5 down to 0.005) on the Bus F
holdout, so a real but under-0.5 signal is visible as recall recovering at a
lower threshold, rather than reading as a flat zero.

Trains a FRESH model per N (does not reuse step41's saved pickles, since the
training data itself changes here). step41's results are kept on disk as the
"before" comparison; do not overwrite them.

Same caveat as step41: one collapse cascade in this dataset. This is a first
signal check, not a validated result.

Run:
    python step42_early_warning_oversampled.py
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample, shuffle as sk_shuffle
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, roc_auc_score, confusion_matrix)

# ----------------------------------------------------------------------
# Paths, must match step33/36/37/38/39/40/41
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
OUTPUT_DIR = MODEL_ROOT / "42_early_warning_oversampled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5
DT_SECONDS = 0.01 * SUBSAMPLE_STRIDE  # 50ms per row after subsampling

RANDOM_STATE = 42
LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]
TARGET_POS_FRACTION = 0.10          # positives made up to 10% of the training set
THRESHOLD_SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

HIDDEN_LAYER_SIZES = (128, 64)
MLP_KWARGS = dict(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation="relu", solver="adam",
    alpha=0.01, batch_size=256, learning_rate="constant", learning_rate_init=0.001,
    max_iter=1000, early_stopping=False, n_iter_no_change=25, tol=1e-4,
    random_state=RANDOM_STATE, verbose=False,
)


# ----------------------------------------------------------------------
# Shared utilities (identical to step36/37/38/39/40/41)
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


def build_bus_frame_with_label(bus_idx, wide, canonical_cols, n_seconds):
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
    return sub


def oversample_positives(X, y, target_pos_fraction, random_state):
    """Resample the positive class with replacement so it makes up
    target_pos_fraction of the returned training set. Does not touch the
    negative class. Returns shuffled (X, y). Prints before/after counts."""
    pos_mask = y == 1
    neg_mask = ~pos_mask
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
    if n_pos == 0:
        raise ValueError("No positive examples to oversample.")
    # solve for desired_pos_count such that desired_pos / (desired_pos + n_neg) = target_pos_fraction
    desired_pos = int(round(target_pos_fraction * n_neg / (1 - target_pos_fraction)))
    desired_pos = max(desired_pos, n_pos)  # never downsample positives
    X_pos, y_pos = X[pos_mask], y[pos_mask]
    X_pos_rs, y_pos_rs = resample(X_pos, y_pos, replace=True, n_samples=desired_pos,
                                   random_state=random_state)
    X_out = np.concatenate([X[neg_mask], X_pos_rs], axis=0)
    y_out = np.concatenate([y[neg_mask], y_pos_rs], axis=0)
    X_out, y_out = sk_shuffle(X_out, y_out, random_state=random_state)
    print(f"    Oversampled: {n_pos} real positives -> {desired_pos} after resampling "
          f"({100*desired_pos/(desired_pos+n_neg):.1f}% of {desired_pos+n_neg:,} training rows, "
          f"target was {target_pos_fraction*100:.0f}%)")
    return X_out, y_out


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
    print("STEP 42: EARLY-WARNING EXPERIMENT, CORRECTED")
    print("(oversampled minority class + threshold sweep)")
    print("=" * 70)
    print("Reminder: one collapse cascade in this dataset. This is a first read on")
    print("whether lead-time signal exists, not yet a validated result.\n")

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    print(f"Working table: {len(wide):,} rows, {DT_SECONDS*1000:.0f}ms per row\n")

    canonical_cols = list(get_local_feature_map(0).keys())
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    print(f"Using strict-precursor feature set: {len(strict_cols)} features "
          f"(voltage, GEI, source power excluded)")
    print(f"Training on every non-F row (see step41 for why block splitting does not apply "
          f"to this single-event task); evaluating on the fully held-out Bus F.\n")

    sweep_results = []
    saved_models = {}

    for n_seconds in LEAD_TIMES_SECONDS:
        print("-" * 70)
        print(f"LEAD TIME N = {n_seconds}s")
        print("-" * 70)

        long_frames = []
        for bus_idx, bus in enumerate(BUSES):
            sub = build_bus_frame_with_label(bus_idx, wide, strict_cols, n_seconds)
            sub["bus"] = bus
            long_frames.append(sub)
        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df[long_df["keep"]]

        train_pool = long_df[long_df["bus"] != HOLDOUT_BUS]
        holdout_pool = long_df[long_df["bus"] == HOLDOUT_BUS].sort_values("time")

        n_pos_train = int(train_pool["y_early"].sum())
        n_pos_holdout = int(holdout_pool["y_early"].sum())
        print(f"  Train rows: {len(train_pool):,} (real positives: {n_pos_train:,}, all from Bus E)  |  "
              f"Holdout rows: {len(holdout_pool):,} (positives: {n_pos_holdout:,})")

        if n_pos_train == 0 or n_pos_holdout == 0:
            print(f"  SKIPPED: no positive examples in train or holdout at N={n_seconds}s.")
            continue

        X_train_raw = train_pool[strict_cols].to_numpy(dtype=np.float64)
        y_train_raw = train_pool["y_early"].to_numpy(dtype=int)
        X_holdout = holdout_pool[strict_cols].to_numpy(dtype=np.float64)
        y_holdout = holdout_pool["y_early"].to_numpy(dtype=int)

        X_train, y_train = oversample_positives(X_train_raw, y_train_raw, TARGET_POS_FRACTION, RANDOM_STATE)

        pipeline = Pipeline([("x_scaler", StandardScaler()), ("mlp", MLPClassifier(**MLP_KWARGS))])
        pipeline.fit(X_train, y_train)
        mlp = pipeline.named_steps["mlp"]
        print(f"  Trained: {mlp.n_iter_} iterations, final loss {mlp.loss_:.6f}")

        proba_holdout = pipeline.predict_proba(X_holdout)[:, 1]
        try:
            auc_holdout = roc_auc_score(y_holdout, proba_holdout)
        except ValueError:
            auc_holdout = None

        thr_rows = sweep_metrics(y_holdout, proba_holdout, THRESHOLD_SWEEP)
        print(f"  Holdout AUC: {auc_holdout if auc_holdout is None else round(auc_holdout,4)}")
        print(f"  {'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'TP':>6s} {'FP':>6s}")
        for r in thr_rows:
            print(f"  {r['threshold']:>10.3f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['tp']:>6d} {r['fp']:>6d}")

        sweep_results.append({
            "n_seconds": n_seconds, "n_pos_train_real": n_pos_train, "n_pos_holdout": n_pos_holdout,
            "auc_holdout": auc_holdout, "threshold_sweep_holdout": thr_rows,
        })
        saved_models[n_seconds] = (pipeline, holdout_pool.reset_index(drop=True), proba_holdout)
        model_path = OUTPUT_DIR / f"early_warning_oversampled_N{n_seconds}s.pkl"
        joblib.dump(pipeline, model_path)
        print(f"  Wrote: {model_path}\n")

    if not sweep_results:
        raise RuntimeError("No lead time produced a usable result.")

    # --------------------------------------------------------------------
    # Plot 1: recall vs threshold, one line per N (the main diagnostic)
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sweep_results)))
    for r, color in zip(sweep_results, colors):
        thrs = [t["threshold"] for t in r["threshold_sweep_holdout"]]
        recs = [t["recall"] for t in r["threshold_sweep_holdout"]]
        ax.plot(thrs, recs, "o-", color=color, linewidth=2, markersize=5,
                label=f"N={r['n_seconds']}s (AUC={r['auc_holdout']:.3f})" if r["auc_holdout"] else f"N={r['n_seconds']}s")
    ax.invert_xaxis()
    ax.set_xlabel("decision threshold (reading right to left = more permissive)")
    ax.set_ylabel("recall (Bus F holdout)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Early warning, oversampled training: recall recovers at lower thresholds\n"
                 "(oversampling changes what the model learned; the threshold sweep changes only how it's read)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = OUTPUT_DIR / "oversampled_recall_vs_threshold.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out1}")

    # --------------------------------------------------------------------
    # Plot 2: AUC vs N, oversampled vs step41 (if step41 results are on disk)
    # --------------------------------------------------------------------
    step41_json = MODEL_ROOT / "41_early_warning_shifted_label" / "early_warning_sweep_results.json"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ns = [r["n_seconds"] for r in sweep_results]
    aucs_new = [r["auc_holdout"] for r in sweep_results]
    ax.plot(ns, aucs_new, "o-", color="#1D9E75", linewidth=2, markersize=7, label="oversampled (this script)")
    if step41_json.exists():
        step41_data = json.loads(step41_json.read_text())
        ns_old = [r["n_seconds"] for r in step41_data["sweep_results"]]
        aucs_old = [r["holdout"]["auc"] for r in step41_data["sweep_results"]]
        ax.plot(ns_old, aucs_old, "o--", color="#85B7EB", linewidth=1.5, markersize=6, label="step41 (no oversampling)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random (AUC=0.5)")
    ax.set_xlabel("lead time N (seconds before onset)")
    ax.set_ylabel("AUC (Bus F holdout)")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUC vs lead time: does oversampling change the ranking quality itself?")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = OUTPUT_DIR / "auc_vs_n_before_after.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out2}")

    # --------------------------------------------------------------------
    # Plot 3: probability zoom, smallest and largest N, oversampled models
    # --------------------------------------------------------------------
    plot_ns = [LEAD_TIMES_SECONDS[0], LEAD_TIMES_SECONDS[-1]]
    plot_ns = [n for n in plot_ns if n in saved_models]
    fig, axes = plt.subplots(len(plot_ns), 1, figsize=(13, 4.2 * len(plot_ns)))
    if len(plot_ns) == 1:
        axes = [axes]
    for ax, n_seconds in zip(axes, plot_ns):
        _, holdout_pool, proba_holdout = saved_models[n_seconds]
        t = holdout_pool["time"].to_numpy()
        y_true = holdout_pool["y_early"].to_numpy().astype(bool)
        onsets = get_onsets(wide["time"].to_numpy(), wide[f"collapsed_{HOLDOUT_BUS}"].to_numpy().astype(bool))
        onset_t = onsets[0] if len(onsets) else np.nan
        mask = (t >= onset_t - max(200, n_seconds * 2)) & (t <= onset_t + 20) if not np.isnan(onset_t) else np.ones(len(t), dtype=bool)
        ax.fill_between(t[mask], 0, 1, where=y_true[mask], color="tab:green", alpha=0.2, step="pre",
                        label=f"within {n_seconds}s of onset (positive label)")
        ax.plot(t[mask], proba_holdout[mask], color="tab:orange", linewidth=1.4, label="predicted probability (oversampled model)")
        if not np.isnan(onset_t):
            ax.axvline(onset_t, color="tab:red", linestyle="--", linewidth=1.2, label=f"true onset (t={onset_t:.0f}s)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="0.5 threshold")
        ax.set_title(f"Bus F (held out), lead time N={n_seconds}s, oversampled training")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(collapse within Ns)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out3 = OUTPUT_DIR / "oversampled_probability_zoom.png"
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out3}")

    # --------------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------------
    meta = {
        "task": "early-warning shifted-label sweep, oversampled training, strict-precursor features, Bus F held out",
        "caveat": "One collapse cascade in the dataset (Bus F then Bus E). This sweep is a first "
                   "signal check, not a validated result. Oversampling reweights training toward the "
                   "existing precursor rows, it does not manufacture new information.",
        "target_pos_fraction": TARGET_POS_FRACTION,
        "threshold_sweep": THRESHOLD_SWEEP,
        "lead_times_seconds": LEAD_TIMES_SECONDS,
        "n_strict_features": len(strict_cols),
        "sweep_results": sweep_results,
    }
    meta_path = OUTPUT_DIR / "oversampled_sweep_results.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. Read oversampled_recall_vs_threshold.png first: if recall recovers to a real")
    print("level (not just 1-2 lucky hits) as the threshold drops, especially for N=60s and N=90s")
    print("where step41 already showed strong AUC, that confirms real precursor signal that was")
    print("simply invisible at the default 0.5 threshold. If recall still stays near zero even at")
    print("very low thresholds, the AUC advantage from step41 was too weak to be practically useful.")


if __name__ == "__main__":
    main()

"""
STEP 41: EARLY-WARNING EXPERIMENT (SHIFTED LABEL, LEAD-TIME SWEEP)
================================================================================
Every model so far answered "is this bus collapsed right now." That question
can only be answered at or after the collapse instant, by construction. This
script asks a different question: "will this bus collapse in the next N
seconds." That is a genuinely different label, built by looking FORWARD from
each row to the next debounced collapse onset for that bus.

Label construction, per bus:
  - Find every debounced collapse onset time (from collapsed_<BUS>).
  - For each row NOT currently collapsed, compute time_to_next_onset = the
    next onset strictly after this row's time (inf if none exists).
  - label = 1 if 0 < time_to_next_onset <= N, else 0.
  - Rows that are CURRENTLY collapsed are dropped entirely from this
    experiment. Recognizing an ongoing collapse is what step36/38 already
    answered; this script is only about the window before onset.

N is swept (5, 15, 30, 60, 90 seconds) rather than fixed, because the useful
output is a curve of recall vs. lead time, not one number. That curve tells
you how far in advance the signal is actually usable.

TWO DESIGN DETAILS THAT MATTER HERE:
  1. NO block-based train/test split on the non-holdout buses. Earlier
     scripts in this project used a 20-block split so both train and test
     saw examples of the "is collapsed" label, which persists for over a
     thousand seconds and spans many blocks. The early-warning label is
     different: it's a short window (5-90s) immediately before Bus E's ONE
     onset, and that window sits entirely inside a single ~250s block.
     Whichever set that block lands in gets 100% of the precursor examples,
     the other set gets none, there is no way to subdivide a single event
     into two meaningful samples. So this script trains on every non-F row
     and evaluates purely on the fully held-out Bus F, which remains the one
     generalization check that means anything given N=1 collapse event.
  2. Only the strict-precursor 83 features are used (voltage, GEI, source
     power excluded). Including voltage would let the model wait for the
     last-second sag and call that "early," which defeats the purpose.

HONEST FRAMING, worth repeating before reading the output: this dataset
contains ONE collapse cascade (Bus F then Bus E, 53s apart). Shifting the
label does not create new collapse events, it relabels the same lead-up
window. Whatever this shows is a real first read on whether lead-time signal
exists at all, not yet a validated result. That still needs the multi-
scenario dataset regeneration from the next-steps list.

Run:
    python step41_early_warning_shifted_label.py
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
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, roc_auc_score, confusion_matrix)

# ----------------------------------------------------------------------
# Paths, must match step33/36/37/38/39/40
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

MODEL_ROOT = PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
OUTPUT_DIR = MODEL_ROOT / "41_early_warning_shifted_label"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
COLLAPSING_BUSES = ["E", "F"]
HEALTHY_CONTRAST_BUSES = ["A", "K"]
SUBSAMPLE_STRIDE = 5
DT_SECONDS = 0.01 * SUBSAMPLE_STRIDE  # 50ms per row after subsampling

N_BLOCKS, TEST_FRAC, PURGE_ROWS_BASE, RANDOM_STATE = 20, 0.20, 10, 42
LEAD_TIMES_SECONDS = [5, 15, 30, 60, 90]

HIDDEN_LAYER_SIZES = (128, 64)
MLP_KWARGS = dict(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation="relu", solver="adam",
    alpha=0.01, batch_size=256, learning_rate="constant", learning_rate_init=0.001,
    max_iter=1000, early_stopping=False, n_iter_no_change=25, tol=1e-4,
    random_state=RANDOM_STATE, verbose=False,
)


# ----------------------------------------------------------------------
# Shared utilities (identical to step36/37/38/39/40)
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
    """Sorted array of debounced collapse onset times for one bus."""
    prev = np.concatenate([[False], collapsed[:-1]])
    onset_mask = collapsed & ~prev
    return t[onset_mask]


def build_lookahead_label(t: np.ndarray, collapsed: np.ndarray, n_seconds: float):
    """
    Returns (label, keep_mask). keep_mask is False for rows currently
    collapsed (dropped from this experiment). label is 1 if the row is
    within n_seconds strictly before the next onset, else 0.
    """
    onsets = get_onsets(t, collapsed)
    keep_mask = ~collapsed
    if len(onsets) == 0:
        return np.zeros(len(t), dtype=int), keep_mask
    # for each t, index of first onset strictly greater than t
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


def evaluate(name, y_true, y_proba, threshold=0.5):
    if len(y_true) == 0 or y_true.sum() == 0:
        print(f"\n{name}: no positive examples in this slice, skipping metrics.")
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
    print(f"{name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} "
          f"f1={f1:.4f} auc={auc if auc is None else round(auc,4)}  TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    print("=" * 70)
    print("STEP 41: EARLY-WARNING EXPERIMENT (SHIFTED LABEL, LEAD-TIME SWEEP)")
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
          f"(voltage, GEI, source power excluded)\n")

    print("NOTE ON SPLIT DESIGN FOR THIS TASK:")
    print("The block-based train/test split used elsewhere in this project does not apply")
    print("here. There is exactly one precursor window (Bus E's, ~90s wide at the largest N),")
    print("and it sits entirely inside a single ~250s time block. Whichever set that block")
    print("lands in gets ALL the precursor examples; the other set gets none. Subdividing a")
    print("single event further does not produce a second meaningful test, it can only starve")
    print("training or leak the answer. So for this experiment: train on every non-F row,")
    print("evaluate purely on the fully held-out Bus F, which is the one generalization check")
    print("that was never affected by this issue.\n")

    sweep_results = []
    saved_models = {}

    for n_seconds in LEAD_TIMES_SECONDS:
        print("-" * 70)
        print(f"LEAD TIME N = {n_seconds}s")
        print("-" * 70)

        # build long-format table across all 10 buses for this N (no block split needed)
        long_frames = []
        for bus_idx, bus in enumerate(BUSES):
            sub = build_bus_frame_with_label(bus_idx, wide, strict_cols, n_seconds)
            sub["bus"] = bus
            long_frames.append(sub)
        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df[long_df["keep"]]  # drop rows currently collapsed

        train_pool = long_df[long_df["bus"] != HOLDOUT_BUS]
        holdout_pool = long_df[long_df["bus"] == HOLDOUT_BUS].sort_values("time")

        n_pos_train = int(train_pool["y_early"].sum())
        n_pos_holdout = int(holdout_pool["y_early"].sum())
        print(f"  Train rows: {len(train_pool):,} (positives: {n_pos_train:,}, all from Bus E)  |  "
              f"Holdout rows: {len(holdout_pool):,} (positives: {n_pos_holdout:,})")

        if n_pos_train == 0 or n_pos_holdout == 0:
            print(f"  SKIPPED: no positive examples in train or holdout at N={n_seconds}s.")
            continue

        X_train = train_pool[strict_cols].to_numpy(dtype=np.float64)
        y_train = train_pool["y_early"].to_numpy(dtype=int)
        X_holdout = holdout_pool[strict_cols].to_numpy(dtype=np.float64)
        y_holdout = holdout_pool["y_early"].to_numpy(dtype=int)

        pipeline = Pipeline([("x_scaler", StandardScaler()), ("mlp", MLPClassifier(**MLP_KWARGS))])
        pipeline.fit(X_train, y_train)
        mlp = pipeline.named_steps["mlp"]
        print(f"  Trained: {mlp.n_iter_} iterations, final loss {mlp.loss_:.6f}")

        proba_train = pipeline.predict_proba(X_train)[:, 1]
        proba_holdout = pipeline.predict_proba(X_holdout)[:, 1]

        r_train = evaluate("  train (fit quality only, not generalization)", y_train, proba_train)
        r_holdout = evaluate("  holdout(F) -- the metric that matters", y_holdout, proba_holdout)

        sweep_results.append({
            "n_seconds": n_seconds,
            "n_pos_train": n_pos_train, "n_pos_holdout": n_pos_holdout,
            "train": r_train, "holdout": r_holdout,
        })
        saved_models[n_seconds] = (pipeline, holdout_pool.reset_index(drop=True), proba_holdout)
        model_path = OUTPUT_DIR / f"early_warning_N{n_seconds}s.pkl"
        joblib.dump(pipeline, model_path)
        print(f"  Wrote: {model_path}\n")

    if not sweep_results:
        raise RuntimeError("No lead time produced a usable result, check LEAD_TIMES_SECONDS "
                            "and the purge/block-width relationship.")

    # --------------------------------------------------------------------
    # Plot 1: recall / precision vs lead time (the main deliverable)
    # --------------------------------------------------------------------
    ns = [r["n_seconds"] for r in sweep_results]
    recalls = [r["holdout"]["recall"] if r["holdout"] else np.nan for r in sweep_results]
    precisions = [r["holdout"]["precision"] if r["holdout"] else np.nan for r in sweep_results]
    aucs = [r["holdout"]["auc"] if r["holdout"] else np.nan for r in sweep_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, recalls, "o-", color="#1D9E75", linewidth=2, markersize=7, label="Recall (Bus F holdout)")
    ax.plot(ns, precisions, "o-", color="#F0997B", linewidth=2, markersize=7, label="Precision (Bus F holdout)")
    ax.plot(ns, aucs, "o--", color="#85B7EB", linewidth=1.5, markersize=6, label="AUC (Bus F holdout)")
    ax.set_xlabel("lead time N (seconds before onset)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Early-warning sweep: strict-precursor features, Bus F held out\n"
                 "(one collapse cascade in the dataset, read as a first signal check)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = OUTPUT_DIR / "early_warning_lead_time_sweep.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out1}")

    # --------------------------------------------------------------------
    # Plot 2: probability vs time for smallest and largest usable N
    # --------------------------------------------------------------------
    plot_ns = [ns[0], ns[-1]] if len(ns) > 1 else ns
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
        ax.plot(t[mask], proba_holdout[mask], color="tab:orange", linewidth=1.4, label="predicted probability")
        if not np.isnan(onset_t):
            ax.axvline(onset_t, color="tab:red", linestyle="--", linewidth=1.2, label=f"true onset (t={onset_t:.0f}s)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Bus F (held out), lead time N={n_seconds}s")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(collapse within Ns)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out2 = OUTPUT_DIR / "early_warning_probability_zoom.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out2}")

    # --------------------------------------------------------------------
    # Save the numeric sweep for the record
    # --------------------------------------------------------------------
    meta = {
        "task": "early-warning shifted-label sweep, strict-precursor features, Bus F held out",
        "caveat": "One collapse cascade in the dataset (Bus F then Bus E). This sweep is a first "
                   "signal check, not a validated result. Needs multi-scenario data before trusting "
                   "any single N's number as representative.",
        "lead_times_seconds": LEAD_TIMES_SECONDS,
        "n_strict_features": len(strict_cols),
        "sweep_results": sweep_results,
    }
    meta_path = OUTPUT_DIR / "early_warning_sweep_results.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\nDone. Read early_warning_lead_time_sweep.png first: if recall stays near 0 for every N,")
    print("there is no usable lead-time signal yet in the strict-precursor features. If it rises")
    print("above 0 for small N and decays as N grows, that is a genuine (if single-event) early")
    print("warning signal, worth pursuing with more collapse scenarios next.")


if __name__ == "__main__":
    main()

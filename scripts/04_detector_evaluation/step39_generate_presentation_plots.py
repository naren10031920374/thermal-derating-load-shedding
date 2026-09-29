"""
STEP 39: PRESENTATION PLOTS FOR THE THREE-MODEL COMPARISON
================================================================================
Produces six figures for the professor/Hayla.ai presentation, all built from
real saved data and real model predictions, nothing illustrative:

  1. bus_E_F_collapse_raw.png
     What the two real collapses in the dataset actually look like: voltage
     and GEI for buses E and F over the full 5000s run, with the 100V
     threshold and the debounced collapsed_<BUS> windows shaded.

  2. bus_F_holdout_split.png
     The blocked-purged time split (20 blocks, this run's actual assignment)
     plus real per-bus row counts, showing exactly what "Bus F held out
     completely" means: F contributes zero rows to training, period.

  3. full_model_detection.png
     Loads the saved 121-feature model (bus_symmetric_collapse_detector.pkl).
     Predicted probability vs actual collapse, buses E, F (holdout), A, K.

  4. raw_ceiling_detection.png
     Loads the saved 51-feature raw-signal model (step38's
     raw_signal_collapse_detector.pkl). Same layout as (3).

  5. strict_precursor_detection.png
     Retrains the 83-feature strict-precursor model IN THIS SCRIPT, since
     bus_symmetric_precursor_only_test.py never saved a pickle, only a JSON
     summary. Same feature list, same MLP_KWARGS, same random_state=42 as
     that script, so this reproduces the exact model behind the 0.659 recall
     number in precursor_only_comparison.json. Same layout as (3) and (4),
     plus a recall-by-time-since-onset bar chart for Bus F to show WHERE in
     the collapse window this model misses (see step38's onset breakdown for
     the same chart on the raw-ceiling model, reused here for contrast).

  6. three_way_recall_comparison.png
     Grouped bar chart, train / test-seen-buses / holdout-Bus-F recall for
     all three models, read directly from the saved JSON result files, not
     retyped. This is the one-slide summary of the table you sketched.

Run:
    python step39_generate_presentation_plots.py
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
from sklearn.metrics import recall_score

# ----------------------------------------------------------------------
# Paths, must match step33 / 36 / 37 / 38 exactly
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
PRECURSOR_JSON = MODEL_ROOT / "37_precursor_only_test" / "precursor_only_comparison.json"
RAW_CEILING_MODEL_PATH = MODEL_ROOT / "38_raw_signal_detector" / "raw_signal_collapse_detector.pkl"
RAW_CEILING_JSON = MODEL_ROOT / "38_raw_signal_detector" / "raw_signal_metadata.json"

OUTPUT_DIR = MODEL_ROOT / "39_presentation_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
COLLAPSING_BUSES = ["E", "F"]
HEALTHY_CONTRAST_BUSES = ["A", "K"]
SUBSAMPLE_STRIDE = 5
COLLAPSE_VOLTAGE_V = 100.0

N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE = 20, 0.20, 10, 42
HIDDEN_LAYER_SIZES = (128, 64)
MLP_KWARGS = dict(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation="relu", solver="adam",
    alpha=0.01, batch_size=256, learning_rate="constant", learning_rate_init=0.001,
    max_iter=1000, early_stopping=False, n_iter_no_change=25, tol=1e-4,
    random_state=RANDOM_STATE, verbose=False,
)

ONSET_BUCKETS = [(0, 5), (5, 30), (30, 100), (100, np.inf)]


# ----------------------------------------------------------------------
# Shared utilities, copied verbatim from step36/37/38 so this script is
# standalone and guaranteed consistent with the saved models' training code.
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
    block_of_row = np.zeros(n_rows, dtype=int)
    for bi, (start, end) in enumerate(blocks):
        block_of_row[start:end] = bi
        p_start = start + purge_rows if bi > 0 else start
        p_end   = end - purge_rows if bi < n_blocks - 1 else end
        if p_end <= p_start:
            continue
        split[p_start:p_end] = "test" if bi in test_block_set else "train"
    return split, block_of_row, blocks, test_block_set


def load_corrected_targets():
    df = pd.read_csv(CORRECTED_TARGETS_CSV)
    bool_cols = [c for c in df.columns if c.startswith("affected_") or c.startswith("collapsed_")]
    for c in bool_cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df[c] = df[c].astype(bool)
    return df


def build_long_df(wide: pd.DataFrame, canonical_cols: list[str]) -> pd.DataFrame:
    """Reshape wide -> bus-symmetric long format using the given canonical column list."""
    long_frames = []
    for bus_idx, bus in enumerate(BUSES):
        fmap = get_local_feature_map(bus_idx)
        actual_cols = [fmap[c] for c in canonical_cols]
        sub = wide[actual_cols].copy()
        sub.columns = canonical_cols
        sub["bus"] = bus
        sub["time"] = wide["time"].to_numpy()
        sub["split"] = wide["_time_split"].to_numpy()
        sub["y_collapsed"] = wide[f"collapsed_{bus}"].to_numpy().astype(int)
        long_frames.append(sub)
    return pd.concat(long_frames, ignore_index=True)


def onset_recall(sub_holdout: pd.DataFrame, y_proba: np.ndarray, threshold: float = 0.5):
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
        label = f"{lo:.0f}-{'inf' if hi == np.inf else f'{hi:.0f}'}s"
        rows.append({"window": label, "n_rows": n, "recall": rec})
    return rows


# ----------------------------------------------------------------------
# Plot 1: raw E/F collapse in the dataset
# ----------------------------------------------------------------------
def plot1_raw_collapse(raw: pd.DataFrame, targets: pd.DataFrame):
    print("\n[1/6] Plotting raw Bus E / Bus F collapse from the dataset ...")
    t = raw["time"].to_numpy()
    colors = {"E": "#D85A30", "F": "#378ADD"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax_v, ax_gei = axes

    for bus in COLLAPSING_BUSES:
        v = raw[f"V_Bus_{bus}"].to_numpy()
        collapsed = targets[f"collapsed_{bus}"].to_numpy()
        ax_v.plot(t, v, color=colors[bus], linewidth=1.1, label=f"Bus {bus} voltage")
        ax_v.fill_between(t, 0, v.max() * 1.05, where=collapsed, color=colors[bus],
                           alpha=0.12, step="pre")
        if collapsed.any():
            onset_t = t[np.argmax(collapsed)]
            ax_v.axvline(onset_t, color=colors[bus], linestyle=":", linewidth=1)
            ax_v.annotate(f"Bus {bus} onset t={onset_t:.0f}s", xy=(onset_t, v.max() * 0.9),
                          xytext=(8, 0), textcoords="offset points", fontsize=9, color=colors[bus])
    ax_v.axhline(COLLAPSE_VOLTAGE_V, color="gray", linestyle="--", linewidth=1,
                 label=f"{COLLAPSE_VOLTAGE_V:.0f}V collapse threshold")
    ax_v.set_ylabel("Bus voltage (V)")
    ax_v.set_title("Buses E and F: voltage over the full 5000s run, shaded = debounced collapse window")
    ax_v.legend(loc="upper right", fontsize=9)
    ax_v.grid(True, alpha=0.3)

    for bus in COLLAPSING_BUSES:
        gei_col = f"GEI_{bus}"
        if gei_col in raw.columns:
            collapsed = targets[f"collapsed_{bus}"].to_numpy()
            ax_gei.plot(t, raw[gei_col].to_numpy(), color=colors[bus], linewidth=1.0,
                        label=f"Bus {bus} GEI")
            ax_gei.fill_between(t, -0.5, 2.0, where=collapsed, color=colors[bus], alpha=0.1, step="pre")
    ax_gei.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="GEI=1.0 (grid average)")
    ax_gei.set_ylabel("GEI")
    ax_gei.set_xlabel("time (s)")
    ax_gei.set_title("Same windows: GEI crashes in the same instant as voltage, not before")
    ax_gei.legend(loc="upper right", fontsize=9)
    ax_gei.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "bus_E_F_collapse_raw.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {out}")


# ----------------------------------------------------------------------
# Plot 2: holdout / split methodology
# ----------------------------------------------------------------------
def plot2_holdout_split(wide: pd.DataFrame, block_of_row, blocks, test_block_set):
    print("\n[2/6] Plotting Bus F holdout and time-block split methodology ...")
    fig, (ax_blocks, ax_counts) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})

    t = wide["time"].to_numpy()
    for bi, (start, end) in enumerate(blocks):
        t0, t1 = t[start], t[min(end, len(t) - 1)]
        color = "#F0997B" if bi in test_block_set else "#85B7EB"
        ax_blocks.axvspan(t0, t1, color=color, alpha=0.6, linewidth=0)
    ax_blocks.set_yticks([])
    ax_blocks.set_xlabel("time (s)")
    ax_blocks.set_title(f"{N_BLOCKS}-block split, {int(N_BLOCKS*TEST_FRAC)} test blocks, "
                         f"{PURGE_ROWS}-row purge at every boundary\n(this run's actual random_state=42 assignment)")
    from matplotlib.patches import Patch
    ax_blocks.legend(handles=[Patch(color="#85B7EB", label="train block"),
                               Patch(color="#F0997B", label="test block")], loc="upper right", fontsize=9)

    bus_labels, train_counts, test_counts, holdout_counts = [], [], [], []
    for bus in BUSES:
        n_train = int((wide["_time_split"] == "train").sum())
        n_test = int((wide["_time_split"] == "test").sum())
        bus_labels.append(bus)
        if bus == HOLDOUT_BUS:
            train_counts.append(0)
            test_counts.append(0)
            holdout_counts.append(n_train + n_test)
        else:
            train_counts.append(n_train)
            test_counts.append(n_test)
            holdout_counts.append(0)

    x = np.arange(len(BUSES))
    ax_counts.bar(x, train_counts, color="#85B7EB", label="train rows")
    ax_counts.bar(x, test_counts, bottom=train_counts, color="#F0997B", label="test rows (seen buses)")
    ax_counts.bar(x, holdout_counts, color="#993C1D", label=f"holdout rows (Bus {HOLDOUT_BUS} only)")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(BUSES)
    ax_counts.set_ylabel("rows contributed")
    ax_counts.set_title(f"Bus {HOLDOUT_BUS}: zero rows in training, ever")
    ax_counts.legend(loc="upper right", fontsize=8)
    ax_counts.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = OUTPUT_DIR / "bus_F_holdout_split.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {out}")


# ----------------------------------------------------------------------
# Plots 3/4/5: predicted probability vs actual, for a given fitted pipeline
# ----------------------------------------------------------------------
def plot_detection(pipeline, canonical_cols, long_df_by_cols, title, filename,
                    include_onset_chart=False):
    print(f"  Building detection plot: {title}")
    plot_buses = list(dict.fromkeys(COLLAPSING_BUSES + HEALTHY_CONTRAST_BUSES))
    bus_results = {}
    for bus in plot_buses:
        sub = long_df_by_cols[long_df_by_cols["bus"] == bus].sort_values("time")
        X = sub[canonical_cols].to_numpy(dtype=np.float64)
        proba = pipeline.predict_proba(X)[:, 1]
        sub = sub.copy()
        sub["proba"] = proba
        bus_results[bus] = sub

    n_rows_extra = 1 if include_onset_chart else 0
    fig, axes = plt.subplots(len(plot_buses) + n_rows_extra, 1,
                              figsize=(14, 2.8 * len(plot_buses) + (2.5 if include_onset_chart else 0)),
                              gridspec_kw={"height_ratios": [1] * len(plot_buses) + ([0.9] if include_onset_chart else [])})
    det_axes = axes[:len(plot_buses)] if include_onset_chart else axes

    for ax, bus in zip(det_axes, plot_buses):
        sub = bus_results[bus]
        t = sub["time"].to_numpy()
        y_true = sub["y_collapsed"].to_numpy().astype(bool)
        ax.fill_between(t, 0, 1, where=y_true, color="tab:blue", alpha=0.2, step="pre", label="actual collapsed")
        ax.plot(t, sub["proba"].to_numpy(), color="tab:orange", linewidth=1.0, label="predicted probability")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        role = "  (HELD OUT)" if bus == HOLDOUT_BUS else ("  (seen in training)" if bus in COLLAPSING_BUSES else "  (healthy, false-positive check)")
        ax.set_title(f"Bus {bus}{role}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(collapsed)")
        ax.grid(True, alpha=0.3)
    det_axes[0].legend(loc="upper right", fontsize=9)
    det_axes[-1].set_xlabel("time (s)")

    if include_onset_chart:
        ax_onset = axes[-1]
        rows = onset_recall(bus_results[HOLDOUT_BUS], bus_results[HOLDOUT_BUS]["proba"].to_numpy())
        labels = [r["window"] for r in rows]
        vals = [0 if np.isnan(r["recall"]) else r["recall"] for r in rows]
        bars = ax_onset.bar(labels, vals, color="#1D9E75")
        for bar, r in zip(bars, rows):
            ax_onset.annotate(f"n={r['n_rows']}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax_onset.set_ylim(0, 1.15)
        ax_onset.set_ylabel("recall")
        ax_onset.set_title(f"Bus {HOLDOUT_BUS}: recall by time-since-onset (0.5 threshold)")
        ax_onset.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Wrote: {out}")


# ----------------------------------------------------------------------
# Plot 6: three-way recall comparison from saved JSON results
# ----------------------------------------------------------------------
def plot6_three_way_comparison():
    print("\n[6/6] Plotting three-way recall comparison from saved results ...")
    if not PRECURSOR_JSON.exists():
        print(f"  SKIPPED: {PRECURSOR_JSON} not found.")
        return
    precursor = json.loads(PRECURSOR_JSON.read_text())
    raw_ceiling = json.loads(RAW_CEILING_JSON.read_text()) if RAW_CEILING_JSON.exists() else None

    models = ["Full\n(121 features)", "Raw ceiling\n(51 features)", "Strict precursor\n(83 features)"]
    splits = ["train", "test_seen", "holdout"]
    split_labels = ["Train", "Test (seen buses)", f"Holdout (Bus {HOLDOUT_BUS})"]

    recall_full = [precursor["results_full"][s]["recall"] for s in splits]
    if raw_ceiling is not None:
        rc = raw_ceiling["results_default_threshold"]
        recall_raw = [rc["train"]["recall"], rc["test_seen_buses"]["recall"],
                      rc[f"holdout_bus_{HOLDOUT_BUS}"]["recall"]]
    else:
        recall_raw = [np.nan, np.nan, np.nan]
    recall_strict = [precursor["results_strict_precursor"][s]["recall"] for s in splits]

    data = np.array([recall_full, recall_raw, recall_strict])  # 3 models x 3 splits

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    colors = ["#85B7EB", "#F0997B", "#97C459"]
    for i, (split_label, color) in enumerate(zip(split_labels, colors)):
        vals = data[:, i]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=split_label, color=color)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Recall across train / test / Bus-{HOLDOUT_BUS}-holdout, three feature sets\n"
                 "(same split, same architecture, only the feature list changes)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = OUTPUT_DIR / "three_way_recall_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {out}")


def main():
    print("=" * 70)
    print("STEP 39: PRESENTATION PLOTS")
    print("=" * 70)

    raw = pd.read_csv(RAW_CSV)
    targets = load_corrected_targets()
    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    wide = pd.merge(features_df, targets, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    print(f"Loaded raw CSV ({len(raw):,} rows) and joined features+targets ({len(wide):,} rows)")

    split, block_of_row, blocks, test_block_set = blocked_purged_split(
        len(wide), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide["_time_split"] = split

    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
        print(f"Subsampled every {SUBSAMPLE_STRIDE} rows for the modeling plots: {len(wide):,} rows")

    # ---- Plot 1: raw collapse (full resolution, not subsampled) ----
    plot1_raw_collapse(raw, targets)

    # ---- Plot 2: split methodology ----
    wide_full_res = pd.merge(features_df, targets, on="time", how="inner", validate="one_to_one").sort_values("time").reset_index(drop=True)
    split_full, block_of_row_full, blocks_full, test_block_set_full = blocked_purged_split(
        len(wide_full_res), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide_full_res["_time_split"] = split_full
    plot2_holdout_split(wide_full_res, block_of_row_full, blocks_full, test_block_set_full)

    # ---- Full 121-feature canonical columns (also used as base for strict) ----
    canonical_cols = list(get_local_feature_map(0).keys())

    # ---- Plot 3: full model ----
    if FULL_MODEL_PATH.exists():
        print(f"\n[3/6] Loading full model: {FULL_MODEL_PATH}")
        full_pipeline = joblib.load(FULL_MODEL_PATH)
        long_full = build_long_df(wide, canonical_cols)
        plot_detection(full_pipeline, canonical_cols, long_full,
                       title="Full model (121 features): same-instant detection, includes own voltage/GEI",
                       filename="full_model_detection.png")
    else:
        print(f"\n[3/6] SKIPPED: full model not found at {FULL_MODEL_PATH}")

    # ---- Plot 4: raw ceiling model ----
    if RAW_CEILING_MODEL_PATH.exists():
        print(f"\n[4/6] Loading raw ceiling model: {RAW_CEILING_MODEL_PATH}")
        raw_pipeline = joblib.load(RAW_CEILING_MODEL_PATH)
        long_raw = build_long_df(wide, RAW_FEATURE_KEYS)
        plot_detection(raw_pipeline, RAW_FEATURE_KEYS, long_raw,
                       title="Raw ceiling model (51 features: voltage, GEI, load, Tj, Th, Tj-Th): voltage/GEI make this easy",
                       filename="raw_ceiling_detection.png", include_onset_chart=True)
    else:
        print(f"\n[4/6] SKIPPED: raw ceiling model not found at {RAW_CEILING_MODEL_PATH}")

    # ---- Plot 5: strict precursor model, RETRAINED here (never saved to disk) ----
    print("\n[5/6] Retraining strict precursor model (83 features, not saved by step37) ...")
    strict_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    long_strict = build_long_df(wide, canonical_cols)  # build once with full cols, select subset below
    train_pool = long_strict[(long_strict["bus"] != HOLDOUT_BUS) & (long_strict["split"] == "train")]
    X_train = train_pool[strict_cols].to_numpy(dtype=np.float64)
    y_train = train_pool["y_collapsed"].to_numpy(dtype=int)
    strict_pipeline = Pipeline([("x_scaler", StandardScaler()), ("mlp", MLPClassifier(**MLP_KWARGS))])
    strict_pipeline.fit(X_train, y_train)
    print(f"  Retrained: {strict_pipeline.named_steps['mlp'].n_iter_} iterations, "
          f"final loss {strict_pipeline.named_steps['mlp'].loss_:.6f}")
    holdout_check = long_strict[long_strict["bus"] == HOLDOUT_BUS]
    proba_check = strict_pipeline.predict_proba(holdout_check[strict_cols].to_numpy(dtype=np.float64))[:, 1]
    recall_check = recall_score(holdout_check["y_collapsed"].to_numpy(), (proba_check >= 0.5).astype(int))
    print(f"  Sanity check, Bus F recall={recall_check:.4f} "
          f"(should match precursor_only_comparison.json results_strict_precursor.holdout.recall ~0.659)")
    strict_model_path = OUTPUT_DIR / "strict_precursor_collapse_detector_retrained.pkl"
    joblib.dump(strict_pipeline, strict_model_path)
    print(f"  Wrote retrained model: {strict_model_path}")
    plot_detection(strict_pipeline, strict_cols, long_strict,
                   title="Strict precursor model (83 features: voltage/GEI/power removed): must learn warning signs",
                   filename="strict_precursor_detection.png", include_onset_chart=True)

    # ---- Plot 6: summary comparison ----
    plot6_three_way_comparison()

    print("\nDone. All figures in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

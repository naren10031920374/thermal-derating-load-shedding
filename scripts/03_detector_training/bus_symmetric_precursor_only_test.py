"""
PRECURSOR-ONLY TEST: IS THIS DETECTING COLLAPSE, OR JUST READING VOLTAGE?
================================================================================
bus_symmetric_collapse_detector.py achieved recall=1.000, precision=1.000 on
Bus F, a bus completely excluded from training. Before treating that as
strong evidence of learned, transferable pattern recognition, worth being
honest about why it was likely this clean: own_Voltage is one of the 121
canonical features, and collapsed_<BUS> is DEFINED as "voltage under 100V for
0.5s". Any reasonably competent classifier given direct access to voltage
would nail this for any bus, since the rule is the same physical threshold
everywhere in the ring. That result mostly confirms the reshaping and holdout
mechanics work correctly, not that the model learned something subtle.

This script removes every canonical feature that is a direct or near-direct
algebraic function of the bus's OWN voltage or GEI (26 of 121 features, see
DIRECT_SIGNAL_KEYS below, built by tracing exactly which canonical keys
derive from Voltage or GEI in get_local_feature_map), and reruns the EXACT
SAME Bus-F holdout experiment on the remaining 95 "precursor" features:
temperature trends, commanded load, power/load imbalance, and the adjacent
converters' thermal/derate/junction signals for both this bus and its two
ring-neighbors. GEI and Voltage are still visible for the two NEIGHBOR buses
(neighbor_prev_Voltage, neighbor_next_GEI, etc.), only the bus's OWN
voltage/GEI and everything derived from them is removed. The neighbor
readings are a legitimate, physically meaningful precursor signal (a
neighboring bus already collapsing is real information available before this
bus's own voltage necessarily crashes), not the same tautology as reading the
bus's own voltage to predict its own voltage-defined label.

Both experiments (full features vs precursor-only) run on the IDENTICAL rows
and split, only the column selection differs, so any performance gap is
attributable to the removed signals, not a different train/test division.

Interpretation:
  - If precursor-only recall on Bus F stays reasonably high (the exact bar is
    a judgment call, but a small drop, not a collapse to near-zero) that is
    real evidence of learned, transferable precursor patterns, not just
    voltage-thresholding, and meaningfully raises confidence this would catch
    a genuinely new bus like K before its voltage crashes.
  - If precursor-only recall drops sharply (toward the false-alarm rate a
    coin flip would produce), the original full-feature result was mostly
    "detected the voltage crash", useful as a same-instant confirmation
    signal, but not evidence of early or transferable pattern recognition.

Run:
    python bus_symmetric_precursor_only_test.py
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
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
FEATURE_DIR = PROJECT_ROOT / "data_d" / "unified_10bus_derating_features"
FEATURE_PARQUET = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.parquet"
FEATURE_CSV     = FEATURE_DIR / "v7_ALfix_GEIfix_10bus_derating_5000s_features.csv"
CORRECTED_TARGETS_CSV = FEATURE_DIR / "corrected_phase_targets_5000s.csv"

OUTPUT_DIR = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
              / "37_precursor_only_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"
SUBSAMPLE_STRIDE = 5   # match bus_symmetric_collapse_detector.py's default for a fair comparison

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
    verbose=False,   # quieter, two full training runs happen in this script
)

# Every canonical feature that is a direct or near-direct algebraic function
# of the bus's OWN voltage or GEI. Traced exactly against get_local_feature_map
# below, not guessed. 14 voltage-derived + 12 GEI-derived = 26 total.
DIRECT_SIGNAL_KEYS = [
    # own voltage and everything derived from it
    "own_Voltage", "own_Voltage_lag1", "own_Voltage_lag2", "own_Voltage_lag3",
    "own_Voltage_roll_mean_5", "own_Voltage_roll_std_5", "own_d_Voltage",
    "own_Voltage_Error", "own_Abs_Voltage_Error",
    "own_Voltage_Imbalance", "own_Voltage_Imbalance_lag1",
    "own_Voltage_Imbalance_lag2", "own_Voltage_Imbalance_lag3",
    "own_Abs_Voltage_Imbalance",
    # own GEI and everything derived from it
    "own_GEI", "own_GEI_lag1", "own_GEI_lag2", "own_GEI_lag3",
    "own_GEI_roll_mean_5", "own_GEI_roll_std_5", "own_d_GEI",
    "own_GEI_Error", "own_GEI_Error_lag1", "own_GEI_Error_lag2",
    "own_GEI_Error_lag3", "own_Abs_GEI_Error",
]

# ADDITIONAL keys for the STRICT precursor test. own_Source_Power was left in
# the original "precursor" set, reasoned as a distinct signal from voltage.
# That reasoning was wrong: physically, a bus at near-zero voltage cannot
# deliver meaningful power (P = V x I), so Source_Power crashes in the SAME
# instant as voltage during a real collapse, it is another same-instant
# symptom, not a precursor. This is exactly what explained the suspicious
# BIT-FOR-BIT IDENTICAL result (same TP/FP/FN/TN to the last row) between the
# full and light-precursor experiments on the real dataset: the model simply
# swapped one perfect same-instant proxy for another equally perfect one.
STRICT_ADDITIONAL_KEYS = [
    "own_Source_Power", "own_Source_Power_lag1", "own_Source_Power_lag2",
    "own_Source_Power_lag3", "own_Source_Power_roll_mean_5",
    "own_Source_Power_roll_std_5", "own_d_Source_Power",
    "own_Power_Imbalance", "own_Power_Imbalance_lag1",
    "own_Power_Imbalance_lag2", "own_Power_Imbalance_lag3",
    "own_Abs_Power_Imbalance",
]


# ----------------------------------------------------------------------
# Canonical local+neighbor feature map, identical to bus_symmetric_collapse_detector.py
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


def evaluate(pipeline, name, X, y):
    if len(y) == 0:
        return None
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    if y.sum() == 0:
        return {"accuracy": acc, "precision": None, "recall": None, "f1": None, "auc": None}
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = None
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    print(f"  [{name}] accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  "
          f"f1={f1:.4f}  auc={auc if auc is None else round(auc,4)}  "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def run_experiment(label, cols, train_pool, test_pool_seen, holdout_pool):
    print(f"\n{'='*70}\nEXPERIMENT: {label}  ({len(cols)} features)\n{'='*70}")
    X_train = train_pool[cols].to_numpy(dtype=np.float64)
    y_train = train_pool["y_collapsed"].to_numpy(dtype=int)
    X_test_seen = test_pool_seen[cols].to_numpy(dtype=np.float64)
    y_test_seen = test_pool_seen["y_collapsed"].to_numpy(dtype=int)
    X_holdout = holdout_pool[cols].to_numpy(dtype=np.float64)
    y_holdout = holdout_pool["y_collapsed"].to_numpy(dtype=int)

    pipeline = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MLP_KWARGS)),
    ])
    pipeline.fit(X_train, y_train)
    mlp = pipeline.named_steps["mlp"]
    print(f"Trained: {mlp.n_iter_} iterations, final loss {mlp.loss_:.6f}")

    print("Evaluation:")
    r_train = evaluate(pipeline, "train", X_train, y_train)
    r_seen = evaluate(pipeline, "seen-buses test", X_test_seen, y_test_seen)
    r_holdout = evaluate(pipeline, f"HOLDOUT BUS {HOLDOUT_BUS}", X_holdout, y_holdout)

    return pipeline, {"train": r_train, "test_seen": r_seen, "holdout": r_holdout}


def main():
    print("=" * 70)
    print("PRECURSOR-ONLY TEST")
    print("=" * 70)

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    print(f"Joined wide table: {len(wide):,} rows")

    time_split = blocked_purged_split(len(wide), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide["_time_split"] = time_split

    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
        print(f"Subsampled every {SUBSAMPLE_STRIDE} rows: {len(wide):,} timesteps remain")

    print("\nReshaping to bus-symmetric long format ...")
    long_frames = []
    canonical_cols = None
    for bus_idx, bus in enumerate(BUSES):
        fmap = get_local_feature_map(bus_idx)
        if canonical_cols is None:
            canonical_cols = list(fmap.keys())
        actual_cols = [fmap[c] for c in canonical_cols]
        missing = [c for c in actual_cols if c not in wide.columns]
        if missing:
            raise ValueError(f"Bus {bus}: missing columns:\n{missing[:10]}")
        sub = wide[actual_cols].copy()
        sub.columns = canonical_cols
        sub["bus"] = bus
        sub["split"] = wide["_time_split"].to_numpy()
        sub["y_collapsed"] = wide[f"collapsed_{bus}"].to_numpy().astype(int)
        long_frames.append(sub)
    long_df = pd.concat(long_frames, ignore_index=True)
    print(f"Total long-format rows: {len(long_df):,}, canonical feature count: {len(canonical_cols)}")

    missing_drop = [k for k in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS if k not in canonical_cols]
    if missing_drop:
        raise ValueError(f"Signal-removal keys reference names not present in the canonical "
                          f"feature map, check for a naming mismatch:\n{missing_drop}")

    full_cols = canonical_cols
    light_precursor_cols = [c for c in canonical_cols if c not in DIRECT_SIGNAL_KEYS]
    strict_precursor_cols = [c for c in canonical_cols
                              if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]
    print(f"\nFull feature set: {len(full_cols)} features")
    print(f"Light precursor (voltage/GEI removed): {len(light_precursor_cols)} features")
    print(f"Strict precursor (voltage/GEI/power removed): {len(strict_precursor_cols)} features "
          f"({len(DIRECT_SIGNAL_KEYS) + len(STRICT_ADDITIONAL_KEYS)} same-instant electrical "
          f"symptoms removed, only thermal/load/converter/neighbor signals remain)")

    train_pool = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "train")]
    test_pool_seen = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "test")]
    holdout_pool = long_df[long_df["bus"] == HOLDOUT_BUS]
    print(f"\nTrain rows: {len(train_pool):,}  Seen-test rows: {len(test_pool_seen):,}  "
          f"Holdout bus {HOLDOUT_BUS} rows: {len(holdout_pool):,}")

    # --------------------------------------------------------------------
    # Run all three experiments on the IDENTICAL rows and split
    # --------------------------------------------------------------------
    _, results_full = run_experiment("FULL FEATURES (includes own voltage/GEI/power)",
                                      full_cols, train_pool, test_pool_seen, holdout_pool)
    _, results_light = run_experiment("LIGHT PRECURSOR (own voltage/GEI removed, power kept)",
                                       light_precursor_cols, train_pool, test_pool_seen, holdout_pool)
    _, results_strict = run_experiment("STRICT PRECURSOR (voltage/GEI/power all removed)",
                                        strict_precursor_cols, train_pool, test_pool_seen, holdout_pool)

    # --------------------------------------------------------------------
    # Three-way comparison
    # --------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON, BUS F HOLDOUT (the result that matters)")
    print("=" * 70)
    hf, hl, hs = results_full["holdout"], results_light["holdout"], results_strict["holdout"]
    print(f"{'Metric':12s} {'Full':>10s} {'LightPrecursor':>15s} {'StrictPrecursor':>16s}")
    for metric in ["recall", "precision", "f1", "auc"]:
        vf = hf.get(metric) if hf else None
        vl = hl.get(metric) if hl else None
        vs = hs.get(metric) if hs else None
        def f(v): return f"{v:.4f}" if v is not None else "n/a"
        print(f"{metric:12s} {f(vf):>10s} {f(vl):>15s} {f(vs):>16s}")

    out = {
        "holdout_bus": HOLDOUT_BUS,
        "direct_signal_keys_removed": DIRECT_SIGNAL_KEYS,
        "strict_additional_keys_removed": STRICT_ADDITIONAL_KEYS,
        "n_full_features": len(full_cols),
        "n_light_precursor_features": len(light_precursor_cols),
        "n_strict_precursor_features": len(strict_precursor_cols),
        "results_full": results_full,
        "results_light_precursor": results_light,
        "results_strict_precursor": results_strict,
    }
    out_path = OUTPUT_DIR / "precursor_only_comparison.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_path}")

    print("\n===== INTERPRETATION =====")
    if hf and hl and hs and all(r and r.get("recall") is not None for r in [hf, hl, hs]):
        if abs(hf["recall"] - hl["recall"]) < 1e-6:
            print("Full and light-precursor recall were IDENTICAL (or nearly so). This is the "
                  "signature of another same-instant electrical symptom (most likely "
                  "own_Source_Power, which crashes in lockstep with voltage) still present in the "
                  "light-precursor set, acting as an equally perfect proxy, not evidence of a "
                  "genuine precursor pattern.")
        strict_drop = hf["recall"] - hs["recall"]
        if hs["recall"] > 0.6 and strict_drop < 0.3:
            print(f"\nStrict-precursor recall ({hs['recall']:.3f}), with ALL same-instant "
                  f"electrical signals removed (voltage, GEI, power), stayed reasonably close to "
                  f"the full result ({hf['recall']:.3f}). THIS is real evidence of a genuine, "
                  f"thermal/load/neighbor-based precursor pattern, worth pursuing as the basis "
                  f"for an actual early-warning detector.")
        else:
            print(f"\nStrict-precursor recall ({hs['recall']:.3f}) dropped sharply from the full "
                  f"result ({hf['recall']:.3f}). With every same-instant electrical symptom "
                  f"removed, the model could not reliably detect F's collapse in advance. This "
                  f"means the earlier perfect scores were detecting the collapse AS it happened "
                  f"electrically, not predicting it from thermal or load precursors. Genuine early "
                  f"warning likely requires shifting the label forward in time and training "
                  f"explicitly for that, rather than assuming this same-instant model already "
                  f"does it.")
    print("\nDone.")


if __name__ == "__main__":
    main()

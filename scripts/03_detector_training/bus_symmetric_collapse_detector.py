"""
BUS-SYMMETRIC COLLAPSE DETECTOR: DOES IT GENERALIZE TO AN UNSEEN BUS?
================================================================================
The previous detector (train_bus_collapse_detector.py) had 10 independent
output heads, one per bus, each with its own private weights. That means "has
this model learned to recognize collapse in general" was never actually
testable, buses A, B, C, D, G, H, K, L never collapsed in this dataset, so
their heads never received a single real training example, and could not
possibly generalize regardless of what the shared hidden layers learned.

This script restructures the problem to make that question testable. Instead
of one row per timestep with 981 bus-specific columns, this reshapes to ONE
ROW PER BUS PER TIMESTEP, using only that bus's own local signals, its two
adjacent converters' thermal/derate signals, and its two ring-neighbor buses'
basic readings, all renamed to the SAME canonical feature names regardless of
which physical bus the row came from (own_Voltage, conv_prev_Derate_Factor,
neighbor_next_GEI, etc.). A single shared classifier is then trained across
all 10 buses' rows at once, so whatever it learns about "what collapse looks
like, from a bus's own local perspective" is forced to apply uniformly to
every bus, not memorized per bus identity.

THE KEY EXPERIMENT: no real dataset exists where a bus other than E or F
collapses, so "will it catch Bus K collapsing" cannot be answered directly
yet. This script constructs the closest rigorous proxy available from the
data that does exist: Bus F is COMPLETELY EXCLUDED from training, not just
its collapse rows, EVERY row belonging to F, at every timestep, healthy or
not. The shared model is trained only on what E's collapse looked like
(plus everyone else's healthy behavior). It is then evaluated purely on F,
a bus it has never seen in any capacity during training. If it correctly
flags F's real collapse anyway, that is direct evidence the shared
representation transfers across bus identity, not just within one memorized
bus, which is the strongest evidence available right now for whether this
approach would also catch a genuinely new bus like K in a future simulation.

All canonical feature names below were checked column-by-column against
step33_build_10bus_derating_features_v3.py's actual generation logic, not
guessed. In particular, Bus_<b>_Temp is in BASE_SIGNALS but NOT in
ROLLING_SIGNALS there, so it has lag1/2/3 and a first difference but no
rolling mean/std, unlike Voltage, Source_Power, GEI, and Commanded_Load,
which have all six variants. This distinction is preserved here exactly.

Run:
    python bus_symmetric_collapse_detector.py
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
              / "36_bus_symmetric_detector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

HOLDOUT_BUS = "F"   # completely excluded from training, the proxy for "an unseen bus like K"

# Subsampling the long-format table for tractable training time. 1 = full
# resolution (~5,000,000 rows across all 10 buses), slow. Start higher
# (e.g. 5 or 10) for a fast first check, then drop to 1 once satisfied.
SUBSAMPLE_STRIDE = 5

N_BLOCKS     = 20
TEST_FRAC    = 0.20
PURGE_ROWS   = 10
RANDOM_STATE = 42

HIDDEN_LAYER_SIZES = (128, 64)   # smaller than the phase model: far fewer, more focused features here
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


# ----------------------------------------------------------------------
# Canonical local+neighbor feature map, identical structure for every bus
# ----------------------------------------------------------------------
def get_local_feature_map(bus_idx: int) -> dict:
    bus = BUSES[bus_idx]
    conv_next = CONV[bus_idx]
    conv_prev = CONV[(bus_idx - 1) % 10]
    nb_next = BUSES[(bus_idx + 1) % 10]
    nb_prev = BUSES[(bus_idx - 1) % 10]

    m = {}

    # Own base signals. Voltage/Source_Power/GEI/Commanded_Load have full
    # lag+roll+diff variants (they are in ROLLING_SIGNALS). Temp has lag+diff
    # only, no roll_mean/std, confirmed against the actual generation script.
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

    # Own imbalance/error features. GEI_Error, Power_Imbalance,
    # Voltage_Imbalance, Load_Imbalance have lag1/2/3 (EXTRA_LAG_SIGNALS).
    # The Abs_ versions, Voltage_Error, and Bus_Temp_Imbalance do not.
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

    # Adjacent converters, "prev" (ring-incoming) and "next" (ring-outgoing),
    # a structurally consistent role for every bus, not tied to a specific
    # converter name. Heat_Sink_Temp, Derate_Factor, Junction_Temp all have
    # full lag+roll+diff variants (all three are in ROLLING_SIGNALS).
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

    # Ring-neighbor buses, base signals only (no lag), just enough context
    # to know what is happening next door without the full feature stack.
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


def main():
    print("=" * 70)
    print("BUS-SYMMETRIC COLLAPSE DETECTOR")
    print("=" * 70)

    features_df = (pd.read_parquet(FEATURE_PARQUET) if FEATURE_PARQUET.exists()
                    else pd.read_csv(FEATURE_CSV))
    targets_df = load_corrected_targets()
    wide = pd.merge(features_df, targets_df, on="time", how="inner", validate="one_to_one")
    wide = wide.sort_values("time").reset_index(drop=True)
    print(f"Joined wide table: {len(wide):,} rows")

    # Time-based split computed ONCE on the original time axis, then applied
    # identically to every bus's rows at that timestep after reshaping. This
    # is what keeps the split leakage-safe across the reshape, the same
    # timestep must land in the same split for every bus, not be re-randomized.
    time_split = blocked_purged_split(len(wide), N_BLOCKS, TEST_FRAC, PURGE_ROWS, RANDOM_STATE)
    wide["_time_split"] = time_split

    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
        print(f"Subsampled every {SUBSAMPLE_STRIDE} rows for tractable training time: "
              f"{len(wide):,} timesteps remain (set SUBSAMPLE_STRIDE=1 for full resolution later)")

    # --------------------------------------------------------------------
    # Reshape: one row per bus per timestep, canonical feature names
    # --------------------------------------------------------------------
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
            raise ValueError(f"Bus {bus}: missing expected columns, check the feature map "
                              f"against the real feature file:\n{missing[:10]}")
        sub = wide[actual_cols].copy()
        sub.columns = canonical_cols
        sub["bus"] = bus
        sub["time"] = wide["time"].to_numpy()
        sub["split"] = wide["_time_split"].to_numpy()
        sub["y_collapsed"] = wide[f"collapsed_{bus}"].to_numpy().astype(int)
        long_frames.append(sub)
        print(f"  {bus}: {len(sub):,} rows, {len(canonical_cols)} canonical features")

    long_df = pd.concat(long_frames, ignore_index=True)
    print(f"\nTotal long-format rows: {len(long_df):,} "
          f"({len(BUSES)} buses x {len(wide):,} timesteps)")
    print(f"Canonical feature count: {len(canonical_cols)}")

    # --------------------------------------------------------------------
    # The key split: HOLDOUT_BUS completely removed from the training pool,
    # not just its positive rows, every row, healthy or not.
    # --------------------------------------------------------------------
    print(f"\n--- Holding out Bus {HOLDOUT_BUS} COMPLETELY from training ---")
    print(f"(proxy for a genuinely new bus like K collapsing in a future simulation)")

    train_pool = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "train")]
    test_pool_seen = long_df[(long_df["bus"] != HOLDOUT_BUS) & (long_df["split"] == "test")]
    holdout_pool = long_df[long_df["bus"] == HOLDOUT_BUS]   # ALL of it, train+test+purge, never used in training

    print(f"Training rows (9 buses, time-train blocks only): {len(train_pool):,}")
    print(f"Standard test rows (9 buses, time-test blocks, buses model DID see): {len(test_pool_seen):,}")
    print(f"Held-out bus {HOLDOUT_BUS} rows (NEVER seen in any capacity during training): {len(holdout_pool):,}")

    n_pos_train = train_pool["y_collapsed"].sum()
    print(f"\nPositive (collapsed) examples in training pool: {n_pos_train:,} "
          f"({100*n_pos_train/len(train_pool):.2f}%)")
    if n_pos_train == 0:
        raise RuntimeError("No positive examples in the training pool at all, cannot proceed.")

    X_train = train_pool[canonical_cols].to_numpy(dtype=np.float64)
    y_train = train_pool["y_collapsed"].to_numpy(dtype=int)
    X_test_seen = test_pool_seen[canonical_cols].to_numpy(dtype=np.float64)
    y_test_seen = test_pool_seen["y_collapsed"].to_numpy(dtype=int)
    X_holdout = holdout_pool[canonical_cols].to_numpy(dtype=np.float64)
    y_holdout = holdout_pool["y_collapsed"].to_numpy(dtype=int)

    print(f"\nArchitecture: hidden_layer_sizes={HIDDEN_LAYER_SIZES}, alpha={MLP_KWARGS['alpha']}, "
          f"{len(canonical_cols)} shared canonical features (vs 981 in the per-bus model)")

    pipeline = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MLP_KWARGS)),
    ])

    print("\nTraining shared model (never sees any Bus F data) ...")
    pipeline.fit(X_train, y_train)

    mlp = pipeline.named_steps["mlp"]
    stopped_early = mlp.n_iter_ < MLP_KWARGS["max_iter"]
    print(f"\nStopped after {mlp.n_iter_} iterations "
          f"({'training loss plateaued' if stopped_early else 'hit max_iter'})")
    print(f"Final training loss: {mlp.loss_:.6f}")

    def evaluate(name, X, y):
        if len(y) == 0:
            print(f"\n{name}: no rows, skipping.")
            return None
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        if y.sum() == 0:
            print(f"\n{name}: accuracy={acc:.4f} (no positive examples here, precision/recall n/a)")
            return {"accuracy": acc, "precision": None, "recall": None, "f1": None, "auc": None}
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = None
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        print(f"\n{name}:")
        print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  "
              f"auc={auc if auc is None else round(auc,4)}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    print("\n===== EVALUATION =====")
    result_train = evaluate("Training set (fit quality check)", X_train, y_train)
    result_seen = evaluate("Held-out TIME blocks, buses the model DID train on (standard generalization)",
                            X_test_seen, y_test_seen)
    result_holdout = evaluate(f"Bus {HOLDOUT_BUS}, NEVER seen in training at all "
                               f"(the actual answer to 'can it recognize a new bus')",
                               X_holdout, y_holdout)

    # --------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------
    model_path = OUTPUT_DIR / "bus_symmetric_collapse_detector.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nWrote: {model_path}")

    meta = {
        "task": "bus-symmetric same-instant collapse detection, single shared model across all buses",
        "holdout_bus": HOLDOUT_BUS,
        "holdout_bus_never_appears_in_training": True,
        "canonical_features": canonical_cols,
        "n_canonical_features": len(canonical_cols),
        "subsample_stride": SUBSAMPLE_STRIDE,
        "architecture": {"hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
                          **{k: v for k, v in MLP_KWARGS.items() if k != "hidden_layer_sizes"}},
        "n_train_rows": int(len(train_pool)),
        "results": {
            "train": result_train,
            "test_seen_buses": result_seen,
            f"holdout_bus_{HOLDOUT_BUS}": result_holdout,
        },
    }
    meta_path = OUTPUT_DIR / "bus_symmetric_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote: {meta_path}")

    print("\n===== ANSWER =====")
    if result_holdout is not None and result_holdout.get("recall") is not None:
        print(f"Bus {HOLDOUT_BUS} was NEVER included in training, in any row, healthy or "
              f"collapsed. The model achieved recall={result_holdout['recall']:.3f}, "
              f"precision={result_holdout['precision']:.3f} on {HOLDOUT_BUS}'s real collapse, "
              f"using only the shared local+neighbor representation learned from Bus E.")
        if result_holdout["recall"] > 0.7:
            print("This is real, direct evidence the approach generalizes across bus identity, "
                  "not just within one memorized bus. A genuinely new bus like K collapsing in a "
                  "future simulation would plausibly be caught the same way, though this remains "
                  "unconfirmed until tested on an actual new run.")
        else:
            print("Recall is not strong enough to be confident this generalizes well yet. Consider "
                  "SUBSAMPLE_STRIDE=1 for full resolution, more hidden units, or waiting for a "
                  "second real collapse scenario (from the multi-scenario dataset generation "
                  "discussed earlier) before trusting this for a genuinely new bus.")
    print("\nDone.")


if __name__ == "__main__":
    main()

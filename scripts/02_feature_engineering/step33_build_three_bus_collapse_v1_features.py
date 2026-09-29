"""
STEP 33 (three_bus_collapse_v1 variant): Build the 10-bus feature dataset for
the NEW three_bus_collapse scenario (formerly "corridor_triad_collapse"),
using the EXACT SAME recipe as step33_build_10bus_derating_features_v3.py
(the script that built the features the 5 baseline early-warning detectors
were trained on) -- identical to the gradual_busC_v4 variant of this script,
just pointed at a different raw CSV.

WHY THIS EXISTS: to test the baseline-trained N=5/15/30/60/90s detectors
against the brand-new three_bus_collapse scenario (Bus G collapses at
t=1081.96s, Bus H at t=1082.05s, Bus K at t=1108.70s -- a genuine regional
collapse, all three within 26.74s of each other), the new scenario's raw
simulation CSV must be turned into features with IDENTICAL column names,
IDENTICAL physics/temporal feature construction, and IDENTICAL feature
ordering as the baseline features -- otherwise the pretrained sklearn
Pipelines (StandardScaler + MLPClassifier) will silently score garbage,
because they index into the feature vector positionally.

Nothing about the feature RECIPE changes here vs. step33_v3 or the
gradual_busC_v4 variant. Only the input/output paths differ.

Run (on the Windows machine, inside the venv):
    python step33_build_three_bus_collapse_v1_features.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_three_bus_collapse"
             / "thermal_derating_three_bus_collapse_v1_5000s.csv")
OUTPUT_DIR = (PROJECT_ROOT / "data_d" / "three_bus_collapse_v1_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLTAGE_REFERENCE = 800.0
GEI_REFERENCE = 1.0
TJ_DERATE_ONSET = 125.0
TJ_SHUTDOWN = 175.0
LAG_STEPS = (1, 2, 3)
ROLL_WINDOW = 5
ROLLING_STD_DDOF = 1

# three_bus_collapse has an 800s preheat + 600s surge lead-in before the first
# collapse (t=1081.96s), similar in shape to gradual_busC_v4's lead-in. Keep
# the same small warm-up tolerance as a sanity net, not a relaxed one.
EXPECTED_MAX_WARMUP_ROWS = 20

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]
RING_BUS_PAIRS  = [(BUSES[i], BUSES[(i+1) % 10]) for i in range(10)]
RING_CONV_PAIRS = [(CONV[i],  CONV[(i+1) % 10])  for i in range(10)]

BASE_SIGNALS = (
    [f"Bus_{b}_GEI" for b in BUSES]
    + [f"Bus_{b}_Voltage" for b in BUSES]
    + [f"Bus_{b}_Source_Power" for b in BUSES]
    + [f"Bus_{b}_Temp" for b in BUSES]
    + [f"DAB_{c}_Heat_Sink_Temp" for c in CONV]
    + [f"Bus_{b}_Commanded_Load" for b in BUSES]
    + [f"DAB_{c}_Derate_Factor" for c in CONV]
    + [f"DAB_{c}_Junction_Temp" for c in CONV]
)
TARGET_COLUMNS = [f"Phase_{c}" for c in CONV]

ROLLING_SIGNALS = (
    [f"Bus_{b}_GEI" for b in BUSES]
    + [f"Bus_{b}_Source_Power" for b in BUSES]
    + [f"Bus_{b}_Voltage" for b in BUSES]
    + [f"DAB_{c}_Heat_Sink_Temp" for c in CONV]
    + [f"Bus_{b}_Commanded_Load" for b in BUSES]
    + [f"DAB_{c}_Derate_Factor" for c in CONV]
    + [f"DAB_{c}_Junction_Temp" for c in CONV]
)
EXTRA_LAG_SIGNALS = (
    [f"GEI_Error_{b}" for b in BUSES]
    + [f"Power_Imbalance_{b}" for b in BUSES]
    + [f"Voltage_Imbalance_{b}" for b in BUSES]
    + [f"Load_Imbalance_{b}" for b in BUSES]
)

def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for b in BUSES:
        m[f"GEI_{b}"]              = f"Bus_{b}_GEI"
        m[f"V_Bus_{b}"]            = f"Bus_{b}_Voltage"
        m[f"Bus_{b}_Src_Pow"]      = f"Bus_{b}_Source_Power"
        m[f"CommandedLoad_kW_{b}"] = f"Bus_{b}_Commanded_Load"
    for c in CONV:
        m[f"HeatSinkTemp_C_{c}"]      = f"DAB_{c}_Heat_Sink_Temp"
        m[f"JunctionTemp_C_{c}"]      = f"DAB_{c}_Junction_Temp"
        m[f"Phase_{c}_cmd_deg"]       = f"Phase_{c}"
    return df.rename(columns=m)

def add_physics(raw: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=raw.index)
    gei  = [f"Bus_{b}_GEI" for b in BUSES]
    vol  = [f"Bus_{b}_Voltage" for b in BUSES]
    pw   = [f"Bus_{b}_Source_Power" for b in BUSES]
    bt   = [f"Bus_{b}_Temp" for b in BUSES]
    hs   = [f"DAB_{c}_Heat_Sink_Temp" for c in CONV]
    load = [f"Bus_{b}_Commanded_Load" for b in BUSES]
    der  = [f"DAB_{c}_Derate_Factor" for c in CONV]
    jt   = [f"DAB_{c}_Junction_Temp" for c in CONV]

    for col in BASE_SIGNALS:
        f[col] = raw[col].astype(np.float64)

    f["Avg_GEI"] = raw[gei].mean(axis=1)
    f["Avg_Bus_Voltage"] = raw[vol].mean(axis=1)
    f["Avg_Source_Power"] = raw[pw].mean(axis=1)
    f["Total_Source_Power"] = raw[pw].sum(axis=1)
    f["Avg_Bus_Temp"] = raw[bt].mean(axis=1)
    f["Avg_DAB_Heat_Sink_Temp"] = raw[hs].mean(axis=1)
    f["Avg_Commanded_Load"] = raw[load].mean(axis=1)
    f["Total_Commanded_Load"] = raw[load].sum(axis=1)
    f["Avg_Derate_Factor"] = raw[der].mean(axis=1)
    f["Min_Derate_Factor"] = raw[der].min(axis=1)
    f["Total_Derate_Deficit"] = (1.0 - raw[der]).sum(axis=1)
    f["Avg_DAB_Junction_Temp"] = raw[jt].mean(axis=1)
    f["Max_Junction_Temp"] = raw[jt].max(axis=1)
    f["N_Converters_Over_125"] = (raw[jt] > TJ_DERATE_ONSET).sum(axis=1).astype(np.float64)
    f["N_Converters_Over_175"] = (raw[jt] > TJ_SHUTDOWN).sum(axis=1).astype(np.float64)

    for b in BUSES:
        f[f"GEI_Error_{b}"] = GEI_REFERENCE - raw[f"Bus_{b}_GEI"]
        f[f"Abs_GEI_Error_{b}"] = f[f"GEI_Error_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"GEI_{x}{y}_Diff"] = raw[f"Bus_{x}_GEI"] - raw[f"Bus_{y}_GEI"]
        f[f"Abs_GEI_{x}{y}_Diff"] = f[f"GEI_{x}{y}_Diff"].abs()
    f["Max_GEI"] = raw[gei].max(axis=1)
    f["Min_GEI"] = raw[gei].min(axis=1)
    f["GEI_Range"] = f["Max_GEI"] - f["Min_GEI"]
    f["Total_Abs_GEI_Error"] = sum(f[f"Abs_GEI_Error_{b}"] for b in BUSES)

    for b in BUSES:
        f[f"Voltage_Error_{b}"] = VOLTAGE_REFERENCE - raw[f"Bus_{b}_Voltage"]
        f[f"Abs_Voltage_Error_{b}"] = f[f"Voltage_Error_{b}"].abs()
        f[f"Voltage_Imbalance_{b}"] = raw[f"Bus_{b}_Voltage"] - f["Avg_Bus_Voltage"]
        f[f"Abs_Voltage_Imbalance_{b}"] = f[f"Voltage_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"V_{x}{y}_Diff"] = raw[f"Bus_{x}_Voltage"] - raw[f"Bus_{y}_Voltage"]
        f[f"Abs_V_{x}{y}_Diff"] = f[f"V_{x}{y}_Diff"].abs()

    for b in BUSES:
        f[f"Power_Imbalance_{b}"] = raw[f"Bus_{b}_Source_Power"] - f["Avg_Source_Power"]
        f[f"Abs_Power_Imbalance_{b}"] = f[f"Power_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"P_{x}{y}_Diff"] = raw[f"Bus_{x}_Source_Power"] - raw[f"Bus_{y}_Source_Power"]
        f[f"Abs_P_{x}{y}_Diff"] = f[f"P_{x}{y}_Diff"].abs()
    f["Total_Abs_Power_Imbalance"] = sum(f[f"Abs_Power_Imbalance_{b}"] for b in BUSES)

    for b in BUSES:
        f[f"Bus_{b}_Temp_Imbalance"] = raw[f"Bus_{b}_Temp"] - f["Avg_Bus_Temp"]
        f[f"Abs_Bus_{b}_Temp_Imbalance"] = f[f"Bus_{b}_Temp_Imbalance"].abs()

    for c in CONV:
        f[f"DAB_{c}_Temp_Imbalance"] = raw[f"DAB_{c}_Heat_Sink_Temp"] - f["Avg_DAB_Heat_Sink_Temp"]
        f[f"Abs_DAB_{c}_Temp_Imbalance"] = f[f"DAB_{c}_Temp_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Temp_Diff"] = raw[f"DAB_{x}_Heat_Sink_Temp"] - raw[f"DAB_{y}_Heat_Sink_Temp"]

    for b in BUSES:
        f[f"Load_Imbalance_{b}"] = raw[f"Bus_{b}_Commanded_Load"] - f["Avg_Commanded_Load"]
        f[f"Abs_Load_Imbalance_{b}"] = f[f"Load_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"Load_{x}{y}_Diff"] = raw[f"Bus_{x}_Commanded_Load"] - raw[f"Bus_{y}_Commanded_Load"]
        f[f"Abs_Load_{x}{y}_Diff"] = f[f"Load_{x}{y}_Diff"].abs()
    f["Total_Abs_Load_Imbalance"] = sum(f[f"Abs_Load_Imbalance_{b}"] for b in BUSES)

    for c in CONV:
        f[f"DAB_{c}_Derate_Imbalance"] = raw[f"DAB_{c}_Derate_Factor"] - f["Avg_Derate_Factor"]
        f[f"Abs_DAB_{c}_Derate_Imbalance"] = f[f"DAB_{c}_Derate_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Derate_Diff"] = raw[f"DAB_{x}_Derate_Factor"] - raw[f"DAB_{y}_Derate_Factor"]

    for c in CONV:
        f[f"DAB_{c}_Junction_Imbalance"] = raw[f"DAB_{c}_Junction_Temp"] - f["Avg_DAB_Junction_Temp"]
        f[f"Abs_DAB_{c}_Junction_Imbalance"] = f[f"DAB_{c}_Junction_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Junction_Diff"] = raw[f"DAB_{x}_Junction_Temp"] - raw[f"DAB_{y}_Junction_Temp"]

    for c in CONV:
        f[f"DAB_{c}_Junction_HeatSink_Diff"] = raw[f"DAB_{c}_Junction_Temp"] - raw[f"DAB_{c}_Heat_Sink_Temp"]

    return f

def add_temporal(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    for col in BASE_SIGNALS:
        for lag in LAG_STEPS:
            f[f"{col}_lag{lag}"] = f[col].shift(lag)
    for col in EXTRA_LAG_SIGNALS:
        for lag in LAG_STEPS:
            f[f"{col}_lag{lag}"] = f[col].shift(lag)
    for col in ROLLING_SIGNALS:
        r = f[col].rolling(window=ROLL_WINDOW, min_periods=ROLL_WINDOW)
        f[f"{col}_roll_mean_5"] = r.mean()
        f[f"{col}_roll_std_5"] = r.std(ddof=ROLLING_STD_DDOF)
    for col in BASE_SIGNALS:
        f[f"d_{col}"] = f[col].diff()
    return f

def main():
    print("=" * 70)
    print("STEP 33 (three_bus_collapse_v1 variant): build features for the new scenario")
    print("=" * 70)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing three_bus_collapse_v1 raw output:\n{INPUT_CSV}")
    raw = pd.read_csv(INPUT_CSV)
    raw = rename_to_canonical(raw)
    print(f"Loaded {len(raw):,} rows.")

    missing = [c for c in BASE_SIGNALS + TARGET_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename:\n{missing}")

    physics = add_physics(raw)
    feats = add_temporal(physics)
    feature_order = list(feats.columns)

    phase_leaks = [c for c in feature_order if "Phase" in c]
    if phase_leaks:
        raise RuntimeError(f"Phase found in the input feature set:\n{phase_leaks}")
    print("Confirmed: no Phase_* column present in the input feature set.")
    print(f"Feature count: {len(feature_order)}")

    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = feats.notna().all(axis=1)
    n_dropped = int((~valid).sum())
    print(f"Warm-up/incomplete-history rows dropped: {n_dropped}")
    if n_dropped > EXPECTED_MAX_WARMUP_ROWS:
        raise RuntimeError(
            f"Dropped {n_dropped} rows, expected at most {EXPECTED_MAX_WARMUP_ROWS}. "
            f"This means the three_bus_collapse_v1 raw CSV has a NaN/Inf block somewhere "
            f"beyond ordinary warm-up (e.g. the Divide-by-zero warning seen during the v1 "
            f"sim actually corrupted a GEI column instead of falling back cleanly). "
            f"Investigate before trusting this feature set -- do not silently proceed."
        )
    print(f"Row drop within expected warm-up range (<= {EXPECTED_MAX_WARMUP_ROWS}).")

    X = feats.loc[valid, feature_order].reset_index(drop=True)
    y = raw.loc[valid, TARGET_COLUMNS].reset_index(drop=True)
    y = y.rename(columns={f"Phase_{c}": f"target_Phase_{c}" for c in CONV})
    tcol = raw.loc[valid, "time"].reset_index(drop=True)

    out = pd.concat([tcol.rename("time"), X, y], axis=1)
    out_parquet = OUTPUT_DIR / "three_bus_collapse_v1_5000s_features.parquet"
    out_csv = OUTPUT_DIR / "three_bus_collapse_v1_5000s_features.csv"
    try:
        out.to_parquet(out_parquet, index=False, compression="zstd")
        saved = out_parquet
    except Exception as e:
        print(f"(parquet failed: {e}; writing csv)")
        out.to_csv(out_csv, index=False); saved = out_csv

    schema = {
        "input_csv": str(INPUT_CSV),
        "n_features": len(feature_order),
        "features": feature_order,
        "targets": [f"target_Phase_{c}" for c in CONV],
        "buses": BUSES, "converters": CONV,
        "rows_dropped_as_warmup": n_dropped,
        "note": "Feature recipe identical to step33_build_10bus_derating_features_v3.py, "
                "applied to the new three_bus_collapse_v1 regional collapse scenario "
                "(Bus G collapses at t=1081.96s, Bus H at t=1082.05s, Bus K at t=1108.70s "
                "per the MATLAB run's own health check + co-collapse window verdict).",
    }
    (OUTPUT_DIR / "feature_schema_three_bus_collapse_v1.json").write_text(json.dumps(schema, indent=2))

    print(f"\nSaved features: {saved}  ({len(out):,} rows x {out.shape[1]} cols)")
    print(f"Saved schema:   {OUTPUT_DIR / 'feature_schema_three_bus_collapse_v1.json'}")

if __name__ == "__main__":
    main()
"""
STEP 33 (corridor_triad_collapse): Build the 10-bus THERMAL DERATING
feature dataset for the new corridor_triad_collapse scenario.
================================================================================
Same recipe as step33_build_10bus_derating_features_v3.py (80 base signals
including junction temperature, grid-wide averages/imbalances, ring-adjacent
diffs, lags, rolling stats, first differences, the Tj-Th physics feature).
Only the input file and output naming differ - the feature engineering
logic itself is byte-for-byte identical to v3.

Unlike the original baseline dataset, this scenario's GEI was computed
correctly INLINE during generation (generate_corridor_triad_collapse.m
implements the same confirmed fix_GEI_v7_ALfix.py formula directly), so
there is no separate GEI-fix step to run first and no equivalent of the
~86,000-row NaN blowup the baseline hit. The NaN/Inf validation already
run inside generate_corridor_triad_collapse.m confirmed zero NaN/Inf in
this CSV, so EXPECTED_MAX_WARMUP_ROWS below should only ever catch the
handful of ordinary lag/rolling warm-up rows - if it trips, something
is wrong with this run, not a known/expected issue.

PHASE IS NOT AN INPUT FEATURE, same as v3, confirmed with the same runtime
assertion, not assumed.

Run:
    python step33_build_corridor_triad_collapse_features.py
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
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
             / "corridor_triad_collapse" / "scenario_corridor_triad_collapse_5000s.csv")
OUTPUT_DIR = (PROJECT_ROOT / "data_d" / "unified_10bus_derating_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLTAGE_REFERENCE = 800.0
GEI_REFERENCE = 1.0
TJ_DERATE_ONSET = 125.0     # matches the model's own thermal_derater threshold
TJ_SHUTDOWN = 175.0         # matches the model's own thermal_derater threshold
LAG_STEPS = (1, 2, 3)
ROLL_WINDOW = 5
ROLLING_STD_DDOF = 1

# Warm-up rows are expected to be small (a handful, from lag/rolling window
# construction only). If the row drop after feature construction is much
# larger than this, something is wrong with the input data, not just
# ordinary warm-up, and this script should stop rather than proceed.
EXPECTED_MAX_WARMUP_ROWS = 20

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]         # 10 buses
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]        # 10 converters (ring)
RING_BUS_PAIRS  = [(BUSES[i], BUSES[(i+1) % 10]) for i in range(10)]
RING_CONV_PAIRS = [(CONV[i],  CONV[(i+1) % 10])  for i in range(10)]

# canonical base-signal names, 80 total: 50 original + 10 commanded load +
# 10 derate factor + 10 junction temperature.
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
TARGET_COLUMNS = [f"Phase_{c}" for c in CONV]     # all 10, AL included, NOT in BASE_SIGNALS

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

# CSV column -> canonical name mapping.
def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for b in BUSES:
        m[f"GEI_{b}"]              = f"Bus_{b}_GEI"
        m[f"V_Bus_{b}"]            = f"Bus_{b}_Voltage"
        m[f"Bus_{b}_Src_Pow"]      = f"Bus_{b}_Source_Power"
        m[f"CommandedLoad_kW_{b}"] = f"Bus_{b}_Commanded_Load"
        # Bus_{b}_Temp already canonical
    for c in CONV:
        m[f"HeatSinkTemp_C_{c}"]      = f"DAB_{c}_Heat_Sink_Temp"
        m[f"JunctionTemp_C_{c}"]      = f"DAB_{c}_Junction_Temp"
        m[f"Phase_{c}_cmd_deg"]       = f"Phase_{c}"
        # DAB_{c}_Derate_Factor already canonical
    return df.rename(columns=m)

# ----------------------------------------------------------------------
# Physics features
# ----------------------------------------------------------------------
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

    # A. base signals
    for col in BASE_SIGNALS:
        f[col] = raw[col].astype(np.float64)

    # B. grid-wide averages / totals
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

    # C. GEI errors, ring-adjacent diffs, extremes
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

    # D. Voltage errors, imbalance, ring-adjacent diffs
    for b in BUSES:
        f[f"Voltage_Error_{b}"] = VOLTAGE_REFERENCE - raw[f"Bus_{b}_Voltage"]
        f[f"Abs_Voltage_Error_{b}"] = f[f"Voltage_Error_{b}"].abs()
        f[f"Voltage_Imbalance_{b}"] = raw[f"Bus_{b}_Voltage"] - f["Avg_Bus_Voltage"]
        f[f"Abs_Voltage_Imbalance_{b}"] = f[f"Voltage_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"V_{x}{y}_Diff"] = raw[f"Bus_{x}_Voltage"] - raw[f"Bus_{y}_Voltage"]
        f[f"Abs_V_{x}{y}_Diff"] = f[f"V_{x}{y}_Diff"].abs()

    # E. Power imbalance, ring-adjacent diffs
    for b in BUSES:
        f[f"Power_Imbalance_{b}"] = raw[f"Bus_{b}_Source_Power"] - f["Avg_Source_Power"]
        f[f"Abs_Power_Imbalance_{b}"] = f[f"Power_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"P_{x}{y}_Diff"] = raw[f"Bus_{x}_Source_Power"] - raw[f"Bus_{y}_Source_Power"]
        f[f"Abs_P_{x}{y}_Diff"] = f[f"P_{x}{y}_Diff"].abs()
    f["Total_Abs_Power_Imbalance"] = sum(f[f"Abs_Power_Imbalance_{b}"] for b in BUSES)

    # F. Bus-temperature imbalance
    for b in BUSES:
        f[f"Bus_{b}_Temp_Imbalance"] = raw[f"Bus_{b}_Temp"] - f["Avg_Bus_Temp"]
        f[f"Abs_Bus_{b}_Temp_Imbalance"] = f[f"Bus_{b}_Temp_Imbalance"].abs()

    # G. DAB heat-sink imbalance + adjacent-converter diffs
    for c in CONV:
        f[f"DAB_{c}_Temp_Imbalance"] = raw[f"DAB_{c}_Heat_Sink_Temp"] - f["Avg_DAB_Heat_Sink_Temp"]
        f[f"Abs_DAB_{c}_Temp_Imbalance"] = f[f"DAB_{c}_Temp_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Temp_Diff"] = raw[f"DAB_{x}_Heat_Sink_Temp"] - raw[f"DAB_{y}_Heat_Sink_Temp"]

    # H. Commanded load imbalance + ring-adjacent bus diffs (mirrors E)
    for b in BUSES:
        f[f"Load_Imbalance_{b}"] = raw[f"Bus_{b}_Commanded_Load"] - f["Avg_Commanded_Load"]
        f[f"Abs_Load_Imbalance_{b}"] = f[f"Load_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"Load_{x}{y}_Diff"] = raw[f"Bus_{x}_Commanded_Load"] - raw[f"Bus_{y}_Commanded_Load"]
        f[f"Abs_Load_{x}{y}_Diff"] = f[f"Load_{x}{y}_Diff"].abs()
    f["Total_Abs_Load_Imbalance"] = sum(f[f"Abs_Load_Imbalance_{b}"] for b in BUSES)

    # I. Derate factor imbalance + ring-adjacent converter diffs (mirrors G)
    for c in CONV:
        f[f"DAB_{c}_Derate_Imbalance"] = raw[f"DAB_{c}_Derate_Factor"] - f["Avg_Derate_Factor"]
        f[f"Abs_DAB_{c}_Derate_Imbalance"] = f[f"DAB_{c}_Derate_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Derate_Diff"] = raw[f"DAB_{x}_Derate_Factor"] - raw[f"DAB_{y}_Derate_Factor"]

    # J. Junction temperature imbalance + ring-adjacent converter diffs (mirrors G)
    for c in CONV:
        f[f"DAB_{c}_Junction_Imbalance"] = raw[f"DAB_{c}_Junction_Temp"] - f["Avg_DAB_Junction_Temp"]
        f[f"Abs_DAB_{c}_Junction_Imbalance"] = f[f"DAB_{c}_Junction_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Junction_Diff"] = raw[f"DAB_{x}_Junction_Temp"] - raw[f"DAB_{y}_Junction_Temp"]

    # K. Junction-to-heatsink differential. Proportional to instantaneous
    # heat flow through R_Jh in the model.
    for c in CONV:
        f[f"DAB_{c}_Junction_HeatSink_Diff"] = raw[f"DAB_{c}_Junction_Temp"] - raw[f"DAB_{c}_Heat_Sink_Temp"]

    return f

# ----------------------------------------------------------------------
# Temporal features
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 33 (corridor_triad_collapse): build 10-bus THERMAL DERATING features")
    print("=" * 70)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Run generate_corridor_triad_collapse.m first (PILOT_MODE=false). Missing:\n{INPUT_CSV}"
        )
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
        raise RuntimeError(
            f"Phase found in the input feature set, this must never happen:\n{phase_leaks}"
        )
    print("Confirmed: no Phase_* column present in the input feature set.")

    print(f"Feature count: {len(feature_order)}")

    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = feats.notna().all(axis=1)
    n_dropped = int((~valid).sum())
    print(f"Warm-up/incomplete-history rows dropped: {n_dropped}")

    if n_dropped > EXPECTED_MAX_WARMUP_ROWS:
        raise RuntimeError(
            f"Dropped {n_dropped} rows, expected at most {EXPECTED_MAX_WARMUP_ROWS} "
            f"(ordinary warm-up only). This scenario's NaN/Inf validation already "
            f"passed during generation, so a row drop this large means something "
            f"changed between generation and here - check INPUT_CSV is pointing at "
            f"the right file before proceeding."
        )
    print(f"Row drop is within the expected warm-up range "
          f"(<= {EXPECTED_MAX_WARMUP_ROWS} rows).")

    X = feats.loc[valid, feature_order].reset_index(drop=True)
    y = raw.loc[valid, TARGET_COLUMNS].reset_index(drop=True)
    y = y.rename(columns={f"Phase_{c}": f"target_Phase_{c}" for c in CONV})
    tcol = raw.loc[valid, "time"].reset_index(drop=True) if "time" in raw.columns else pd.Series(np.arange(len(X)), name="time")

    out = pd.concat([tcol.rename("time"), X, y], axis=1)
    out_parquet = OUTPUT_DIR / "v7_corridor_triad_collapse_10bus_derating_5000s_features.parquet"
    out_csv = OUTPUT_DIR / "v7_corridor_triad_collapse_10bus_derating_5000s_features.csv"
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
        "ring_bus_pairs": RING_BUS_PAIRS, "ring_conv_pairs": RING_CONV_PAIRS,
        "voltage_reference": VOLTAGE_REFERENCE, "gei_reference": GEI_REFERENCE,
        "tj_derate_onset": TJ_DERATE_ONSET, "tj_shutdown": TJ_SHUTDOWN,
        "lag_steps": list(LAG_STEPS), "rolling_window": ROLL_WINDOW,
        "rolling_std_ddof": ROLLING_STD_DDOF,
        "rows_dropped_as_warmup": n_dropped,
        "note": (
            "corridor_triad_collapse scenario: E and F collapsed together at "
            "t~1149.1-1149.2s (0.1s apart); G did not collapse. GEI was computed "
            "correctly inline during generation, no separate GEI-fix step needed. "
            "Phase confirmed absent from input features by a runtime assertion."
        ),
    }
    (OUTPUT_DIR / "feature_schema_corridor_triad_collapse_v3.json").write_text(json.dumps(schema, indent=2))

    print(f"\nSaved features: {saved}  ({len(out):,} rows x {out.shape[1]} cols)")
    print(f"Saved schema:   {OUTPUT_DIR / 'feature_schema_corridor_triad_collapse_v3.json'}")
    print(f"\n{len(feature_order)} features  ->  {len(TARGET_COLUMNS)} phase targets")
    print("AL target range:", float(y['target_Phase_AL'].min()), "to", float(y['target_Phase_AL'].max()))
    print("Min derate factor observed anywhere:", float(physics['Min_Derate_Factor'].min()))
    print("Max junction temperature observed anywhere:", float(physics['Max_Junction_Temp'].max()),
          f" C (derate onset is {TJ_DERATE_ONSET} C, shutdown threshold is {TJ_SHUTDOWN} C)")

if __name__ == "__main__":
    main()
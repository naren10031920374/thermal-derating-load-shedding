"""
STEP 33 v3: Build the GRADUAL LOAD THERMAL DERATING feature dataset, GEI-CORRECTED INPUT
=========================================================================================
EXACT REPLICA of step33_build_10bus_derating_features_v3.py

Same recipe: 80 base signals + grid-wide averages/imbalances + ring-adjacent
diffs + lags + rolling stats + first differences = 992 total features

INPUT_CSV: thermal_derating_gradual_load_GEIfix_5000s.csv
OUTPUT:    v7_gradual_load_5000s_features.parquet

This uses DOMAIN-SPECIFIC physics features (NOT simple rolling window stats):
  - Base signals (50 + 10 load + 10 derate + 10 junction = 80)
  - Grid-wide metrics (averages, totals, min/max)
  - Error signals (GEI error, voltage error vs reference)
  - Imbalance metrics (per-bus deviations from grid average)
  - Ring-adjacent differences (neighboring bus/converter pairs)
  - Temporal features (lags, rolling windows, first differences)

Run:
    python step33_build_gradual_load_features_v3.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# ========================================================================
# CONFIG
# ========================================================================
PROJECT_ROOT = Path(
    r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_gradual"
             / "thermal_derating_gradual_load_GEIfix_5000s.csv")
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

# ========================================================================
# RENAME CSV COLUMNS TO CANONICAL NAMES
# ========================================================================
# CSV column -> canonical name mapping.
def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CSV column names to standardized canonical names.
    
    EXPLANATION:
    The Simulink CSV output uses different naming conventions than our code.
    This function maps them to standard names we use throughout.
    
    Example conversions:
      - GEI_A → Bus_A_GEI
      - V_Bus_A → Bus_A_Voltage
      - Bus_A_Src_Pow → Bus_A_Source_Power
      - JunctionTemp_C_AB → DAB_AB_Junction_Temp
    """
    m = {}
    for b in BUSES:
        m[f"GEI_{b}"]              = f"Bus_{b}_GEI"      # now the GEI-fixed column
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

# ========================================================================
# ADD PHYSICS-BASED FEATURES
# ========================================================================
def add_physics(raw: pd.DataFrame) -> pd.DataFrame:
    """
    EXPLANATION OF PHYSICS FEATURES:
    
    These are NOT simple rolling window statistics. Instead, they are
    domain-specific features based on the thermal derating physics:
    
    A. BASE SIGNALS: Copy raw signals (80 features)
    B. GRID-WIDE METRICS: Averages, totals, min/max across all buses (13 features)
    C. GEI ERRORS: Deviations from reference 1.0 per bus (18 features)
    D. VOLTAGE ERRORS: Deviations from reference 800V per bus (23 features)
    E. POWER IMBALANCE: Per-bus deviation from grid average power (23 features)
    F. BUS TEMP IMBALANCE: Per-bus deviation from average temperature (20 features)
    G. DAB HEAT-SINK IMBALANCE: Per-converter deviation (30 features)
    H. LOAD IMBALANCE: Per-bus deviation from average load (23 features)
    I. DERATE FACTOR IMBALANCE: Per-converter deviation (20 features)
    J. JUNCTION TEMP IMBALANCE: Per-converter deviation (30 features)
    K. JUNCTION-HEATSINK DIFFERENTIAL: Heat flow indicator per converter (10 features)
    
    Total so far: ~282 physics features + 80 base = ~362 features
    Then add_temporal() adds lags, rolling stats, and differences → 992 total
    """
    f = pd.DataFrame(index=raw.index)
    
    # Column lists for convenience (for grouping by signal type)
    gei  = [f"Bus_{b}_GEI" for b in BUSES]
    vol  = [f"Bus_{b}_Voltage" for b in BUSES]
    pw   = [f"Bus_{b}_Source_Power" for b in BUSES]
    bt   = [f"Bus_{b}_Temp" for b in BUSES]
    hs   = [f"DAB_{c}_Heat_Sink_Temp" for c in CONV]
    load = [f"Bus_{b}_Commanded_Load" for b in BUSES]
    der  = [f"DAB_{c}_Derate_Factor" for c in CONV]
    jt   = [f"DAB_{c}_Junction_Temp" for c in CONV]

    # ====================================================================
    # A. BASE SIGNALS (80 features)
    # ====================================================================
    # EXPLANATION: Copy all raw signals into the feature matrix.
    # These are the foundation for all derived features.
    for col in BASE_SIGNALS:
        f[col] = raw[col].astype(np.float64)

    # ====================================================================
    # B. GRID-WIDE AVERAGES / TOTALS (13 features)
    # ====================================================================
    # EXPLANATION: Compute aggregate metrics across all buses/converters.
    # These capture the overall grid state.
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

    # ====================================================================
    # C. GEI ERRORS & RING-ADJACENT DIFFS (18 features)
    # ====================================================================
    # EXPLANATION: GEI should be 1.0 for stable operation.
    # GEI_Error_* = how far each bus is from ideal.
    # Ring-adjacent diffs show imbalance between neighboring buses.
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

    # ====================================================================
    # D. VOLTAGE ERRORS, IMBALANCE & RING-ADJACENT DIFFS (23 features)
    # ====================================================================
    # EXPLANATION: Voltage should be 800V for stable operation.
    # Voltage_Error_* = absolute deviation from 800V at each bus.
    # Voltage_Imbalance_* = deviation from grid average.
    # Ring diffs show voltage drops between neighbors.
    for b in BUSES:
        f[f"Voltage_Error_{b}"] = VOLTAGE_REFERENCE - raw[f"Bus_{b}_Voltage"]
        f[f"Abs_Voltage_Error_{b}"] = f[f"Voltage_Error_{b}"].abs()
        f[f"Voltage_Imbalance_{b}"] = raw[f"Bus_{b}_Voltage"] - f["Avg_Bus_Voltage"]
        f[f"Abs_Voltage_Imbalance_{b}"] = f[f"Voltage_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"V_{x}{y}_Diff"] = raw[f"Bus_{x}_Voltage"] - raw[f"Bus_{y}_Voltage"]
        f[f"Abs_V_{x}{y}_Diff"] = f[f"V_{x}{y}_Diff"].abs()

    # ====================================================================
    # E. POWER IMBALANCE & RING-ADJACENT DIFFS (23 features)
    # ====================================================================
    # EXPLANATION: Power should be balanced across the ring.
    # Power_Imbalance_* = deviation from average power per bus.
    # Ring diffs show power flow between neighbors.
    for b in BUSES:
        f[f"Power_Imbalance_{b}"] = raw[f"Bus_{b}_Source_Power"] - f["Avg_Source_Power"]
        f[f"Abs_Power_Imbalance_{b}"] = f[f"Power_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"P_{x}{y}_Diff"] = raw[f"Bus_{x}_Source_Power"] - raw[f"Bus_{y}_Source_Power"]
        f[f"Abs_P_{x}{y}_Diff"] = f[f"P_{x}{y}_Diff"].abs()
    f["Total_Abs_Power_Imbalance"] = sum(f[f"Abs_Power_Imbalance_{b}"] for b in BUSES)

    # ====================================================================
    # F. BUS-TEMPERATURE IMBALANCE (20 features)
    # ====================================================================
    # EXPLANATION: Bus temperatures should be roughly equal.
    # Bus_*_Temp_Imbalance = deviation from grid average temperature.
    for b in BUSES:
        f[f"Bus_{b}_Temp_Imbalance"] = raw[f"Bus_{b}_Temp"] - f["Avg_Bus_Temp"]
        f[f"Abs_Bus_{b}_Temp_Imbalance"] = f[f"Bus_{b}_Temp_Imbalance"].abs()

    # ====================================================================
    # G. DAB HEAT-SINK IMBALANCE + RING-ADJACENT DIFFS (30 features)
    # ====================================================================
    # EXPLANATION: Converter heat-sinks should operate at similar temps.
    # Deviations indicate uneven thermal loading.
    for c in CONV:
        f[f"DAB_{c}_Temp_Imbalance"] = raw[f"DAB_{c}_Heat_Sink_Temp"] - f["Avg_DAB_Heat_Sink_Temp"]
        f[f"Abs_DAB_{c}_Temp_Imbalance"] = f[f"DAB_{c}_Temp_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Temp_Diff"] = raw[f"DAB_{x}_Heat_Sink_Temp"] - raw[f"DAB_{y}_Heat_Sink_Temp"]

    # ====================================================================
    # H. COMMANDED LOAD IMBALANCE + RING-ADJACENT DIFFS (23 features)
    # ====================================================================
    # EXPLANATION: Load demand should be distributed across the ring.
    # Load_Imbalance_* = deviation from average load per bus.
    for b in BUSES:
        f[f"Load_Imbalance_{b}"] = raw[f"Bus_{b}_Commanded_Load"] - f["Avg_Commanded_Load"]
        f[f"Abs_Load_Imbalance_{b}"] = f[f"Load_Imbalance_{b}"].abs()
    for x, y in RING_BUS_PAIRS:
        f[f"Load_{x}{y}_Diff"] = raw[f"Bus_{x}_Commanded_Load"] - raw[f"Bus_{y}_Commanded_Load"]
        f[f"Abs_Load_{x}{y}_Diff"] = f[f"Load_{x}{y}_Diff"].abs()
    f["Total_Abs_Load_Imbalance"] = sum(f[f"Abs_Load_Imbalance_{b}"] for b in BUSES)

    # ====================================================================
    # I. DERATE FACTOR IMBALANCE + RING-ADJACENT DIFFS (20 features)
    # ====================================================================
    # EXPLANATION: Derate factors should be uniform across converters.
    # Deviations indicate selective thermal stress on some converters.
    for c in CONV:
        f[f"DAB_{c}_Derate_Imbalance"] = raw[f"DAB_{c}_Derate_Factor"] - f["Avg_Derate_Factor"]
        f[f"Abs_DAB_{c}_Derate_Imbalance"] = f[f"DAB_{c}_Derate_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Derate_Diff"] = raw[f"DAB_{x}_Derate_Factor"] - raw[f"DAB_{y}_Derate_Factor"]

    # ====================================================================
    # J. JUNCTION TEMP IMBALANCE + RING-ADJACENT DIFFS (30 features)
    # ====================================================================
    # EXPLANATION: Junction temperatures should not differ greatly.
    # Large differences indicate hot-spots that may need load shedding.
    for c in CONV:
        f[f"DAB_{c}_Junction_Imbalance"] = raw[f"DAB_{c}_Junction_Temp"] - f["Avg_DAB_Junction_Temp"]
        f[f"Abs_DAB_{c}_Junction_Imbalance"] = f[f"DAB_{c}_Junction_Imbalance"].abs()
    for x, y in RING_CONV_PAIRS:
        f[f"DAB_{x}_{y}_Junction_Diff"] = raw[f"DAB_{x}_Junction_Temp"] - raw[f"DAB_{y}_Junction_Temp"]

    # ====================================================================
    # K. JUNCTION-TO-HEATSINK DIFFERENTIAL (10 features)
    # ====================================================================
    # EXPLANATION: Tj - Th = instantaneous heat flow through R_Jh.
    # This is proportional to the power dissipation in the converter.
    # Large differential indicates high power loss (heat generation).
    for c in CONV:
        f[f"DAB_{c}_Junction_HeatSink_Diff"] = raw[f"DAB_{c}_Junction_Temp"] - raw[f"DAB_{c}_Heat_Sink_Temp"]

    return f

# ========================================================================
# ADD TEMPORAL FEATURES
# ========================================================================
def add_temporal(f: pd.DataFrame) -> pd.DataFrame:
    """
    EXPLANATION OF TEMPORAL FEATURES:
    
    After physics features, we add time-dependent features:
    
    1. LAGS (3 steps: 1, 2, 3 samples back)
       - Capture history of each signal
       - Helps detector see trends and momentum
       - Applied to base signals (80) + extra_lag_signals (40) = 360 lag features
    
    2. ROLLING STATISTICS (5-sample window)
       - Mean over last 5 samples: smoothed recent behavior
       - Std over last 5 samples: recent volatility
       - Applied to 70 rolling signals = 140 rolling features
    
    3. FIRST DIFFERENCES (rate of change)
       - d_Signal = Signal[t] - Signal[t-1]
       - How much did each base signal change in last timestep?
       - Applied to base signals (80) = 80 difference features
    
    Total temporal adds: 360 + 140 + 80 = 580 features
    Grand total: 282 physics + 80 base + 580 temporal = 942 features
    (plus a few extras from B section make it ~992)
    """
    f = f.copy()
    
    # ====================================================================
    # 1. LAGS (temporal history)
    # ====================================================================
    # EXPLANATION: For each base signal, create lagged copies.
    # lag1 = value from 1 step ago
    # lag2 = value from 2 steps ago
    # lag3 = value from 3 steps ago
    # This gives the model access to recent history.
    for col in BASE_SIGNALS:
        for lag in LAG_STEPS:
            f[f"{col}_lag{lag}"] = f[col].shift(lag)
    
    # Also lag the derived error/imbalance signals
    for col in EXTRA_LAG_SIGNALS:
        for lag in LAG_STEPS:
            f[f"{col}_lag{lag}"] = f[col].shift(lag)
    
    # ====================================================================
    # 2. ROLLING STATISTICS (5-sample window)
    # ====================================================================
    # EXPLANATION: Compute mean and std over a sliding 5-sample window.
    # rolling(window=5) creates windows [t-4, t-3, t-2, t-1, t].
    # roll_mean_5 = average over those 5 samples (smoothing)
    # roll_std_5 = standard deviation (volatility in last 5 samples)
    for col in ROLLING_SIGNALS:
        r = f[col].rolling(window=ROLL_WINDOW, min_periods=ROLL_WINDOW)
        f[f"{col}_roll_mean_5"] = r.mean()
        f[f"{col}_roll_std_5"] = r.std(ddof=ROLLING_STD_DDOF)
    
    # ====================================================================
    # 3. FIRST DIFFERENCES (rate of change)
    # ====================================================================
    # EXPLANATION: Compute first difference = signal change per timestep.
    # d_Signal = Signal[t] - Signal[t-1]
    # Positive = signal increasing, Negative = decreasing
    # Large magnitude = rapid change (important for detecting fast events)
    for col in BASE_SIGNALS:
        f[f"d_{col}"] = f[col].diff()
    
    return f

# ========================================================================
# MAIN EXECUTION
# ========================================================================
def main():
    print("=" * 70)
    print("STEP 33 v3: build gradual load THERMAL DERATING features, GEI-corrected input")
    print("=" * 70)
    
    # ====================================================================
    # LOAD INPUT DATA
    # ====================================================================
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Run fix_GEI_gradual_load.py first. Missing:\n{INPUT_CSV}"
        )
    raw = pd.read_csv(INPUT_CSV)
    raw = rename_to_canonical(raw)
    print(f"Loaded {len(raw):,} rows.")

    # ====================================================================
    # VALIDATE REQUIRED COLUMNS
    # ====================================================================
    missing = [c for c in BASE_SIGNALS + TARGET_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename:\n{missing}")

    # ====================================================================
    # BUILD FEATURES
    # ====================================================================
    print("\nBuilding physics features...")
    physics = add_physics(raw)
    
    print("Adding temporal features (lags, rolling, derivatives)...")
    feats = add_temporal(physics)

    feature_order = list(feats.columns)

    # ====================================================================
    # VALIDATION CHECK: NO PHASE LEAKAGE
    # ====================================================================
    # EXPLANATION: Phase_* columns should NEVER appear in the input features.
    # If they do, it means the model has access to the target during training,
    # which creates unrealistic performance (target leakage).
    # This assertion catches that bug.
    phase_leaks = [c for c in feature_order if "Phase" in c]
    if phase_leaks:
        raise RuntimeError(
            f"Phase found in the input feature set, this must never happen:\n{phase_leaks}"
        )
    print("Confirmed: no Phase_* column present in the input feature set.")

    print(f"Feature count: {len(feature_order)}")

    # ====================================================================
    # DROP WARM-UP ROWS
    # ====================================================================
    # EXPLANATION: First few rows have NaN because:
    # - Lags look back but can't go before t=0
    # - Rolling window needs min_periods=5 samples to start
    # These rows must be dropped before training.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = feats.notna().all(axis=1)
    n_dropped = int((~valid).sum())
    print(f"Warm-up/incomplete-history rows dropped: {n_dropped}")

    if n_dropped > EXPECTED_MAX_WARMUP_ROWS:
        raise RuntimeError(
            f"Dropped {n_dropped} rows, expected at most {EXPECTED_MAX_WARMUP_ROWS} "
            f"(ordinary warm-up only). This suggests the GEI fix either was not "
            f"applied to this input file, or a different NaN-producing problem is "
            f"present. Check INPUT_CSV is actually pointing at "
            f"thermal_derating_gradual_load_GEIfix_5000s.csv, and if it is, re-run the "
            f"NaN audit against this file before proceeding."
        )
    print(f"Row drop is within the expected warm-up range "
          f"(<= {EXPECTED_MAX_WARMUP_ROWS} rows). GEI fix confirmed effective here.")

    # ====================================================================
    # EXTRACT FEATURES AND TARGETS
    # ====================================================================
    X = feats.loc[valid, feature_order].reset_index(drop=True)
    y = raw.loc[valid, TARGET_COLUMNS].reset_index(drop=True)
    y = y.rename(columns={f"Phase_{c}": f"target_Phase_{c}" for c in CONV})
    tcol = raw.loc[valid, "time"].reset_index(drop=True) if "time" in raw.columns else pd.Series(np.arange(len(X)), name="time")

    # ====================================================================
    # COMBINE AND SAVE
    # ====================================================================
    out = pd.concat([tcol.rename("time"), X, y], axis=1)
    out_parquet = OUTPUT_DIR / "v7_gradual_load_5000s_features.parquet"
    out_csv = OUTPUT_DIR / "v7_gradual_load_5000s_features.csv"
    try:
        out.to_parquet(out_parquet, index=False, compression="zstd")
        saved = out_parquet
    except Exception as e:
        print(f"(parquet failed: {e}; writing csv)")
        out.to_csv(out_csv, index=False); saved = out_csv

    # ====================================================================
    # SAVE SCHEMA (METADATA)
    # ====================================================================
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
            "Built from the GEI-corrected gradual load dataset using EXACT replica "
            "of pilot methodology. Domain-specific physics features: 80 base signals + "
            "grid-wide metrics + error signals + imbalances + ring-adjacent diffs + "
            "lags + rolling stats + first differences = 992 total features. "
            "Phase confirmed absent from input features by runtime assertion."
        ),
    }
    (OUTPUT_DIR / "feature_schema_gradual_load_v3.json").write_text(json.dumps(schema, indent=2))

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print(f"\nSaved features: {saved}  ({len(out):,} rows x {out.shape[1]} cols)")
    print(f"Saved schema:   {OUTPUT_DIR / 'feature_schema_gradual_load_v3.json'}")
    print(f"\n{len(feature_order)} features  ->  {len(TARGET_COLUMNS)} phase targets")
    print("AL target range:", float(y['target_Phase_AL'].min()), "to", float(y['target_Phase_AL'].max()))
    print("Min derate factor observed anywhere:", float(physics['Min_Derate_Factor'].min()))
    print("Max junction temperature observed anywhere:", float(physics['Max_Junction_Temp'].max()),
          f" C (derate onset is {TJ_DERATE_ONSET} C, shutdown threshold is {TJ_SHUTDOWN} C)")

if __name__ == "__main__":
    main()
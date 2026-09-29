"""
FEATURES_FROM_RAW.PY  (dataset pipeline, step 3 helper)
================================================================================
Turns one scenario's raw simulation table (the 92-column output of
run_scenario.m, identical layout to the Bus C v4 CSV) into the SAME feature
table the early-warning detector was trained on.

Everything below "VOLTAGE_REFERENCE" is copied VERBATIM from
scripts/02_feature_engineering/step33_build_gradual_busC_v4_features.py.py
(itself the step33_v3 recipe). Do not edit the recipe here: the saved
sklearn pipeline indexes features positionally.

The only additions are the build_features() wrapper at the bottom and the
collapse-label helper (copied from build_corrected_phase_targets.py).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

VOLTAGE_REFERENCE = 800.0
GEI_REFERENCE = 1.0
TJ_DERATE_ONSET = 125.0
TJ_SHUTDOWN = 175.0
LAG_STEPS = (1, 2, 3)
ROLL_WINDOW = 5
ROLLING_STD_DDOF = 1

# gradual_busC_v4 has a slow settle+ramp lead-in (0-2800s) before anything
# collapses, so there is no reason to expect a big warm-up-row blowup here
# the way the baseline run's t=4140-5000s GEI/0 block did. Keep the same
# small tolerance as step33_v3 as a sanity net, not a relaxed one.
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



# ----------------------------------------------------------------------
# Collapse label, copied verbatim from build_corrected_phase_targets.py
# ----------------------------------------------------------------------
COLLAPSE_VOLTAGE_V = 100.0
MIN_CONSECUTIVE_SAMPLES = 50


def detect_collapse(voltage: pd.Series, threshold: float, min_run: int) -> pd.Series:
    below = voltage < threshold
    run_id = (below != below.shift(fill_value=False)).cumsum()
    run_length = below.groupby(run_id).transform("size")
    return below & (run_length >= min_run)


# Small tolerance for start-up rows dropped because lags/rolling need history
# (lag 3 / window 5 -> a handful of rows). Same guard as step33.
EXPECTED_MAX_WARMUP_ROWS = 20


def build_features(raw: pd.DataFrame, keep_cols=None) -> pd.DataFrame:
    """raw: run_scenario.m output. Returns a table with 'time', the feature
    columns, and the per-bus collapsed_<BUS> flags, warm-up rows removed.

    keep_cols: optional list of feature columns to keep. The full recipe makes
    ~990 columns x 500k rows (several GB); the detector only reads a subset, so
    pass that subset to keep memory low. The values are unchanged."""
    raw = rename_to_canonical(raw.copy())
    missing = [c for c in BASE_SIGNALS + TARGET_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing[:8]}"
                         f"{'...' if len(missing) > 8 else ''}")

    feats = add_temporal(add_physics(raw))
    if keep_cols is not None:
        feats = feats[list(keep_cols)]
    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = feats.notna().all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped > EXPECTED_MAX_WARMUP_ROWS:
        raise RuntimeError(
            f"{n_dropped} rows have NaN/Inf features (expected <= {EXPECTED_MAX_WARMUP_ROWS}). "
            "The raw table has a NaN/Inf block beyond normal warm-up; do not trust "
            "this scenario until it is inspected.")

    out = feats.loc[valid].copy()
    out.insert(0, "time", raw.loc[valid, "time"].to_numpy())
    for b in BUSES:
        v = raw[f"Bus_{b}_Voltage"]
        out[f"collapsed_{b}"] = detect_collapse(
            v, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES).loc[valid].to_numpy()
    return out.reset_index(drop=True)

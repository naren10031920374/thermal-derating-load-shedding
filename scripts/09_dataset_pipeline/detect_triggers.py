"""
DETECT_TRIGGERS.PY  (dataset pipeline, step 3)
================================================================================
For ONE scenario's no-shed run, find each bus's early-warning trigger time
using the saved step41 N=60 s detector, exactly as
scripts/07_closed_loop_validation/find_detector_trigger_time.py does for the
single fixed dataset -- but generalised to any scenario table from
run_scenario.m.

Same rules as the original (all copied, none re-tuned):
  - pipeline.predict_proba(X) on the 83 strict-precursor features, no
    separate scaler (it is inside the pickle)
  - threshold 5e-6, stride-5 subsampling (50 ms), trigger = first run of
    >= 10 consecutive samples (0.5 s) at/above threshold
  - for a bus that collapses, only samples BEFORE its collapse onset count
  - for a bus that never collapses, the whole run is scored (a trigger there
    is a FALSE POSITIVE and is reported as such)

Usage (from anywhere; paths are relative to the project root)
    python detect_triggers.py S041
    python detect_triggers.py S041 --model path/to/early_warning_N60s.pkl
    python detect_triggers.py --all                 # every scenario folder found

Reads   <out_root>/<ID>/noshed_<ID>_5000s.parquet  (or .csv)
Writes  <out_root>/<ID>/detector_triggers_<ID>.json
<out_root> = env DATASET_OUTPUT_ROOT, else <project>/model_outputs/dataset_pipeline
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features_from_raw as ffr   # noqa: E402

PROJECT_ROOT = HERE.parents[1]                       # scripts/09_dataset_pipeline -> project root
DEFAULT_MODEL = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
                 / "41_early_warning_shifted_label" / "early_warning_N60s.pkl")

DECISION_THRESHOLD = 5e-6
MIN_CONSECUTIVE_TRIGGER_SAMPLES = 10   # x 50 ms = 0.5 s sustained
SUBSAMPLE_STRIDE = 5
LEAD_SECONDS = 60.0                    # the N of the N=60 s model


# ---- copied verbatim from find_detector_trigger_time.py ---------------------
def get_local_feature_map(bus_idx: int) -> dict:
    BUSES, CONV = ffr.BUSES, ffr.CONV
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


def first_sustained_crossing(proba, threshold, min_run, search_mask):
    """Index of the first sample that starts a run of >= min_run consecutive
    samples with proba >= threshold inside search_mask; None if none.
    (Same rule as the original, written as a plain scan.)"""
    above = (proba >= threshold) & search_mask
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            return i - min_run + 1
    return None
# -----------------------------------------------------------------------------


def strict_columns():
    canonical = list(get_local_feature_map(0).keys())
    return [c for c in canonical if c not in DIRECT_SIGNAL_KEYS + STRICT_ADDITIONAL_KEYS]


def load_raw(scn_dir: Path, scn_id: str) -> pd.DataFrame:
    pq = scn_dir / f"noshed_{scn_id}_5000s.parquet"
    cs = scn_dir / f"noshed_{scn_id}_5000s.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    raise FileNotFoundError(f"No noshed_{scn_id}_5000s.(parquet|csv) in {scn_dir}")


def score_scenario(raw: pd.DataFrame, pipeline) -> dict:
    wide = ffr.build_features(raw)
    if SUBSAMPLE_STRIDE > 1:
        wide = wide.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)
    t_all = wide["time"].to_numpy()
    strict = strict_columns()
    assert len(strict) == 83, f"expected 83 strict-precursor features, got {len(strict)}"

    buses = {}
    for bus_idx, bus in enumerate(ffr.BUSES):
        fmap = get_local_feature_map(bus_idx)
        cols = [fmap[c] for c in strict]
        missing = [c for c in cols if c not in wide.columns]
        if missing:
            raise ValueError(f"Bus {bus}: missing feature columns {missing[:5]}")
        proba = pipeline.predict_proba(wide[cols].to_numpy(dtype=np.float64))[:, 1]

        collapsed = wide[f"collapsed_{bus}"].to_numpy().astype(bool)
        onset = float(t_all[np.argmax(collapsed)]) if collapsed.any() else None
        mask = (t_all < onset) if onset is not None else np.ones_like(t_all, dtype=bool)

        idx = first_sustained_crossing(proba, DECISION_THRESHOLD,
                                       MIN_CONSECUTIVE_TRIGGER_SAMPLES, mask)
        trig = float(t_all[idx]) if idx is not None else None
        if onset is not None and trig is not None:
            status = "caught"
        elif onset is not None:
            status = "MISS"
        elif trig is not None:
            status = "false_positive"
        else:
            status = "correctly_quiet"
        buses[bus] = {
            "collapse_onset": onset,
            "trigger_time": trig,
            "lead_time": (onset - trig) if (onset is not None and trig is not None) else None,
            "max_probability": float(proba[mask].max()) if mask.any() else None,
            "status": status,
        }
    return buses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", help="scenario id, e.g. S041")
    ap.add_argument("--all", action="store_true", help="every scenario folder under the output root")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--output-root", default=None)
    a = ap.parse_args()

    out_root = Path(a.output_root or os.environ.get("DATASET_OUTPUT_ROOT")
                    or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    model_path = Path(a.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Detector model not found: {model_path}\n"
            "It is untracked in git; copy early_warning_N60s.pkl there (or pass --model).")
    pipeline = joblib.load(model_path)

    if a.all:
        ids = sorted(p.name for p in out_root.iterdir()
                     if p.is_dir() and not p.name.endswith("_smoke"))
    elif a.scenario:
        ids = [a.scenario]
    else:
        ap.error("give a scenario id or --all")

    for scn in ids:
        d = out_root / scn
        try:
            raw = load_raw(d, scn)
        except FileNotFoundError as e:
            print(f"[{scn}] skipped: {e}")
            continue
        print(f"[{scn}] {len(raw):,} rows -> features -> detector ...")
        buses = score_scenario(raw, pipeline)
        for b, r in buses.items():
            if r["collapse_onset"] is not None or r["trigger_time"] is not None:
                print(f"   Bus {b}: onset={r['collapse_onset']}  trigger={r['trigger_time']}  "
                      f"lead={r['lead_time']}  [{r['status']}]")
        outp = d / f"detector_triggers_{scn}.json"
        outp.write_text(json.dumps({
            "scenario_id": scn, "threshold": DECISION_THRESHOLD,
            "min_consecutive_samples": MIN_CONSECUTIVE_TRIGGER_SAMPLES,
            "subsample_stride": SUBSAMPLE_STRIDE, "model_path": str(model_path),
            "buses": buses}, indent=2))
        print(f"   wrote {outp}")


if __name__ == "__main__":
    main()

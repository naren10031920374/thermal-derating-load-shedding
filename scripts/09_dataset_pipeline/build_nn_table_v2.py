"""
BUILD_NN_TABLE_V2.PY  (dataset pipeline, step 5a, version 2)
================================================================================
Same job as build_nn_table.py (one row per scenario: grid state at the decision
moment t*, plus the minimum safe ShedFrac as the label), but with 12 EXTRA
features that describe how FAST things are changing at t*.

Why
---
Batch-2 results (107 rows) showed that more rows alone did not fix the model.
The 25 v1 features are mostly "levels" at the alarm moment, and at that moment
the loads are still climbing, so the model cannot see how bad it will get. The
extra features give it the speed and shape of the climb and of the heating.

What is new (all computed only from data up to t*, nothing from the future;
5 s trailing means like v1, so 800 W of load noise does not leak in)
---------------------------------------------------------------------
  Load ramp (the most heavily loaded bus at t*, so one bus climbing fast is not
  watered down by two buses that are not climbing)
    load_max_rate_10s       kW/s over the last 10 s
    load_max_curvature      rise in the last 30 s minus rise in the 30 s before
                            (kW): about 0 = steady climb, negative = flattening
    load_flatten_ratio      rate over last 10 s / rate over last 60 s, clipped to
                            [-3, 3]: about 1 = steady, about 0 = flat, >1 = speeding up
  Heating
    tj_max_rate_10s         deg C/s, hottest junction, last 10 s
    tj_time_to_125_s        estimated seconds until the hottest junction reaches the
                            125 C derating onset (0 if already there, capped at 600)
    hs_max_rise_30s         deg C, hottest heat sink, last 30 s
    bus_temp_max_rise_30s   deg C, hottest bus, last 30 s
    derate_worst_change_30s change in the derate factor of the worst converter, 30 s
                            (negative = derating getting deeper)
    p_max_rise_30s          W, busiest source-power bus, last 30 s
  How long it has been going on
    ramp_age_s              seconds since the first bus load passed 40 kW
    heavy_age_100kW_s       seconds since the first bus load passed 100 kW (0 if none)
    ramp_avg_rate_kw_s      (highest load - 40 kW) / ramp_age_s, the average climb speed

t* itself is still NOT a feature. The three "how long" features are durations
measured back from t*, which a real controller can track, not the scenario clock.

The 25 v1 feature columns are produced by the SAME function as before
(build_nn_table.state_features), so they are unchanged.

Run (on the cluster, inside the venv; reads the big parquet files):
    python build_nn_table_v2.py
Writes  <out_root>/nn_table_v2.csv  and  <out_root>/nn_table_v2_skipped.csv
(it does not touch nn_table.csv, so the batch-2 results stay as they are).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from build_nn_table import (BUSES, CONV, TS, AVG_S, RAW_COLS, META_COLS, HERE,
                            PROJECT_ROOT, TJ_DERATE_ONSET, level, state_features,
                            decide_moment)

NEW_FEATURES = [
    "load_max_rate_10s", "load_max_curvature", "load_flatten_ratio",
    "tj_max_rate_10s", "tj_time_to_125_s", "hs_max_rise_30s", "bus_temp_max_rise_30s",
    "derate_worst_change_30s", "p_max_rise_30s",
    "ramp_age_s", "heavy_age_100kW_s", "ramp_avg_rate_kw_s",
]
RAMP_START_KW = 40.0        # a bus is "ramping" once its load passes this (ambient is 27 kW)
HEAVY_KW = 100.0
TJ_CAP_S = 600.0            # cap for the time-to-125 C estimate
FLAT_FLOOR_KW_S = 0.02      # keeps the flatten ratio finite when the 60 s rate is about 0


def _first_cross_age(loads: np.ndarray, i: int, thr_kw: float) -> float:
    """Seconds from the earliest time ANY bus load passed thr_kw until sample i
    (0 if none has by i). loads is [n_bus x n_samples]; only samples <= i are read.

    The load is smoothed with the same 5 s trailing mean as the other features
    first: the raw load has 0.8 kW of noise, and a single noisy sample would
    otherwise "cross" the threshold 10-15 s too early. A trailing mean lags the
    true crossing by about half its window, so half a window is added back."""
    w = int(round(AVG_S / TS))
    cs = np.cumsum(loads[:, : i + 1], axis=1)
    sm = np.empty_like(cs)
    sm[:, :w] = cs[:, :w] / np.arange(1, w + 1)          # window still filling at the start
    sm[:, w:] = (cs[:, w:] - cs[:, :-w]) / w
    above = sm > thr_kw
    hit = above.any(axis=1)
    if not hit.any():
        return 0.0
    first = np.argmax(above[hit], axis=1).min()
    return float((i - first) * TS + AVG_S / 2.0)


def extra_features(cols: dict, i: int) -> dict:
    """The 12 new features at sample index i. Reads only samples <= i."""
    w = int(round(AVG_S / TS))

    def lag_idx(sec):                                   # same clamping as v1
        return max(w, i - int(round(sec / TS)))

    def elapsed(j):                                     # seconds between lagged moment j and i
        return (i - j) * TS

    load_n = [f"CommandedLoad_kW_{b}" for b in BUSES]
    p_n  = [f"Bus_{b}_Src_Pow" for b in BUSES]
    bt_n = [f"Bus_{b}_Temp" for b in BUSES]
    tj_n = [f"JunctionTemp_C_{c}" for c in CONV]
    hs_n = [f"HeatSinkTemp_C_{c}" for c in CONV]
    dr_n = [f"DAB_{c}_Derate_Factor" for c in CONV]

    j10, j30, j60 = lag_idx(10.0), lag_idx(30.0), lag_idx(60.0)
    L0, L10, L30, L60 = (level(cols, load_n, i, w), level(cols, load_n, j10, w),
                         level(cols, load_n, j30, w), level(cols, load_n, j60, w))
    kmax = int(np.argmax(L0))                            # the most heavily loaded bus at t*
    f = {}

    # ---- load ramp (heaviest bus) ----
    e10, e60 = elapsed(j10), elapsed(j60)
    rate10 = float((L0[kmax] - L10[kmax]) / e10) if e10 >= 1 else 0.0
    rate60 = float((L0[kmax] - L60[kmax]) / e60) if e60 >= 1 else 0.0
    f["load_max_rate_10s"] = rate10
    f["load_max_curvature"] = float((L0[kmax] - L30[kmax]) - (L30[kmax] - L60[kmax]))
    f["load_flatten_ratio"] = float(np.clip(rate10 / max(abs(rate60), FLAT_FLOOR_KW_S), -3.0, 3.0))

    # ---- heating ----
    TJ0, TJ10, TJ30 = (level(cols, tj_n, i, w), level(cols, tj_n, j10, w), level(cols, tj_n, j30, w))
    k = int(np.argmax(TJ0))
    e30 = elapsed(j30)
    f["tj_max_rate_10s"] = float((TJ0[k] - TJ10[k]) / e10) if e10 >= 1 else 0.0
    slope30 = float((TJ0[k] - TJ30[k]) / e30) if e30 >= 1 else 0.0
    if TJ0[k] >= TJ_DERATE_ONSET:
        tt = 0.0
    elif slope30 <= 1e-3:
        tt = TJ_CAP_S
    else:
        tt = min(TJ_CAP_S, (TJ_DERATE_ONSET - float(TJ0[k])) / slope30)
    f["tj_time_to_125_s"] = float(tt)

    HS0, HS30 = level(cols, hs_n, i, w), level(cols, hs_n, j30, w)
    kh = int(np.argmax(HS0))
    f["hs_max_rise_30s"] = float(HS0[kh] - HS30[kh])

    BT0, BT30 = level(cols, bt_n, i, w), level(cols, bt_n, j30, w)
    kb = int(np.argmax(BT0))
    f["bus_temp_max_rise_30s"] = float(BT0[kb] - BT30[kb])

    DR0, DR30 = level(cols, dr_n, i, w), level(cols, dr_n, j30, w)
    kd = int(np.argmin(DR0))
    f["derate_worst_change_30s"] = float(DR0[kd] - DR30[kd])

    P0, P30 = level(cols, p_n, i, w), level(cols, p_n, j30, w)
    kp = int(np.argmax(P0))
    f["p_max_rise_30s"] = float(P0[kp] - P30[kp])

    # ---- how long it has been going on ----
    loads = np.vstack([cols[n] for n in load_n])         # raw 10 ms loads, kW
    age40 = _first_cross_age(loads, i, RAMP_START_KW)
    f["ramp_age_s"] = age40
    f["heavy_age_100kW_s"] = _first_cross_age(loads, i, HEAVY_KW)
    f["ramp_avg_rate_kw_s"] = float((L0.max() - RAMP_START_KW) / age40) if age40 >= 5.0 else 0.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--scenarios", nargs="*", default=None)
    a = ap.parse_args()

    out_root = Path(a.out_root or os.environ.get("DATASET_OUTPUT_ROOT")
                    or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    tbl = pd.read_csv(a.table)
    tbl = tbl[~tbl["spare"].astype(str).str.lower().isin(["true", "1"])]
    if a.scenarios:
        tbl = tbl[tbl["scenario_id"].isin(a.scenarios)]

    rows, skipped = [], []
    for _, r in tbl.iterrows():
        sid = r["scenario_id"]
        d = out_root / sid
        det_p = d / f"detector_triggers_{sid}.json"
        sig_p = d / f"noshed_{sid}_5000s.parquet"
        if not det_p.exists() or not sig_p.exists():
            skipped.append((sid, "missing detector json or signal parquet"))
            continue
        det = json.loads(det_p.read_text())
        bis_p = d / f"bisect_{sid}_result.json"
        bis = json.loads(bis_p.read_text()) if bis_p.exists() else None
        dec = decide_moment(det, bis)
        if dec[0] is None:
            skipped.append((sid, dec[1]))
            continue
        t_star, label, src, shed, n_trig = dec

        df = pd.read_parquet(sig_p, columns=RAW_COLS)
        t = df["time"].to_numpy()
        i = int(np.searchsorted(t, t_star))
        cols = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS if c != "time"}
        feats = state_features(cols, i, n_trig, t_star)      # the 25 v1 features, unchanged
        feats.update(extra_features(cols, i))                # + the 12 new ones

        row = {"scenario_id": sid, "kind": r["kind"], "split": r["split"],
               "label": label, "label_source": src, "t_star": t_star,
               "n_shed_buses": len(shed), "shed_buses": ";".join(shed)}
        row.update({k: v for k, v in feats.items() if k != "t_star"})
        rows.append(row)
        print(f"[{sid}] {r['kind']:10s} split={r['split']:5s} t*={t_star:8.2f}  label={label:.2f} ({src})")

    out = pd.DataFrame(rows)
    feat_cols = [c for c in out.columns if c not in META_COLS]
    bad = out[feat_cols].isna().any(axis=1)
    if bad.any():
        print("WARNING: NaN features in", out.loc[bad, "scenario_id"].tolist())
    out_p = out_root / "nn_table_v2.csv"
    out.to_csv(out_p, index=False)
    pd.DataFrame(skipped, columns=["scenario_id", "reason"]).to_csv(out_root / "nn_table_v2_skipped.csv", index=False)
    print(f"\nWrote {out_p}: {len(out)} rows x {len(feat_cols)} features "
          f"({len(feat_cols) - len(NEW_FEATURES)} v1 + {len(NEW_FEATURES)} new; "
          f"train {int((out.split=='train').sum())}, test {int((out.split=='test').sum())}); "
          f"skipped {len(skipped)}")
    for s in skipped:
        print("  skipped", s)


if __name__ == "__main__":
    main()

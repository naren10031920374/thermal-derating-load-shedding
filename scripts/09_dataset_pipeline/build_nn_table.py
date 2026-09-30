"""
BUILD_NN_TABLE.PY  (dataset pipeline, step 5a)
================================================================================
Turns the finished dataset (steps 2-4) into ONE training table for the shed NN:
one row per scenario.

  DECISION MOMENT  t* = the earliest detector trigger time among the buses that
                   are shed in that scenario. This is when the NN would be asked
                   "how much should we cut?" in a real deployment.
  INPUTS           grid state visible AT t* (and 30 s / 60 s before it):
                   bus loads, voltages, source powers, converter junction /
                   heat-sink temperatures, derate factors, GEI error, plus how
                   many detector triggers have fired by t*. Nothing from after
                   t* and nothing from the collapse itself is used, so the same
                   inputs exist at run time. The features are permutation
                   invariant (sorted / min / max / counts) so one model works for
                   any bus or bus combination.
  LABEL            best_shedfrac from bisect_scenario.m = mildest ShedFrac (fraction
                   of load KEPT) that keeps all 10 buses alive. One shared cut per
                   scenario (what the first batch of bisections measured).
  MILD SCENARIOS   no collapse -> no shed needed -> label 1.0. They only get a row
                   if the detector FALSE-ALARMED (that is the moment the NN would
                   be asked); they teach it not to cut a stressed-but-safe grid.
                   Mild scenarios with no detector trigger never call the NN and
                   are skipped.

Run (on the cluster, inside the venv; reads the big parquet files):
    python build_nn_table.py
    python build_nn_table.py --out-root /some/path --scenarios S001 S002
Writes  <out_root>/nn_table.csv   and   <out_root>/nn_table_skipped.csv
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
CONV  = ["AB", "BC", "CD", "DE", "EF", "FG", "GH", "HK", "KL", "AL"]
TS = 0.01
AVG_S = 5.0            # level = trailing 5 s mean (the loads carry 800 W noise)
LAGS_S = (30.0, 60.0)
TJ_DERATE_ONSET = 125.0

RAW_COLS = (["time"]
            + [f"CommandedLoad_kW_{b}" for b in BUSES]
            + [f"V_Bus_{b}" for b in BUSES]
            + [f"Bus_{b}_Src_Pow" for b in BUSES]
            + [f"Bus_{b}_Temp" for b in BUSES]
            + [f"GEI_{b}" for b in BUSES]
            + [f"JunctionTemp_C_{c}" for c in CONV]
            + [f"HeatSinkTemp_C_{c}" for c in CONV]
            + [f"DAB_{c}_Derate_Factor" for c in CONV])

META_COLS = ["scenario_id", "kind", "split", "label", "label_source", "t_star",
             "n_shed_buses", "shed_buses"]


def tmean(arr: np.ndarray, i: int, w: int) -> float:
    """Trailing mean of arr over the w samples ending at index i (no future data)."""
    lo = max(0, i - w + 1)
    return float(np.nanmean(arr[lo:i + 1]))


def level(df_cols: dict, names: list, i: int, w: int) -> np.ndarray:
    return np.array([tmean(df_cols[n], i, w) for n in names])


def state_features(cols: dict, i: int, n_trig: int, t_star: float) -> dict:
    w = int(round(AVG_S / TS))
    lag = {L: max(w, i - int(round(L / TS))) for L in LAGS_S}     # index of the lagged moment

    load_n = [f"CommandedLoad_kW_{b}" for b in BUSES]
    v_n    = [f"V_Bus_{b}" for b in BUSES]
    p_n    = [f"Bus_{b}_Src_Pow" for b in BUSES]
    bt_n   = [f"Bus_{b}_Temp" for b in BUSES]
    g_n    = [f"GEI_{b}" for b in BUSES]
    tj_n   = [f"JunctionTemp_C_{c}" for c in CONV]
    hs_n   = [f"HeatSinkTemp_C_{c}" for c in CONV]
    dr_n   = [f"DAB_{c}_Derate_Factor" for c in CONV]

    L0 = level(cols, load_n, i, w)
    V0 = level(cols, v_n, i, w)
    P0 = level(cols, p_n, i, w)
    BT0 = level(cols, bt_n, i, w)
    G0 = level(cols, g_n, i, w)
    TJ0 = level(cols, tj_n, i, w)
    HS0 = level(cols, hs_n, i, w)
    DR0 = level(cols, dr_n, i, w)
    L30 = level(cols, load_n, lag[30.0], w)
    V30 = level(cols, v_n, lag[30.0], w)
    TJ30 = level(cols, tj_n, lag[30.0], w)
    TJ60 = level(cols, tj_n, lag[60.0], w)
    L60 = level(cols, load_n, lag[60.0], w)

    top3 = np.argsort(-L0)[:3]
    vmin_bus = int(np.argmin(V0))
    tjmax_conv = int(np.argmax(TJ0))

    f = {}
    f["t_star"] = t_star
    f["n_triggers_by_tstar"] = float(n_trig)
    # loads (kW)
    f["load_max"] = float(L0.max())
    f["load_top3_mean"] = float(L0[top3].mean())
    f["load_total"] = float(L0.sum())
    f["n_load_gt_100kW"] = float((L0 > 100).sum())
    f["n_load_gt_150kW"] = float((L0 > 150).sum())
    f["load_top3_rise_30s"] = float((L0[top3] - L30[top3]).mean())
    f["load_top3_rise_60s"] = float((L0[top3] - L60[top3]).mean())
    # voltages (V)
    f["v_min"] = float(V0.min())
    f["v_mean"] = float(V0.mean())
    f["v_std"] = float(V0.std())
    f["v_min_bus_drop_30s"] = float(V0[vmin_bus] - V30[vmin_bus])
    # source power (W)
    f["p_max"] = float(P0.max())
    f["p_min"] = float(P0.min())
    f["p_spread"] = float(P0.max() - P0.min())
    # temperatures (C)
    f["bus_temp_max"] = float(BT0.max())
    f["tj_max"] = float(TJ0.max())
    f["tj_mean"] = float(TJ0.mean())
    f["n_tj_gt_125"] = float((TJ0 > TJ_DERATE_ONSET).sum())
    f["tj_max_rise_30s"] = float(TJ0[tjmax_conv] - TJ30[tjmax_conv])
    f["tj_max_rise_60s"] = float(TJ0[tjmax_conv] - TJ60[tjmax_conv])
    f["heatsink_max"] = float(HS0.max())
    # derating and balance
    f["derate_min"] = float(DR0.min())
    f["derate_deficit_sum"] = float((1.0 - DR0).sum())
    f["gei_abs_err_max"] = float(np.abs(1.0 - G0).max())
    return f


def decide_moment(det: dict, bis: dict | None):
    """Returns (t_star, label, label_source, shed_buses, n_trig_at_tstar) or (None, reason)."""
    trig = {b: r["trigger_time"] for b, r in det["buses"].items() if r.get("trigger_time") is not None}
    if bis is not None and bis.get("status") == "safe_found":
        shed = bis["shed_buses"]
        if isinstance(shed, str):
            shed = [shed]
        tt = bis["trigger_time_by_bus"]
        tt = [tt] if not isinstance(tt, list) else tt
        t_star = float(min(tt))
        n_trig = sum(1 for t in trig.values() if t <= t_star + 1e-9)
        n_trig = max(n_trig, 1)
        return t_star, float(bis["best_shedfrac"]), "bisection", shed, n_trig
    if bis is not None:
        return None, f"bisect_status={bis.get('status')}"
    # no bisection file: mild scenario
    if trig:
        t_star = float(min(trig.values()))
        n_trig = sum(1 for t in trig.values() if t <= t_star + 1e-9)
        return t_star, 1.0, "mild_false_alarm_no_shed", [], n_trig
    return None, "mild_no_detector_trigger"


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
        feats = state_features(cols, i, n_trig, t_star)

        row = {"scenario_id": sid, "kind": r["kind"], "split": r["split"],
               "label": label, "label_source": src, "t_star": t_star,
               "n_shed_buses": len(shed), "shed_buses": ";".join(shed)}
        row.update({k: v for k, v in feats.items() if k != "t_star"})
        rows.append(row)
        print(f"[{sid}] {r['kind']:7s} split={r['split']:5s} t*={t_star:8.2f}  label={label:.2f} ({src})")

    out = pd.DataFrame(rows)
    feat_cols = [c for c in out.columns if c not in META_COLS]
    bad = out[feat_cols].isna().any(axis=1)
    if bad.any():
        print("WARNING: NaN features in", out.loc[bad, "scenario_id"].tolist())
    out_p = out_root / "nn_table.csv"
    out.to_csv(out_p, index=False)
    pd.DataFrame(skipped, columns=["scenario_id", "reason"]).to_csv(out_root / "nn_table_skipped.csv", index=False)
    print(f"\nWrote {out_p}: {len(out)} rows x {len(feat_cols)} features "
          f"(train {int((out.split=='train').sum())}, test {int((out.split=='test').sum())}); "
          f"skipped {len(skipped)}")
    for s in skipped:
        print("  skipped", s)


if __name__ == "__main__":
    main()

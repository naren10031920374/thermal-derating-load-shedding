"""
BUILD_CAUSAL_NN_TABLE.PY  (causal cap network, step 1: the stage-by-stage training table)
================================================================================
Why this exists
---------------
nn_table_v2.csv has ONE row per scenario, taken at the first alarm (t*). At that
moment exactly one bus is flagged in every collapse scenario, but the FINAL group
turns out to be 1, 2 or 3 buses. The earlier cap network was given the final group
size, which nobody knows at the first alarm. So it was not causal.

The staged limiter (run_staged_limiter.m, STAGED=1) already works stage by stage:
every time the detector flags another shed bus, the cap changes to the one for
"k buses flagged so far, E/F among them or not". This script builds the matching
training table: ONE ROW PER ALARM STAGE.

For every collapse scenario (bisection status safe_found, label < 0.95):
  - the shed buses are sorted by detector trigger time  t_1 <= t_2 <= ... <= t_n
  - stage k (k = 1..n) is the moment t_k, when k buses have been flagged
  - the row holds only what is known at t_k:
        n_flagged            k
        n_ef_flagged         how many of the k flagged buses are E or F
        has_ef_flagged       1 if n_ef_flagged > 0
        since_prev_alarm_s   t_k - t_(k-1)   (0 for the first alarm)
        + the grid measurements at t_k: the same 24 v1 features and 12 "how fast
          is it changing" features as nn_table_v2.csv (trailing 5 s means,
          nothing from after t_k). The old column n_triggers_by_tstar is dropped
          because n_flagged replaces it.
  - the TARGET is kept_kw = label x plateau (kW kept per shed bus after the
    minimum safe cut of the FINAL group), plus the final-group facts as META
    columns (final_n, final_has_ef, is_final_stage). Those META columns must
    NEVER be used as inputs; they only say which rows may be used for training:

        stage k of a scenario whose final group has k buses  (is_final_stage = 1)
        is the only row whose target is really the cap for "k buses".
        Rows with k < final_n (is_final_stage = 0) have no true target for "k
        buses" (the bisection only measured the final group). They are kept for
        evaluation: they show what the cap does while the group is still growing.

Mild false-alarm scenarios (no shed needed) are not included here.

The feature code is IMPORTED from build_nn_table.py / build_nn_table_v2.py, so the
numbers are computed by exactly the same functions as before.

Run (on the cluster, inside the venv; reads the big parquet files):
    python build_causal_nn_table.py
    python build_causal_nn_table.py --scenarios S001 S012 H01
Writes  <out_root>/nn_stage_table.csv  and  <out_root>/nn_stage_table_skipped.csv
(it does not touch nn_table.csv / nn_table_v2.csv).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from build_nn_table import RAW_COLS, HERE, PROJECT_ROOT, state_features
from build_nn_table_v2 import extra_features, NEW_FEATURES

MAX_LABEL = 0.95          # same rule as make_staged_caps.py: above this no cut was needed
TRIG_TOL_S = 0.05         # warn if detector json and bisection json disagree by more than this

META_COLS = ["scenario_id", "kind", "split", "stage", "final_n", "final_has_ef",
             "is_final_stage", "label", "plateau_kw", "kept_kw", "t_stage",
             "flagged_buses", "shed_buses_final"]
GROUP_COLS = ["n_flagged", "n_ef_flagged", "has_ef_flagged", "since_prev_alarm_s"]


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
        bis_p = d / f"bisect_{sid}_result.json"
        if not (det_p.exists() and sig_p.exists() and bis_p.exists()):
            skipped.append((sid, "missing detector json, signal parquet or bisection json"))
            continue
        det = json.loads(det_p.read_text())
        bis = json.loads(bis_p.read_text())
        if bis.get("status") != "safe_found":
            skipped.append((sid, f"bisect_status={bis.get('status')}"))
            continue
        label = float(bis["best_shedfrac"])
        if label >= MAX_LABEL:
            skipped.append((sid, f"label {label} >= {MAX_LABEL}: no cut needed"))
            continue

        shed = bis["shed_buses"]
        shed = [shed] if isinstance(shed, str) else list(shed)
        tt_bis = bis["trigger_time_by_bus"]
        tt_bis = [tt_bis] if not isinstance(tt_bis, list) else list(tt_bis)

        # trigger time per shed bus: detector json first, bisection json as fallback
        trig = {}
        for b, tb in zip(shed, tt_bis):
            td = det.get("buses", {}).get(b, {}).get("trigger_time")
            t_use = td if td is not None else tb
            if td is not None and tb is not None and abs(float(td) - float(tb)) > TRIG_TOL_S:
                print(f"WARNING [{sid}] bus {b}: detector {td} vs bisection {tb}")
            trig[b] = None if t_use is None else float(t_use)
        if any(v is None for v in trig.values()):
            skipped.append((sid, "a shed bus has no trigger time"))
            continue
        order = sorted(shed, key=lambda b: trig[b])
        n_final = len(order)
        final_ef = int(any(b in ("E", "F") for b in order))
        plateau_kw = float(r["plateau_w"]) / 1000.0
        kept_kw = label * plateau_kw

        df = pd.read_parquet(sig_p, columns=RAW_COLS)
        t = df["time"].to_numpy()
        cols = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS if c != "time"}

        prev_t = None
        for k in range(1, n_final + 1):
            t_k = trig[order[k - 1]]
            flagged = order[:k]
            n_ef = sum(b in ("E", "F") for b in flagged)
            i = int(np.searchsorted(t, t_k))
            feats = state_features(cols, i, k, t_k)             # 24 v1 features + t_star + n_triggers
            feats.update(extra_features(cols, i))               # + 12 new
            feats.pop("t_star", None)
            feats.pop("n_triggers_by_tstar", None)              # replaced by n_flagged
            row = {
                "scenario_id": sid, "kind": r["kind"], "split": r["split"],
                "stage": k, "final_n": n_final, "final_has_ef": final_ef,
                "is_final_stage": int(k == n_final),
                "label": label, "plateau_kw": plateau_kw, "kept_kw": kept_kw,
                "t_stage": t_k, "flagged_buses": ";".join(flagged),
                "shed_buses_final": ";".join(order),
                "n_flagged": k, "n_ef_flagged": n_ef, "has_ef_flagged": int(n_ef > 0),
                "since_prev_alarm_s": 0.0 if prev_t is None else float(t_k - prev_t),
            }
            row.update(feats)
            rows.append(row)
            prev_t = t_k
        print(f"[{sid}] {r['kind']:10s} split={r['split']:7s} final_n={n_final} "
              f"label={label:.2f} kept={kept_kw:6.1f} kW  trig={[round(trig[b], 1) for b in order]}")

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise SystemExit("no rows written: check that bisection results exist")
    snap_cols = [c for c in out.columns if c not in META_COLS and c not in GROUP_COLS]
    bad = out[GROUP_COLS + snap_cols].isna().any(axis=1)
    if bad.any():
        print("WARNING: NaN features in", sorted(set(out.loc[bad, "scenario_id"])))
    out = out[META_COLS + GROUP_COLS + snap_cols]
    out_p = out_root / "nn_stage_table.csv"
    out.to_csv(out_p, index=False)
    pd.DataFrame(skipped, columns=["scenario_id", "reason"]).to_csv(
        out_root / "nn_stage_table_skipped.csv", index=False)

    print(f"\nWrote {out_p}: {len(out)} stage rows from {out['scenario_id'].nunique()} scenarios; "
          f"{len(GROUP_COLS)} group/timing inputs + {len(snap_cols)} snapshot inputs "
          f"({len(snap_cols) - len(NEW_FEATURES)} v1 + {len(NEW_FEATURES)} new); skipped {len(skipped)}")
    print("\nrows by split / stage / final group size (is_final_stage rows are the training rows):")
    print(out.groupby(["split", "stage", "final_n"]).size().to_string())
    for s in skipped:
        print("  skipped", s)


if __name__ == "__main__":
    main()

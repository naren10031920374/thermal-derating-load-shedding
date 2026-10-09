"""
MAKE_STAGED_CAPS.PY   (Option B, step 1: the "load limiter" caps)
================================================================================
Option B = staged shedding. Instead of one fixed cut at the trigger time, each
overloaded bus is held at a power CAP (kW). Load below the cap is untouched; as the
load climbs past the cap it is cut back to the cap. So the cut grows by itself as the
load rises and nobody has to predict the final load height.

Where the caps come from (data we already have):
    kept_load = label x plateau            (kW kept on a bus after the minimum safe cut)
Across the collapse scenarios this number is almost the same for every scenario with the
same bus group, whatever the plateau was (std about 3 kW; correlation with plateau ~ 0):

    1 bus,  no E/F : ~129 kW        2 buses, no E/F : ~110 kW        3 buses, no E/F : ~102 kW
    1 bus,  E or F : ~104 kW        2 buses, E/F    : ~84 kW         3 buses, E/F    : ~76 kW

This script builds that lookup from the TRAIN rows only (so the 10 test scenarios are a
fair check), subtracts a safety margin (default 5 kW), and writes staged_caps.csv with one
row per scenario in scenario_table.csv. run_staged_limiter.m reads that file.

Run (from scripts/09_dataset_pipeline):
    python make_staged_caps.py --nn-table path/to/nn_table_v2.csv
Options:
    --scenarios scenario_table.csv   --out staged_caps.csv   --margin-kw 5
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-table", required=True, help="nn_table_v2.csv (has label and split)")
    ap.add_argument("--scenarios", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--out", default=str(HERE / "staged_caps.csv"))
    ap.add_argument("--margin-kw", type=float, default=5.0)
    ap.add_argument("--max-label", type=float, default=0.95,
                    help="rows with label above this did not need a cut and are ignored")
    a = ap.parse_args()

    sc = pd.read_csv(a.scenarios)
    nn = pd.read_csv(a.nn_table)[["scenario_id", "label", "split"]].rename(columns={"split": "nn_split"})
    d = sc.merge(nn, on="scenario_id", how="left")
    d["n_ef"] = d["targets"].map(lambda s: sum(b in ("E", "F") for b in str(s).split(";")))
    d["has_ef"] = (d["n_ef"] > 0).astype(int)
    d["plateau_kw"] = d["plateau_w"] / 1000.0
    d["kept_kw"] = d["label"] * d["plateau_kw"]

    fit = d[(d["nn_split"] == "train") & (d["label"] < a.max_label) & d["kept_kw"].notna()]
    table = fit.groupby(["n_target", "has_ef"])["kept_kw"].agg(["mean", "std", "count"]).round(2)
    print("Cap lookup built from", len(fit), "train collapse scenarios (kW kept per shed bus):")
    print(table.to_string())

    cap_mean = fit.groupby(["n_target", "has_ef"])["kept_kw"].mean()
    # fall back to a straight-line fit for any (n, has_ef) group with no training rows
    X = np.c_[np.ones(len(fit)), fit["n_target"], fit["has_ef"]]
    beta = np.linalg.lstsq(X, fit["kept_kw"].to_numpy(), rcond=None)[0]

    def cap_for(n, ef):
        if (n, ef) in cap_mean.index:
            return float(cap_mean[(n, ef)])
        return float(beta[0] + beta[1] * n + beta[2] * ef)

    d["cap_kw"] = [cap_for(n, e) for n, e in zip(d["n_target"], d["has_ef"])]
    d["cap_kw_margin"] = d["cap_kw"] - a.margin_kw
    out = d[["scenario_id", "kind", "split", "targets", "n_target", "n_ef", "has_ef",
             "plateau_w", "label", "kept_kw", "cap_kw", "cap_kw_margin"]].copy()
    out.round(3).to_csv(a.out, index=False)
    print(f"\nWrote {a.out} ({len(out)} rows). Margin used: {a.margin_kw} kW.")
    t = out[out["split"] == "test"]
    print("\nTest scenarios:")
    print(t[["scenario_id", "targets", "plateau_w", "label", "kept_kw", "cap_kw_margin"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()


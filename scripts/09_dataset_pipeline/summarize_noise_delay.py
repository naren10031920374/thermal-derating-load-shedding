#!/usr/bin/env python3
"""Summarise the sensor noise / delay test (standard library only).

Reads, for every scenario and setting, <root>/<ID>/staged_<ID>_<tag>_result.json and
..._iterations.json written by run_staged_limiter.m, and prints

  1. stability: does the limiter still keep all 10 buses alive at the FIRST cap, or only after
     retries with a lower cap (offset in kW), or never;
  2. energy shed in kWh and its change against the perfect-measurement run (tag 'staged');
  3. the lowest bus voltage reached.

Usage (from the repo root, on the cluster or the laptop):
    python3 scripts/09_dataset_pipeline/summarize_noise_delay.py [--root model_outputs/dataset_pipeline]
Writes noise_delay_summary.csv in the current folder.
"""
import argparse
import csv
import json
import math
import os

SCEN = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035"]
SETTINGS = [  # tag, label
    ("staged", "perfect"),
    ("nd_d30", "delay 30 s"),
    ("nd_d60", "delay 60 s"),
    ("nd_d120", "delay 120 s"),
    ("nd_n5", "noise 5%"),
    ("nd_n10", "noise 10%"),
    ("nd_n5d30", "noise 5% + delay 30 s"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def clean(x):
    if x is None:
        return math.nan
    return float(x)


def one(root, sid, tag):
    res = load(os.path.join(root, sid, f"staged_{sid}_{tag}_result.json"))
    its = load(os.path.join(root, sid, f"staged_{sid}_{tag}_iterations.json"))
    if res is None or its is None:
        return None
    iters = its["iterations"]
    if isinstance(iters, dict):
        iters = [iters]
    # only the limiter runs (a validation run of the constant cut may also be stored)
    lim = [it for it in iters if str(it.get("label", "")).startswith(("staged", "limiter"))]
    iters = lim or iters
    status = res.get("status")
    if status == "no_collapse":
        return None
    # the iteration that decided the outcome: the first safe one, else the last one tried
    chosen = None
    for it in iters:
        if it.get("all_safe") in (True, 1):
            chosen = it
            break
    if chosen is None:
        chosen = iters[-1]
    min_v = [clean(v) for v in (chosen.get("min_v") or [])]
    min_v = [v for v in min_v if not math.isnan(v)]
    safe = chosen is not None and chosen.get("all_safe") in (True, 1)
    offset = res.get("first_safe_offset_kw") if status == "safe_found" else None
    return {
        "safe": bool(safe),
        "offset_kw": None if offset is None else float(offset),
        "n_sims": len(iters),
        "shed_kwh": clean(chosen.get("shed_kwh")),
        "min_v": min(min_v) if min_v else math.nan,
    }


def verdict(r):
    if r is None:
        return "-"
    if not r["safe"]:
        return "FAIL"
    if abs(r["offset_kw"] or 0.0) < 1e-9:
        return "OK"
    return f"retry {r['offset_kw']:+.0f} kW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("model_outputs", "dataset_pipeline"))
    ap.add_argument("--out", default="noise_delay_summary.csv")
    a = ap.parse_args()

    data = {(s, t): one(a.root, s, t) for s in SCEN for t, _ in SETTINGS}

    def table(title, cell, width=22):
        print("\n" + title)
        print("scenario " + "".join(f"{lab:<{width}}" for _, lab in SETTINGS))
        for s in SCEN:
            print(f"{s:<9}" + "".join(f"{cell(data[(s, t)], s, t):<{width}}" for t, _ in SETTINGS))

    table("1. STABILITY (OK = all 10 buses survive at the first cap; retry = only after lowering the cap; FAIL = never)",
          lambda r, s, t: verdict(r))

    def energy(r, s, t):
        if r is None or not r["safe"]:
            return "-"
        base = data[(s, "staged")]
        txt = f"{r['shed_kwh']:.1f}"
        if base is not None and base["safe"] and t != "staged":
            txt += f" ({100 * (r['shed_kwh'] / base['shed_kwh'] - 1):+.1f}%)"
        return txt

    table("2. ENERGY SHED in kWh of the safe run (change against perfect measurement in brackets)", energy)
    table("3. LOWEST BUS VOLTAGE (V) of the deciding run (collapse = below 100 V)",
          lambda r, s, t: "-" if r is None or math.isnan(r["min_v"]) else f"{r['min_v']:.0f}")

    print("\nTOTALS")
    print(f"{'setting':<26}{'stable at 1st cap':<20}{'stable after retry':<20}{'fail':<8}{'total shed kWh (safe runs)'}")
    rows = []
    for t, lab in SETTINGS:
        rs = [data[(s, t)] for s in SCEN]
        have = [r for r in rs if r is not None]
        first = sum(1 for r in have if r["safe"] and abs(r["offset_kw"] or 0.0) < 1e-9)
        retry = sum(1 for r in have if r["safe"] and abs(r["offset_kw"] or 0.0) >= 1e-9)
        fail = sum(1 for r in have if not r["safe"])
        kwh = sum(r["shed_kwh"] for r in have if r["safe"])
        print(f"{lab:<26}{str(first) + '/' + str(len(have)):<20}{retry:<20}{fail:<8}{kwh:.1f}")
        rows.append([lab, len(have), first, retry, fail, round(kwh, 1)])

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario"] + [f"{lab} | {k}" for _, lab in SETTINGS for k in ("verdict", "shed_kwh", "min_v")])
        for s in SCEN:
            row = [s]
            for t, _ in SETTINGS:
                r = data[(s, t)]
                row += [verdict(r), "" if r is None else round(r["shed_kwh"], 2), "" if r is None else round(r["min_v"], 1)]
            w.writerow(row)
        w.writerow([])
        w.writerow(["setting", "scenarios found", "stable at first cap", "stable after retry", "fail", "total shed kWh"])
        w.writerows(rows)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

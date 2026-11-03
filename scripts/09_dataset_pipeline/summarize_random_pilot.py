"""
SUMMARIZE_RANDOM_PILOT.PY  (random scenarios for the probabilistic cap network, Network B)
================================================================================
Reads the staged-limiter results written by submit_random_pilot.sh (tag rnd_g10_w1s, or
rnd_ideal) for the random scenarios (RT / RC / RF / RZ) and shows, for each scenario:

  status     no_collapse   nothing collapses without shedding -> replace by the next RZ reserve of that set
             safe_found    a cap that keeps all 10 buses alive was found
             no_safe       even the lowest cap tried (-10 kW) failed
             pending       no result file yet
  offset     largest safe offset (kW added to every network cap). margin needed = 5 - offset.
             Offset +5 = margin 0 (the caps sit at the network's own prediction).
  bracket    how well the true limit is known (what Network B needs to learn from):
               bracketed      a cap that collapsed AND the next lower cap that survived were both run
                              -> the limit sits in a known 5 kW step.   GOOD LABEL
               censored_high  the first cap tried (+15) was already safe -> the limit is above +15 (ladder too low)
               none_safe      all caps collapsed -> the limit is below the lowest cap tried (ladder too high)

It also writes random_labels_<tag>.csv: one row per simulation that was RUN (scenario, cap, offset,
survived 1/0), plus the scenario settings. Caps that were not run (lower than the first safe one)
are not guessed; if you want them as "survived", that is an assumption (a lower cap never hurts)
and is made later in the training script, not here.

Run (project root, on the cluster after  module load python/3.11.5):
    python3 scripts/09_dataset_pipeline/summarize_random_pilot.py                  # the 30 pilot scenarios
    python3 scripts/09_dataset_pipeline/summarize_random_pilot.py --set rtrain     # a whole set (rtrain, rcheck, rtest, all)
    python3 scripts/09_dataset_pipeline/summarize_random_pilot.py --tag rnd_ideal  # another sensor setting
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PRESENT = 5.0                       # offset that equals margin 0


def listify(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARNING: could not read {p}: {e}")
        return None


def classify(res):
    """-> dict(status, offset, bracket, runs=[(offset, cap, safe)])"""
    if res is None:
        return dict(status="pending", offset=None, bracket="", runs=[])
    st = res.get("status")
    if st == "no_collapse":
        return dict(status="no_collapse", offset=None, bracket="", runs=[])
    base = res.get("base_cap_kw")
    caps = [float(c) for c in listify(res.get("cap_tried_kw"))]
    safe = [bool(s) for s in listify(res.get("cap_safe"))]
    runs = [(c - base, c, s) for c, s in zip(caps, safe)] if base is not None else []
    if st == "safe_found":
        off = float(res["first_safe_offset_kw"])
        br = "bracketed" if len(runs) >= 2 else "censored_high"
        return dict(status="safe_found", offset=off, bracket=br, runs=runs)
    return dict(status="no_safe", offset=None, bracket="none_safe", runs=runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="rnd_g10_w1s")
    ap.add_argument("--set", default="pilot", help="pilot (RT001-RT030), rtrain, rcheck, rtest or all")
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--out", default=None, help="labels csv (default: random_labels_<tag>.csv in the current folder)")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT") or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")

    with open(a.table, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scenario_id"][:2] in ("RT", "RC", "RF", "RZ")]
    if a.set == "pilot":
        rows = [r for r in rows if r["scenario_id"].startswith("RT") and int(r["scenario_id"][2:]) <= 30]
    elif a.set != "all":
        rows = [r for r in rows if r["split"] == a.set]
    if not rows:
        raise SystemExit("No random scenarios found in the table (git pull?).")

    info = {}
    for r in rows:
        sid = r["scenario_id"]
        res = load(root / sid / f"staged_{sid}_{a.tag}_result.json")
        c = classify(res)
        c["row"] = r
        c["buses"] = listify(res.get("shed_buses")) if res else []
        info[sid] = c

    print(f"Root: {root}\nTag: {a.tag}   set: {a.set}   scenarios: {len(rows)}\n")
    print("A. PER SCENARIO   (offset = largest safe offset in kW; margin = 5 - offset)")
    hdr = f"{'id':7s}{'kind':8s}{'targets':8s}{'plat kW':>8s}  {'status':12s}{'offset':>7s}{'margin':>7s}  {'bracket':14s}tried (offset:S=survived,C=collapsed)"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        c = info[r["scenario_id"]]
        off = "" if c["offset"] is None else f"{c['offset']:+.0f}"
        mar = "" if c["offset"] is None else f"{PRESENT - c['offset']:+.0f}"
        tried = " ".join(f"{o:+.0f}:{'S' if s else 'C'}" for o, _, s in c["runs"])
        print(f"{r['scenario_id']:7s}{r['kind']:8s}{r['targets']:8s}{int(r['plateau_w'])/1000:>8.0f}  "
              f"{c['status']:12s}{off:>7s}{mar:>7s}  {c['bracket']:14s}{tried}")

    cnt = Counter(c["status"] for c in info.values())
    br = Counter(c["bracket"] for c in info.values() if c["bracket"])
    print("\nB. SUMMARY")
    print(f"  done: {len(rows) - cnt['pending']} of {len(rows)}   pending: {cnt['pending']}")
    print(f"  no_collapse (replace by a reserve): {cnt['no_collapse']}")
    print(f"  safe_found: {cnt['safe_found']}   no_safe: {cnt['no_safe']}")
    print(f"  bracketed (good labels): {br['bracketed']}   censored_high (limit above +15): {br['censored_high']}   none_safe: {br['none_safe']}")
    offs = Counter(int(c["offset"]) for c in info.values() if c["offset"] is not None)
    if offs:
        print("  largest safe offset -> number of scenarios:  " + "   ".join(f"{o:+d}: {offs[o]}" for o in sorted(offs, reverse=True)))
        need = [PRESENT - o for o in offs.elements()]
        print(f"  margin needed: worst {max(need):+.0f} kW, scenarios safe at margin 0 (offset >= +5): {sum(v for o, v in offs.items() if o >= 5)} of {sum(offs.values())}")

    print("\nC. IS THE DATA USABLE FOR NETWORK B?")
    done_ok = cnt["safe_found"] + cnt["no_safe"]
    if done_ok == 0:
        print("  No finished sweeps yet.")
    else:
        if br["censored_high"] > 0.2 * done_ok:
            print("  MANY scenarios were safe at the very first cap (+15): the limit is above the ladder. Add higher offsets (e.g. 30:25:20:15:...) for those rows.")
        if br["none_safe"] > 0.1 * done_ok:
            print("  SOME scenarios collapsed at every cap: the ladder needs lower offsets (e.g. ...:-10:-15:-20) for those rows.")
        if br["bracketed"] >= 0.7 * done_ok:
            print("  Most labels are bracketed: a collapsed and a survived cap were both run. Good for Network B.")
        if cnt["no_collapse"]:
            print(f"  {cnt['no_collapse']} scenario(s) did not collapse without shedding: use the next RZ reserve of the same set.")

    out = Path(a.out or f"random_labels_{a.tag}.csv")
    cols = ["scenario_id", "split", "kind", "targets", "n_target", "has_ef", "plateau_w", "ramp_start_s", "ramp_end_s",
            "boundary_low_w", "seed", "cap_kw", "offset_kw", "survived"]
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            c = info[r["scenario_id"]]
            has_ef = int(any(b in ("E", "F") for b in r["targets"].split(";")))
            for o, cap, s in c["runs"]:
                w.writerow([r["scenario_id"], r["split"], r["kind"], r["targets"], r["n_target"], has_ef, r["plateau_w"],
                            r["ramp_start_s"], r["ramp_end_s"], r["boundary_low_w"], r["seed"],
                            round(cap, 3), round(o, 3), int(s)])
                n += 1
    print(f"\nWrote {out}  ({n} simulation rows: one per cap that was run)")


if __name__ == "__main__":
    main()

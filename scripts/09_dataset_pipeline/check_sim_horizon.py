"""
CHECK_SIM_HORIZON.PY  (does a shorter simulation give the same safe / not-safe answer?)
================================================================================
Question (from the professor): does a candidate cap really need the full 5000 s
simulation? If every collapse happens well before 5000 s, a shorter run would
give the same answer and a search (bisection, PSO, GA, ...) would be cheaper.

What this script does
---------------------
It does NOT run any simulation. It re-reads the results of runs that already
finished and replays the question "what if the run had been stopped at time H?":

  * a run stopped at H gives the same answer as the full run, unless a bus
    collapses AFTER H. Then the short run would wrongly say "safe".
  * so for every candidate that FAILED in a full run we look at the time its
    first bus collapsed (the 'onset'), and count how many onsets are later than H.

Sources (all already on the cluster, all small text files):
  <root>/<ID>/bisect_<ID>_iterations.json         every cut tried by the bisection
  <root>/<ID>/staged_<ID>[_<TAG>]_iterations.json  every cap tried by the staged limiter
  <root>/<ID>/noshed_<ID>_collapse.csv             collapse time with no limit at all
  scripts/09_dataset_pipeline/scenario_table.csv   ramp start / end of each scenario

Run (login node is fine, it only reads small files, a few seconds):
    python scripts/09_dataset_pipeline/check_sim_horizon.py
Writes  <root>/sim_horizon_failing_candidates.csv  and prints the summary.
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
TEND_S = 5000.0

ABS_HORIZONS = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
RAMP_END_PLUS = [0, 100, 200, 400, 600, 800, 1000, 1500, 2000]
CUT_PLUS = [120, 400, 800, 1200, 1600, 2000, 2500, 3000]


def num(x):
    """float or None (None for null, '', 'NaN', non-numbers)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def first_of(x):
    """Smallest number inside x (a number, a list of numbers, or empty)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        vals = [num(v) for v in x]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None
    return num(x)


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def fmt(v, nd=0):
    return "n/a" if v is None else f"{v:.{nd}f}"


def load_table(path: Path):
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["scenario_id"]] = {"ramp_start": num(r.get("ramp_start_s")),
                                     "ramp_end": num(r.get("ramp_end_s"))}
    return out


def noshed_onset(d: Path, sid: str):
    p = d / f"noshed_{sid}_collapse.csv"
    if not p.exists():
        return None
    best = None
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("collapsed", "0")).strip() in ("1", "1.0") and \
               str(r.get("is_target", "1")).strip() in ("1", "1.0"):
                v = num(r.get("onset_s"))
                if v is not None and (best is None or v < best):
                    best = v
    return best


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:                      # keep going, report at the end
        print(f"WARNING: could not read {p}: {e}")
        return None


def collect(root: Path, table):
    recs, problems = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = d.name
        files = [("bisect", f) for f in sorted(d.glob("bisect_*_iterations.json"))]
        files += [("staged", f) for f in sorted(d.glob("staged_*_iterations.json"))]
        if not files:
            continue
        ons_noshed = noshed_onset(d, sid)
        ramp = table.get(sid, {})
        for kind, f in files:
            data = read_json(f)
            if not data:
                problems.append(str(f))
                continue
            if kind == "staged":
                tag = f.name[len(f"staged_{sid}"):-len("_iterations.json")].lstrip("_")
                source = "staged" + (":" + tag if tag else "")
            else:
                source = "bisect"
            trig = first_of(data.get("trigger_time_by_bus"))
            for it in data.get("iterations", []):
                label = it.get("shedfrac", it.get("label"))
                cut = num(it.get("first_cut_s")) if kind == "staged" else None
                if cut is None:
                    cut = trig
                recs.append({
                    "scenario": sid, "source": source, "label": label,
                    "safe": bool(it.get("all_safe")),
                    "onset": first_of(it.get("onsets")),
                    "cut_start": cut,
                    "ramp_end": ramp.get("ramp_end"),
                    "noshed_onset": ons_noshed,
                    "wall_s": num(it.get("wall_clock_s")),
                })
    return recs, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset_pipeline folder")
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    table = load_table(Path(a.table))
    recs, problems = collect(root, table)

    fails = [r for r in recs if (not r["safe"]) and r["onset"] is not None]
    fail_no_onset = [r for r in recs if (not r["safe"]) and r["onset"] is None]
    safes = [r for r in recs if r["safe"]]
    print(f"Root: {root}")
    print(f"Scenarios with results: {len({r['scenario'] for r in recs})}")
    print(f"Candidates (each = one full-length simulation): {len(recs)}  "
          f"safe {len(safes)} | failed {len(fails)} | failed without an onset time {len(fail_no_onset)}")
    for s in sorted({r['source'] for r in recs}):
        sub = [r for r in recs if r['source'] == s]
        print(f"   {s:<22} {len(sub):5d} runs, {sum(1 for r in sub if not r['safe']):4d} failed")

    walls = sorted(r["wall_s"] for r in recs if r["wall_s"] is not None)
    if walls:
        print(f"\nWall-clock time of one full simulation (min): "
              f"min {walls[0]/60:.1f} | median {pct(walls,50)/60:.1f} | 90th pct {pct(walls,90)/60:.1f} | max {walls[-1]/60:.1f}")

    if not fails:
        print("\nNo failed candidates with an onset time were found, nothing to analyse.")
        return

    # --- when do the failing candidates collapse? ---
    on = sorted(r["onset"] for r in fails)
    lag_cut = sorted(r["onset"] - r["cut_start"] for r in fails if r["cut_start"] is not None)
    lag_ramp = sorted(r["onset"] - r["ramp_end"] for r in fails if r["ramp_end"] is not None)
    lag_ns = sorted(r["onset"] - r["noshed_onset"] for r in fails if r["noshed_onset"] is not None)
    print("\nWHEN DO FAILING CANDIDATES COLLAPSE? (seconds of simulated time)")
    print(f"{'':44s}{'n':>5s}{'min':>8s}{'median':>8s}{'90th':>8s}{'max':>8s}")
    for name, v in [("collapse time (absolute)", on),
                    ("... after the cut / first alarm", lag_cut),
                    ("... after the end of the load ramp", lag_ramp),
                    ("... after the no-limit collapse time", lag_ns)]:
        print(f"{name:44s}{len(v):5d}{fmt(v[0] if v else None):>8s}{fmt(pct(v,50)):>8s}{fmt(pct(v,90)):>8s}{fmt(v[-1] if v else None):>8s}")
    print("(a negative 'after the end of the ramp' value = it collapsed before the ramp ended)")

    # --- the replay: what would a shorter run get wrong? ---
    n = len(fails)
    print(f"\nREPLAY: a run stopped at time H wrongly says 'safe' for every failed candidate whose collapse is after H.")
    print(f"Failed candidates in total: {n}")
    print("\n(a) stop at a fixed time H")
    print(f"{'H (s)':>8s}{'missed':>8s}{'% of failed':>13s}")
    for H in ABS_HORIZONS:
        m = sum(1 for r in fails if r["onset"] > H)
        print(f"{H:8d}{m:8d}{100.0*m/n:12.1f}%")
    withramp = [r for r in fails if r["ramp_end"] is not None]
    print(f"\n(b) stop at (end of the load ramp + X)   [{len(withramp)} failed candidates have a ramp end]")
    print(f"{'X (s)':>8s}{'missed':>8s}{'% of failed':>13s}")
    for X in RAMP_END_PLUS:
        m = sum(1 for r in withramp if r["onset"] > r["ramp_end"] + X)
        print(f"{X:8d}{m:8d}{100.0*m/max(len(withramp),1):12.1f}%")
    withcut = [r for r in fails if r["cut_start"] is not None]
    print(f"\n(c) stop at (first cut + X)   [{len(withcut)} failed candidates have a cut time]")
    print(f"{'X (s)':>8s}{'missed':>8s}{'% of failed':>13s}")
    for X in CUT_PLUS:
        m = sum(1 for r in withcut if r["onset"] > r["cut_start"] + X)
        print(f"{X:8d}{m:8d}{100.0*m/max(len(withcut),1):12.1f}%")

    print("\nLATEST 8 COLLAPSES AMONG FAILED CANDIDATES (these decide how short a run can be)")
    for r in sorted(fails, key=lambda r: -r["onset"])[:8]:
        print(f"  {r['scenario']:5s} {r['source']:<18s} label {r['label']}  collapse at {r['onset']:.0f} s"
              f" | cut at {fmt(r['cut_start'])} s | ramp ends {fmt(r['ramp_end'])} s")

    H = min(TEND_S, math.ceil((on[-1] + 500.0) / 100.0) * 100.0)
    saving = 1.0 - H / TEND_S
    print(f"\nREADING: the latest collapse in the data is at {on[-1]:.0f} s. With a 500 s safety allowance a run to "
          f"about {H:.0f} s would have reached the same verdict for all {n} failed candidates "
          f"({100*saving:.0f}% less simulated time than {TEND_S:.0f} s).")
    if walls:
        print(f"If wall-clock time scales with simulated time, one run would take about "
              f"{pct(walls,50)/60*H/TEND_S:.1f} min instead of {pct(walls,50)/60:.1f} min. "
              f"This is an estimate from past runs, not a promise: the data only holds cuts the bisection and the limiter tried.")

    out = root / "sim_horizon_failing_candidates.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "source", "label", "collapse_onset_s", "cut_start_s", "ramp_end_s",
                    "noshed_onset_s", "onset_minus_cut_s", "onset_minus_ramp_end_s", "wall_clock_s"])
        for r in sorted(fails, key=lambda r: (r["scenario"], r["source"], str(r["label"]))):
            w.writerow([r["scenario"], r["source"], r["label"], r["onset"], r["cut_start"], r["ramp_end"],
                        r["noshed_onset"],
                        None if r["cut_start"] is None else r["onset"] - r["cut_start"],
                        None if r["ramp_end"] is None else r["onset"] - r["ramp_end"], r["wall_s"]])
    print(f"\nWrote {out} ({len(fails)} rows)")
    if problems:
        print("Files that could not be read:", *problems, sep="\n  ")


if __name__ == "__main__":
    main()

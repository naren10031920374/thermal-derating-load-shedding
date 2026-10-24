"""
SUMMARIZE_SLACK_MARGIN2.PY  (margin fit: noise trend and second noise seed)
================================================================================
Offset = kW added to ALL six caps; the caps contain a 5 kW margin below the cell mean, so the margin a
scenario REQUIRES (below the cell mean) is 5 - largest_safe_offset (kW). 5 kW offset grid; 'none' = a lower
bound (shown with +).

A  Training scenarios (18, 3 per cap cell): required margin for every sensor setting.
B  Per setting, fitted on training only: fixed margin, proportional factor r (margin = r x cell cap), per cell.
C  Each rule checked once on the 12 held-out/example scenarios (safe if rule margin >= required margin).
D  Noise trend: r against the noise level (10 s average, seed 1), and the interpolation test:
   r at 5 % predicted from the 2 % and 10 % fits, checked on the 5 % held-out results.
E  Seed check: rules fitted on training seed 1 applied to held-out seed 2, and the reverse.
Run:  python3 scripts/09_dataset_pipeline/summarize_slack_margin2.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, csv, json, os, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CUR = 5.0
TRAIN = ["S003", "S006", "S030", "S010", "S031", "T024", "S002", "S011", "S027", "S009", "S026", "T004",
         "S001", "S016", "T008", "S023", "T005", "T048"]
TEST = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035", "H01", "H02", "H03"]
# label, noise %, train tag, test tag
SET = [("ideal", 0, "slacktr_ideal", "slackb"),
       ("2%+avg", 2, "slacktr_n2f", "slack_n2f"),
       ("5%+avg", 5, "slacktr_n5f", "slack_n5f"),
       ("10%+avg", 10, "slacktr_n10f", "slack_n10f"),
       ("10%+avg s2", 10, "slacktr_n10f_s2", "slack_n10f_s2")]
CELLNAME = {(1, 0): "1 bus", (1, 1): "1 bus E/F", (2, 0): "2 bus", (2, 1): "2 bus E/F", (3, 0): "3 bus", (3, 1): "3 bus E/F"}


def load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARNING: could not read {p}: {e}")
        return None


def listify(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def req(root, sid, tag):
    r = load(root / sid / f"staged_{sid}_{tag}_result.json")
    if r is None:
        return None
    if r.get("status") == "safe_found":
        return CUR - float(r["first_safe_offset_kw"]), False
    base, tried = r.get("base_cap_kw"), listify(r.get("cap_tried_kw"))
    if base is None or not tried:
        return None
    return CUR - (min(tried) - base), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--scripts", default=str(HERE))
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    sdir = Path(a.scripts)
    cell_of, cap = {}, {}
    with open(sdir / "staged_caps.csv", newline="") as f:
        for r in csv.DictReader(f):
            cell_of[r["scenario_id"]] = (int(r["n_target"]), int(r["has_ef"]))
    with open(sdir / "staged_cap_groups_nn.csv", newline="") as f:
        for r in csv.DictReader(f):
            cap[(int(r["n_target"]), int(r["has_ef"]))] = float(r["cap_kw"])

    tr = {lab: {s: req(root, s, t) for s in TRAIN} for lab, _, t, _ in SET}
    te = {lab: {s: req(root, s, h) for s in TEST} for lab, _, _, h in SET}

    print(f"Root: {root}\nRequired margin (kW below the cell mean cap) = 5 - largest safe offset. '+' after a number = lower bound.\n")
    print("A. TRAINING SCENARIOS: required margin (kW)")
    hdr = f"{'scenario':9s}{'cell':>11s}" + "".join(f"{lab:>12s}" for lab, *_ in SET)
    print(hdr); print("-" * len(hdr))
    for s in TRAIN:
        row = f"{s:9s}{CELLNAME.get(cell_of.get(s), '?'):>11s}"
        for lab, *_ in SET:
            v = tr[lab][s]
            row += f"{('--' if v is None else f'{v[0]:+.0f}' + ('+' if v[1] else '')):>12s}"
        print(row)

    def fit(table):
        got = {s: v for s, v in table.items() if v is not None and cell_of.get(s) in cap}
        if not got:
            return None
        fixed = max(v[0] for v in got.values())
        pc = {}
        for s, v in got.items():
            pc[cell_of[s]] = max(pc.get(cell_of[s], -99.0), v[0])
        r = max(v[0] / cap[cell_of[s]] for s, v in got.items())
        return {"n": len(got), "fixed": fixed, "r": r, "pc": pc}

    def rules_from(f):
        return {"fixed": lambda c, m=f["fixed"]: m,
                "per cell": lambda c, d=f["pc"], m=f["fixed"]: d.get(c, m),
                "proportional": lambda c, r=f["r"]: r * cap[c]}

    def check(fn, table):
        got = {s: v for s, v in table.items() if v is not None and cell_of.get(s) in cap}
        ok, bad, ms = 0, [], []
        for s, v in got.items():
            m = fn(cell_of[s]); ms.append(m)
            if m + 1e-9 >= v[0]:
                ok += 1
            else:
                bad.append(f"{s}(needs {v[0]:+.0f}, has {m:+.1f})")
        return ok, len(got), (statistics.mean(ms) if ms else float('nan')), bad

    fits = {lab: fit(tr[lab]) for lab, *_ in SET}
    print("\nB. FITTED ON TRAINING ONLY")
    print(f"{'setting':12s}{'n':>3s}{'fixed kW':>10s}{'r (% of cap)':>14s}   per-cell margins (kW)")
    for lab, *_ in SET:
        f = fits[lab]
        if f is None:
            print(f"{lab:12s}  no training results yet"); continue
        pcs = "  ".join(f"{CELLNAME[c]}:{f['pc'][c]:+.0f}" for c in sorted(f['pc']))
        print(f"{lab:12s}{f['n']:>3d}{f['fixed']:>+10.1f}{100*f['r']:>+13.2f}%   {pcs}")

    print("\nC. EACH RULE CHECKED ON THE 12 HELD-OUT / EXAMPLE SCENARIOS (same setting)")
    print(f"{'setting':12s}{'rule':14s}{'safe':>7s}{'avg margin':>12s}  not safe")
    for lab, *_ in SET:
        if fits[lab] is None:
            continue
        for name, fn in rules_from(fits[lab]).items():
            ok, n, am, bad = check(fn, te[lab])
            if n:
                print(f"{lab:12s}{name:14s}{ok:>4d}/{n:<2d}{am:>+10.1f} kW  {', '.join(bad) if bad else '-'}")
        ok, n, am, bad = check(lambda c: CUR, te[lab])
        if n:
            print(f"{lab:12s}{'current 5 kW':14s}{ok:>4d}/{n:<2d}{CUR:>+10.1f} kW  {', '.join(b.split('(')[0] for b in bad) if bad else '-'}")

    print("\nD. NOISE TREND (10 s average, seed 1) and INTERPOLATION TEST")
    pts = [(n, fits[lab]) for lab, n, *_ in SET if lab in ("ideal", "2%+avg", "5%+avg", "10%+avg") and fits[lab]]
    for n, f in pts:
        print(f"   noise {n:>2d} %  ->  r = {100*f['r']:+.2f} % of the cap   (fixed margin {f['fixed']:+.1f} kW)")
    f2, f5, f10 = fits["2%+avg"], fits["5%+avg"], fits["10%+avg"]
    if f2 and f10:
        r_pred = f2["r"] + (f10["r"] - f2["r"]) * (5 - 2) / (10 - 2)
        print(f"   r at 5 % predicted by a straight line between 2 % and 10 %: {100*r_pred:+.2f} % of the cap"
              + (f"   (fitted directly: {100*f5['r']:+.2f} %)" if f5 else "   (no 5 % training results yet)"))
        ok, n, am, bad = check(lambda c: r_pred * cap[c], te["5%+avg"])
        if n:
            print(f"   predicted-r rule on the 5 % held-out results: safe {ok}/{n}, average margin {am:+.1f} kW  {', '.join(bad) if bad else ''}")
        else:
            print("   no 5 % held-out results yet")

    print("\nE. SEED CHECK (10 % + avg)")
    f1, fs2 = fits["10%+avg"], fits["10%+avg s2"]
    if f1 and fs2:
        pairs = [("fitted on seed 1 training -> seed 2 held-out", f1, "10%+avg s2"),
                 ("fitted on seed 2 training -> seed 1 held-out", fs2, "10%+avg")]
        for title, f, lab in pairs:
            print(f"   {title}")
            for name, fn in rules_from(f).items():
                ok, n, am, bad = check(fn, te[lab])
                if n:
                    print(f"      {name:14s} safe {ok}/{n}, average margin {am:+.1f} kW  {', '.join(bad) if bad else ''}")
        both = {s: (None if (tr['10%+avg'][s] is None or tr['10%+avg s2'][s] is None)
                    else (max(tr['10%+avg'][s][0], tr['10%+avg s2'][s][0]), False)) for s in TRAIN}
        fb = fit(both)
        if fb:
            print(f"   fitted on BOTH seeds (worst case per training scenario): fixed {fb['fixed']:+.1f} kW, r = {100*fb['r']:+.2f} % of the cap")
            for lab in ("10%+avg", "10%+avg s2"):
                for name, fn in rules_from(fb).items():
                    ok, n, am, bad = check(fn, te[lab])
                    if n:
                        print(f"      on held-out {lab:11s} {name:14s} safe {ok}/{n}, average margin {am:+.1f} kW  {', '.join(bad) if bad else ''}")
        print("   scenarios whose requirement changed between seeds (training):")
        ch = [f"{s}: {tr['10%+avg'][s][0]:+.0f} -> {tr['10%+avg s2'][s][0]:+.0f}" for s in TRAIN
              if tr['10%+avg'][s] and tr['10%+avg s2'][s] and tr['10%+avg'][s][0] != tr['10%+avg s2'][s][0]]
        print("      " + (", ".join(ch) if ch else "none"))
    else:
        print("   no seed-2 training results yet")
    print("\nREADING: D tells us whether the margin follows noise smoothly (a straight line between 2 % and 10 % predicts 5 %); E tells us"
          "\nwhether the result depends on the noise seed. Both together decide whether a 'noise level -> margin' table can be trusted."
          "\nSame caveats: 5 kW grid, 18 training and 12 test scenarios, detector trained on the same training scenarios, noise level unknown.")


if __name__ == "__main__":
    main()

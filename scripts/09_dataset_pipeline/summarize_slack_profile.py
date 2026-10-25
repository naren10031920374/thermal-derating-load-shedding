"""
SUMMARIZE_SLACK_PROFILE.PY  (does the margin fitted on training scenarios also hold on profile-shaped scenarios?)
================================================================================
The 15 profile-shaped scenarios H01-H15 follow Don's example load profile instead of a plain ramp + plateau.
H01-H03 were already tested (held-out tags); H04-H15 are the new ones (tags slackh_*). Same offset sweeps,
same required-margin definition as summarize_slack_margin2.py:
    required margin (kW below the cell mean cap) = 5 - largest safe offset;  '+' after a number = lower bound.

A  required margin of every profile scenario, per sensor setting, with its shape and cap cell
B  rules fitted on the 18 TRAINING scenarios only (same numbers as summarize_slack_margin2.py)
C  each rule checked on the 12 NEW scenarios (H04-H15) and on all 15 (H01-H15)
D  by shape: average and largest required margin, compared with the largest training requirement
Run:  python3 scripts/09_dataset_pipeline/summarize_slack_profile.py
Only the Python standard library is used (summarize_slack_margin2.py must sit in the same folder).
"""
from __future__ import annotations
import argparse, csv, os, statistics
from pathlib import Path
from summarize_slack_margin2 import req, TRAIN, SET, CELLNAME, CUR

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SHAPE = {"H01": "deep-late", "H02": "deep-late", "H03": "deep-late",
         "H04": "near", "H07": "near", "H11": "near",
         "H05": "in-ramp", "H09": "in-ramp", "H12": "in-ramp",
         "H06": "shallow", "H08": "shallow", "H13": "shallow", "H15": "shallow",
         "H10": "flat", "H14": "flat"}
OLD = ["H01", "H02", "H03"]
NEW = [f"H{i:02d}" for i in range(4, 16)]
# label -> tag used for H04-H15 (H01-H03 use the held-out tag in SET)
NEWTAG = {"ideal": "slackh_ideal", "2%+avg": "slackh_n2f", "5%+avg": "slackh_n5f",
          "10%+avg": "slackh_n10f", "10%+avg s2": "slackh_n10f_s2"}


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

    labels = [lab for lab, *_ in SET]
    tr = {lab: {s: req(root, s, ttag) for s in TRAIN} for lab, _, ttag, _ in SET}
    pr = {}
    for lab, _, _, htag in SET:
        d = {s: req(root, s, htag) for s in OLD}
        d.update({s: req(root, s, NEWTAG[lab]) for s in NEW})
        pr[lab] = d

    print(f"Root: {root}\nRequired margin (kW below the cell mean cap) = 5 - largest safe offset. '+' = lower bound, -- = no result.\n")
    print("A. PROFILE-SHAPED SCENARIOS: required margin (kW)")
    hdr = f"{'scenario':9s}{'shape':11s}{'cell':>11s}" + "".join(f"{lab:>12s}" for lab in labels)
    print(hdr); print("-" * len(hdr))
    for s in OLD + NEW:
        row = f"{s:9s}{SHAPE[s]:11s}{CELLNAME.get(cell_of.get(s), '?'):>11s}"
        for lab in labels:
            v = pr[lab][s]
            row += f"{('--' if v is None else f'{v[0]:+.0f}' + ('+' if v[1] else '')):>12s}"
        print(row)

    def fit(table):
        got = {s: v for s, v in table.items() if v is not None and cell_of.get(s) in cap}
        if not got:
            return None
        pc = {}
        for s, v in got.items():
            pc[cell_of[s]] = max(pc.get(cell_of[s], -99.0), v[0])
        return {"n": len(got), "fixed": max(v[0] for v in got.values()), "pc": pc,
                "r": max(v[0] / cap[cell_of[s]] for s, v in got.items())}

    def rules(f):
        return {"fixed": lambda c, m=f["fixed"]: m,
                "per cell": lambda c, d=f["pc"], m=f["fixed"]: d.get(c, m),
                "proportional": lambda c, r=f["r"]: r * cap[c]}

    def check(fn, table, ids):
        got = {s: table[s] for s in ids if table.get(s) is not None and cell_of.get(s) in cap}
        ok, bad, ms = 0, [], []
        for s, v in got.items():
            m = fn(cell_of[s]); ms.append(m)
            if m + 1e-9 >= v[0]:
                ok += 1
            else:
                bad.append(f"{s}(needs {v[0]:+.0f}, has {m:+.1f})")
        return ok, len(got), (statistics.mean(ms) if ms else float("nan")), bad

    fits = {lab: fit(tr[lab]) for lab in labels}
    # worst case over the two 10 % seeds, per training scenario
    both = {s: (None if (tr["10%+avg"][s] is None or tr["10%+avg s2"][s] is None)
                else (max(tr["10%+avg"][s][0], tr["10%+avg s2"][s][0]), False)) for s in TRAIN}
    fits["10% both seeds"] = fit(both)
    pr["10% both seeds"] = None

    print("\nB. RULES FITTED ON THE 18 TRAINING SCENARIOS ONLY")
    print(f"{'setting':16s}{'n':>3s}{'fixed kW':>10s}{'r (% of cap)':>14s}   per-cell margins (kW)")
    for lab in labels + ["10% both seeds"]:
        f = fits[lab]
        if f is None:
            print(f"{lab:16s}  no training results"); continue
        pcs = "  ".join(f"{CELLNAME[c]}:{f['pc'][c]:+.0f}" for c in sorted(f["pc"]))
        print(f"{lab:16s}{f['n']:>3d}{f['fixed']:>+10.1f}{100*f['r']:>+13.2f}%   {pcs}")

    print("\nC. EACH RULE CHECKED ON THE PROFILE-SHAPED SCENARIOS (same setting; safe if rule margin >= required margin)")
    print(f"{'setting':21s}{'rule':14s}{'new 12':>9s}{'all 15':>9s}{'avg margin':>12s}  not safe (among all 15)")
    for lab in labels + ["10% both seeds"]:
        f = fits[lab]
        if f is None:
            continue
        for tab_lab in ([lab] if lab != "10% both seeds" else ["10%+avg", "10%+avg s2"]):
            for name, fn in list(rules(f).items()) + [("current 5 kW", lambda c: CUR)]:
                ok12, n12, _, _ = check(fn, pr[tab_lab], NEW)
                ok15, n15, am, bad = check(fn, pr[tab_lab], OLD + NEW)
                if n15:
                    tag = lab if lab != "10% both seeds" else f"both->{tab_lab}"
                    print(f"{tag:21s}{name:14s}{ok12:>5d}/{n12:<3d}{ok15:>5d}/{n15:<3d}{am:>+9.1f} kW  {', '.join(bad) if bad else '-'}")

    print("\nD. BY SHAPE: required margin (kW), average / largest, and the largest training requirement for comparison")
    shapes = ["deep-late", "near", "in-ramp", "shallow", "flat"]
    print(f"{'setting':12s}{'train max':>10s}" + "".join(f"{sh:>16s}" for sh in shapes))
    for lab in labels:
        row = f"{lab:12s}"
        tmax = [v[0] for v in tr[lab].values() if v is not None]
        row += f"{(max(tmax) if tmax else float('nan')):>+9.0f} "
        for sh in shapes:
            vals = [pr[lab][s][0] for s in OLD + NEW if SHAPE[s] == sh and pr[lab][s] is not None]
            row += f"{(f'{statistics.mean(vals):+.1f} / {max(vals):+.0f}' if vals else '--'):>16s}"
        print(row)
    print("\nREADING: C says whether the margin fitted on the plain-ramp training scenarios is also safe when the load follows the example profile;"
          "\nD says which profile shape (if any) needs more margin than the plain ramps. Same caveats: 5 kW grid, one noise seed per cell,"
          "\ndetector trained on the training scenarios, noise level unknown, profile time/size scale assumed (1 x-unit = 1 s, 100 % = plateau).")


if __name__ == "__main__":
    main()

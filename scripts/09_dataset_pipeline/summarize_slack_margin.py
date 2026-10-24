"""
SUMMARIZE_SLACK_MARGIN.PY  (fit a safety margin on training scenarios, check it on the held-out ones)
================================================================================
Offset = kW added to ALL six caps; the caps already contain a 5 kW margin below the cell mean,
so the margin a scenario REQUIRES (below the cell mean) is  5 - largest_safe_offset  (kW).
Because the offset grid is 5 kW, this is a safe (slightly conservative) number. 'none' = even
the lowest offset tried failed: the requirement is then at least 5 - (lowest offset) and is
shown as a lower bound (counted at that value).

Step A  training scenarios (tags slacktr_ideal / slacktr_n2f / slacktr_n10f): required margin per scenario.
Step B  three margin rules, FITTED ON THE TRAINING SCENARIOS ONLY:
          fixed         one margin for everything       = max required margin
          per cell      one margin per cap cell          = max required margin in the cell
          proportional  margin = r x cell cap            = r is the max of (required margin / cell cap)
Step C  check each rule once on the 12 held-out/profile scenarios (tags slackb, slack_n2f, slack_n10f):
          safe if rule margin >= the scenario's required margin. Reports how many of 12 are safe and the
          average margin in kW (smaller margin = less load cut).
Run:  python3 scripts/09_dataset_pipeline/summarize_slack_margin.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, csv, json, os, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CURRENT_MARGIN = 5.0
TRAIN = ["S003", "S006", "S030", "S010", "S031", "T024", "S002", "S011", "S027", "S009", "S026", "T004",
         "S001", "S016", "T008", "S023", "T005", "T048"]
TEST = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035", "H01", "H02", "H03"]
# (label, training tag, test tag)
SETTINGS = [("ideal", "slacktr_ideal", "slackb"), ("2%+avg", "slacktr_n2f", "slack_n2f"), ("10%+avg", "slacktr_n10f", "slack_n10f")]
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


def required_margin(root: Path, sid: str, tag: str):
    """(margin_kW, is_lower_bound) or None if there is no result."""
    r = load(root / sid / f"staged_{sid}_{tag}_result.json")
    if r is None:
        return None
    if r.get("status") == "safe_found":
        return CURRENT_MARGIN - float(r["first_safe_offset_kw"]), False
    base = r.get("base_cap_kw")
    tried = listify(r.get("cap_tried_kw"))
    if base is None or not tried:
        return None
    return CURRENT_MARGIN - (min(tried) - base), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--scripts", default=str(HERE), help="folder with staged_caps.csv and staged_cap_groups_nn.csv")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    sdir = Path(a.scripts)
    cell_of = {}
    with open(sdir / "staged_caps.csv", newline="") as f:
        for r in csv.DictReader(f):
            cell_of[r["scenario_id"]] = (int(r["n_target"]), int(r["has_ef"]))
    cap = {}
    with open(sdir / "staged_cap_groups_nn.csv", newline="") as f:
        for r in csv.DictReader(f):
            cap[(int(r["n_target"]), int(r["has_ef"]))] = float(r["cap_kw"])

    req_tr = {lab: {s: required_margin(root, s, ttag) for s in TRAIN} for lab, ttag, _ in SETTINGS}
    req_te = {lab: {s: required_margin(root, s, htag) for s in TEST} for lab, _, htag in SETTINGS}

    print(f"Root: {root}\nRequired margin (kW below the cell mean cap) = 5 - largest safe offset. A trailing + = lower bound ('none').\n")
    print("A. TRAINING SCENARIOS: required margin (kW)")
    hdr = f"{'scenario':9s}{'cell':>11s}{'cell cap':>9s}" + "".join(f"{lab:>10s}" for lab, _, _ in SETTINGS)
    print(hdr); print("-" * len(hdr))
    for s in TRAIN:
        c = cell_of.get(s)
        row = f"{s:9s}{CELLNAME.get(c, '?'):>11s}{cap.get(c, float('nan')):>9.1f}"
        for lab, _, _ in SETTINGS:
            v = req_tr[lab][s]
            row += f"{('--' if v is None else f'{v[0]:+.0f}' + ('+' if v[1] else '')):>10s}"
        print(row)

    print("\nB. FITTED ON TRAINING ONLY")
    rules = {}
    for lab, _, _ in SETTINGS:
        got = {s: v for s, v in req_tr[lab].items() if v is not None and cell_of.get(s) in cap}
        if not got:
            print(f"\n  {lab}: no training results yet")
            continue
        fixed = max(v[0] for v in got.values())
        per_cell = {}
        for s, v in got.items():
            per_cell[cell_of[s]] = max(per_cell.get(cell_of[s], -99.0), v[0])
        r_prop = max(v[0] / cap[cell_of[s]] for s, v in got.items())
        rules[lab] = {"fixed": lambda c, m=fixed: m,
                      "per cell": lambda c, d=per_cell, m=fixed: d.get(c, m),
                      "proportional": lambda c, r=r_prop: r * cap[c]}
        print(f"\n  {lab}   ({len(got)} training scenarios{', some lower bounds' if any(v[1] for v in got.values()) else ''})")
        print(f"    fixed margin            : {fixed:+.1f} kW")
        print(f"    proportional factor r   : {100*r_prop:+.2f} % of the cell cap")
        print(f"    {'cell':11s}{'n':>3s}{'max required':>14s}{'(as % of cell cap)':>20s}{'proportional rule':>19s}")
        for c in sorted(per_cell):
            n = sum(1 for s in got if cell_of[s] == c)
            print(f"    {CELLNAME[c]:11s}{n:>3d}{per_cell[c]:>+13.1f} kW{100*per_cell[c]/cap[c]:>17.1f} %{r_prop*cap[c]:>+16.1f} kW")

    print("\nC. CHECK ON THE 12 HELD-OUT / PROFILE SCENARIOS (each rule is applied as fitted above)")
    print(f"{'setting':9s}{'rule':14s}{'safe':>7s}{'avg margin':>12s}  not safe")
    for lab, _, _ in SETTINGS:
        if lab not in rules:
            continue
        got = {s: v for s, v in req_te[lab].items() if v is not None and cell_of.get(s) in cap}
        if not got:
            continue
        for name, fn in rules[lab].items():
            ok, bad, ms = 0, [], []
            for s, v in got.items():
                m = fn(cell_of[s])
                ms.append(m)
                if m + 1e-9 >= v[0]:
                    ok += 1
                else:
                    bad.append(f"{s}(needs {v[0]:+.0f}{'+' if v[1] else ''}, has {m:+.1f})")
            print(f"{lab:9s}{name:14s}{ok:>4d}/{len(got):<2d}{statistics.mean(ms):>+10.1f} kW  {', '.join(bad) if bad else '-'}")
        cur = [s for s, v in got.items() if v[0] > CURRENT_MARGIN + 1e-9]
        print(f"{lab:9s}{'current 5 kW':14s}{len(got)-len(cur):>4d}/{len(got):<2d}{CURRENT_MARGIN:>+10.1f} kW  {', '.join(cur) if cur else '-'}")
    print("\nREADING: a good rule is safe in all 12 and has a smaller average margin than the current 5 kW where the sensor is clean."
          "\nOne noise seed, 5 kW offset grid (requirements are rounded up to the grid), 18 training scenarios (3 per cell). The training"
          "\nscenarios were also used to build the cap table and train the detector, so detector misses are likely rarer there than on new data.")


if __name__ == "__main__":
    main()

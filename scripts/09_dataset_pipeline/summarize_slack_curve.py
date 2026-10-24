"""
SUMMARIZE_SLACK_CURVE.PY  (how does the cap room shrink as sensor noise grows?)
Puts all slack tests side by side: ideal sensor (slackb), noise 1/2/5/10 % (no average)
and noise 1/2/10 % with the 10 s average. Offset = kW added to ALL six caps; the value shown
is the largest offset at which all 10 buses survive ('none' = even -10 kW failed, '--' = no result).
Run:  python3 scripts/09_dataset_pipeline/summarize_slack_curve.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CELL = {"S012": 3, "S013": 3, "S014": 3, "S019": 2, "S020": 2, "S024": 3, "S032": 3, "S034": 1,
        "S035": 2, "H01": 1, "H02": 2, "H03": 3}
SCEN = list(CELL)
HELD_OUT = SCEN[:9]
SETTINGS = [("ideal", "slackb"), ("n1%", "slack_n1"), ("n2%", "slack_n2"), ("n5%", "slack_n5"),
            ("n10%", "slack_n10"), ("n1%+avg", "slack_n1f"), ("n2%+avg", "slack_n2f"), ("n10%+avg", "slack_n10f")]
BASE_TAG = "nncausal"


def load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARNING: could not read {p}: {e}")
        return None


def read_run(root: Path, sid: str, tag: str):
    r = load(root / sid / f"staged_{sid}_{tag}_result.json")
    if r is None:
        return None
    found = r.get("status") == "safe_found"
    tried = r.get("cap_tried_kw") or []
    tried = tried if isinstance(tried, list) else [tried]
    return {"found": found, "off": r.get("first_safe_offset_kw") if found else None, "n": len(tried),
            "kwh": (r.get("safe_policy") or {}).get("shed_kwh") if found else None}


def cell(r):
    if r is None:
        return "--"
    return f"{r['off']:+.0f}" if r["found"] else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    data = {s: {t: read_run(root, s, t) for _, t in SETTINGS} for s in SCEN}
    base = {}
    for s in SCEN:
        b = load(root / s / f"staged_{s}_{BASE_TAG}_result.json")
        ok = b and b.get("status") == "safe_found" and abs(b.get("first_safe_offset_kw", 99)) < 1e-6
        base[s] = (b.get("safe_policy") or {}).get("shed_kwh") if ok else None

    print(f"Root: {root}\nLargest safe offset (kW added to all six caps). '--' no result yet, 'none' = even -10 kW failed.\n")
    print("A. PER SCENARIO   (bus = number of buses in the final group)")
    hdr = f"{'scenario':9s}{'bus':>4s}" + "".join(f"{t:>10s}" for t, _ in SETTINGS)
    print(hdr); print("-" * len(hdr))
    for s in SCEN:
        print(f"{s:9s}{CELL[s]:>4d}" + "".join(f"{cell(data[s][tag]):>10s}" for _, tag in SETTINGS))

    def offs(ids, tag):
        rs = [data[s][tag] for s in ids if data[s][tag] is not None]
        if not rs:
            return None
        return [r["off"] if r["found"] else -15.0 for r in rs], any(not r["found"] for r in rs), len(rs)

    print("\nB. SAFE IN ALL SCENARIOS OF THE BLOCK (= the smallest of the largest safe offsets; * = at least one 'none', counted as -15)")
    hdr = f"{'block':22s}" + "".join(f"{t:>10s}" for t, _ in SETTINGS)
    print(hdr); print("-" * len(hdr))
    blocks = [("all 12", SCEN), ("9 held-out", HELD_OUT)] + \
             [(f"{k}-bus cases", [s for s in SCEN if CELL[s] == k]) for k in (1, 2, 3)]
    for name, ids in blocks:
        row = f"{name:22s}"
        for _, tag in SETTINGS:
            o = offs(ids, tag)
            if o is None:
                row += f"{'--':>10s}"
            else:
                v, star, n = o
                row += f"{(f'{min(v):+.0f}' + ('*' if star else '')):>10s}"
        print(row)

    print("\nC. MEDIAN largest safe offset over all 12")
    print("".join(f"{t:>10s}" for t, _ in SETTINGS))
    row = ""
    for _, tag in SETTINGS:
        o = offs(SCEN, tag)
        row += f"{(f'{statistics.median(o[0]):+.0f}' if o else '--'):>10s}"
    print(row)

    print("\nD. SHED ENERGY of the 9 held-out at each scenario's own largest safe offset vs offset 0 with an ideal sensor")
    print(f"{'setting':10s}{'done':>6s}{'kWh @0':>10s}{'kWh @own':>10s}{'change':>9s}{'sims':>6s}")
    for t, tag in SETTINGS:
        rs = [(base[s], data[s][tag]) for s in HELD_OUT if data[s][tag] is not None]
        both = [(b, r["kwh"]) for b, r in rs if b and r["kwh"] is not None]
        n_sims = sum(r["n"] for _, r in rs)
        if not both:
            print(f"{t:10s}{len(rs):>6d}{'n/a':>10s}{'n/a':>10s}{'n/a':>9s}{n_sims:>6d}")
            continue
        t0 = sum(b for b, _ in both); tx = sum(k for _, k in both)
        part = "" if len(both) == len(HELD_OUT) else f"  (only {len(both)} safe scenarios)"
        print(f"{t:10s}{len(rs):>6d}{t0:>10.1f}{tx:>10.1f}{100*(tx-t0)/t0:>+8.1f}%{n_sims:>6d}{part}")
    print("\nREADING: follow one row of table B from left to right: the offset that is safe in every scenario falls as noise grows."
          "\nThe avg columns show how much of the room the 10 s average gives back. The same 12 scenarios as before: this tells us how"
          "\nthe room depends on noise, it must not be used to pick the final margin. One noise seed, 5 kW grid.")


if __name__ == "__main__":
    main()

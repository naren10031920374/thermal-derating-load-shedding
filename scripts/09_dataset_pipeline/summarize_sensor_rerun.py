"""
SUMMARIZE_SENSOR_RERUN.PY  (margin needed with Don's sensor numbers, by noise model and averaging window)
Reads the tags written by submit_sensor_rerun.sh for H01 and H09 and prints, per tag:
  largest safe offset (kW added to all six caps) and the margin needed = 5 kW - offset,
  what was tried (+15S = safe at +15, +10F = collapse at +10 ...), and the shed energy at that offset.
Old results (1 s sampling) are shown for comparison where they exist.
'none' = even the lowest offset tried (-10 kW) failed, so the margin needed is more than 15 kW.
Run:  python3 scripts/09_dataset_pipeline/summarize_sensor_rerun.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SCEN = ["H01", "H09"]
PRESENT_MARGIN_KW = 5.0   # the caps already contain a 5 kW margin below the network's cell mean

# (title, {scenario: tag}) -- the same tag for every scenario unless a dict is given
NEW = [
    ("0.5% bounded, no avg",      "sr_b05_w0"),
    ("0.5% bounded, 100 ms",      "sr_b05_w100ms"),
    ("0.5% bounded, 1 s",         "sr_b05_w1s"),
    ("0.5% bounded, 10 s",        "sr_b05_w10s"),
    ("0.5% Gaussian, no avg",     "sr_g05_w0"),
    ("0.5% Gaussian, 100 ms",     "sr_g05_w100ms"),
    ("0.5% Gaussian, 1 s",        "sr_g05_w1s"),
    ("0.5% Gaussian, 10 s",       "sr_g05_w10s"),
    ("10% Gaussian, 100 ms",      "sr_g10_w100ms"),
    ("10% Gaussian, 1 s",         "sr_g10_w1s"),
    ("10% Gaussian, 10 s",        "sr_g10_w10s"),
]
OLD = [
    ("OLD ideal sensor",                {"H01": "slack_f10", "H09": "slackh_ideal"}),
    ("OLD 2% + 10 s (1 s sampling)",    {"H01": "slack_n2f", "H09": "slackh_n2f"}),
    ("OLD 5% + 10 s (1 s sampling)",    {"H01": "slack_n5f", "H09": "slackh_n5f"}),
    ("OLD 10% + 10 s (1 s sampling)",   {"H01": "slack_n10f", "H09": "slackh_n10f"}),
]


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


def read_run(root: Path, sid: str, tag: str):
    r = load(root / sid / f"staged_{sid}_{tag}_result.json")
    if r is None:
        return None
    base = r.get("base_cap_kw")
    tried = listify(r.get("cap_tried_kw"))
    safe = listify(r.get("cap_safe"))
    found = r.get("status") == "safe_found"
    pol = r.get("safe_policy") or {}
    return {
        "runs": " ".join(f"{c - base:+.0f}{'S' if s else 'F'}" for c, s in zip(tried, safe)),
        "found": found,
        "off": r.get("first_safe_offset_kw") if found else None,
        "kwh": pol.get("shed_kwh") if found else None,
    }


def cell(r):
    if r is None:
        return "--"
    if not r["found"]:
        return "none"
    return f"{r['off']:+.0f} -> {PRESENT_MARGIN_KW - r['off']:+.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    rows = []
    for title, tag in NEW:
        rows.append((title, {s: read_run(root, s, tag) for s in SCEN}))
    for title, tags in OLD:
        rows.append((title, {s: read_run(root, s, tags[s]) for s in SCEN}))

    print(f"Root: {root}")
    print("Cell = largest safe offset (kW) -> margin needed (kW) = 5 - offset.  '--' = no result yet,"
          " 'none' = even -10 kW failed (margin > 15 kW).\n")
    print("A. LARGEST SAFE OFFSET -> MARGIN NEEDED")
    hdr = f"{'setting':32s}" + "".join(f"{s:>14s}" for s in SCEN)
    print(hdr); print("-" * len(hdr))
    for title, d in rows:
        print(f"{title:32s}" + "".join(f"{cell(d[s]):>14s}" for s in SCEN))

    print("\nB. WHAT WAS TRIED (offset + S safe / F collapse)")
    for title, d in rows:
        print(f"  {title}")
        for s in SCEN:
            r = d[s]
            print(f"    {s:5s} {r['runs'] if r else '--'}")

    print("\nC. SHED ENERGY (kWh) at the largest safe offset  [less is better; compare rows of the same scenario]")
    hdr = f"{'setting':32s}" + "".join(f"{s:>14s}" for s in SCEN)
    print(hdr); print("-" * len(hdr))
    for title, d in rows:
        vals = []
        for s in SCEN:
            r = d[s]
            vals.append(f"{r['kwh']:.1f}" if r and r["kwh"] is not None else "--")
        print(f"{title:32s}" + "".join(f"{v:>14s}" for v in vals))

    print("\nREADING: compare the windows within one noise model. If the margin needed does not change between the"
          "\nwindows, the window does not matter much for that noise. The OLD rows are the earlier tests with a 1 s sampling"
          "\nperiod. Only two scenarios and one noise seed: this is a check on whether the results change, not a new margin table.")


if __name__ == "__main__":
    main()

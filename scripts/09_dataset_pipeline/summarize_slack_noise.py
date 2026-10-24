"""
SUMMARIZE_SLACK_NOISE.PY  (how much of the cap room survives a noisy sensor?)
Reads the noise-aware slack test results (tags slack_n5, slack_n10, slack_n10f), puts them
next to the ideal-sensor slack test (slackb) and the baseline at offset 0 (nncausal).
Offset = kW added to ALL six caps. 'none' = even the lowest offset tried (-10 kW) failed.
Run:  python3 scripts/09_dataset_pipeline/summarize_slack_noise.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SCEN = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035", "H01", "H02", "H03"]
HELD_OUT = SCEN[:9]
PROFILE = SCEN[9:]
SETTINGS = [("ideal", "slackb"), ("noise5%", "slack_n5"), ("noise10%", "slack_n10"), ("n10%+filt10s", "slack_n10f")]
BASE_TAG = "nncausal"


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
        "n": len(tried),
        "found": found,
        "off": r.get("first_safe_offset_kw") if found else None,
        "kwh": pol.get("shed_kwh") if found else None,
    }


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
    data = {sid: {t: read_run(root, sid, t) for _, t in SETTINGS} for sid in SCEN}
    base = {}
    for sid in SCEN:
        b = load(root / sid / f"staged_{sid}_{BASE_TAG}_result.json")
        ok = b and b.get("status") == "safe_found" and abs(b.get("first_safe_offset_kw", 99)) < 1e-6
        base[sid] = (b.get("safe_policy") or {}).get("shed_kwh") if ok else None

    print(f"Root: {root}\nOffset = kW added to all six caps. '--' = no result yet, 'none' = even -10 kW failed.\n")
    print("A. LARGEST SAFE OFFSET (kW) per scenario and sensor setting")
    hdr = f"{'scenario':9s}" + "".join(f"{t:>14s}" for t, _ in SETTINGS)
    print(hdr); print("-" * len(hdr))
    for sid in SCEN:
        print(f"{sid:9s}" + "".join(f"{cell(data[sid][tag]):>14s}" for _, tag in SETTINGS))

    print("\nB. WHAT WAS TRIED (offset + S safe / F collapse)")
    for title, tag in SETTINGS[1:]:
        print(f"\n  {title}  (tag {tag})")
        for sid in SCEN:
            r = data[sid][tag]
            print(f"    {sid:6s} {r['runs'] if r else '--'}")

    print("\nC. SUMMARY per setting (offset safe in ALL scenarios of the block = the smallest 'largest safe offset')")
    for name, ids in [("9 held-out", HELD_OUT), ("3 example-profile (H01-H03)", PROFILE)]:
        print(f"\n  {name}")
        print(f"  {'setting':16s}{'done':>6s}{'min':>7s}{'median':>8s}{'max':>6s}{'kWh @0':>9s}{'kWh @own max':>14s}{'change':>8s}{'sims':>6s}")
        for title, tag in SETTINGS:
            rs = [data[s][tag] for s in ids if data[s][tag] is not None]
            if not rs:
                print(f"  {title:16s}{0:>6d}"); continue
            offs = [r["off"] if r["found"] else -15.0 for r in rs]
            both = [(base[s], data[s][tag]["kwh"]) for s in ids
                    if data[s][tag] is not None and data[s][tag]["kwh"] is not None and base[s]]
            t0 = sum(b for b, _ in both); tx = sum(k for _, k in both)
            chg = f"{100*(tx-t0)/t0:+.1f}%" if both else "n/a"
            mn = f"{min(offs):+.0f}" + ("*" if any(not r["found"] for r in rs) else "")
            print(f"  {title:16s}{len(rs):>6d}{mn:>7s}{statistics.median(offs):>+8.0f}{max(offs):>+6.0f}"
                  f"{(t0 if both else float('nan')):>9.1f}{(tx if both else float('nan')):>14.1f}{chg:>8s}{sum(r['n'] for r in rs):>6d}")
    print("\n  (* = at least one scenario failed even at -10 kW; shown as -15 in the median/min)")
    print("  (kWh @0 is the ideal-sensor run at offset 0; the change compares each setting at its own largest safe offset)")
    print("\nREADING: compare the 'min' column across settings. The drop from 'ideal' to a noisy setting is the room that noise uses up;"
          "\nif 'n10%+filt10s' is close to 'ideal', averaging the readings gives most of the room back. These are the SAME 12 scenarios"
          "\nas the first slack test, so they decide whether tuning is worthwhile; they must not be used to pick the final margin.")


if __name__ == "__main__":
    main()

"""
SUMMARIZE_SLACK.PY  (how much room do the caps have?)
================================================================================
Reads the results of the slack test (run_staged_array.sh with OFFSETS=25:20:15:10:5,
STAGED=1, TAG=slacknn) and the earlier network run (TAG=nncausal, offset 0) and prints,
per scenario:

  * every cap that was tried (offset in kW added to ALL six caps), safe (S) or failed (F)
  * the largest safe offset = the "slack" of that scenario
  * the shed energy at offset 0 and at the largest safe offset, and the change in %

The slack test tries the most permissive cap first and stops at the first safe one,
so the largest safe offset is the first S in the list. Offset 0 was already safe on
the first cap in all 12 scenarios (ideal sensor), so "no S in the list" means the
slack is between 0 and the smallest offset tried.

Run (login node is fine, it only reads small files):
    python scripts/09_dataset_pipeline/summarize_slack.py
    python scripts/09_dataset_pipeline/summarize_slack.py --tag slacknn --base-tag nncausal
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
HELD_OUT = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035"]
PROFILE = ["H01", "H02", "H03"]


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


def fnum(x, nd=1, plus=False):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="slacknn")
    ap.add_argument("--base-tag", default="nncausal")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")

    rows = []
    for sid in HELD_OUT + PROFILE:
        d = root / sid
        sl = load(d / f"staged_{sid}_{a.tag}_result.json")
        bs = load(d / f"staged_{sid}_{a.base_tag}_result.json")
        if sl is None:
            rows.append({"sid": sid, "missing": True})
            continue
        base_cap = sl.get("base_cap_kw")
        tried = listify(sl.get("cap_tried_kw"))
        safe = listify(sl.get("cap_safe"))
        runs = " ".join(f"{c - base_cap:+.0f}{'S' if s else 'F'}" for c, s in zip(tried, safe))
        found = sl.get("status") == "safe_found"
        off = sl.get("first_safe_offset_kw") if found else None
        k_x = (sl.get("safe_policy") or {}).get("shed_kwh") if found else None
        k_0 = None
        if bs and bs.get("status") == "safe_found" and abs(bs.get("first_safe_offset_kw", 99)) < 1e-6:
            k_0 = (bs.get("safe_policy") or {}).get("shed_kwh")
        mv = listify((sl.get("safe_policy") or {}).get("min_v")) if found else []
        rows.append({"sid": sid, "missing": False, "base_cap": base_cap, "runs": runs,
                     "n_runs": len(tried), "found": found, "off": off, "k0": k_0, "kx": k_x,
                     "minv": min(mv) if mv else None})

    print(f"Root: {root}   slack tag: {a.tag}   baseline tag (offset 0): {a.base_tag}\n")
    print("Offset = kW added to all six caps. Tried most permissive first. S = all 10 buses survive, F = a bus collapses.\n")
    hdr = f"{'scenario':9s}{'tried (offset, verdict)':32s}{'largest safe':>13s}{'kWh @0':>9s}{'kWh @max':>10s}{'change':>8s}{'min V':>8s}"
    print(hdr)
    print("-" * len(hdr))
    missing = []
    for r in rows:
        if r["missing"]:
            missing.append(r["sid"])
            continue
        if r["found"]:
            lab = f"{r['off']:+.0f} kW"
        else:
            lab = "< smallest"
        chg = None
        if r["k0"] and r["kx"] is not None:
            chg = 100.0 * (r["kx"] - r["k0"]) / r["k0"]
        print(f"{r['sid']:9s}{r['runs']:32s}{lab:>13s}{fnum(r['k0'],1):>9s}{fnum(r['kx'],1):>10s}"
              f"{(fnum(chg,1,True)+'%') if chg is not None else 'n/a':>8s}{fnum(r['minv'],0):>8s}")
    if missing:
        print("\nNo slack result yet for:", ", ".join(missing))

    def block(name, ids):
        rs = [r for r in rows if r["sid"] in ids and not r["missing"]]
        offs = [r["off"] if r["found"] else 0.0 for r in rs]   # not found -> between 0 and the smallest offset
        if not rs:
            return
        both = [r for r in rs if r["k0"] and r["kx"] is not None]
        print(f"\n{name}: {len(rs)} scenarios | runs used {sum(r['n_runs'] for r in rs)}")
        print(f"   largest safe offset: min {min(offs):+.0f} | median {statistics.median(offs):+.0f} | max {max(offs):+.0f} kW"
              f"   (a value of +0 here can also mean 'between 0 and the smallest offset tried')")
        if both:
            t0 = sum(r["k0"] for r in both); tx = sum(r["kx"] for r in both)
            print(f"   shed energy over {len(both)} scenarios: offset 0 = {t0:.1f} kWh, each at its own largest safe offset = {tx:.1f} kWh "
                  f"({100.0*(tx-t0)/t0:+.1f}%)")

    block("9 held-out scenarios", HELD_OUT)
    block("3 example-profile scenarios (H01-H03)", PROFILE)

    print("\nREADING: the 'min' largest-safe offset is how much ALL six caps could rise at once and still be safe in every scenario,"
          "\nwith an ideal sensor (no noise, no delay) and one pass. It is the room the 5 kW safety margin leaves unused here, not a"
          "\nrecommendation: under noisy or late sensors the room is smaller (S034 and H01 needed -10 kW at 10% noise).")


if __name__ == "__main__":
    main()

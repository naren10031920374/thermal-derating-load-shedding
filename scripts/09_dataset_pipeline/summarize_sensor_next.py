"""
SUMMARIZE_SENSOR_NEXT.PY  (two follow-up tests with Don's sensor numbers)
  Part 1  shortest averaging window: tags from submit_sensor_window.sh next to the known 10 % + 1 s result.
  Part 2  finer offset grid: tag sr_fine_g10_w1s from submit_sensor_fine.sh (1 kW steps inside the last 5 kW step).
Offset = kW added to all six caps. Margin needed = 5 kW - offset. 'none' = even the lowest offset tried failed.
Run:  python3 scripts/09_dataset_pipeline/summarize_sensor_next.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SCEN = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035"] + [f"H{i:02d}" for i in range(1, 16)]
PRESENT = 5.0

# Part 1: (title, noise left after averaging in %, tag)
WINDOWS = [
    ("10% + 100 ms", 3.2, "sr_g10_w100ms"),     # only H01 and H09 were run
    ("10% + 300 ms", 1.8, "sr_w_g10_w300ms"),
    ("5% + 100 ms",  1.6, "sr_w_g05_w100ms"),
    ("10% + 500 ms", 1.4, "sr_w_g10_w500ms"),
    ("10% + 1 s",    1.0, "sr_g10_w1s"),         # reference, seed 1
]
COARSE_TAG = "sr_g10_w1s"
FINE_TAG = "sr_fine_g10_w1s"


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
    found = r.get("status") == "safe_found"
    base = r.get("base_cap_kw")
    tried = listify(r.get("cap_tried_kw"))
    return {"found": found, "off": r.get("first_safe_offset_kw") if found else None,
            "lowest_tried": (min(tried) - base) if tried and base is not None else None}


def cell(r):
    if r is None:
        return "--"
    return f"{r['off']:+.0f}->{PRESENT - r['off']:+.0f}" if r["found"] else "none"


def safe_at(r, margin):
    return r is not None and r["found"] and r["off"] >= PRESENT - margin - 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    print(f"Root: {root}\n")

    # ---------------- Part 1 ----------------
    print("PART 1. SHORTEST AVERAGING WINDOW  (cell = largest safe offset -> margin needed, kW; seed 1)")
    data = {s: [read_run(root, s, t) for _, _, t in WINDOWS] for s in SCEN}
    hdr = f"{'scenario':9s}" + "".join(f"{t:>15s}" for t, _, _ in WINDOWS)
    print(hdr)
    print(f"{'noise left':9s}" + "".join(f"{str(n) + ' %':>15s}" for _, n, _ in WINDOWS))
    print("-" * len(hdr))
    for s in SCEN:
        print(f"{s:9s}" + "".join(f"{cell(r):>15s}" for r in data[s]))
    print(f"\n  {'setting':16s}{'noise left':>11s}{'done':>6s}{'max M':>8s}{'M=0':>6s}{'M=5':>6s}{'M=10':>6s}")
    for i, (title, nl, _) in enumerate(WINDOWS):
        rs = [data[s][i] for s in SCEN if data[s][i] is not None]
        if not rs:
            print(f"  {title:16s}{nl:>10.1f}%{0:>6d}"); continue
        mar = [PRESENT - r["off"] if r["found"] else 99 for r in rs]
        mx = ">15" if max(mar) == 99 else f"{max(mar):+.0f}"
        print(f"  {title:16s}{nl:>10.1f}%{len(rs):>6d}{mx:>8s}{sum(safe_at(r, 0) for r in rs):>6d}"
              f"{sum(safe_at(r, 5) for r in rs):>6d}{sum(safe_at(r, 10) for r in rs):>6d}")
    print("  (10% + 100 ms was only run on H01 and H09: 'done' is 2)")
    print("  READING: if rows with about the same 'noise left' (5% + 100 ms and 10% + 500 ms) give about the same M, the noise left decides.")

    # ---------------- Part 2 ----------------
    print("\nPART 2. FINER OFFSET GRID, 10% + 1 s, seed 1  (1 kW steps inside the last 5 kW step)")
    print("  coarse = largest safe offset on the 5 kW grid; fine = largest safe offset found in 1 kW steps (a lower bound:")
    print("  the real limit is below fine + 1). Spare = fine - 5 = how far above margin 0 the cap could sit.")
    print(f"\n  {'scenario':9s}{'coarse':>8s}{'fine':>8s}{'spare kW':>10s}{'  note'}")
    spare_all = []
    for s in SCEN:
        c = read_run(root, s, COARSE_TAG)
        f = read_run(root, s, FINE_TAG)
        if c is None or not c["found"]:
            print(f"  {s:9s}{'--':>8s}{'--':>8s}{'--':>10s}  no coarse result"); continue
        if f is None:
            print(f"  {s:9s}{c['off']:>+8.0f}{'--':>8s}{'--':>10s}  fine run not done"); continue
        if f["found"]:
            fine, note = f["off"], ""
        else:
            fine, note = c["off"], f"none of the fine offsets safe: limit is between {c['off']:+.0f} and {c['off'] + 1:+.0f}"
        spare = fine - PRESENT
        spare_all.append(spare)
        print(f"  {s:9s}{c['off']:>+8.0f}{fine:>+8.0f}{spare:>+10.0f}  {note}")
    if spare_all:
        print(f"\n  scenarios with a fine result: {len(spare_all)}")
        print(f"  smallest spare: {min(spare_all):+.0f} kW   largest spare: {max(spare_all):+.0f} kW")
        print(f"  so a fixed margin of {-min(spare_all):+.0f} kW (offset {PRESENT + min(spare_all):+.0f}) is safe on all {len(spare_all)} of them, at 1 kW resolution")
    print("\nREADING: the spare room tells how much of the cap range is really unused at margin 0. Seed 1 only; the 5 kW grid result"
          "\nof the coarse run is unchanged. A fine offset equal to the coarse offset means the limit is within 1 kW of the coarse value.")


if __name__ == "__main__":
    main()

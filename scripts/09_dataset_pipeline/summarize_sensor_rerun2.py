"""
SUMMARIZE_SENSOR_RERUN2.PY  (margin needed on the 24 test + profile scenarios, Don's sensor numbers, two noise seeds)
Reads the tags written by submit_sensor_rerun2.sh (and, for H01 and H09 seed 1, by submit_sensor_rerun.sh) and puts them
next to the old results (1 s sampling, 10 s average, noise 10 %) and the ideal sensor.
Offset = kW added to all six caps. Margin needed = 5 kW - offset. 'none' = even -10 kW failed (margin > 15 kW).
A scenario is safe with a fixed margin M if its largest safe offset is at least 5 - M.
Run:  python3 scripts/09_dataset_pipeline/summarize_sensor_rerun2.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SCEN = ["S012", "S013", "S014", "S019", "S020", "S024", "S032", "S034", "S035"] + [f"H{i:02d}" for i in range(1, 16)]
PRESENT = 5.0


def first12(sid):
    return sid.startswith("S") or sid in ("H01", "H02", "H03")


# (title, function(scenario) -> list of tags tried in order)
SETTINGS = [
    ("0.5% bounded s1", lambda s: ["sr_b05_w1s"]),
    ("0.5% bounded s2", lambda s: ["sr_b05_w1s_s2"]),
    ("10% + 1 s avg s1", lambda s: ["sr_g10_w1s"]),
    ("10% + 1 s avg s2", lambda s: ["sr_g10_w1s_s2"]),
    ("OLD 10%+10s s1", lambda s: ["slack_n10f"] if first12(s) else ["slackh_n10f"]),
    ("OLD 10%+10s s2", lambda s: ["slack_n10f_s2"] if first12(s) else ["slackh_n10f_s2"]),
    ("OLD ideal", lambda s: ["slackb", "slack_f10"] if first12(s) else ["slackh_ideal"]),
]
PAIRS = [("0.5% bounded, worst of 2 seeds", 0, 1), ("10% + 1 s avg, worst of 2 seeds", 2, 3), ("OLD 10%+10s, worst of 2 seeds", 4, 5)]


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


def read_run(root: Path, sid: str, tags):
    for tag in tags:
        r = load(root / sid / f"staged_{sid}_{tag}_result.json")
        if r is None:
            continue
        found = r.get("status") == "safe_found"
        pol = r.get("safe_policy") or {}
        return {"found": found, "off": r.get("first_safe_offset_kw") if found else None,
                "kwh": pol.get("shed_kwh") if found else None}
    return None


def cell(r):
    if r is None:
        return "--"
    return f"{r['off']:+.0f}->{PRESENT - r['off']:+.0f}" if r["found"] else "none"


def worst(a, b):
    if a is None or b is None:
        return None
    if not (a["found"] and b["found"]):
        return {"found": False, "off": None, "kwh": None}
    return {"found": True, "off": min(a["off"], b["off"]), "kwh": None}


def safe_at(r, margin):
    return r is not None and r["found"] and r["off"] >= PRESENT - margin - 1e-9


def block(title, runs):
    rs = [r for r in runs if r is not None]
    if not rs:
        print(f"  {title:34s}{0:>5d}"); return
    mar = [PRESENT - r["off"] if r["found"] else 99 for r in rs]
    n = len(rs)
    print(f"  {title:34s}{n:>5d}{('>15' if max(mar) == 99 else f'{max(mar):+.0f}'):>9s}"
          f"{sum(safe_at(r, 0) for r in rs):>7d}{sum(safe_at(r, 5) for r in rs):>7d}{sum(safe_at(r, 10) for r in rs):>7d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT")
                or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    data = {s: [read_run(root, s, f(s)) for _, f in SETTINGS] for s in SCEN}

    print(f"Root: {root}")
    print("Cell = largest safe offset (kW) -> margin needed (kW) = 5 - offset.  '--' = no result yet, 'none' = even -10 kW failed.\n")
    print("A. PER SCENARIO")
    hdr = f"{'scenario':9s}" + "".join(f"{t:>18s}" for t, _ in SETTINGS)
    print(hdr); print("-" * len(hdr))
    for s in SCEN:
        print(f"{s:9s}" + "".join(f"{cell(r):>18s}" for r in data[s]))

    print("\nB. SUMMARY (done = scenarios with a result; 'safe @ M' = scenarios safe with a fixed margin of M kW)")
    print(f"  {'setting':34s}{'done':>5s}{'max M':>9s}{'M=0':>7s}{'M=5':>7s}{'M=10':>7s}")
    for i, (title, _) in enumerate(SETTINGS):
        block(title, [data[s][i] for s in SCEN])
    for title, i, j in PAIRS:
        block(title, [worst(data[s][i], data[s][j]) for s in SCEN])

    print("\nC. SHED ENERGY (kWh) at each scenario's own largest safe offset, paired with the OLD 10%+10s result of the same seed")
    print(f"  {'setting':34s}{'pairs':>6s}{'new kWh':>10s}{'old kWh':>10s}{'change':>9s}")
    for i, j in [(2, 4), (3, 5)]:
        pr = [(data[s][i]["kwh"], data[s][j]["kwh"]) for s in SCEN
              if data[s][i] and data[s][j] and data[s][i]["kwh"] is not None and data[s][j]["kwh"] is not None]
        if pr:
            tn = sum(p[0] for p in pr); to = sum(p[1] for p in pr)
            print(f"  {SETTINGS[i][0]:34s}{len(pr):>6d}{tn:>10.1f}{to:>10.1f}{100*(tn-to)/to:>+8.1f}%")
        else:
            print(f"  {SETTINGS[i][0]:34s}{0:>6d}")

    print("\nREADING: 'max M' is the margin the worst scenario needs. If M=0 is safe in all scenarios for a setting and both seeds,"
          "\na margin of 0 kW looks possible for that setting. The 5 kW grid means every number is a multiple of 5.")


if __name__ == "__main__":
    main()

"""
SUMMARIZE_DELAY.PY  (sensor delay test: does a delay change the margin the caps need?)
Required margin (kW below the cell mean cap) = 5 - largest safe offset; '+' after a number = lower bound.
Columns: reference runs without delay (tags slack_f10 = ideal + 10 s average, slack_n10f = noise 10 % + 10 s
average), then the delay runs (tags dly*). 'same' = identical to the reference in every scenario.
Run:  module load python/3.11.5 ; python3 scripts/09_dataset_pipeline/summarize_delay.py
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from summarize_slack_margin2 import req, TEST

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
COLS = [("ideal+avg", "slack_f10"), ("100ns ideal", "dly100ns_ideal"),
        ("10%+avg", "slack_n10f"), ("100ns 10%", "dly100ns_n10f"),
        ("1s 10%", "dly1s_n10f"), ("5s 10%", "dly5s_n10f"), ("10s 10%", "dly10s_n10f")]


def fmt(v):
    return "--" if v is None else f"{v[0]:+.0f}{'+' if v[1] else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT") or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    R = {t: {s: req(root, s, t) for s in TEST} for _, t in COLS}
    print("Required margin (kW below the cell mean cap) = 5 - largest safe offset. '+' = lower bound, -- = no result.\n")
    print("scenario " + "".join(f"{lab:>13}" for lab, _ in COLS))
    print("-" * (9 + 13 * len(COLS)))
    for s in TEST:
        print(f"{s:<9}" + "".join(f"{fmt(R[t][s]):>13}" for _, t in COLS))
    print()
    for lab, t, ref in (("100 ns, ideal sensor", "dly100ns_ideal", "slack_f10"), ("100 ns, 10 % noise", "dly100ns_n10f", "slack_n10f")):
        same = sum(1 for s in TEST if R[t][s] is not None and R[t][s] == R[ref][s])
        have = sum(1 for s in TEST if R[t][s] is not None)
        print(f"{lab}: identical to the no-delay run in {same}/{len(TEST)} scenarios ({have} have a result)")
    ok = lambda t: [s for s in TEST if R[t][s] is not None]
    for lab, t in (("1 s", "dly1s_n10f"), ("5 s", "dly5s_n10f"), ("10 s", "dly10s_n10f")):
        r = ok(t)
        if not r:
            print(f"delay {lab}, 10 % noise: no results"); continue
        worse = [s for s in r if R["slack_n10f"][s] is not None and R[t][s][0] > R["slack_n10f"][s][0]]
        mx = max(R[t][s][0] for s in r)
        print(f"delay {lab}, 10 % noise + avg: largest requirement {mx:+.0f} kW over {len(r)} scenarios; "
              f"needs more margin than no delay in {len(worse)}: {', '.join(worse) if worse else '-'}")
    print("\nREAD: a margin above +5 means the present 5 kW margin is not enough; 5 kW grid, one noise seed, 12 scenarios.")


if __name__ == "__main__":
    main()

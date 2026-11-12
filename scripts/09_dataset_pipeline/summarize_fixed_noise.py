"""
SUMMARIZE_FIXED_NOISE.PY  (fixed-size noise check on H09, 2026-11-11)
Reads the two tags written by submit_fixed_noise.sh and prints them next to the earlier 10 % noise results for H09:
largest safe offset (kW added to all six caps) -> margin needed = 5 kW - offset, what was tried, and the shed energy.
Run:  python3 scripts/09_dataset_pipeline/summarize_fixed_noise.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from summarize_sensor_rerun import read_run, cell, PROJECT_ROOT

SID = "H09"
ROWS = [
    ("fixed 2400 W (3 A), 1 s avg",    "fx_a2400_w1s"),
    ("fixed 2400 W (3 A), 100 ms avg", "fx_a2400_w100ms"),
    ("10% Gaussian, 1 s avg",          "sr_g10_w1s"),
    ("10% Gaussian, 100 ms avg",       "sr_g10_w100ms"),
    ("0.5% bounded, 1 s avg",          "sr_b05_w1s"),
    ("OLD ideal sensor",               "slackh_ideal"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT") or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    print(f"Root: {root}\nScenario {SID}. Cell = largest safe offset (kW) -> margin needed (kW) = 5 - offset."
          "  '--' = no result yet, 'none' = even -10 kW failed.\n")
    hdr = f"{'setting':34s}{'offset -> margin':>18s}{'shed kWh':>10s}   tried (S safe / F collapse)"
    print(hdr); print("-" * len(hdr))
    for title, tag in ROWS:
        r = read_run(root, SID, tag)
        kwh = f"{r['kwh']:.1f}" if r and r["kwh"] is not None else "--"
        print(f"{title:34s}{cell(r):>18s}{kwh:>10s}   {r['runs'] if r else '--'}")
    print("\nREADING: 2400 W is about 2 % of an H09 cap (about 110 kW), so it is a smaller noise than 10 %."
          "\nOne scenario, one noise seed, and the 3 A figure is not confirmed: a rough check only.")


if __name__ == "__main__":
    main()

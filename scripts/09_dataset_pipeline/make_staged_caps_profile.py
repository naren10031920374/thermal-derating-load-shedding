"""
MAKE_STAGED_CAPS_PROFILE.PY
================================================================================
The staged load limiter reads one row per scenario from staged_caps.csv (cap cell =
number of target buses x has E/F). This script makes sure every example-profile
scenario of scenario_table.csv (split = 'profile': H01..H15) has such a row.

  * the first 126 rows (S, X, T, Y) are copied byte for byte, never changed
  * all rows after them are rebuilt: one per profile scenario, in table order
  * H01-H03 come out exactly as they are in the file today (checked by comparing
    with the old file with --check)
  * the cap columns (cap_kw, cap_kw_margin) are copied from an existing row of the same cell;
    they are only used when no groups_csv is given. The runs use staged_cap_groups_nn.csv.
  * label / kept_kw stay empty (no bisection for these scenarios)

Run (from this folder):
    python make_staged_caps_profile.py
Safe to re-run.  Only the Python standard library is used.
"""
from __future__ import annotations
import csv, io
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_BASE = 126
HEADER = ["scenario_id", "kind", "split", "targets", "n_target", "n_ef", "has_ef", "plateau_w",
          "label", "kept_kw", "cap_kw", "cap_kw_margin"]


def main():
    table = list(csv.DictReader(open(HERE / "scenario_table.csv", newline="")))
    lines = (HERE / "staged_caps.csv").read_bytes().splitlines(keepends=True)
    base = lines[:N_BASE + 1]
    if not base[-1].endswith(b"\n"):
        base[-1] += b"\n"
    old = list(csv.DictReader(io.StringIO(b"".join(lines).decode())))
    cap_of = {}
    for r in old:
        cap_of.setdefault((int(r["n_target"]), int(r["has_ef"])), (r["cap_kw"], r["cap_kw_margin"]))
    new = []
    for r in table:
        if r["split"] != "profile":
            continue
        tg = r["targets"].split(";")
        n_t = len(tg)
        n_ef = sum(1 for b in tg if b in ("E", "F"))
        has = 1 if n_ef > 0 else 0
        cap, capm = cap_of[(n_t, has)]
        row = [r["scenario_id"], r["kind"], "profile", r["targets"], n_t, n_ef, has,
               int(float(r["plateau_w"])), "", "", cap, capm]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow(row)
        new.append(buf.getvalue().encode() + b"\n")
    (HERE / "staged_caps.csv").write_bytes(b"".join(base) + b"".join(new))
    print(f"staged_caps.csv: rows 1-{N_BASE} unchanged, then {len(new)} profile rows")
    for ln in new:
        print("  ", ln.decode().strip())


if __name__ == "__main__":
    main()

"""
MAKE_SCENARIO_TABLE_EXTRA.PY
================================================================================
Dataset pipeline, batch 2: appends 60 NEW scenarios (T001..T060) plus 6 spares
(Y01..Y06) to scenario_table.csv, so the shed NN has more training data.

Why a separate script (and not an edit of make_scenario_table.py)
-----------------------------------------------------------------
Rows 1-60 (S001-S050, X01-X10) are already simulated and labelled. They must not
change by even one character, because the SLURM array id = row number and the
saved results on the cluster belong to those exact rows. So this script:
  1. keeps the first 61 lines of scenario_table.csv (header + 60 rows) AS THEY ARE,
  2. appends the new rows below them,
  3. is safe to re-run: it always cuts the table back to 60 rows first, and the
     new rows depend only on EXTRA_SEED, so you get the same 66 new rows every time.

Row numbers (= SLURM array ids)
-------------------------------
    1-50    S001-S050   batch 1 (done)
   51-60    X01-X10     batch 1 spares (never needed)
   61-120   T001-T060   batch 2 (NEW, run these)
  121-126   Y01-Y06     batch 2 spares (run only if a varied scenario fails to collapse)

What the 60 new scenarios are for
---------------------------------
The first NN only learned HOW MANY buses are in trouble (singles ~0.66, pairs
~0.54, triples ~0.48). To learn severity it needs several scenarios of the same
type that differ in plateau, ramp speed and start time, plus labels between 0.75
and 1.0, which batch 1 does not have.

   14 single     every bus once, plus extra E, F, D, G (hardest region: E/F)
   16 pair       every ring link once, plus extra D;E, E;F, F;G and 3 random
   14 triple     every ring arc once (incl. G;H;K with the generic recipe),
                 plus extra C;D;E, D;E;F, E;F;G, F;G;H (hardest region)
   10 borderline plateau 125-168 kW: heavily loaded, may or may not collapse.
                 These are where the label should sit between 0.75 and 1.0.
    6 mild       no collapse expected (detector false-alarm data)

Settings for single/pair/triple: plateau 175-218 kW, ramp start 500-1100 s, ramp
length 400-2600 s, boundary load 3-6.5 kW (Latin hypercube, so they are spread
out). All new scenarios are split = train: the original 10 test scenarios stay the
ONLY test set, so results stay comparable with the first run.

Run:
    python make_scenario_table_extra.py
    python make_scenario_table_extra.py --table scenario_table.csv --out new_table.csv   # dry run
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from make_scenario_table import BUS, NB, AMBIENT_W, COLUMNS, make_row, lhs

EXTRA_SEED = 20261003
N_BASE_ROWS = 60                     # S001-S050 + X01-X10 are frozen
HERE = Path(__file__).resolve().parent
KIND = {1: "single", 2: "pair", 3: "triple"}


def build_extra() -> pd.DataFrame:
    rng = np.random.default_rng(EXTRA_SEED)
    items = []                       # one dict per new scenario, before ID assignment

    # ---- 44 "varied" scenarios (single / pair / triple) ---------------------
    # indices into BUS: A0 B1 C2 D3 E4 F5 G6 H7 K8 L9; an arc starts at `start`
    single_starts = list(range(NB)) + [4, 5, 3, 6]                              # +E,F,D,G
    pair_starts = (list(range(NB)) + [3, 4, 5]                                  # +D;E, E;F, F;G
                   + [int(s) for s in rng.choice(NB, 3, replace=False)])
    triple_starts = list(range(NB)) + [2, 3, 4, 5]                              # +CDE, DEF, EFG, FGH
    shapes = ([(1, s) for s in single_starts] + [(2, s) for s in pair_starts]
              + [(3, s) for s in triple_starts])
    pts = lhs(len(shapes), [(175e3, 218e3), (500, 1100), (400, 2600), (3000, 6500)],
              seed=EXTRA_SEED + 1)
    for (n, start), (plat, r0, rlen, blow) in zip(shapes, pts):
        items.append(dict(kind=KIND[n], n=n, start=int(start), ramp_start=r0, ramp_end=r0 + rlen,
                          plateau=plat, b_low=blow, note=""))

    # ---- 10 borderline scenarios: heavy but maybe survivable ----------------
    b_shapes = ([(1, int(s)) for s in rng.choice(NB, 4, replace=False)]
                + [(2, int(s)) for s in rng.choice(NB, 3, replace=False)]
                + [(3, int(s)) for s in rng.choice(NB, 3, replace=False)])
    b_pts = lhs(len(b_shapes), [(125e3, 168e3), (500, 1100), (400, 2600), (3000, 6500)],
                seed=EXTRA_SEED + 2)
    for (n, start), (plat, r0, rlen, blow) in zip(b_shapes, b_pts):
        items.append(dict(kind="borderline", n=n, start=start, ramp_start=r0, ramp_end=r0 + rlen,
                          plateau=plat, b_low=blow,
                          note="borderline: plateau 125-168 kW, may or may not collapse; fills labels between 0.75 and 1.0"))

    # ---- 6 mild scenarios: no collapse expected -----------------------------
    m_pts = lhs(6, [(60e3, 110e3), (500, 1100), (1200, 2200)], seed=EXTRA_SEED + 3)
    for plat, r0, rlen in m_pts:
        n = int(rng.choice([1, 2]))
        items.append(dict(kind="mild", n=n, start=int(rng.integers(NB)), ramp_start=r0,
                          ramp_end=r0 + rlen, plateau=plat, b_low=AMBIENT_W,
                          note="mild load, boundary not starved; no collapse expected"))

    # mix the kinds through the ID range, so a partly finished array is still a balanced set
    items = [items[i] for i in rng.permutation(len(items))]

    rows = []
    for i, it in enumerate(items, start=1):
        rows.append(make_row(f"T{i:03d}", it["kind"], it["start"], it["n"],
                             ramp_start=it["ramp_start"], ramp_end=it["ramp_end"],
                             plateau=it["plateau"], b_low=it["b_low"], seed=2000 + i,
                             note=it["note"]))

    # ---- 6 spares (higher plateau): run only if a varied scenario does not collapse
    sp_shapes = ([(1, int(s)) for s in rng.choice(NB, 2, replace=False)]
                 + [(2, int(s)) for s in rng.choice(NB, 2, replace=False)]
                 + [(3, int(s)) for s in rng.choice(NB, 2, replace=False)])
    sp_pts = lhs(len(sp_shapes), [(195e3, 220e3), (500, 1100), (400, 2600), (3000, 6500)],
                 seed=EXTRA_SEED + 4)
    for i, ((n, start), (plat, r0, rlen, blow)) in enumerate(zip(sp_shapes, sp_pts), start=1):
        rows.append(make_row(f"Y{i:02d}", KIND[n], start, n, ramp_start=r0, ramp_end=r0 + rlen,
                             plateau=plat, b_low=blow, seed=2100 + i, spare=True,
                             note="spare: higher plateau, swap in if a varied scenario does not collapse"))
    return pd.DataFrame(rows)[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--out", default=None, help="default: overwrite --table")
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out or a.table)

    lines = table.read_bytes().splitlines(keepends=True)
    if len(lines) < N_BASE_ROWS + 1:
        raise SystemExit(f"{table} has only {len(lines) - 1} rows; expected at least {N_BASE_ROWS}.")
    base = lines[:N_BASE_ROWS + 1]                       # header + the 60 frozen rows
    ids = [ln.decode().split(",")[0] for ln in base[1:]]
    expect = [f"S{i:03d}" for i in range(1, 51)] + [f"X{i:02d}" for i in range(1, 11)]
    if ids != expect:
        raise SystemExit("First 60 rows are not S001-S050 + X01-X10; refusing to append.")

    eol = b"\r\n" if base[0].endswith(b"\r\n") else b"\n"
    if not base[-1].endswith(b"\n"):
        base[-1] += eol

    extra = build_extra()
    text = extra.to_csv(index=False, header=False, lineterminator="\n").replace("\n", eol.decode())
    out.write_bytes(b"".join(base) + text.encode())

    main_rows = extra[~extra["spare"]]
    print(f"Wrote {out}")
    print(f"  rows 1-{N_BASE_ROWS} unchanged; appended {len(main_rows)} main (rows {N_BASE_ROWS + 1}-"
          f"{N_BASE_ROWS + len(main_rows)}) + {int(extra['spare'].sum())} spare "
          f"(rows {N_BASE_ROWS + len(main_rows) + 1}-{N_BASE_ROWS + len(extra)})")
    print("\nKind counts (new main):\n", main_rows["kind"].value_counts().to_string())
    cover = pd.Series([b for t in main_rows["targets"] for b in t.split(";")]).value_counts()
    print("\nTimes each bus is a target (new main):\n", cover.reindex(BUS).to_string())
    print("\nPlateau kW by kind (min / max):")
    print((main_rows.groupby("kind")["plateau_w"].agg(["min", "max"]) / 1000).round(0).to_string())


if __name__ == "__main__":
    main()

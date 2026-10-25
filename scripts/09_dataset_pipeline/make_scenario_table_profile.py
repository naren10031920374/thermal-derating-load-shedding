"""
MAKE_SCENARIO_TABLE_PROFILE.PY
================================================================================
Appends 15 EXAMPLE-PROFILE scenarios (H01..H15) to scenario_table.csv as rows
127-141 and writes profile_scenarios.csv (which profile file / start offset each
one uses).  H01-H03 are the first three (rows 127-129, unchanged); H04-H15 (rows
130-141) are the extra reshaped scenarios for the margin test: two per cap cell,
each with a different piece of the example profile (see SHAPES below).

What these scenarios are
------------------------
Same buses, boundary buses, plateau, ramp start and ramp end as one existing
scenario each (H01 = S034 single C, H02 = S020 pair A;B, H03 = S012 triple C;D;E,
H04.. = the source named in NEW). The difference: the target buses do NOT follow a
plain ramp + plateau. They follow the shape of the example load profile
(example_load_profile_traced.csv, traced from Figure 1 of the vendor PDF), read from a
chosen start x0. Time scale (1 x-unit = 1 s) and size (100 % = plateau_w) are OUR
assumptions; the PDF gives neither.

Shapes (what the load looks like; the profile has only four real dips, so every shape
is a different place to start reading it). H04-H15 reach >= 93 % of plateau at the
end of the ramp (H02 reaches 91 %), so a collapse is still expected:
  deep-late  H01-H03   deep dip (43-48 %) about 1,300 s after the ramp ends
  near       H04 H07 H11   deep dip (44 %) starts 0-700 s after the ramp ends
  shallow    H06 H08 H13 H15   shallow dip (62-66 %) 1,400-2,300 s after the ramp ends
  in-ramp    H05 H09 H12   deep dip (43 %) in the MIDDLE of the ramp, back to 95 % at ramp end
  flat       H10 H14   stays above 85 % after the ramp ends (control: almost a plain plateau)

Rules kept from make_scenario_table_extra.py
--------------------------------------------
  * rows 1-126 (S, X, T, Y) are copied byte for byte, never changed
  * rows 127-129 (H01-H03) are written exactly as before (same x0, same seed)
  * safe to re-run: it always cuts the table back to 126 rows first
  * split = 'profile', so train_shed_models.py (uses only 'train' / 'test') never
    trains or tests on these rows by accident

Run (from this folder):
    python make_scenario_table_profile.py
    python make_scenario_table_profile.py --table scenario_table.csv --out new_table.csv   # dry run
"""
from __future__ import annotations
import argparse, csv, io
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_BASE_ROWS = 126                      # S001-S050, X01-X10, T001-T060, Y01-Y06 are frozen
PROFILE_FILE = "example_load_profile_traced.csv"

# new id, geometry copied from this existing row, noise seed, profile start x (x-units), why this start
NEW = [
    ("H01", "S034", 3001, 11610, "single bus C; deep dip (44 %) about 1,300 s after the ramp ends"),
    ("H02", "S020", 3002, 16380, "pair A;B; deep dip (48 %) about 1,300 s after the ramp ends"),
    ("H03", "S012", 3003, 20820, "triple C;D;E; deep dip (43 %) about 1,300 s after the ramp ends"),
    ("H04", "S004", 3004, 12660, "single C; near: deep dip (44 %) 690 s after the ramp ends"),
    ("H05", "S008", 3005, 2520,  "single D; in-ramp: deep dip (43 %) in the middle of the ramp"),
    ("H06", "S010", 3006, 24510, "single E; shallow dip (63 %) 1,400 s after the ramp ends"),
    ("H07", "S031", 3007, 11070, "single F; near: deep dip (44 %) 670 s after the ramp ends"),
    ("H08", "S002", 3008, 3930,  "pair K;L; shallow dip (62 %) 1,700 s after the ramp ends"),
    ("H09", "S022", 3009, 3240,  "pair H;K; in-ramp: deep dip (43 %) in the middle of the ramp"),
    ("H10", "S009", 3010, 19230, "pair D;E; flat: stays above 84 %"),
    ("H11", "S017", 3011, 11520, "pair E;F; near: deep dip (44 %) 680 s after the ramp ends"),
    ("H12", "S001", 3012, 2430,  "triple A;B;C; in-ramp: deep dip (43 %) in the middle of the ramp"),
    ("H13", "S016", 3013, 4140,  "triple B;C;D; shallow dip (62 %) 1,900 s after the ramp ends"),
    ("H14", "S013", 3014, 9060,  "triple D;E;F; flat: stays above 84 %"),
    ("H15", "S014", 3015, 9480,  "triple E;F;G; shallow dip (66 %) 2,300 s after the ramp ends"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--out", default=None, help="default: overwrite --table")
    a = ap.parse_args()
    table, out = Path(a.table), Path(a.out or a.table)

    lines = table.read_bytes().splitlines(keepends=True)
    if len(lines) < N_BASE_ROWS + 1:
        raise SystemExit(f"{table} has only {len(lines) - 1} rows; expected at least {N_BASE_ROWS}.")
    base = lines[:N_BASE_ROWS + 1]
    ids = [ln.decode().split(",")[0] for ln in base[1:]]
    expect = ([f"S{i:03d}" for i in range(1, 51)] + [f"X{i:02d}" for i in range(1, 11)]
              + [f"T{i:03d}" for i in range(1, 61)] + [f"Y{i:02d}" for i in range(1, 7)])
    if ids != expect:
        raise SystemExit("First 126 rows are not S001-S050, X01-X10, T001-T060, Y01-Y06; refusing to append.")

    eol = b"\r\n" if base[0].endswith(b"\r\n") else b"\n"
    if not base[-1].endswith(b"\n"):
        base[-1] += eol

    header = next(csv.reader(io.StringIO(base[0].decode())))
    rows = {r[0]: dict(zip(header, r)) for r in csv.reader(io.StringIO(b"".join(base[1:]).decode()))}

    new_lines = []
    for new_id, src, seed, x0, why in NEW:
        r = dict(rows[src])                              # raw strings, copied exactly
        r.update(scenario_id=new_id, seed=str(seed), split="profile", spare="False",
                 note=f"example profile; geometry of {src}; x0={x0}")
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow([r[c] for c in header])
        new_lines.append(buf.getvalue().encode() + eol)
    out.write_bytes(b"".join(base) + b"".join(new_lines))

    prof = out.parent / "profile_scenarios.csv" if a.out is None else Path(a.out).with_name("profile_scenarios.csv")
    prof.write_text("scenario_id,profile_file,profile_x0_s\n" +
                    "".join(f"{i},{PROFILE_FILE},{x0}\n" for i, _, _, x0, _ in NEW))
    print(f"Wrote {out}  (rows 1-{N_BASE_ROWS} unchanged; appended rows {N_BASE_ROWS + 1}-{N_BASE_ROWS + len(NEW)})")
    print(f"Wrote {prof}")
    for i, (new_id, src, seed, x0, why) in enumerate(NEW):
        print(f"  row {N_BASE_ROWS + 1 + i}: {new_id}  <- {src}  seed {seed}  x0 {x0}  ({why})")

if __name__ == "__main__":
    main()

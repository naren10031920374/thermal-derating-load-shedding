"""
MAKE_SCENARIO_TABLE_RANDOM.PY
================================================================================
Adds 220 RANDOM scenarios (plus 22 reserves) to scenario_table.csv, for the
probabilistic "confidence network" (Network B) plan.

Why random (and not hand-built like S, T, H)
--------------------------------------------
The safety statement "all N new scenarios stayed safe, so at 95 % confidence at most
about 1 - 0.05^(1/N) of similar new scenarios fail" needs scenarios that are drawn at
random from one fixed distribution. Hand-built scenarios do not give that. So here every
setting is an independent random draw (no Latin hypercube, no hand-picking), from a seed
written below. Nothing is simulated by this script; it only writes table rows.

The three sets (rows follow the order of the table, so a partly finished array is still
a valid set)
-----------------------------------------------------------------------------------
    rows 142-241   RT001-RT100   split 'rtrain'  train Network B
                                  (rows 142-171 = RT001-RT030 = the PILOT)
    rows 242-301   RC001-RC060   split 'rcheck'  check / fix overconfidence of B
    rows 302-361   RF001-RF060   split 'rtest'   FINAL test, never used before the end
    rows 362-383   RZ01-RZ22     split = the set it backs up ('rtrain' 10, 'rcheck' 6,
                                  'rtest' 6), spare = True. Use one ONLY if a scenario of
                                  that set does not collapse in its no-shed run (same rule
                                  as the X and Y spares). Take them in order.

What a random scenario is (the distribution; each value is an independent draw)
-------------------------------------------------------------------------------
    number of limited buses  1, 2 or 3, equally likely
    first bus of the arc     any of the 10 buses, equally likely (arc runs round the ring)
    plateau                  uniform 170-218 kW          (batch 1 was 170-210, batch 2 175-218)
    ramp start               uniform 500-1100 s
    ramp length              uniform 400-2600 s
    boundary buses' load     uniform 3.0-6.5 kW
    load shape               80 % plain ramp + plateau, 20 % follow the traced example
                             load profile, starting at a random x0 (multiple of 10).
                             A start x0 is accepted only if the profile is still at least
                             92 % of peak at the end of the ramp (same idea as H01-H15),
                             so a collapse is still expected.
So the population is "load growth that would collapse a bus without shedding" (that is
what the limiter is for). Scenarios that do NOT collapse in the no-shed run are replaced by
the reserves above, which come from the same distribution.

Rules kept from make_scenario_table_extra.py / _profile.py
----------------------------------------------------------
  * rows 1-141 (S, X, T, Y, H) are copied byte for byte, never changed
  * safe to re-run: it always cuts scenario_table.csv back to 141 rows first, and the
    new rows depend only on RANDOM_SEED, so you get the same 242 rows every time
  * also appends the matching rows to staged_caps.csv (the staged limiter stops with
    "not in caps_csv" if a scenario is missing there) and the 20 % profile scenarios to
    profile_scenarios.csv. Rows 1-141 / H01-H15 of those two files stay unchanged.
  * split values 'rtrain' / 'rcheck' / 'rtest' (not 'train' / 'test'), so
    train_shed_models.py and the older NN code never pick these rows up by accident
  * noise seeds (load noise) 4001-4242, a new range

WARNING: make_scenario_table_profile.py and make_staged_caps_profile.py cut their files
back to 126 rows (and rewrite profile_scenarios.csv with H01-H15 only). If you ever re-run
one of them, re-run THIS script afterwards.

Run (from scripts/09_dataset_pipeline). Only numpy and the Python standard library:
    python make_scenario_table_random.py
    python make_scenario_table_random.py --dry-run       # writes new_*.csv next to the tables, changes nothing else
"""
from __future__ import annotations
import argparse, csv, io
from pathlib import Path
import numpy as np

RANDOM_SEED = 20261102
N_BASE_ROWS = 141                    # S001-S050, X01-X10, T001-T060, Y01-Y06, H01-H15 are frozen
N_PROFILE_LINES = 16                 # profile_scenarios.csv: header + H01-H15
HERE = Path(__file__).resolve().parent
PROFILE_FILE = "example_load_profile_traced.csv"

BUS = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
NB = len(BUS)
AMBIENT_W = 27000
KIND = {1: "single", 2: "pair", 3: "triple"}

# (prefix, how many, split)
SETS = [("RT", 100, "rtrain"), ("RC", 60, "rcheck"), ("RF", 60, "rtest")]
RESERVES = [("rtrain", 10), ("rcheck", 6), ("rtest", 6)]

PLATEAU_KW = (170.0, 218.0)
RAMP_START_S = (500.0, 1100.0)
RAMP_LEN_S = (400.0, 2600.0)
BOUNDARY_KW = (3.0, 6.5)
P_PROFILE = 0.20
PROFILE_MIN_AT_RAMP_END = 0.92
SIM_END_S = 5000

COLUMNS = ["scenario_id", "kind", "targets", "boundary", "n_target", "ramp_start_s", "ramp_end_s",
           "plateau_w", "boundary_low_w", "preheat_w", "boundary_preheat_w", "jitter_frac",
           "stepdown_start_s", "stepdown_end_s", "recovery_w", "boundary_recovery_w",
           "seed", "split", "spare", "note"]
CAPS_HEADER = ["scenario_id", "kind", "split", "targets", "n_target", "n_ef", "has_ef", "plateau_w",
               "label", "kept_kw", "cap_kw", "cap_kw_margin"]


def arc(start, n):
    return [BUS[(start + i) % NB] for i in range(n)]


def boundary_of(start, n):
    return [BUS[(start - 1) % NB], BUS[(start + n) % NB]]


def draw_one(rng, prof_x, prof_p):
    """One random scenario (a dict of settings). The order of the draws is fixed."""
    n = int(rng.integers(1, 4))
    start = int(rng.integers(NB))
    plateau = rng.uniform(*PLATEAU_KW) * 1000.0
    r0 = rng.uniform(*RAMP_START_S)
    rlen = rng.uniform(*RAMP_LEN_S)
    blow = rng.uniform(*BOUNDARY_KW) * 1000.0
    ramp_start = int(round(r0))
    ramp_end = int(round(r0 + rlen))
    x0 = None
    if rng.random() < P_PROFILE:
        x_hi = float(prof_x[-1]) - SIM_END_S
        for _ in range(10000):
            cand = 10 * int(round(rng.uniform(float(prof_x[0]), x_hi) / 10.0))
            if np.interp(cand + ramp_end, prof_x, prof_p) / 100.0 >= PROFILE_MIN_AT_RAMP_END:
                x0 = cand
                break
        if x0 is None:
            raise SystemExit("could not find a profile start that satisfies the rule")
    return dict(n=n, start=start, plateau=int(round(plateau, -3)), blow=int(round(blow, -2)),
                ramp_start=ramp_start, ramp_end=ramp_end, x0=x0)


def build(prof_x, prof_p):
    rng = np.random.default_rng(RANDOM_SEED)
    items = []                                           # (id, split, spare, setting dict)
    for prefix, count, split in SETS:
        for i in range(1, count + 1):
            items.append((f"{prefix}{i:03d}", split, False, draw_one(rng, prof_x, prof_p)))
    z = 1
    for split, count in RESERVES:
        for _ in range(count):
            items.append((f"RZ{z:02d}", split, True, draw_one(rng, prof_x, prof_p)))
            z += 1
    rows = []
    for k, (sid, split, spare, s) in enumerate(items, start=1):
        n, start = s["n"], s["start"]
        note = "random scenario"
        if s["x0"] is not None:
            note += f"; example profile x0={s['x0']}"
        if spare:
            note += f"; reserve for {split}: use only if a scenario of that set does not collapse"
        rows.append({
            "scenario_id": sid, "kind": KIND[n], "targets": ";".join(arc(start, n)),
            "boundary": ";".join(boundary_of(start, n)), "n_target": n,
            "ramp_start_s": s["ramp_start"], "ramp_end_s": s["ramp_end"],
            "plateau_w": s["plateau"], "boundary_low_w": s["blow"],
            "preheat_w": AMBIENT_W, "boundary_preheat_w": AMBIENT_W, "jitter_frac": "0.0",
            "stepdown_start_s": "", "stepdown_end_s": "", "recovery_w": "", "boundary_recovery_w": "",
            "seed": 4000 + k, "split": split, "spare": "True" if spare else "False", "note": note,
            "_x0": s["x0"],
        })
    return rows


def csv_line(values):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(values)
    return buf.getvalue()


def read_base(path, n_lines, label):
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < n_lines:
        raise SystemExit(f"{path} has only {len(lines)} lines; expected at least {n_lines}.")
    base = lines[:n_lines]
    if not base[-1].endswith(b"\n"):
        base[-1] += b"\n"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE), help="folder with the three tables (default: this folder)")
    ap.add_argument("--dry-run", action="store_true", help="write new_<name>.csv instead of changing the tables")
    a = ap.parse_args()
    d = Path(a.dir)
    f_tab, f_cap, f_prof = d / "scenario_table.csv", d / "staged_caps.csv", d / "profile_scenarios.csv"

    def target(p):
        return p.with_name("new_" + p.name) if a.dry_run else p

    # ---- profile shape --------------------------------------------------------
    with open(d / PROFILE_FILE, newline="") as fh:
        pr = list(csv.DictReader(fh))
    prof_x = np.array([float(r["x_units"]) for r in pr])
    prof_p = np.array([float(r["power_units"]) for r in pr])

    # ---- frozen base of scenario_table.csv -------------------------------------
    base = read_base(f_tab, N_BASE_ROWS + 1, "scenario_table")
    ids = [ln.decode().split(",")[0] for ln in base[1:]]
    expect = ([f"S{i:03d}" for i in range(1, 51)] + [f"X{i:02d}" for i in range(1, 11)]
              + [f"T{i:03d}" for i in range(1, 61)] + [f"Y{i:02d}" for i in range(1, 7)]
              + [f"H{i:02d}" for i in range(1, 16)])
    if ids != expect:
        raise SystemExit("First 141 rows are not S, X, T, Y, H as expected; refusing to append.")
    header = next(csv.reader(io.StringIO(base[0].decode())))
    if header != COLUMNS:
        raise SystemExit("scenario_table.csv header differs from the expected columns; refusing to append.")

    rows = build(prof_x, prof_p)
    first_new = N_BASE_ROWS + 1                      # table row number of the first new scenario (header not counted)
    new_tab = [(csv_line([r[c] for c in COLUMNS]) + "\n").encode() for r in rows]
    target(f_tab).write_bytes(b"".join(base) + b"".join(new_tab))

    # ---- staged_caps.csv: one row per new scenario -------------------------------
    cbase = read_base(f_cap, N_BASE_ROWS + 1, "staged_caps")
    cold = list(csv.DictReader(io.StringIO(b"".join(cbase).decode())))
    if list(cold[0].keys()) != CAPS_HEADER or [r["scenario_id"] for r in cold] != expect:
        raise SystemExit("staged_caps.csv does not hold exactly the 141 expected rows; refusing to append.")
    cap_of = {}
    for r in cold:
        cap_of.setdefault((int(r["n_target"]), int(r["has_ef"])), (r["cap_kw"], r["cap_kw_margin"]))
    new_caps = []
    for r in rows:
        tg = r["targets"].split(";")
        n_ef = sum(b in ("E", "F") for b in tg)
        has = 1 if n_ef else 0
        cap, capm = cap_of[(len(tg), has)]
        new_caps.append((csv_line([r["scenario_id"], r["kind"], r["split"], r["targets"], len(tg), n_ef,
                                   has, r["plateau_w"], "", "", cap, capm]) + "\n").encode())
    target(f_cap).write_bytes(b"".join(cbase) + b"".join(new_caps))

    # ---- profile_scenarios.csv: keep header + H01-H15, add the random profile ones ---
    pbase = read_base(f_prof, N_PROFILE_LINES, "profile_scenarios")
    new_prof = [f"{r['scenario_id']},{PROFILE_FILE},{r['_x0']}\n".encode() for r in rows if r["_x0"] is not None]
    target(f_prof).write_bytes(b"".join(pbase) + b"".join(new_prof))

    # ---- summary ---------------------------------------------------------------
    print(f"Wrote {target(f_tab).name}: rows 1-{N_BASE_ROWS} unchanged, appended {len(rows)} rows "
          f"(rows {first_new}-{first_new + len(rows) - 1})")
    print(f"Wrote {target(f_cap).name}: {len(new_caps)} rows appended")
    print(f"Wrote {target(f_prof).name}: {len(new_prof)} profile rows appended after H01-H15")
    pos = first_new
    for prefix, count, split in SETS:
        sub = rows[pos - first_new: pos - first_new + count]
        kinds = {k: sum(r["kind"] == k for r in sub) for k in KIND.values()}
        ef = sum(any(b in ("E", "F") for b in r["targets"].split(";")) for r in sub)
        prof = sum(r["_x0"] is not None for r in sub)
        pl = [r["plateau_w"] / 1000 for r in sub]
        cover = {b: sum(b in r["targets"].split(";") for r in sub) for b in BUS}
        print(f"\n{prefix}{1:03d}-{prefix}{count:03d}  rows {pos}-{pos + count - 1}  split {split}")
        print(f"  kinds {kinds}   with E or F: {ef}   profile-shaped: {prof}   plateau {min(pl):.0f}-{max(pl):.0f} kW")
        print("  times each bus is a target: " + " ".join(f"{b}{cover[b]}" for b in BUS))
        pos += count
    print(f"\nReserves RZ01-RZ22: rows {pos}-{pos + sum(c for _, c in RESERVES) - 1}")
    print("PILOT = rows 142-171 (RT001-RT030).")


if __name__ == "__main__":
    main()

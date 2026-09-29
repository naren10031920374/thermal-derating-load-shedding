"""
MAKE_SCENARIO_TABLE.PY
================================================================================
Phase 1, step 1 of the dataset-generation pipeline.

Builds scenario_table.csv: one row per scenario. A later MATLAB function
(run_scenario.m, step 2) reads one row and builds that scenario's load
profile, so ONE script drives all 50 scenarios.

What a row means
----------------
The load recipe is the one already used in the project:
  - "target" buses ramp from ambient (27 kW) up to plateau_w (heavy load)
  - the ring neighbours of the target arc ("boundary" buses) are starved
    down to boundary_low_w (their rescue paths are removed)
  - every other bus stays near ambient (slow sine + noise)
  - loads are clipped to [2000, 220000] W, as in the v4 / three-bus scripts

Ring order comes from the converter list AB,BC,CD,DE,EF,FG,GH,HK,KL,AL:
    A - B - C - D - E - F - G - H - K - L - (back to A)

Scenario mix (50 scenarios)
---------------------------
  35 "varied"   : 15 single-bus, 12 adjacent pairs, 8 regional triples
   5 "mild"     : no collapse expected (false-alarm checks for the detector)
  10 "repeat"   : known recipes (Bus C v4, G/H/K) and Bus F, new seeds
plus 10 spare rows (X01..X10) to swap in if a scenario turns out NOT to
collapse in its no-shed run.

Continuous settings (plateau, ramp timing, boundary load) are spread with
Latin hypercube sampling so scenarios are not clustered.

IMPORTANT: none of these settings has been simulated yet. The no-shed run in
the pilot decides which scenarios really collapse. Plateau values are centred
on the two recipes that did collapse (200 kW for Bus C v4, 190 kW for G/H/K).

Run:
    python make_scenario_table.py
Writes scenario_table.csv next to this script (deterministic: fixed seed).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

MASTER_SEED = 20260929
BUS = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]
NB = len(BUS)
AMBIENT_W = 27000
OUT = Path(__file__).with_name("scenario_table.csv")

COLUMNS = [
    "scenario_id", "kind", "targets", "boundary", "n_target",
    "ramp_start_s", "ramp_end_s", "plateau_w", "boundary_low_w",
    "preheat_w", "boundary_preheat_w", "jitter_frac",
    "stepdown_start_s", "stepdown_end_s", "recovery_w", "boundary_recovery_w",
    "seed", "split", "spare", "note",
]


def arc(start: int, n: int) -> list[str]:
    """n consecutive buses on the ring starting at index `start`."""
    return [BUS[(start + i) % NB] for i in range(n)]


def boundary_of(start: int, n: int) -> list[str]:
    """The two ring neighbours just outside the arc."""
    return [BUS[(start - 1) % NB], BUS[(start + n) % NB]]


def make_row(sid, kind, start, n, *, ramp_start, ramp_end, plateau, b_low,
             seed, note="", preheat=AMBIENT_W, b_preheat=AMBIENT_W, jitter=0.0,
             sd_start=np.nan, sd_end=np.nan, recovery=np.nan, b_recovery=np.nan,
             spare=False):
    return {
        "scenario_id": sid, "kind": kind,
        "targets": ";".join(arc(start, n)),
        "boundary": ";".join(boundary_of(start, n)),
        "n_target": n,
        "ramp_start_s": int(round(ramp_start)), "ramp_end_s": int(round(ramp_end)),
        "plateau_w": int(round(plateau, -3)), "boundary_low_w": int(round(b_low, -2)),
        "preheat_w": preheat, "boundary_preheat_w": b_preheat, "jitter_frac": jitter,
        "stepdown_start_s": sd_start, "stepdown_end_s": sd_end,
        "recovery_w": recovery, "boundary_recovery_w": b_recovery,
        "seed": int(seed), "split": "train", "spare": spare, "note": note,
    }


def lhs(n_points, bounds, seed):
    """Latin hypercube sample scaled to [(lo, hi), ...] -> array (n, d)."""
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n_points)
    return qmc.scale(unit, lo, hi)


def build() -> pd.DataFrame:
    rng = np.random.default_rng(MASTER_SEED)
    rows = []

    # ---- 35 varied scenarios: which buses fail -----------------------------
    # singles: every bus once, plus 5 buses picked a second time
    single_starts = list(range(NB)) + list(rng.choice(NB, 5, replace=False))
    # pairs: every ring link once (10), plus 2 repeats
    pair_starts = list(range(NB)) + list(rng.choice(NB, 2, replace=False))
    # triples: 8 distinct arcs, excluding G-H-K (start 6), which is a known case
    triple_starts = list(rng.choice([i for i in range(NB) if i != 6], 8, replace=False))
    shapes = ([(1, s) for s in single_starts] + [(2, s) for s in pair_starts]
              + [(3, s) for s in triple_starts])
    order = rng.permutation(len(shapes))          # mix kinds through the ID range
    shapes = [shapes[i] for i in order]

    # plateau 170-210 kW | ramp start 600-1000 s | ramp length 500-2200 s | boundary 3-6 kW
    pts = lhs(len(shapes), [(170e3, 210e3), (600, 1000), (500, 2200), (3000, 6000)],
              seed=MASTER_SEED + 1)
    for i, ((n, start), (plat, r0, rlen, blow)) in enumerate(zip(shapes, pts), start=1):
        kind = {1: "single", 2: "pair", 3: "triple"}[n]
        rows.append(make_row(f"S{i:03d}", kind, start, n,
                             ramp_start=r0, ramp_end=r0 + rlen, plateau=plat,
                             b_low=blow, seed=1000 + i))

    # ---- 5 mild scenarios: no collapse expected ----------------------------
    mild_pts = lhs(5, [(60e3, 100e3), (600, 1000), (1200, 2200)], seed=MASTER_SEED + 2)
    for j, (plat, r0, rlen) in enumerate(mild_pts):
        n = int(rng.choice([1, 2]))
        start = int(rng.integers(NB))
        rows.append(make_row(f"S{36 + j:03d}", "mild", start, n,
                             ramp_start=r0, ramp_end=r0 + rlen, plateau=plat,
                             b_low=AMBIENT_W, seed=1100 + j,
                             note="mild load, boundary not starved; no collapse expected"))

    # ---- 10 repeats of known cases -----------------------------------------
    sid = 41
    # Bus C v4: exact recipe (seed 104) + 3 new seeds
    for k, seed in enumerate([104, 1201, 1202, 1203]):
        note = ("EXACT repeat of generate_gradual_busC_scenario_v4 (seed 104)"
                if seed == 104 else "Bus C v4 recipe, new seed")
        rows.append(make_row(f"S{sid:03d}", "repeat", 2, 1, ramp_start=800, ramp_end=2800,
                             plateau=200e3, b_low=3000, seed=seed, note=note))
        sid += 1
    # G/H/K: exact recipe (seed 201) + 2 new seeds. Includes preheat, surge
    # 800-1400 s, plateau to 3400 s, step-down to 3700 s (as in the script).
    for seed in [201, 1211, 1212]:
        note = ("EXACT repeat of generate_three_bus_collapse_scenario_v1 (seed 201)"
                if seed == 201 else "G/H/K recipe, new seed")
        rows.append(make_row(f"S{sid:03d}", "repeat", 6, 3, ramp_start=800, ramp_end=1400,
                             plateau=190e3, b_low=3000, seed=seed, note=note,
                             preheat=33000, b_preheat=31000, jitter=0.03,
                             sd_start=3400, sd_end=3700, recovery=45000, b_recovery=27000))
        sid += 1
    # Bus F: generic single-bus recipe, 3 seeds. NOTE: the original baseline
    # Bus F scenario uses a different generator that is not in this recipe yet.
    for seed in [1221, 1222, 1223]:
        rows.append(make_row(f"S{sid:03d}", "repeat", 5, 1, ramp_start=800, ramp_end=2800,
                             plateau=200e3, b_low=3000, seed=seed,
                             note="Bus F via the generic single-bus recipe (NOT the original baseline generator)"))
        sid += 1

    # ---- 10 spare rows (used only if a scenario fails the no-shed screen) ---
    spare_shapes = [(1, int(s)) for s in rng.choice(NB, 4, replace=False)] \
        + [(2, int(s)) for s in rng.choice(NB, 3, replace=False)] \
        + [(3, int(s)) for s in rng.choice([i for i in range(NB) if i != 6], 3, replace=False)]
    sp_pts = lhs(len(spare_shapes), [(190e3, 220e3), (600, 1000), (500, 2200), (3000, 6000)],
                 seed=MASTER_SEED + 3)
    for i, ((n, start), (plat, r0, rlen, blow)) in enumerate(zip(spare_shapes, sp_pts), start=1):
        kind = {1: "single", 2: "pair", 3: "triple"}[n]
        rows.append(make_row(f"X{i:02d}", kind, start, n, ramp_start=r0, ramp_end=r0 + rlen,
                             plateau=plat, b_low=blow, seed=1300 + i, spare=True,
                             note="spare: higher plateau, swap in if a varied scenario does not collapse"))

    df = pd.DataFrame(rows)[COLUMNS]
    df = assign_test_split(df, rng)
    return df


def assign_test_split(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Hold out 10 of the 50 main scenarios, chosen so every bus appears as a
    target in at least one held-out scenario. Spares are never held out."""
    main = df[~df["spare"]]
    chosen: list[str] = []
    covered: set[str] = set()
    candidates = main[main["kind"].isin(["single", "pair", "triple"])]
    while covered != set(BUS):
        best_sid, best_gain = None, 0
        for _, r in candidates.sample(frac=1, random_state=int(rng.integers(1e9))).iterrows():
            if r["scenario_id"] in chosen:
                continue
            gain = len(set(r["targets"].split(";")) - covered)
            if gain > best_gain:
                best_sid, best_gain = r["scenario_id"], gain
        chosen.append(best_sid)
        covered |= set(main.loc[main["scenario_id"] == best_sid, "targets"].iloc[0].split(";"))
    # make sure at least one mild scenario is held out, then fill up to 10
    mild_ids = list(main.loc[main["kind"] == "mild", "scenario_id"])
    if not any(s in mild_ids for s in chosen):
        chosen.append(str(rng.choice(mild_ids)))
    pool = [s for s in main["scenario_id"] if s not in chosen]
    need = 10 - len(chosen)
    if need > 0:
        chosen += [str(s) for s in rng.choice(pool, need, replace=False)]
    df.loc[df["scenario_id"].isin(chosen), "split"] = "test"
    return df


if __name__ == "__main__":
    table = build()
    table.to_csv(OUT, index=False)
    main = table[~table["spare"]]
    print(f"Wrote {OUT}  ({len(table)} rows: {len(main)} main + {int(table['spare'].sum())} spare)")
    print("\nKind counts (main):\n", main["kind"].value_counts().to_string())
    print("\nSplit counts (main):\n", main["split"].value_counts().to_string())
    cover = pd.Series([b for t in main["targets"] for b in t.split(";")]).value_counts()
    print("\nTimes each bus is a target (main):\n", cover.reindex(BUS).to_string())
    test_cover = pd.Series([b for t in main[main['split']=='test']["targets"] for b in t.split(";")])
    print("\nBuses covered by the test split:", sorted(set(test_cover)))

"""
SUMMARIZE_RANDOM_FINE.PY  (fine 1 kW test on the 30 random pilot scenarios, noise seeds 1 and 2)
================================================================================
Reads the pilot result (tag rnd_g10_w1s, 5 kW steps, seed 1) and the fine results written by
submit_random_fine.sh (tags rnd_fine_g10_w1s_s1 and _s2) and shows, for each scenario, the largest
safe offset to the nearest kW, for both noise seeds.

  offset +5 = margin 0 (caps at the network's own prediction). margin needed = 5 - offset.
  L1 / L2  = largest safe offset found with noise seed 1 / seed 2 (kW). The true limit lies in [L, L+1).
  'below'  = seed 2 collapsed even at the bottom of the bracket (+5 or +10): its limit is below the bracket.
  diff     = L2 - L1  (how much a different noise pattern moves the limit)

Also writes random_labels_fine.csv: one row per simulation that was run (pilot rows, seed 1, plus the
fine rows of both seeds): scenario, cap, offset, noise seed, survived 1/0.

Run (project root, after  module load python/3.11.5):
    python3 scripts/09_dataset_pipeline/summarize_random_fine.py
Only the Python standard library is used.
"""
from __future__ import annotations
import argparse, csv, json, os, statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
COARSE = "rnd_g10_w1s"
FINE = {1: "rnd_fine_g10_w1s_s1", 2: "rnd_fine_g10_w1s_s2"}
PRESENT = 5.0


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


def runs_of(res):
    if not res or res.get("base_cap_kw") is None:
        return []
    caps = [float(c) for c in listify(res.get("cap_tried_kw"))]
    safe = [bool(s) for s in listify(res.get("cap_safe"))]
    return [(c - res["base_cap_kw"], c, s) for c, s in zip(caps, safe)]


def limit(fine, c):
    """-> ('ok', L) | ('below', None) | ('pending', None)"""
    if fine is None:
        return "pending", None
    if fine.get("status") == "safe_found":
        return "ok", float(fine["first_safe_offset_kw"])
    if fine.get("status") == "no_safe_cap_tried":
        return "ok_floor", float(c)          # every fine cap collapsed: limit in [c, c+1)
    return "pending", None


def fmt(v):
    return "" if v is None else f"{v:+.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--table", default=str(HERE / "scenario_table.csv"))
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=30)
    ap.add_argument("--out", default="random_labels_fine.csv")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("DATASET_OUTPUT_ROOT") or PROJECT_ROOT / "model_outputs" / "dataset_pipeline")
    with open(a.table, newline="") as fh:
        tab = {r["scenario_id"]: r for r in csv.DictReader(fh)}

    ids = [f"RT{n:03d}" for n in range(a.first, a.last + 1)]
    info = {}
    for sid in ids:
        r = tab.get(sid)
        if r is None:
            continue
        coarse = load(root / sid / f"staged_{sid}_{COARSE}_result.json")
        c = float(coarse["first_safe_offset_kw"]) if coarse and coarse.get("status") == "safe_found" else None
        f1 = load(root / sid / f"staged_{sid}_{FINE[1]}_result.json")
        f2 = load(root / sid / f"staged_{sid}_{FINE[2]}_result.json")
        L1 = limit(f1, c)[1] if c is not None else None
        L2 = None
        s2 = "pending"
        if c is not None and f2 is not None:
            if f2.get("status") == "safe_found":
                L2, s2 = float(f2["first_safe_offset_kw"]), "ok"
            elif f2.get("status") == "no_safe_cap_tried":
                s2 = "below"
        has_ef = int(any(b in ("E", "F") for b in r["targets"].split(";")))
        info[sid] = dict(row=r, c=c, L1=L1, L2=L2, s2=s2, f1=f1, f2=f2, coarse=coarse, has_ef=has_ef)

    print(f"Root: {root}\nCell = largest safe offset in kW (the true limit is in [L, L+1)).  margin needed = 5 - L.\n")
    print("A. PER SCENARIO")
    hdr = f"{'id':7s}{'targets':8s}{'E/F':>4s}{'plat kW':>8s}{'pilot':>7s}{'L seed1':>9s}{'L seed2':>9s}{'diff':>6s}"
    print(hdr); print("-" * len(hdr))
    for sid in ids:
        d = info.get(sid)
        if d is None:
            continue
        l2 = "below" if d["s2"] == "below" else fmt(d["L2"])
        diff = "" if d["L1"] is None or d["L2"] is None else f"{d['L2'] - d['L1']:+.0f}"
        print(f"{sid:7s}{d['row']['targets']:8s}{d['has_ef']:>4d}{int(d['row']['plateau_w'])/1000:>8.0f}{fmt(d['c']):>7s}"
              f"{fmt(d['L1']):>9s}{l2:>9s}{diff:>6s}")

    done1 = [d for d in info.values() if d["L1"] is not None]
    done2 = [d for d in info.values() if d["L2"] is not None or d["s2"] == "below"]
    both = [d for d in info.values() if d["L1"] is not None and d["L2"] is not None]
    print("\nB. SUMMARY")
    print(f"  seed 1 done: {len(done1)} of {len(info)}    seed 2 done: {len(done2)} of {len(info)}    both: {len(both)}")
    if done1:
        cnt = Counter(int(d["L1"]) for d in done1)
        print("  seed 1, largest safe offset -> scenarios:  " + "   ".join(f"{k:+d}: {cnt[k]}" for k in sorted(cnt, reverse=True)))
        print(f"  seed 1 limit: min {min(d['L1'] for d in done1):+.0f}, max {max(d['L1'] for d in done1):+.0f}, "
              f"std {statistics.pstdev([d['L1'] for d in done1]):.2f} kW over {len(done1)} scenarios")
    if done2:
        cnt2 = Counter(int(d["L2"]) for d in done2 if d["L2"] is not None)
        print("  seed 2, largest safe offset -> scenarios:  " + "   ".join(f"{k:+d}: {cnt2[k]}" for k in sorted(cnt2, reverse=True))
              + f"   below bracket: {sum(d['s2'] == 'below' for d in done2)}")
    if both:
        diffs = [d["L2"] - d["L1"] for d in both]
        same = sum(x == 0 for x in diffs)
        print(f"  seed 2 minus seed 1 (kW): same limit in {same} of {len(both)}; mean |diff| {statistics.mean(abs(x) for x in diffs):.2f}; "
              f"largest |diff| {max(abs(x) for x in diffs):.0f}")
    worst = []
    for d in info.values():
        if d["L1"] is None:
            continue
        if d["s2"] == "below":
            worst.append(None)                        # worse than the bracket bottom
        elif d["L2"] is not None:
            worst.append(min(d["L1"], d["L2"]))
        else:
            worst.append(d["L1"])
    if worst:
        unsafe0 = sum(w is None or w < PRESENT for w in worst)
        print(f"  worst of the seeds below +5 (not safe at margin 0): {unsafe0} of {len(worst)}")
    grp = defaultdict(list)
    for d in done1:
        grp[(int(d["row"]["n_target"]), d["has_ef"])].append(d["L1"])
    if grp:
        print("\n  seed 1 limit by group (buses limited, E/F involved): count, min, mean, max (kW)")
        for k in sorted(grp):
            v = grp[k]
            print(f"    {k[0]} bus{'es' if k[0] > 1 else ' '}  {'with E/F' if k[1] else 'no E/F  '}  n={len(v):>2d}  {min(v):+.0f}  {statistics.mean(v):+.1f}  {max(v):+.0f}")

    print("\nC. READING")
    if len(both) >= 5:
        if all(x == 0 for x in diffs):
            print("  Both noise seeds give the SAME limit in every scenario: the noise barely moves the limit at 1 kW. A confidence number")
            print("  would be nearly 0 % or 100 %, so a fixed rule (margin 0, or one margin per group) is probably enough.")
        else:
            print("  The noise seeds move the limit by up to " + f"{max(abs(x) for x in diffs):.0f} kW: a confidence number has something real to describe.")
            print("  With only two seeds this is a first look; more seeds near the limit would be needed to train Network B.")
    else:
        print("  Not enough finished scenarios with both seeds yet.")

    out = Path(a.out)
    cols = ["scenario_id", "split", "kind", "targets", "n_target", "has_ef", "plateau_w", "ramp_start_s", "ramp_end_s",
            "boundary_low_w", "noise_seed", "cap_kw", "offset_kw", "survived", "tag"]
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for sid in ids:
            d = info.get(sid)
            if d is None:
                continue
            r = d["row"]
            for tag, res, seed in [(COARSE, d["coarse"], 1), (FINE[1], d["f1"], 1), (FINE[2], d["f2"], 2)]:
                for o, cap, s in runs_of(res):
                    w.writerow([sid, r["split"], r["kind"], r["targets"], r["n_target"], d["has_ef"], r["plateau_w"],
                                r["ramp_start_s"], r["ramp_end_s"], r["boundary_low_w"], seed, round(cap, 3), round(o, 3), int(s), tag])
                    n += 1
    print(f"\nWrote {out}  ({n} simulation rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Result figures in the same style as the S012/S020 figure: for every case, the 10 bus voltages (top) and the load
asked for / load the limited bus got (bottom). Every curve comes from the trace_<ID>_<tag>_*.csv files written by
run_staged_limiter.m with save_trace on (see submit_trace_figures.sh). Nothing is simulated or invented here.
    python3 make_result_graphs.py --dir model_outputs/dataset_pipeline --out result_graphs
"""
import argparse, glob, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_slide_graphs as g

def trace(d, sid, tag):
    hits = []
    for pat in (f"trace_{sid}_{tag}_staged_*kW.csv", f"trace_{sid}_{tag}_limiter_*kW.csv"):
        hits += glob.glob(os.path.join(d, sid, pat)) + glob.glob(os.path.join(d, pat))
    if not hits:
        raise SystemExit(f"no trace for {sid} with tag {tag} under {d}")
    return g.load_trace(sorted(hits)[0])

# (file name, [(scenario, tag, panel title), ...])
FIGS = [
    ("r10_ideal_room",     [("H01", "tr_idp10",   "H01, ideal sensor, caps +10 kW"),
                            ("H01", "tr_idp15",   "H01, ideal sensor, caps +15 kW")]),
    ("r09_averaging",      [("H01", "tr_raw10",   "H01, 10% noise, readings not averaged"),
                            ("H01", "tr_n10fm5",  "H01, 10% noise, 10 s average")]),
    ("r11_noise",          [("H01", "tr_n2f0",    "H01, 2% noise, 5 kW margin"),
                            ("H01", "tr_n10f0",   "H01, 10% noise, 5 kW margin")]),
    ("r12_group",          [("H01", "tr_n10f0",   "H01 (1 bus), 10% noise, 5 kW margin"),
                            ("H12", "tr_n10f0",   "H12 (3 buses), 10% noise, 5 kW margin")]),
    ("r13_seed",           [("H04", "tr_n10f0",   "H04, 10% noise, seed 1, 5 kW margin"),
                            ("H04", "tr_n10f0s2", "H04, 10% noise, seed 2, 5 kW margin")]),
    ("r16_profile_margin", [("H09", "tr_n10f0",   "H09, 10% noise, 5 kW margin"),
                            ("H09", "tr_n10fm5",  "H09, 10% noise, 10 kW margin")]),
    ("r17_shape",          [("H09", "tr_n10f0",   "H09 (dip in the ramp), 10% noise, 5 kW margin"),
                            ("H10", "tr_n10f0",   "H10 (almost flat), 10% noise, 5 kW margin")]),
]

def fig(d, out, name, panels):
    f, ax = plt.subplots(2, len(panels), figsize=(7 * len(panels), 7), sharex="col",
                         gridspec_kw=dict(height_ratios=[3, 2]))
    for j, (sid, tag, title) in enumerate(panels):
        df = trace(d, sid, tag)
        coll = g.draw_voltage(ax[0, j], df, title, show_ylabel=(j == 0))
        ax[0, j].set_xlabel("")
        if not coll:
            ax[0, j].text(0.5, 0.12, "all 10 buses stay up", transform=ax[0, j].transAxes, ha="center",
                          fontsize=13, color=g.BLUE, fontweight="bold")
        g.draw_load(ax[1, j], df, show_ylabel=(j == 0))
        print(f"{name}: {title}: " + ("COLLAPSE " + str(coll) if coll else "all up") +
              f"  min V {min(df[c].min() for c in df.columns if c.startswith('V_')):.0f}")
    f.tight_layout()
    p = os.path.join(out, name + ".png")
    f.savefig(p, dpi=200)
    plt.close(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="model_outputs/dataset_pipeline")
    ap.add_argument("--out", default="result_graphs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, panels in FIGS:
        fig(a.dir, a.out, name, panels)

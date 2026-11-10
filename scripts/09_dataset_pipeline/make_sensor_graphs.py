#!/usr/bin/env python3
"""
Sensor-margin result figures in the same style as the S012/S020 figure: for every case, the 10 bus voltages (top)
and the load asked for / load the limited bus got (bottom). Every curve comes from the trace_<ID>_<tag>_*.csv files
written by run_staged_limiter.m with save_trace on (see submit_trace_sensor.sh). Nothing is simulated or invented here.

The solid line is the limiter's command signal (true load x min(1, cap / noisy reading)), not measured power at the
bus. After a bus collapses it no longer describes that bus, so the solid line stops at the collapse time (orange
dotted line); the dashed line (load asked for) continues.

    python3 make_sensor_graphs.py --dir model_outputs/dataset_pipeline --out sensor_graphs
"""
import argparse, glob, os, sys
import matplotlib, matplotlib.text
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_slide_graphs as g

MINUS = "−"

def trace(d, sid, tag):
    hits = []
    for pat in ("trace_%s_%s_staged_*kW.csv" % (sid, tag), "trace_%s_%s_limiter_*kW.csv" % (sid, tag)):
        hits += glob.glob(os.path.join(d, sid, pat)) + glob.glob(os.path.join(d, pat))
    if not hits:
        raise SystemExit("no trace for %s with tag %s under %s" % (sid, tag, d))
    return g.load_trace(sorted(hits)[0])

# (file name, [(scenario, tag, panel title), ...])
FIGS = [
    ("s1_test1_window", [
        ("H01", "tr_s_h01_old",       "H01, margin 0, 10% noise\nold test: 10 s average, 1 s sampling"),
        ("H01", "tr_s_h01_g10w1",     "H01, margin 0, 10% noise\n1 s average, fast sampling"),
        ("H01", "tr_s_h01_g10w100ms", "H01, margin 0, 10% noise\n100 ms average, fast sampling")]),
    ("s2_test2_tight", [
        ("S035", "tr_s_s035_old",     "S035, margin 0, 10% noise\nold test: 10 s average, 1 s sampling"),
        ("S035", "tr_s_s035_g10w1",   "S035, margin 0, 10% noise\n1 s average, fast sampling"),
        ("S035", "tr_s_s035_b05w1",   "S035, margin 0, 0.5% bounded noise\n1 s average, fast sampling")]),
    ("s3_test3_windows", [
        ("H09", "tr_s_h09_g10w1_o5",     "H09, margin 0, 10% noise\n1 s average"),
        ("H09", "tr_s_h09_g10w500ms_o5", "H09, margin 0, 10% noise\n500 ms average"),
        ("H09", "tr_s_h09_g10w500ms_o0", "H09, margin 5 kW, 10% noise\n500 ms average")]),
    ("s4_test4_edge", [
        ("H09", "tr_s_h09_g10w1_o5", "H09, 10% noise, 1 s average\nmargin 0"),
        ("H09", "tr_s_h09_g10w1_o6", "H09, 10% noise, 1 s average\nmargin " + MINUS + "1 kW")]),
]

def draw_load(ax, df, coll, show_ylabel=True):
    """load panel; each limited bus's solid line stops at that bus's collapse time"""
    t = df["time_s"].values
    palette = [g.BLUE, g.ORANGE, g.NAVY]
    for k, b in enumerate(g.limited_buses(df)):
        c = palette[k % len(palette)]
        ax.plot(t, df["L_true_%s_kW" % b], color=c, lw=1.2, ls="--", alpha=0.55)
        used = df["L_used_%s_kW" % b].copy()
        if b in coll:
            used[df["time_s"] >= coll[b]] = float("nan")
        ax.plot(t, used, color=c, lw=2.2, label="Bus " + b)
    ax.plot([], [], color=g.NAVY, lw=1.2, ls="--", alpha=0.55, label="dashed: load asked for")
    ax.plot([], [], color=g.NAVY, lw=2.2, label="solid: load the bus got")
    if coll:
        for on in coll.values():
            ax.axvline(on, color=g.ORANGE, ls=":", lw=1.3, zorder=1)
        first = min(coll.values())
        txt = "solid line stops:\nbus collapsed" if len(coll) == 1 else "solid lines stop:\nbuses collapsed"
        ax.text(first - 0.012 * t[-1], 0.04, txt, transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=11, color=g.ORANGE)
    ax.set_xlim(t[0], t[-1])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.22)     # headroom so the legend does not sit on the curves
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Load (kW)")
    ax.grid(axis="y", color=g.GRID, lw=0.8)
    ax.legend(loc="upper left", fontsize=10.5, ncol=1)

def fig(d, out, name, panels):
    n = len(panels)
    f, ax = plt.subplots(2, n, figsize=(7 * n, 7), sharex="col", gridspec_kw=dict(height_ratios=[3, 2]))
    for j, (sid, tag, title) in enumerate(panels):
        df = trace(d, sid, tag)
        coll = g.draw_voltage(ax[0, j], df, title, show_ylabel=(j == 0))
        ax[0, j].set_xlabel("")
        if not coll:
            ax[0, j].text(0.5, 0.4, "all 10 buses stay up", transform=ax[0, j].transAxes, ha="center",
                          fontsize=14, color=g.BLUE, fontweight="bold")
        elif len(coll) > 1:      # one label for several buses instead of overlapping arrows
            for t in [t for t in ax[0, j].texts if isinstance(t, matplotlib.text.Annotation)]:
                t.remove()
            t0 = df["time_s"].values[-1]
            first = min(coll.values())
            names = ", ".join(sorted(coll, key=coll.get))
            ax[0, j].annotate("Buses %s collapse\nat " % names + " and ".join("{:,.0f} s".format(v) for v in sorted(coll.values())),
                              xy=(first, 380), xytext=(first - 0.02 * t0, 520), ha="right", fontsize=11, color=g.ORANGE,
                              arrowprops=dict(arrowstyle="-", color=g.ORANGE, lw=1))
        draw_load(ax[1, j], df, coll, show_ylabel=(j == 0))
        print("%s: %r: %s  min V %.0f" % (name, title, ("COLLAPSE " + str(coll)) if coll else "all up",
              min(df[c].min() for c in df.columns if c.startswith("V_"))))
    f.tight_layout()
    p = os.path.join(out, name + ".png")
    f.savefig(p, dpi=200)
    plt.close(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="model_outputs/dataset_pipeline")
    ap.add_argument("--out", default="sensor_graphs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, panels in FIGS:
        fig(a.dir, a.out, name, panels)

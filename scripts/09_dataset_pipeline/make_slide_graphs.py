#!/usr/bin/env python3
"""
Figures for the Option B results deck (white background, high contrast).

Reads the trace_<ID>_<tag>_<label>.csv files written by run_staged_limiter.m with save_trace on
(per-second minimum bus voltage V_<bus>, and for each limited bus the true load L_true_<bus>_kW and the
load the plant received L_used_<bus>_kW). Nothing here is simulated or invented: every curve comes
from those files.

Usage (from the folder that holds the trace files, or pass --dir):
    python make_slide_graphs.py --dir model_outputs/dataset_pipeline --tag trace --out slide_graphs
Outputs in --out:
    slide14_S012_validation.png        no limit (collapse) next to known safe cut (all 10 alive)
    slide15_S012_limiter.png           voltages + load for S012 under the limiter
    slide15_S020_limiter.png           voltages + load for S020 under the limiter
    slide15_S012_S020_limiter.png      both scenarios in one 2-column figure
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, BLUE, ORANGE, GREY, GRID = "#14213D", "#1D4E89", "#C2410C", "#8FA3BF", "#E2E0D8"
COLLAPSE_V = 100.0
BUSES = list("ABCDEFGHKL")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
    "text.color": NAVY, "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def find_trace(d, sid, tag, label_glob):
    pats = [os.path.join(d, sid, f"trace_{sid}_{tag}_{label_glob}.csv"),
            os.path.join(d, f"trace_{sid}_{tag}_{label_glob}.csv")]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits
    return []


def load_trace(path):
    df = pd.read_csv(path)
    vcols = [c for c in df.columns if c.startswith("V_")]
    if not vcols:
        raise SystemExit(f"{path}: no V_ columns")
    return df


def collapsed_buses(df, min_run=1):
    """buses whose per-second minimum voltage stays below 100 V for >= min_run seconds; returns {bus: onset_s}."""
    out = {}
    for c in [c for c in df.columns if c.startswith("V_")]:
        below = (df[c].values < COLLAPSE_V).astype(int)
        run = 0
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= min_run:
                out[c[2:]] = float(df["time_s"].values[i - run + 1])
                break
    return out


def limited_buses(df):
    return [c[len("L_used_"):-3] for c in df.columns if c.startswith("L_used_") and c.endswith("_kW")]


def draw_voltage(ax, df, title, show_ylabel=True, limited_style=True):
    coll = collapsed_buses(df)
    lim = limited_buses(df)
    t = df["time_s"].values
    for b in BUSES:
        col = f"V_{b}"
        if col not in df:
            continue
        if b in coll:
            ax.plot(t, df[col], color=ORANGE, lw=2.4, zorder=4)
        elif b in lim and limited_style:
            ax.plot(t, df[col], color=BLUE, lw=1.8, zorder=3)
        else:
            ax.plot(t, df[col], color=GREY, lw=1.1, zorder=2)
    ax.axhline(COLLAPSE_V, color=ORANGE, ls="--", lw=1.4, zorder=1)
    ax.text(t[-1], COLLAPSE_V + 12, "collapse line (100 V)", ha="right", va="bottom", color=ORANGE, fontsize=11)
    for b, on in coll.items():
        ax.axvline(on, color=ORANGE, ls=":", lw=1.3, zorder=1)
        ax.annotate(f"Bus {b} collapses\nat {on:,.0f} s", xy=(on, 380), xytext=(on - 0.02 * t[-1], 520),
                    ha="right", fontsize=11, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
    ax.set_ylim(0, 860)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Bus voltage (V)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    return coll


def draw_load(ax, df, show_ylabel=True):
    t = df["time_s"].values
    lim = limited_buses(df)
    palette = [BLUE, ORANGE, NAVY]
    for k, b in enumerate(lim):
        c = palette[k % len(palette)]
        ax.plot(t, df[f"L_true_{b}_kW"], color=c, lw=1.2, ls="--", alpha=0.55)
        ax.plot(t, df[f"L_used_{b}_kW"], color=c, lw=2.2, label=f"Bus {b}")
    ax.plot([], [], color=NAVY, lw=1.2, ls="--", alpha=0.55, label="dashed: load asked for")
    ax.plot([], [], color=NAVY, lw=2.2, label="solid: load the bus got")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Load (kW)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper left", fontsize=10.5, ncol=1)


def pick_limiter_trace(d, sid, tag):
    """the first limiter simulation whose traces keep every bus above the collapse line"""
    hits = find_trace(d, sid, tag, "staged_*kW") + find_trace(d, sid, tag, "limiter_*kW")
    if not hits:
        raise SystemExit(f"no limiter trace found for {sid} (tag '{tag}') under {d}")
    def cap_of(p):
        return float(os.path.basename(p).rsplit("_", 1)[1].replace("kW.csv", ""))
    for p in sorted(hits, key=cap_of, reverse=True):      # largest cap first = the first one tried
        df = load_trace(p)
        if not collapsed_buses(df):
            return df, cap_of(p), p
    raise SystemExit(f"{sid}: every limiter trace collapsed; nothing safe to plot")


def read_json(d, sid, name):
    for p in (os.path.join(d, sid, name), os.path.join(d, name)):
        if os.path.exists(p):
            return json.load(open(p))
    return None


def fig_slide14(d, tag, out, sid="S012"):
    h1 = find_trace(d, sid, tag, "validate_noshed")
    h2 = find_trace(d, sid, tag, "validate_const_best")
    if not h1 or not h2:
        raise SystemExit(f"{sid}: need the validate_noshed and validate_const_best traces (run with VALIDATE=1 and SAVE_TRACE=1)")
    a, b = load_trace(h1[0]), load_trace(h2[0])
    best = read_json(d, sid, f"bisect_{sid}_result.json")
    kept = best.get("best_shedfrac") if best else None
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    ca = draw_voltage(ax[0], a, "Check 1: no limit at all (cap = infinity)")
    draw_voltage(ax[1], b, "Check 2: a known safe cut" + (f" ({round((1-kept)*100)}% off)" if kept else ""),
                 show_ylabel=False)
    cb = collapsed_buses(b)
    if not cb:
        ax[1].text(0.5, 0.12, "All 10 buses stay up", transform=ax[1].transAxes, ha="center", fontsize=14,
                   color=BLUE, fontweight="bold")
    # original no-shed onset, for the 'identical' claim
    cs = read_json(d, sid, f"noshed_{sid}_summary.json")
    if cs and cs.get("collapse_onset_s") and ca:
        ax[0].text(0.03, 0.06, f"Original no-shed run collapses at {cs['collapse_onset_s']:,.1f} s",
                   transform=ax[0].transAxes, fontsize=11, color=NAVY)
    fig.suptitle(f"Scenario {sid}: the test reproduces the original collapse, and a safe cut stays safe",
                 x=0.01, ha="left", fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(out, f"slide14_{sid}_validation.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_limiter(d, tag, out, sid):
    df, cap, src = pick_limiter_trace(d, sid, tag)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    draw_voltage(ax[0], df, f"{sid}: all 10 bus voltages stay far above the collapse line")
    ax[0].set_xlabel("")
    draw_load(ax[1], df)
    fig.tight_layout()
    p = os.path.join(out, f"slide15_{sid}_limiter.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p, df


def fig_pair(d, tag, out, sids):
    fig, ax = plt.subplots(2, len(sids), figsize=(7 * len(sids), 7), sharex="col",
                           gridspec_kw=dict(height_ratios=[3, 2]))
    for j, sid in enumerate(sids):
        df, cap, _ = pick_limiter_trace(d, sid, tag)
        draw_voltage(ax[0, j], df, f"{sid}: all 10 buses stay up", show_ylabel=(j == 0))
        ax[0, j].set_xlabel("")
        draw_load(ax[1, j], df, show_ylabel=(j == 0))
    fig.tight_layout()
    p = os.path.join(out, "slide15_" + "_".join(sids) + "_limiter.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="model_outputs/dataset_pipeline")
    ap.add_argument("--tag", default="trace")
    ap.add_argument("--out", default="slide_graphs")
    ap.add_argument("--scenarios", nargs="+", default=["S012", "S020"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []
    try:
        made.append(fig_slide14(a.dir, a.tag, a.out, a.scenarios[0]))
    except SystemExit as e:
        print("slide 14 skipped:", e)
    for s in a.scenarios:
        try:
            made.append(fig_limiter(a.dir, a.tag, a.out, s)[0])
        except SystemExit as e:
            print(f"slide 15 {s} skipped:", e)
    try:
        made.append(fig_pair(a.dir, a.tag, a.out, a.scenarios))
    except SystemExit as e:
        print("pair figure skipped:", e)
    print("\n".join(made))

"""
STEP 35: Diagnose the derating dataset, row-drop cause and possible leakage
================================================================================
Two things flagged from the step34 training run, both investigated here
against the real data rather than guessed at:

  PROBLEM 1: 85,960 of 500,001 rows (17.2%) were dropped as incomplete
  history in step33, far more than the "first ~4 warm-up rows" the pipeline
  should drop. This section audits every column in the RAW CSV for NaN/Inf,
  reports counts, and reports WHERE in time they occur (scattered at the
  start only, vs clustered elsewhere in the run). GEI is the leading
  suspect, since it's a division (power / average power), and a division
  can blow up whenever a bus's power passes very close to zero, which is
  more likely at this dataset's higher 1.25 load scale.

  PROBLEM 2: near-perfect validation R^2 (0.9998) alongside a 20-100x
  spread in per-converter MAE is consistent with a same-instant coupling
  between phase and the voltage/power features, since Simulink can compute
  the DAB power-transfer equation from phase within the same solver step,
  with no delay. If true, the network is partly being fed the electrical
  consequence of the very phase command it is predicting, an easier
  problem than real deployment, where only PAST measurements exist. This
  section checks that empirically: for every converter, it compares how
  strongly its phase correlates with same-instant voltage/power versus
  one-step-lagged voltage/power. A same-instant correlation that is
  consistently and substantially higher than the lagged one is evidence of
  this coupling, independent of needing to inspect Simulink's own
  execution order.

  Also included: a plot of AL's (and the two worst-MAE converters', AB and
  BC's) phase over the FULL 5000 s run, since AL touching exactly 90.0 deg
  in the target-range summary could mean brief legitimate derating
  activation or the same sustained saturation-windup pattern seen in the
  earlier broken V3 model, and summary statistics alone cannot tell the
  two apart. And a range-normalized MAE table, since raw MAE in degrees
  does not say whether a converter's error is large relative to how far it
  actually moves.

This script only reads data and prints/plots findings. It does not modify
the dataset, the features, or the trained model.

Run:
    python step35_diagnose_derating_dataset.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026"
)
RAW_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
           / "thermal_derating_v7_5000s.csv")
MAE_CSV = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
           / "34_train_mlp" / "per_converter_mae.csv")

OUTPUT_DIR = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
              / "35_dataset_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

# Which bus each converter's phase most directly acts on, for the leakage
# check. Ring adjacency: converter XY sits between bus X and bus Y.
CONV_TO_BUSES = {
    "AB": ("A","B"), "BC": ("B","C"), "CD": ("C","D"), "DE": ("D","E"),
    "EF": ("E","F"), "FG": ("F","G"), "GH": ("G","H"), "HK": ("H","K"),
    "KL": ("K","L"), "AL": ("A","L"),
}

TIME_BUCKET_S = 50   # bucket width for the "where do the NaNs cluster" plot


# ----------------------------------------------------------------------
# PROBLEM 1: NaN/Inf audit
# ----------------------------------------------------------------------
def audit_nan_inf(raw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PROBLEM 1: why 17.2% of rows were dropped")
    print("=" * 70)

    n_rows = len(raw)
    bad = raw.replace([np.inf, -np.inf], np.nan).isna()
    col_counts = bad.sum().sort_values(ascending=False)
    col_counts = col_counts[col_counts > 0]

    if col_counts.empty:
        print("No NaN or Inf found in any raw column. The row loss must be "
              "happening inside step33's feature construction, not the raw "
              "CSV. Re-check the lag/rolling logic there instead.")
        return

    print(f"\nColumns with NaN/Inf, out of {n_rows:,} rows:")
    for col, cnt in col_counts.items():
        pct = 100 * cnt / n_rows
        print(f"  {col:30s} {cnt:8,d}  ({pct:5.2f}%)")

    gei_cols = [c for c in col_counts.index if "GEI" in c]
    if gei_cols:
        print(f"\n{len(gei_cols)} GEI column(s) affected. GEI is computed as a "
              f"division (power / average power), the leading suspect since a "
              f"division blows up whenever a bus's power passes very close to "
              f"zero, more likely at this dataset's 1.25 load scale.")

    # Where in time do the bad rows cluster? Scattered near t=0 only would
    # be ordinary warm-up. Clustered elsewhere points to a real event.
    if "time" in raw.columns:
        any_bad = bad.any(axis=1)
        t = raw["time"].to_numpy()
        bucket = (t // TIME_BUCKET_S).astype(int)
        bucket_bad = pd.Series(any_bad.to_numpy(), index=bucket).groupby(level=0).sum()
        bucket_total = pd.Series(np.ones(len(t)), index=bucket).groupby(level=0).sum()
        bucket_frac = (bucket_bad / bucket_total).fillna(0)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(bucket_frac.index * TIME_BUCKET_S, bucket_frac.values * 100)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("% of rows with NaN/Inf in this bucket")
        ax.set_title(f"Where NaN/Inf rows occur across the {int(t.max())} s run "
                     f"({TIME_BUCKET_S}s buckets)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = OUTPUT_DIR / "nan_inf_time_distribution.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"\nWrote: {out_png}")

        first_bad_bucket = bucket_frac[bucket_frac > 0].index.min() if (bucket_frac > 0).any() else None
        last_bad_bucket = bucket_frac[bucket_frac > 0].index.max() if (bucket_frac > 0).any() else None
        if first_bad_bucket is not None:
            print(f"Bad rows span from t~{first_bad_bucket*TIME_BUCKET_S}s to "
                  f"t~{(last_bad_bucket+1)*TIME_BUCKET_S}s.")
            if first_bad_bucket > 2:   # more than ~100s in, not just startup
                print("This is NOT confined to the start of the run. This looks "
                      "like a real event partway through, not ordinary warm-up. "
                      "Check the plot for where it spikes.")
            else:
                print("Concentrated near the start of the run, consistent with "
                      "ordinary warm-up, though the total percentage is still "
                      "far higher than a few warm-up rows would explain, worth "
                      "checking the plot regardless.")


# ----------------------------------------------------------------------
# PROBLEM 2: same-instant vs lagged correlation (leakage check)
# ----------------------------------------------------------------------
def leakage_check(raw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PROBLEM 2: same-instant coupling check (possible leakage)")
    print("=" * 70)

    results = []
    for conv, (b1, b2) in CONV_TO_BUSES.items():
        phase_col = f"Phase_{conv}_cmd_deg"
        if phase_col not in raw.columns:
            continue
        phase = raw[phase_col]

        for sig_name, cols in [
            ("Voltage", [f"V_Bus_{b1}", f"V_Bus_{b2}"]),
            ("Power",   [f"Bus_{b1}_Src_Pow", f"Bus_{b2}_Src_Pow"]),
        ]:
            cols = [c for c in cols if c in raw.columns]
            if not cols:
                continue
            sig = raw[cols].mean(axis=1)

            same_instant_corr = phase.corr(sig)
            lagged_corr = phase.corr(sig.shift(1))

            results.append({
                "converter": conv,
                "signal": sig_name,
                "same_instant_corr": same_instant_corr,
                "lag1_corr": lagged_corr,
                "abs_diff": abs(same_instant_corr) - abs(lagged_corr),
            })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("Could not find matching voltage/power columns to check. "
              "Confirm raw CSV column names match V_Bus_<bus> and "
              "Bus_<bus>_Src_Pow.")
        return res_df

    res_df = res_df.sort_values("abs_diff", ascending=False)
    print("\nCorrelation of each converter's phase with same-instant vs "
          "one-step-lagged voltage/power (sorted by biggest gap):\n")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

    out_csv = OUTPUT_DIR / "leakage_check_correlations.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    n_suspect = (res_df["abs_diff"] > 0.15).sum()
    print(f"\n{n_suspect} of {len(res_df)} converter/signal pairs show a same-instant "
          f"correlation more than 0.15 higher than the lagged correlation.")
    if n_suspect > len(res_df) * 0.3:
        print("A substantial share of pairs show this gap. Consistent with the "
              "same-instant coupling hypothesis: the network may be seeing the "
              "electrical consequence of the phase command at the same step it "
              "is trying to predict. This would not show up in real deployment, "
              "where only past measurements are available, so the training MAE "
              "would not be trustworthy as a deployment estimate without fixing "
              "the feature timing.")
    else:
        print("No strong pattern of same-instant coupling found here. This does "
              "not fully rule out leakage, this is a correlation-based proxy, "
              "not a check of the model's actual execution order, but it does "
              "not support the hypothesis either.")
    return res_df


# ----------------------------------------------------------------------
# AL / worst-MAE converters: full-run phase plot
# ----------------------------------------------------------------------
def plot_phase_traces(raw: pd.DataFrame, converters_to_plot):
    print("\n" + "=" * 70)
    print("Full-run phase plot: is AL's saturation a brief event or sustained?")
    print("=" * 70)

    if "time" not in raw.columns:
        print("No time column found, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    for conv in converters_to_plot:
        col = f"Phase_{conv}_cmd_deg"
        if col in raw.columns:
            ax.plot(raw["time"], raw[col], label=conv, linewidth=1.0)
    ax.axhline(90, color="r", linestyle=":", linewidth=1)
    ax.axhline(-90, color="r", linestyle=":", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title("Full 5000s phase trace: AL and the two worst-MAE converters (AB, BC)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "full_run_phase_trace.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_png}")

    al_col = "Phase_AL_cmd_deg"
    if al_col in raw.columns:
        al = raw[al_col]
        pct_at_limit = (al.abs() > 89).mean() * 100
        print(f"\nAL spends {pct_at_limit:.1f}% of the run within 1 deg of the +/-90 limit.")
        if pct_at_limit > 10:
            print("That is a substantial share of the run pinned at the limit, "
                  "worth checking the plot for the same flat-topped, "
                  "rail-to-rail shape seen in the earlier broken model, versus "
                  "brief spikes during genuine derating events.")
        else:
            print("A small share of the run, consistent with brief, event-driven "
                  "activation rather than sustained saturation, though the plot "
                  "is worth a direct look regardless.")


# ----------------------------------------------------------------------
# Range-normalized MAE
# ----------------------------------------------------------------------
def range_normalized_mae(raw: pd.DataFrame, mae_csv_path: Path):
    print("\n" + "=" * 70)
    print("Range-normalized MAE: is the AB/BC error actually worse, or just a wider range?")
    print("=" * 70)

    if not mae_csv_path.exists():
        print(f"MAE file not found: {mae_csv_path}\nSkipping this section.")
        return

    mae_df = pd.read_csv(mae_csv_path)
    rows = []
    for _, row in mae_df.iterrows():
        conv = row["converter"]
        col = f"Phase_{conv}_cmd_deg"
        if col not in raw.columns:
            continue
        phase_range = raw[col].max() - raw[col].min()
        pct_of_range = 100 * row["mae_deg"] / phase_range if phase_range > 0 else np.nan
        rows.append({
            "converter": conv,
            "mae_deg": row["mae_deg"],
            "phase_range_deg": phase_range,
            "mae_pct_of_range": pct_of_range,
        })

    out_df = pd.DataFrame(rows).sort_values("mae_pct_of_range", ascending=False)
    print("\n" + out_df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    out_csv = OUTPUT_DIR / "range_normalized_mae.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 35: dataset diagnostics")
    print("=" * 70)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found:\n{RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded raw CSV: {len(raw):,} rows x {raw.shape[1]} cols")

    audit_nan_inf(raw)
    leakage_check(raw)
    plot_phase_traces(raw, ["AL", "AB", "BC"])
    range_normalized_mae(raw, MAE_CSV)

    print("\nDone. Review the two PNGs and three CSVs in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

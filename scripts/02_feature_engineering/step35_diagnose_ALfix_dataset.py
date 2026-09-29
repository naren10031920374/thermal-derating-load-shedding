"""
STEP 35 v2: Diagnose the NaN row loss in the AL-fixed dataset
================================================================================
The feature build (step33 v2) dropped 85,960 of 500,001 rows as incomplete
history, almost exactly matching the 85,956 NaN/Inf rows found in the
earlier, AL-broken dataset at the same grid_load_scale (1.25). That earlier
NaN blowup was traced to all ten GEI columns going NaN simultaneously
starting at t=4140.45s, caused by a shared denominator (Src_Pow_Avg, the
grid-wide average source power) crossing zero, most likely tied to a bus
collapse event.

The AL wiring fix did not touch the GEI calculation at all, they are
unrelated parts of the model, so there is no reason to expect this problem
to have gone away on its own. This script checks, directly against the new
raw CSV, whether the same failure is still happening, and if so, whether it
is happening at the same time as before or has shifted, which would tell us
whether AL being genuinely connected now changes which bus collapses first
under this load profile.

This script only reads data. It does not modify the dataset or the model.

Run:
    python step35_diagnose_ALfix_dataset.py
"""
from __future__ import annotations
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
           / "thermal_derating_v7_ALfix_5000s.csv")

OUTPUT_DIR = (PROJECT_ROOT / "model_outputs" / "unified_controller_10bus_derating"
              / "35_dataset_diagnostics_ALfix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSES = ["A","B","C","D","E","F","G","H","K","L"]
CONV  = ["AB","BC","CD","DE","EF","FG","GH","HK","KL","AL"]

TIME_BUCKET_S = 50   # bucket width for the "where do the NaNs cluster" plot

# From the earlier, AL-broken dataset at the same grid_load_scale (1.25),
# for direct comparison, not assumed to still be correct here.
PREVIOUS_NAN_ONSET_S = 4140.45
PREVIOUS_NAN_ROW_COUNT = 85956


# ----------------------------------------------------------------------
# NaN / Inf audit
# ----------------------------------------------------------------------
def audit_nan_inf(raw: pd.DataFrame):
    print("\n" + "=" * 70)
    print("NaN/Inf AUDIT, AL-FIXED DATASET")
    print("=" * 70)

    n_rows = len(raw)
    bad = raw.replace([np.inf, -np.inf], np.nan).isna()
    col_counts = bad.sum().sort_values(ascending=False)
    col_counts = col_counts[col_counts > 0]

    if col_counts.empty:
        print("No NaN or Inf found in any raw column. The row loss reported by "
              "step33 v2 must be coming from somewhere else, recheck the lag/"
              "rolling window logic there instead of the raw data.")
        return

    print(f"\nColumns with NaN/Inf, out of {n_rows:,} rows:")
    for col, cnt in col_counts.items():
        pct = 100 * cnt / n_rows
        print(f"  {col:30s} {cnt:8,d}  ({pct:5.2f}%)")

    total_bad_rows = bad.any(axis=1).sum()
    print(f"\nTotal rows affected: {total_bad_rows:,} "
          f"({100*total_bad_rows/n_rows:.2f}%)")
    print(f"For comparison, the earlier AL-broken dataset (same grid_load_scale "
          f"1.25) had {PREVIOUS_NAN_ROW_COUNT:,} NaN/Inf rows "
          f"({100*PREVIOUS_NAN_ROW_COUNT/n_rows:.2f}%).")
    diff = total_bad_rows - PREVIOUS_NAN_ROW_COUNT
    print(f"Difference: {diff:+,} rows.")
    if abs(diff) < 100:
        print("Essentially the same magnitude as before the AL fix. This problem "
              "is very likely still present and unrelated to AL specifically, "
              "since the AL fix never touched the GEI calculation.")

    gei_cols = [c for c in col_counts.index if "GEI" in c]
    if gei_cols and len(gei_cols) == len(BUSES):
        print(f"\nAll {len(BUSES)} GEI columns affected simultaneously, consistent "
              f"with the shared-denominator hypothesis (Src_Pow_Avg crossing near "
              f"zero), same as the earlier dataset.")
    elif gei_cols:
        print(f"\n{len(gei_cols)} of {len(BUSES)} GEI columns affected, NOT all ten. "
              f"This is DIFFERENT from the earlier dataset, worth a closer look, "
              f"the shared-denominator explanation predicts all ten going bad "
              f"together, a partial pattern suggests something else is going on.")

    non_gei_bad = [c for c in col_counts.index if "GEI" not in c]
    if non_gei_bad:
        print(f"\nNon-GEI columns also affected: {non_gei_bad}")
        print("This did NOT happen in the earlier dataset, where only the GEI "
              "columns were affected. If the underlying event now also corrupts "
              "voltage, power, or temperature columns directly, that points to a "
              "more serious numerical divergence, not just a clean 0/0 in GEI.")
    else:
        print("\nOnly GEI columns affected, same as the earlier dataset, consistent "
              "with a clean division-by-zero rather than a broader numerical "
              "divergence.")

    # Where in time do the bad rows cluster?
    if "time" in raw.columns:
        any_bad = bad.any(axis=1)
        t = raw["time"].to_numpy()
        bucket = (t // TIME_BUCKET_S).astype(int)
        bucket_bad = pd.Series(any_bad.to_numpy(), index=bucket).groupby(level=0).sum()
        bucket_total = pd.Series(np.ones(len(t)), index=bucket).groupby(level=0).sum()
        bucket_frac = (bucket_bad / bucket_total).fillna(0)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(bucket_frac.index * TIME_BUCKET_S, bucket_frac.values * 100)
        ax.axvline(PREVIOUS_NAN_ONSET_S, color="r", linestyle=":", linewidth=1.2,
                   label=f"previous onset (t={PREVIOUS_NAN_ONSET_S}s)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("% of rows with NaN/Inf in this bucket")
        ax.set_title(f"Where NaN/Inf rows occur, AL-fixed dataset, {TIME_BUCKET_S}s buckets")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = OUTPUT_DIR / "nan_inf_time_distribution_ALfix.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"\nWrote: {out_png}")

        first_bad_idx = np.argmax(any_bad.to_numpy()) if any_bad.any() else None
        if first_bad_idx is not None and any_bad.iloc[first_bad_idx]:
            onset_t = t[first_bad_idx]
            print(f"\nFirst NaN/Inf row at t={onset_t:.2f}s "
                  f"(previous dataset: t={PREVIOUS_NAN_ONSET_S}s).")
            shift = onset_t - PREVIOUS_NAN_ONSET_S
            if abs(shift) < 10:
                print("Essentially the same onset time as before. The AL fix does "
                      "not appear to have changed when or why this happens.")
            else:
                print(f"Onset shifted by {shift:+.1f}s relative to the earlier "
                      f"dataset. Worth checking which bus is collapsing at this "
                      f"new time, since AL being genuinely connected now may have "
                      f"changed which bus fails first under this load profile.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 35 v2: diagnose the AL-fixed dataset's row loss")
    print("=" * 70)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found:\n{RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded raw CSV: {len(raw):,} rows x {raw.shape[1]} cols")

    audit_nan_inf(raw)

    print("\nDone. Review the plot in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

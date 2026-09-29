"""
FIX GEI: recompute GEI defensively from raw bus power, no re-simulation needed
================================================================================
Problem confirmed via check_nan_block_location.py against the AL-fixed
dataset: GEI_A through GEI_L go NaN simultaneously for 85,955 rows, t=4140.46s
to t=5000s (the run's final 859.54s, never recovers). Root cause: GEI_X =
Bus_X_Src_Pow / Src_Pow_Avg, and all ten GEI columns share the same
denominator (Src_Pow_Avg, the grid-wide mean source power). When that
denominator crosses zero, every GEI column blows up at once, which is
exactly the observed pattern.

This is fixable entirely in post-processing, no Simulink re-run needed,
because the raw CSV already contains every bus's source power
(Bus_<bus>_Src_Pow), the only ingredient GEI is built from.

THE FIX IS NOT AN ARBITRARY EPSILON PATCH. It follows directly from a
property of the model itself: every bus's source power is floored at zero
(a MinMax block against Constant(0) inside source_bus_model_<bus>, confirmed
in the model file). Src_Pow_Avg is the mean of ten numbers that can never be
negative, so it can only be zero if ALL TEN bus powers are simultaneously
zero. That means whenever the denominator collapses, there is no ambiguity
about what is physically happening: the entire grid has stopped delivering
power, uniformly, not one bus doing something unusual relative to the
others. The correct fallback value follows from that fact rather than being
guessed:

  - When Src_Pow_Avg is meaningfully above zero (normal operation): GEI_X is
    computed exactly as originally intended, Bus_X_Src_Pow / Src_Pow_Avg.
  - When Src_Pow_Avg falls below EPSILON_W (the grid has effectively
    collapsed): GEI_X is set to 0.0 for every bus at that row, not 1.0.
    A GEI of 1.0 would claim "this bus is operating at exactly the grid
    average," which is false when the grid has actually collapsed. 0.0
    correctly represents "this bus is contributing nothing," which is what
    is actually happening.

EPSILON_W is set well above ordinary floating-point/solver noise but well
below typical operating power (bus source power runs in the tens of
kilowatts under normal load per the dataset's own printed ranges), so
genuine near-zero collapse is caught without falsely triggering on ordinary
low-load conditions.

This script does NOT touch the original CSV or the Simulink model. It reads
the existing dataset, recomputes GEI, and writes a new CSV with a clearly
different name so this can never be silently confused with the unfixed
version.

Run:
    python fix_GEI_v7_ALfix.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(
       r"D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject"
)
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
             / "thermal_derating_v7_ALfix_5000s.csv")
OUTPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_v7"
              / "thermal_derating_v7_ALfix_GEIfix_5000s.csv")

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]

# Well above floating-point/solver noise, well below typical operating power
# (tens of kilowatts per the dataset's own printed load ranges). Grid-wide
# average power below this is treated as a genuine collapse, not noise.
EPSILON_W = 500.0


def main():
    print("=" * 70)
    print("FIX GEI: recompute defensively from raw bus power")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")
    raw = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(raw):,} rows.")

    power_cols = [f"Bus_{b}_Src_Pow" for b in BUSES]
    missing = [c for c in power_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected power columns:\n{missing}")

    # --------------------------------------------------------------------
    # Before: audit the existing (flawed) GEI columns, so the fix's effect
    # is measured against a real baseline, not assumed.
    # --------------------------------------------------------------------
    old_gei_cols = [f"GEI_{b}" for b in BUSES]
    old_bad = raw[old_gei_cols].replace([np.inf, -np.inf], np.nan).isna()
    old_bad_rows = old_bad.any(axis=1).sum()
    print(f"\nBefore fix: {old_bad_rows:,} rows ({100*old_bad_rows/len(raw):.2f}%) "
          f"had NaN/Inf in at least one GEI column.")

    # --------------------------------------------------------------------
    # Recompute Src_Pow_Avg directly from the raw power columns, not trusting
    # any previously logged average, since that average is exactly what
    # produced the original blowup.
    # --------------------------------------------------------------------
    src_pow_avg = raw[power_cols].mean(axis=1)
    print(f"\nRecomputed Src_Pow_Avg range: {src_pow_avg.min():.4f} W to "
          f"{src_pow_avg.max():.4f} W")

    collapsed = src_pow_avg < EPSILON_W
    n_collapsed = collapsed.sum()
    print(f"Rows where Src_Pow_Avg < {EPSILON_W:.0f} W (treated as grid collapse): "
          f"{n_collapsed:,} ({100*n_collapsed/len(raw):.2f}%)")

    # --------------------------------------------------------------------
    # Recompute each bus's GEI with the safe fallback.
    # --------------------------------------------------------------------
    new_gei = pd.DataFrame(index=raw.index)
    for b in BUSES:
        power = raw[f"Bus_{b}_Src_Pow"]
        ratio = power / src_pow_avg.where(~collapsed, other=np.nan)
        ratio = ratio.where(~collapsed, other=0.0)
        new_gei[f"GEI_{b}"] = ratio

    # --------------------------------------------------------------------
    # After: confirm no NaN/Inf remain anywhere in the new GEI columns.
    # --------------------------------------------------------------------
    new_bad = new_gei.replace([np.inf, -np.inf], np.nan).isna()
    new_bad_rows = new_bad.any(axis=1).sum()
    print(f"\nAfter fix: {new_bad_rows:,} rows with NaN/Inf in any GEI column.")
    if new_bad_rows > 0:
        raise RuntimeError(
            "Fix did not eliminate all NaN/Inf in GEI. This should not happen given "
            "the collapsed-row fallback is unconditional, check EPSILON_W and the "
            "power columns for unexpected NaN of their own."
        )
    print("Confirmed: zero NaN/Inf remain in the recomputed GEI columns.")

    # --------------------------------------------------------------------
    # Show the actual transition, both directions, so this is checkable
    # directly rather than trusted on the row counts alone.
    # --------------------------------------------------------------------
    if "time" in raw.columns and n_collapsed > 0:
        collapse_start_idx = np.where(collapsed.to_numpy())[0][0]
        print(f"\n--- GEI_A around the collapse onset (t={raw['time'].iloc[collapse_start_idx]:.2f}s) ---")
        window = 5
        lo = max(0, collapse_start_idx - window)
        hi = min(len(raw), collapse_start_idx + window)
        compare = pd.DataFrame({
            "time": raw["time"].iloc[lo:hi],
            "Src_Pow_Avg": src_pow_avg.iloc[lo:hi],
            "GEI_A_old": raw["GEI_A"].iloc[lo:hi],
            "GEI_A_new": new_gei["GEI_A"].iloc[lo:hi],
        })
        print(compare.to_string(index=False))

    # --------------------------------------------------------------------
    # Write the corrected dataset. Original GEI columns are replaced, but
    # the rest of the file is untouched. A separate output filename keeps
    # this from ever being confused with the unfixed version.
    # --------------------------------------------------------------------
    out = raw.copy()
    for b in BUSES:
        out[f"GEI_{b}"] = new_gei[f"GEI_{b}"]
    out["Src_Pow_Avg_recomputed"] = src_pow_avg   # kept for transparency/debugging

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")
    d_size = OUTPUT_CSV.stat().st_size / 1e6
    print(f"File size: {d_size:.1f} MB")

    print(f"\nDone. {old_bad_rows:,} previously-NaN rows recovered "
          f"({100*old_bad_rows/len(raw):.2f}% of the dataset).")
    print("Next step: point step33 v2's INPUT_CSV at this new file "
          f"({OUTPUT_CSV.name}) and rerun feature engineering. The row-drop "
          "count reported there should now be close to the original ~4 "
          "warm-up rows, not ~86,000.")


if __name__ == "__main__":
    main()

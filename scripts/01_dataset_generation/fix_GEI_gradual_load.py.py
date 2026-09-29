"""
FIX GEI FOR THE GRADUAL LOAD SCENARIO: recompute GEI defensively from raw
bus power, no re-simulation needed
============================================================================
This is the same fix used on the pilot dataset (fix_GEI_v7_ALfix.py),
pointed at the gradual-load CSV instead.

WHY THIS IS NEEDED (not just NaN-dropping):
  GEI_X = Bus_X_Src_Pow / Src_Pow_Avg, and all ten GEI columns share the
  same denominator (Src_Pow_Avg, the grid-wide mean source power). If the
  grid partially or fully collapses under the heavy phase of this
  scenario (buses' source power drops toward 0), that denominator can
  approach zero and every GEI column blows up (NaN/Inf) at once, exactly
  like it did in the pilot run around its own collapse.

  Dropping those rows (what the previous version of this script did)
  would delete the exact rows around and after a collapse -- the most
  important rows in the whole dataset. This version instead recomputes
  GEI directly from the raw per-bus source power, with a physically
  grounded fallback:
    - Src_Pow_Avg meaningfully above zero (normal operation): GEI_X is
      computed exactly as intended, Bus_X_Src_Pow / Src_Pow_Avg.
    - Src_Pow_Avg below EPSILON_W (grid has effectively collapsed):
      GEI_X = 0.0 for every bus at that row (not 1.0 -- 1.0 would
      falsely claim "operating at grid average"; 0.0 correctly says
      "contributing nothing").
  No rows are dropped. The dataset keeps its full timeline, collapse
  region included.

Run:
    python fix_GEI_gradual_load.py.py
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
INPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_gradual"
             / "thermal_derating_gradual_load_5000s.csv")
OUTPUT_CSV = (PROJECT_ROOT / "model_outputs" / "thermal_derating_gradual"
              / "thermal_derating_gradual_load_GEIfix_5000s.csv")

BUSES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"]

# Same threshold as the pilot fix: well above floating-point/solver
# noise, well below typical operating power (tens of kW per bus).
EPSILON_W = 500.0


def main():
    print("=" * 70)
    print("FIX GEI (GRADUAL LOAD): recompute defensively from raw bus power")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}\n"
            "Run step36_generate_gradual_load_dataset.m first."
        )
    raw = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(raw):,} rows.")

    power_cols = [f"Bus_{b}_Src_Pow" for b in BUSES]
    missing = [c for c in power_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected power columns:\n{missing}")

    old_gei_cols = [f"GEI_{b}" for b in BUSES]
    missing_gei = [c for c in old_gei_cols if c not in raw.columns]
    if missing_gei:
        raise ValueError(
            f"Missing expected GEI columns:\n{missing_gei}\n"
            "This means the CSV still came from the old, broken generator. "
            "Re-run step36_generate_gradual_load_dataset.m with the "
            "corrected version first."
        )

    # --------------------------------------------------------------------
    # Before: audit the existing (possibly flawed) GEI columns.
    # --------------------------------------------------------------------
    old_bad = raw[old_gei_cols].replace([np.inf, -np.inf], np.nan).isna()
    old_bad_rows = old_bad.any(axis=1).sum()
    print(f"\nBefore fix: {old_bad_rows:,} rows ({100*old_bad_rows/len(raw):.2f}%) "
          f"had NaN/Inf in at least one GEI column.")

    # --------------------------------------------------------------------
    # Recompute Src_Pow_Avg directly from the raw power columns.
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
    # After: confirm no NaN/Inf remain.
    # --------------------------------------------------------------------
    new_bad = new_gei.replace([np.inf, -np.inf], np.nan).isna()
    new_bad_rows = new_bad.any(axis=1).sum()
    print(f"\nAfter fix: {new_bad_rows:,} rows with NaN/Inf in any GEI column.")
    if new_bad_rows > 0:
        raise RuntimeError(
            "Fix did not eliminate all NaN/Inf in GEI. Check EPSILON_W and "
            "the power columns for unexpected NaN of their own."
        )
    print("Confirmed: zero NaN/Inf remain in the recomputed GEI columns.")

    if "time" in raw.columns and n_collapsed > 0:
        collapse_start_idx = np.where(collapsed.to_numpy())[0][0]
        print(f"\n--- GEI_A around the first collapse-like point "
              f"(t={raw['time'].iloc[collapse_start_idx]:.2f}s) ---")
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
    elif n_collapsed == 0:
        print("\nNo collapse-level rows found -- every bus stayed above the "
              "power floor for the whole run. (Worth double-checking against "
              "the MATLAB health-check output if you expected a collapse "
              "in the heavy phase.)")

    # --------------------------------------------------------------------
    # Write the corrected dataset.
    # --------------------------------------------------------------------
    out = raw.copy()
    for b in BUSES:
        out[f"GEI_{b}"] = new_gei[f"GEI_{b}"]
    out["Src_Pow_Avg_recomputed"] = src_pow_avg

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV}")
    print(f"File size: {OUTPUT_CSV.stat().st_size / 1e6:.1f} MB")
    print(f"Rows: {len(out):,} (same as input -- no rows dropped)")

    print("\nNext step: rerun step33_build_gradual_load_features.py "
          "(unchanged) -- it already points at this exact output file.")


if __name__ == "__main__":
    main()
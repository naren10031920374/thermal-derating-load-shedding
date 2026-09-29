#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 gradual-ramp sweep, SERIAL (all 8 durations
# in one job, exactly like running it locally).
#
# Submit from the project root:
#   sbatch scripts/00_environment/run_gradual_ramp_serial.sh
#
# This is the simple, low-risk option: one MATLAB+Simulink session, one
# license checkout, ~2-2.25 hours wall clock (matches the local estimate).
# Prefer run_gradual_ramp_array.sh instead if you want the 8 durations to
# run in parallel and finish in well under an hour.
# ============================================================================

#SBATCH --job-name=gradual_ramp_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/gradual_ramp_serial_%j.log
#SBATCH --error=logs/gradual_ramp_serial_%j_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 gradual-ramp sweep -- SERIAL"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "Host: $(hostname)"
echo "========================================"

mkdir -p logs

# ----------------------------------------------------------------------
# Pinned to R2025b: it's the newest MATLAB Great Lakes has installed
# (module avail matlab showed up to R2025b, default R2024b), and the
# .slx models were re-exported to R2025b format via
# export_models_for_greatlakes.m specifically to match this. If you
# later re-export to a different target version, update this to match.
#
# VERIFY BEFORE RELYING ON THIS: confirm your MATLAB module actually
# includes a Simulink license before submitting a real job.
#   module load matlab/R2025b
#   matlab -batch "disp(license('test','simulink'))"
# should print 1. If it prints 0, ARC-TS support (arc-support@umich.edu)
# needs to confirm Simulink entitlement on your account/allocation --
# this script will otherwise fail at the sim() call, not at startup.
# ----------------------------------------------------------------------
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/06_gradual_ramp"

echo ""
echo "[STEP 1] Verifying the ramped model is a no-op at defaults ..."
matlab -batch "compare_sheddable_vs_ramped_corridor_triad_collapse"

echo ""
echo "[STEP 2] Running the full serial gradual-ramp sweep (8 durations) ..."
matlab -batch "test_shed_gradual_ramp_corridor_triad_collapse"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in: model_outputs/thermal_derating_v7/corridor_triad_collapse_gradual_ramp_sweep/"
echo "========================================"

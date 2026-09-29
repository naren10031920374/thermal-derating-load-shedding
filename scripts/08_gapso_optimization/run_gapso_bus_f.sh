#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 GA-PSO load-shedding search, Bus F scenario.
#
# Submit from the project root:
#   sbatch scripts/08_gapso_optimization/run_gapso_bus_f.sh
#
# Modeled directly on run_gradual_ramp_serial.sh's conventions (see
# GREAT_LAKES_SETUP_ThermalProject.md) -- one MATLAB+Simulink session, one
# license checkout. This job is expected to run MUCH longer than that one:
# the local timing test measured ~15 minutes per candidate evaluation, and
# with POP_SIZE=8/MAX_GENERATIONS=5 (~80 evaluations across the PSO+GA
# passes plus the final full confirmation run) that's roughly 20-24 hours
# worst case -- hence --time=30:00:00 below, with margin. Adjust if your
# own run comes in faster or slower.
# ============================================================================

#SBATCH --job-name=gapso_bus_f
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/gapso_bus_f_%j.log
#SBATCH --error=logs/gapso_bus_f_%j_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 GA-PSO load-shedding search -- Bus F"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "Host: $(hostname)"
echo "========================================"

mkdir -p logs

# ----------------------------------------------------------------------
# Pinned to R2025b to match this project's other Great Lakes jobs (see
# run_gradual_ramp_serial.sh) -- the .slx models here were re-exported to
# R2025b format specifically for Great Lakes. If you re-export to a
# different target version, update this to match.
#
# VERIFY BEFORE RELYING ON THIS -- two separate checks, both needed here
# (the gradual-ramp job only needed the Simulink one; this job also needs
# Global Optimization Toolbox for particleswarm/ga):
#   module load matlab/R2025b
#   matlab -batch "disp(license('test','simulink')); disp(license('test','gads_toolbox'))"
# should print 1 twice. If either prints 0, this job will fail partway
# through (at the sim() call, or at the optimoptions('particleswarm',...)
# call) rather than at startup -- confirm this BEFORE submitting a real
# 20+ hour job, not after.
# ----------------------------------------------------------------------
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/08_gapso_optimization"

echo ""
echo "[STEP 1] Confirming Global Optimization Toolbox + Simulink licenses ..."
matlab -batch "assert(license('test','gads_toolbox')==1, 'Global Optimization Toolbox not licensed/available on this node'); assert(license('test','simulink')==1, 'Simulink not licensed/available on this node'); disp('Both licenses OK.')"

echo ""
echo "[STEP 2] Running the GA-PSO search (SCENARIO_NAME='bus_f' inside the script) ..."
matlab -batch "gapso_load_shedding_reset"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in: model_outputs/gapso_reset_2026_09_22/bus_f/"
echo "========================================"

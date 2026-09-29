#!/bin/bash
# ============================================================================
# ThermalProject -- combines the 8 per-duration partial results from
# run_gradual_ramp_array.sh into the same summary table + 3 plots the
# serial script produces. Runs in seconds -- no Simulink involved.
#
# Normally you don't submit this by hand: submit_gradual_ramp_array.sh
# chains it automatically after the array job. Use this directly only if
# you need to re-run the aggregation (e.g. after resubmitting one failed
# array index).
# ============================================================================

#SBATCH --job-name=combine_gradual_ramp
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --output=logs/combine_gradual_ramp_%j.log
#SBATCH --error=logs/combine_gradual_ramp_%j_error.log

set -euo pipefail

echo "========================================"
echo "Combining gradual-ramp array results"
echo "Time: $(date)"
echo "========================================"

mkdir -p logs

# Pinned to R2025b -- matches the other job scripts; harmless here since
# combine_gradual_ramp_results.m never touches Simulink, but keeps every
# stage on the same MATLAB version for consistency.
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/06_gradual_ramp"
matlab -batch "combine_gradual_ramp_results"

echo ""
echo "Done: $(date)"

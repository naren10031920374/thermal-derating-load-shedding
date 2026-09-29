#!/bin/bash
# ============================================================================
# ThermalProject -- mandatory no-op verification for the ramped Simulink
# model, before trusting ANY gradual-ramp result (serial or array). Fails
# loudly (nonzero exit) if any bus doesn't match to floating-point noise.
#
# Submit from the project root:
#   sbatch scripts/00_environment/run_verify_ramp_model.sh
# ============================================================================

#SBATCH --job-name=verify_ramp_model
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/verify_ramp_model_%j.log
#SBATCH --error=logs/verify_ramp_model_%j_error.log

set -euo pipefail

echo "========================================"
echo "Verifying ramped model is a no-op at defaults"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "========================================"

mkdir -p logs

# Pinned to R2025b -- the newest MATLAB installed on Great Lakes, matching
# the version the .slx models were re-exported to via
# export_models_for_greatlakes.m (see that script's header for why).
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/06_gradual_ramp"
matlab -batch "compare_sheddable_vs_ramped_corridor_triad_collapse"

echo ""
echo "PASSED. Done: $(date)"

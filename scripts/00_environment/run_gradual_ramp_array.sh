#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 gradual-ramp sweep, PARALLEL (one SLURM array
# task per ramp duration -- 8 tasks instead of one 2-2.25 hour serial job).
#
# Do NOT submit this directly the first time -- submit_gradual_ramp_array.sh
# does this AND chains the combine job after it, in the right order. Use
# this file directly only if you want to resubmit a single failed duration
# (e.g. `sbatch --array=3 run_gradual_ramp_array.sh` to redo duration_idx=3).
#
# IMPORTANT -- verify before relying on this:
# Each array task opens its own MATLAB+Simulink session, so 8 tasks means
# 8 CONCURRENT Simulink license checkouts. If your account only has a
# handful of concurrent Simulink seats, tasks beyond that limit will queue
# or error out waiting on a license. The %3 below caps this job to 3
# concurrent tasks as a conservative default -- raise it (e.g. %8 for no
# cap) only after confirming your license pool can support it, or just
# watch the first run's logs for license-wait errors and back off if seen.
# ============================================================================

#SBATCH --job-name=gradual_ramp_array
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --array=1-8%3
#SBATCH --output=logs/gradual_ramp_array_%A_%a.log
#SBATCH --error=logs/gradual_ramp_array_%A_%a_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 gradual-ramp sweep -- ARRAY TASK $SLURM_ARRAY_TASK_ID of 8"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID  Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Host: $(hostname)"
echo "========================================"

mkdir -p logs

# Pinned to R2025b -- the newest MATLAB installed on Great Lakes, matching
# the version the .slx models were re-exported to via
# export_models_for_greatlakes.m (see that script's header for why).
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/06_gradual_ramp"

echo ""
echo "Running duration_idx=$SLURM_ARRAY_TASK_ID (see the script header for the RAMP_DURATIONS list) ..."
matlab -batch "test_shed_gradual_ramp_corridor_triad_collapse($SLURM_ARRAY_TASK_ID)"

echo ""
echo "Done: $(date)"

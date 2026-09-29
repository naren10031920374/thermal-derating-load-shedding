#!/bin/bash
# ============================================================================
# ThermalProject -- one-command submission of the full parallel Stage 06
# pipeline: verify -> array sweep (8 durations, parallel) -> combine.
# Each stage only runs if the previous one succeeds (SLURM --dependency).
#
# Run from the project root (NOT from inside scripts/00_environment):
#   cd /path/to/ThermalProject
#   bash scripts/00_environment/submit_gradual_ramp_array.sh
#
# This only submits the jobs -- it returns immediately. Track progress with:
#   squeue -u $USER
# and check scripts/00_environment/logs/ for each stage's output once it
# starts. If combine_gradual_ramp never runs, check squeue/sacct: it means
# either verify or one of the 8 array tasks failed, and the chain stopped
# rather than combining an incomplete result.
# ============================================================================

set -euo pipefail

mkdir -p logs

echo "Submitting verification job ..."
VERIFY_JOB=$(sbatch --parsable scripts/00_environment/run_verify_ramp_model.sh)
echo "  -> job $VERIFY_JOB"

echo "Submitting array job (8 parallel durations, depends on verify passing) ..."
ARRAY_JOB=$(sbatch --parsable --dependency=afterok:$VERIFY_JOB scripts/00_environment/run_gradual_ramp_array.sh)
echo "  -> job $ARRAY_JOB"

echo "Submitting combine job (depends on ALL 8 array tasks passing) ..."
COMBINE_JOB=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB scripts/00_environment/run_combine_gradual_ramp.sh)
echo "  -> job $COMBINE_JOB"

echo ""
echo "Chain submitted: verify ($VERIFY_JOB) -> array ($ARRAY_JOB, 8 tasks) -> combine ($COMBINE_JOB)"
echo "Watch with: squeue -u \$USER"
echo "If a stage is skipped/cancelled, check 'sacct -j <jobid>' for that stage --"
echo "it means the previous stage did not fully succeed."

#!/bin/bash
# ============================================================================
# ThermalProject -- Option B (staged shedding / load limiter), closed-loop check.
# One array task = one row of scenario_table.csv. Needs the earlier pipeline outputs
# for that scenario (no-shed run, detector triggers, bisection) in DATASET_OUTPUT_ROOT.
#
# Step 0 (once, on the login node or your laptop, from scripts/09_dataset_pipeline):
#   python make_staged_caps.py --nn-table <path>/nn_table_v2.csv
#
# Step 1: a small pilot with the two validation checks switched on (2 scenarios):
#   sbatch --array=34,12 --export=ALL,VALIDATE=1 scripts/09_dataset_pipeline/run_staged_array.sh
#
# Step 2: the 10 held-out test scenarios (rows 12,13,14,19,20,24,32,34,35,36):
#   sbatch --array=12,13,14,19,20,24,32,34,35,36%3 scripts/09_dataset_pipeline/run_staged_array.sh
#
# Wiring check only (200 s, one sim):   sbatch --array=34 --export=ALL,SMOKE=1 ...
# Re-submitting is safe: finished simulations are reused (staged_<ID>_iterations.json).
# ============================================================================
#SBATCH --job-name=staged_limiter
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --array=34
#SBATCH --output=logs/staged_%A_%a.log
#SBATCH --error=logs/staged_%A_%a_error.log

set -euo pipefail
ROW="${SLURM_ARRAY_TASK_ID}"
echo "Staged limiter row ${ROW}  Started: $(date)  Host: $(hostname)  Job: ${SLURM_JOBID}"
mkdir -p logs

OPTS="struct()"
if [ "${SMOKE:-0}" = "1" ]; then
  OPTS="struct('smoke_test', true)"
elif [ "${VALIDATE:-0}" = "1" ]; then
  OPTS="struct('validate', true)"
fi

module load matlab/R2025b
cd "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline"

if [ ! -f staged_caps.csv ]; then
  echo "staged_caps.csv missing: run make_staged_caps.py first." >&2
  exit 1
fi

T=$(date +%s)
matlab -batch "run_staged_limiter(${ROW}, ${OPTS})"
echo "Done in $(( $(date +%s) - T )) s  ($(date))"


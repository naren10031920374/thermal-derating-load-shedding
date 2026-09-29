#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline, NO-SHED runs for many scenarios as a
# SLURM job array. One array task = one row of scenario_table.csv.
#
# Row numbers: 1-50 are S001-S050, 51-60 are the spare rows X01-X10.
#
# Submit from the project root (create logs/ first, SLURM will not):
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_scenarios_array.sh               # rows 1-50
#   sbatch --array=1-10%3 scripts/09_dataset_pipeline/run_scenarios_array.sh # rows 1-10 only
#   sbatch --array=12,19,33 scripts/09_dataset_pipeline/run_scenarios_array.sh  # re-run specific rows
#
# IMPORTANT -- concurrency: every task opens its own MATLAB + Simulink
# session, so N running tasks = N concurrent Simulink license checkouts.
# The %3 below caps this at 3 at a time. Raise it only after the pilot shows
# your Simulink seat count allows more (watch the first logs for license-wait
# errors). Nothing has been confirmed about the seat count yet.
#
# Time limit: 04:00:00 per task is a guess. The pilot log prints the real
# wall-clock time of one full run -- set --time from that number.
# ============================================================================

#SBATCH --job-name=scenario_noshed
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --array=1-50%3
#SBATCH --output=logs/scenario_noshed_%A_%a.log
#SBATCH --error=logs/scenario_noshed_%A_%a_error.log

set -euo pipefail

echo "========================================"
echo "Dataset pipeline -- no-shed run, table row ${SLURM_ARRAY_TASK_ID}"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID  Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Host: $(hostname)"
echo "Output root: ${DATASET_OUTPUT_ROOT:-<default: model_outputs/dataset_pipeline>}"
echo "========================================"

mkdir -p logs

module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline"

T0=$(date +%s)
matlab -batch "run_scenario(${SLURM_ARRAY_TASK_ID}, struct('format', 'parquet'))"
T1=$(date +%s)

echo ""
echo "Row ${SLURM_ARRAY_TASK_ID} wall clock: $((T1 - T0)) s"
echo "Done: $(date)"

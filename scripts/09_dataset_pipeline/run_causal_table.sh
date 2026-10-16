#!/bin/bash
# ============================================================================
# ThermalProject -- causal cap network, step 1: build the stage-by-stage table
# (build_causal_nn_table.py). Reads the saved 5000 s signal tables, so run it
# on a compute node:
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_causal_table.sh
#
# Writes nn_stage_table.csv and nn_stage_table_skipped.csv next to nn_table_v2.csv.
# Does not touch any existing result. Same Python venv as the detector step.
# ============================================================================
#SBATCH --job-name=nn_stage_table
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --output=logs/nn_stage_table_%j.log
#SBATCH --error=logs/nn_stage_table_%j_error.log

set -euo pipefail
mkdir -p logs
module load python/3.11.5
source ~/venv_detector/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "Started: $(date)  Host: $(hostname)  Job: ${SLURM_JOBID}"
python scripts/09_dataset_pipeline/build_causal_nn_table.py
echo "Done: $(date)"

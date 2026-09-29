#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline step 3: detector trigger times.
# Small batch job (the login node kills it for memory).
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_detect_triggers.sh S041
#   sbatch scripts/09_dataset_pipeline/run_detect_triggers.sh --all
#
# Needs the venv from setup:  ~/venv_detector  (scikit-learn 1.9.0 -- must match
# the version the early_warning_N60s.pkl was saved with).
# ============================================================================
#SBATCH --job-name=detect_triggers
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --output=logs/detect_triggers_%j.log
#SBATCH --error=logs/detect_triggers_%j_error.log

set -euo pipefail
mkdir -p logs
module load python/3.11.5
source ~/venv_detector/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "Started: $(date)  args: $*"
python scripts/09_dataset_pipeline/detect_triggers.py "$@"
echo "Done: $(date)"

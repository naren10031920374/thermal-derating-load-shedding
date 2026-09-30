#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline step 5: build the NN training table and
# train / evaluate the shed models. Reads the saved 5000 s signal tables, so run
# it on a compute node:
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_nn_pipeline.sh
#
# Uses the same Python venv as the detector step (~/venv_detector).
# ============================================================================
#SBATCH --job-name=nn_table_train
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --output=logs/nn_pipeline_%j.log
#SBATCH --error=logs/nn_pipeline_%j_error.log

set -euo pipefail
mkdir -p logs
module load python/3.11.5
source ~/venv_detector/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "Started: $(date)"
python scripts/09_dataset_pipeline/build_nn_table.py
python scripts/09_dataset_pipeline/train_shed_models.py
echo "Done: $(date)"

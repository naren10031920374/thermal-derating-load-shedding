#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline step 5, VERSION 2: rebuild the NN table with
# 12 extra "how fast is it changing" features (build_nn_table_v2.py), then train /
# evaluate the same six shed models on it. Reads the saved 5000 s signal tables,
# so run it on a compute node:
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_nn_pipeline_v2.sh
#
# Writes nn_table_v2.csv and shed_models_v2/ next to the existing nn_table.csv and
# shed_models/, so the batch-2 results (25 features) are NOT overwritten.
# Uses the same Python venv as the detector step (~/venv_detector).
# ============================================================================
#SBATCH --job-name=nn_v2_train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --output=logs/nn_pipeline_v2_%j.log
#SBATCH --error=logs/nn_pipeline_v2_%j_error.log

set -euo pipefail
mkdir -p logs
module load python/3.11.5
source ~/venv_detector/bin/activate
cd "$SLURM_SUBMIT_DIR"
ROOT="${DATASET_OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/model_outputs/dataset_pipeline}"
echo "Started: $(date)"
python scripts/09_dataset_pipeline/build_nn_table_v2.py
python scripts/09_dataset_pipeline/train_shed_models.py \
    --table "$ROOT/nn_table_v2.csv" --out-dir "$ROOT/shed_models_v2"
echo "Done: $(date)"

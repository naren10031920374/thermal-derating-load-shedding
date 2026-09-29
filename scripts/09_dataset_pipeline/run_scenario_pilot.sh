#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline, PILOT: one scenario, no-shed run.
#
# Runs the quick 200 s wiring check first, then the full 5000 s no-shed run.
# Prints how long each takes -- that number decides the cluster time estimates
# in the plan, so read it from the log.
#
# Submit from the project root (create logs/ first, SLURM will not):
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_scenario_pilot.sh            # default S041
#   sbatch scripts/09_dataset_pipeline/run_scenario_pilot.sh S045       # another scenario
#
# S041 is an exact repeat of Bus C v4: Bus C should collapse at about
# t = 2094.47 s. If it does, the pipeline reproduces the original result.
#
# Optional: keep the big output files off your home quota, e.g.
#   export DATASET_OUTPUT_ROOT=/scratch/<your_account>/<uniqname>/dataset_pipeline
# ============================================================================

#SBATCH --job-name=scenario_pilot
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/scenario_pilot_%j.log
#SBATCH --error=logs/scenario_pilot_%j_error.log

set -euo pipefail

SCENARIO="${1:-S041}"

echo "========================================"
echo "Dataset pipeline pilot -- no-shed run for scenario ${SCENARIO}"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "Host: $(hostname)"
echo "Output root: ${DATASET_OUTPUT_ROOT:-<default: model_outputs/dataset_pipeline>}"
echo "========================================"

mkdir -p logs

# Same pin as the other job scripts: newest MATLAB on Great Lakes, matching
# the re-exported models in the greatlakes_export folders.
module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline"

echo ""
echo "[STEP 1] Confirming Simulink license ..."
matlab -batch "assert(license('test','simulink')==1, 'Simulink not licensed/available on this node'); disp('Simulink license OK.')"

echo ""
echo "[STEP 2] 200 s wiring check (smoke test) ..."
T0=$(date +%s)
matlab -batch "run_scenario('${SCENARIO}', struct('smoke_test', true))"
T1=$(date +%s)
echo "Smoke test wall clock: $((T1 - T0)) s"

echo ""
echo "[STEP 3] Full 5000 s no-shed run ..."
T2=$(date +%s)
matlab -batch "run_scenario('${SCENARIO}', struct('format', 'parquet'))"
T3=$(date +%s)
echo "Full run wall clock: $((T3 - T2)) s   <-- record this number"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: model_outputs/dataset_pipeline/${SCENARIO}/  (or DATASET_OUTPUT_ROOT)"
echo "========================================"

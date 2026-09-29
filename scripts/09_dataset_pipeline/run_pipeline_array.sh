#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline, FULL per-scenario chain as a job array.
# One array task = one row of scenario_table.csv (1-50 = S001-S050, 51-60 = X01-X10)
# and does all three steps in order:
#   [A] no-shed 5000 s run          (MATLAB, run_scenario)
#   [B] detector trigger times      (Python, detect_triggers.py)
#   [C] bisection for minimum safe shed (MATLAB, bisect_scenario)
# A scenario that does not collapse stops after [A]/[B] (nothing to bisect).
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_pipeline_array.sh                       # rows 1-50, 3 at a time
#   sbatch --array=1-6%3 scripts/09_dataset_pipeline/run_pipeline_array.sh          # a first small batch
#   sbatch --array=12,19 scripts/09_dataset_pipeline/run_pipeline_array.sh          # re-run specific rows
#
# Every step skips work that is already done (parquet exists / result json exists),
# so re-submitting is safe. Concurrency %3 = 3 Simulink sessions at once; raise
# it only after the first batch shows no licence-wait errors.
# ============================================================================
#SBATCH --job-name=scenario_pipeline
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --array=1-50%3
#SBATCH --output=logs/pipeline_%A_%a.log
#SBATCH --error=logs/pipeline_%A_%a_error.log

set -euo pipefail
ROW="${SLURM_ARRAY_TASK_ID}"
echo "Pipeline row ${ROW}  Started: $(date)  Host: $(hostname)  Job: ${SLURM_JOBID}"
mkdir -p logs

# Scenario id for this row (S001.. / X01..) from the table, so we can test for outputs.
ID=$(python3 - "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline/scenario_table.csv" "$ROW" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(rows[int(sys.argv[2]) - 1]["scenario_id"])
PY
)
OUT_ROOT="${DATASET_OUTPUT_ROOT:-$SLURM_SUBMIT_DIR/model_outputs/dataset_pipeline}"
OUT="$OUT_ROOT/$ID"
echo "Scenario ${ID}  ->  ${OUT}"

module load matlab/R2025b
cd "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline"

# ---- [A] no-shed run -------------------------------------------------------
if [ -f "$OUT/noshed_${ID}_summary.json" ] && [ -f "$OUT/noshed_${ID}_5000s.parquet" ]; then
  echo "[A] no-shed output exists, skipping"
else
  T=$(date +%s)
  matlab -batch "run_scenario(${ROW}, struct('format','parquet'))"
  echo "[A] done in $(( $(date +%s) - T )) s"
fi

# ---- [B] detector trigger times (Python venv) ------------------------------
if [ -f "$OUT/detector_triggers_${ID}.json" ]; then
  echo "[B] detector json exists, skipping"
else
  T=$(date +%s)
  ( module load python/3.11.5; source ~/venv_detector/bin/activate; \
    python "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline/detect_triggers.py" "$ID" )
  echo "[B] done in $(( $(date +%s) - T )) s"
fi

# ---- does anything collapse? (from the step-2 collapse table) --------------
# [B] runs for EVERY scenario (mild ones too: that is the detector false-alarm
# check); only the bisection is skipped when nothing collapses.
if ! grep -q -i -E ',(1|true),' "$OUT/noshed_${ID}_collapse.csv"; then
  echo "No bus collapses in ${ID}: no bisection needed."
  echo "Done: $(date)"
  exit 0
fi

# ---- [C] bisection ---------------------------------------------------------
T=$(date +%s)
matlab -batch "bisect_scenario(${ROW}, struct('verify_monotonic', true))"
echo "[C] done in $(( $(date +%s) - T )) s"
echo "Done: $(date)"

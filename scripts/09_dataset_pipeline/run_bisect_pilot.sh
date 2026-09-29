#!/bin/bash
# ============================================================================
# ThermalProject -- dataset pipeline step 4 PILOT: bisection for ONE scenario.
# Needs step 2 (no-shed run) and step 3 (detector triggers) already done for it.
#
#   mkdir -p logs
#   sbatch scripts/09_dataset_pipeline/run_bisect_pilot.sh            # default S041
#   sbatch scripts/09_dataset_pipeline/run_bisect_pilot.sh S045
#
# Runs the 19-point grid (0.05..0.95) and one extra "one step harder" sim to
# check the safe/collapse pattern is a single clean cutoff (monotonic).
# Expect ~5 search sims + 1 check, ~12 min each -> about 70 min.
# Re-submitting the same command resumes: finished fractions are not re-run.
# ============================================================================
#SBATCH --job-name=bisect_pilot
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/bisect_pilot_%j.log
#SBATCH --error=logs/bisect_pilot_%j_error.log

set -euo pipefail
SCENARIO="${1:-S041}"
echo "Bisection pilot for ${SCENARIO}  Started: $(date)  Host: $(hostname)"
mkdir -p logs
module load matlab/R2025b
cd "$SLURM_SUBMIT_DIR/scripts/09_dataset_pipeline"
T0=$(date +%s)
matlab -batch "bisect_scenario('${SCENARIO}', struct('verify_monotonic', true))"
T1=$(date +%s)
echo "Bisection wall clock: $((T1 - T0)) s"
echo "Done: $(date)"

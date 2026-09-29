#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 GA-PSO load-shedding search, THREE-BUS ONLY
# Time-boxed, WIDENED 400-second evaluation window, run alone with no other
# scenario competing for the job's time budget.
#
# Submit from the project root:
#   sbatch scripts/08_gapso_optimization/run_gapso_timeboxed_widened_threebus_only.sh
#
# WHY THIS VERSION EXISTS: job 61977687 ran Bus F, then Bus C, then three-bus
# back-to-back in one job. Bus F overran its own 180-minute target by 106
# minutes (MaxTime in MATLAB's particleswarm/ga is only checked between full
# iterations, not between individual candidate evaluations -- with a
# population of 6 evaluated serially, one expensive iteration can blow the
# budget). That overrun ate into the shared 10-hour job limit, so three-bus
# was killed before it ever started -- it has never once gotten to run its
# own search under the time-boxed approach.
#
# This job runs three-bus ALONE, so it can't be starved by anything else.
# It also gets a bigger time budget than the original 3 hours (6 hours here)
# since there's no need to share -- more candidates evaluated means a better
# shot at a real, confirmed answer on the first clean attempt.
#
# OUTPUT LOCATION: same folder as job 61977687's widened run, so results
# land alongside Bus F and Bus C's:
#   model_outputs/gapso_widened_2026_09_26/three_bus/
# ============================================================================

#SBATCH --job-name=gapso_threebus
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/gapso_threebus_%j.log
#SBATCH --error=logs/gapso_threebus_%j_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 GA-PSO load-shedding search -- THREE-BUS ONLY"
echo "(time-boxed, 6h, WIDENED 400s evaluation window, no competition)"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "Host: $(hostname)"
echo "========================================"

mkdir -p logs

module load matlab/R2025b

cd "$SLURM_SUBMIT_DIR/scripts/08_gapso_optimization"

echo ""
echo "[STEP 1] Confirming Global Optimization Toolbox + Simulink licenses ..."
matlab -batch "assert(license('test','gads_toolbox')==1, 'Global Optimization Toolbox not licensed/available on this node'); assert(license('test','simulink')==1, 'Simulink not licensed/available on this node'); disp('Both licenses OK.')"

echo ""
echo "=========================================================="
echo "[STEP 2] Three-bus scenario -- time-boxed to 6 hours, widened window, ALONE ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed_widened('three_bus', 6)"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in:"
echo "  model_outputs/gapso_widened_2026_09_26/three_bus/"
echo "========================================"

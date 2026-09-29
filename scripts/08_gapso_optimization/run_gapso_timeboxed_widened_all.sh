#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 GA-PSO load-shedding search, ALL THREE scenarios
# (Bus F, Bus C, three-bus), each TIME-BOXED to 3 hours, using the WIDENED
# 400-second truncated-evaluation window, run back-to-back in one job.
#
# Submit from the project root:
#   sbatch scripts/08_gapso_optimization/run_gapso_timeboxed_widened_all.sh
#
# WHY THIS VERSION EXISTS: the first time-boxed run (120s window) landed
# Bus C and three-bus both on the 30% shed floor, and both FAILED the full
# confirmation run -- Bus C actually collapses at 2745s, three-bus at
# 1260s, both well past that old 120s window. Widening the window to 400s
# (gapso_load_shedding_timeboxed_widened.m) lets the search actually see
# those delayed collapses instead of being fooled by a short healthy-
# looking window. Bus F never got a time-boxed attempt at all -- its
# original run was the one that hit the unbounded 30-hour timeout with
# zero results -- so this is also its first real time-boxed shot.
#
# TRADE-OFF: each candidate now costs ~3.3x longer to evaluate (400s of
# simulated time instead of 120s), so within the same 3-hour budget the
# search will try FEWER candidates per scenario than the first run did.
# That's expected and is the whole point -- fewer candidates, but each one
# honestly evaluated.
#
# THREE SCENARIOS x ~3 HOURS EACH = ~9 HOURS TARGET, with --time=10:00:00
# below as a safety margin (not a guarantee -- if a confirmation run's
# 30-minute reserve turns out too small more than once, this could still
# run past 10 hours and get killed by SLURM; check each scenario's own
# printed "TOTAL WALL-CLOCK TIME" line in the log before the next one
# starts, and scancel + resubmit with a bigger --time if the pattern looks
# like it will blow the budget).
#
# OUTPUT LOCATION: results go to a NEW folder so they never overwrite the
# original 120s-window run:
#   model_outputs/gapso_widened_2026_09_26/bus_f/
#   model_outputs/gapso_widened_2026_09_26/bus_c/
#   model_outputs/gapso_widened_2026_09_26/three_bus/
# ============================================================================

#SBATCH --job-name=gapso_widened
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/gapso_widened_%j.log
#SBATCH --error=logs/gapso_widened_%j_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 GA-PSO load-shedding search -- Bus F + Bus C + three-bus"
echo "(time-boxed, 3h each, WIDENED 400s evaluation window)"
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
echo "[STEP 2a] Bus F scenario -- time-boxed to 3 hours, widened window ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed_widened('bus_f', 3)"

echo ""
echo "=========================================================="
echo "[STEP 2b] Bus C scenario -- time-boxed to 3 hours, widened window ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed_widened('bus_c', 3)"

echo ""
echo "=========================================================="
echo "[STEP 2c] Three-bus scenario -- time-boxed to 3 hours, widened window ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed_widened('three_bus', 3)"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in:"
echo "  model_outputs/gapso_widened_2026_09_26/bus_f/"
echo "  model_outputs/gapso_widened_2026_09_26/bus_c/"
echo "  model_outputs/gapso_widened_2026_09_26/three_bus/"
echo "========================================"

#!/bin/bash
# ============================================================================
# ThermalProject -- Stage 06 GA-PSO load-shedding search, Bus C + three-bus
# scenarios, each TIME-BOXED to ~3 hours, run back-to-back in one job.
#
# Submit from the project root:
#   sbatch scripts/08_gapso_optimization/run_gapso_timeboxed_remaining.sh
#
# HOW THIS DIFFERS FROM run_gapso_remaining_scenarios.sh (the earlier
# POP_SIZE/MAX_GENERATIONS-guessing version): this one uses
# gapso_load_shedding_timeboxed.m, which hard-caps each scenario's PSO and
# GA phases with MATLAB's 'MaxTime' option, so each scenario's search
# itself is capped by wall-clock time, not by a guessed number of
# evaluations. Total time per scenario should land close to 3 hours
# (~30 min of that reserved for the one full confirmation run at the end,
# which is NOT time-capped since a partial confirmation run would be
# useless -- if that step runs long, total time for that scenario can run
# a bit past 3 hours).
#
# TWO SCENARIOS x ~3 HOURS EACH = ~6 HOURS TARGET, with --time=07:00:00
# below as a safety margin above that target (not a guarantee -- if the
# confirmation run reserve turns out to be too small for both scenarios,
# this could still run past 7 hours and get killed by SLURM; check the
# log's own printed "TOTAL WALL-CLOCK TIME" line after Bus C finishes to
# see if the reserve needs adjusting before three_bus starts).
#
# NOTE ON RUNNING ALONGSIDE THE BUS F JOB: if the big Bus F search (job
# 61826551 or its resubmission) is still running when this is submitted,
# both jobs will want a Global Optimization Toolbox + Simulink license
# checkout at the same time. If this job's STEP 1 license check hangs,
# that's the most likely reason -- check 'squeue -u narenv' for the Bus F
# job's status.
# ============================================================================

#SBATCH --job-name=gapso_timeboxed
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --output=logs/gapso_timeboxed_%j.log
#SBATCH --error=logs/gapso_timeboxed_%j_error.log

set -euo pipefail

echo "========================================"
echo "Stage 06 GA-PSO load-shedding search -- Bus C + three-bus (time-boxed, 3h each)"
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
echo "[STEP 2a] Bus C scenario -- time-boxed to 3 hours ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed('bus_c', 3)"

echo ""
echo "=========================================================="
echo "[STEP 2b] Three-bus scenario -- time-boxed to 3 hours ..."
echo "=========================================================="
matlab -batch "gapso_load_shedding_timeboxed('three_bus', 3)"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in:"
echo "  model_outputs/gapso_reset_2026_09_22/bus_c/"
echo "  model_outputs/gapso_reset_2026_09_22/three_bus/"
echo "========================================"

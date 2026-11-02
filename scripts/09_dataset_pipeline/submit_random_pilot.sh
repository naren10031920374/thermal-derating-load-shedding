#!/bin/bash
# ============================================================================
# Probabilistic cap network (Network B), PILOT: the first 30 random scenarios
# RT001-RT030 = rows 142-171 of scenario_table.csv.
#
# What it submits (two job arrays, the second waits for the first, task by task):
#   1. SETUP   no-shed 5000 s run + detector trigger times (bisection is skipped).
#              About 15 min per scenario.
#   2. SWEEP   staged limiter with the network caps at offsets +15, +10, +5, 0, -5, -10 kW
#              (most permissive first, stops at the first offset that keeps all 10 buses alive),
#              with the sensor setting below. Up to 6 simulations of about 11 min each.
#              A scenario that does not collapse in the no-shed run is recorded as
#              "no_collapse" and costs no simulation (then use the next RZ reserve of the set).
# Task N of the sweep starts only after task N of the setup finished OK
# (--dependency=aftercorr). If a setup task fails, its sweep task stays pending
# ("DependencyNeverSatisfied"): cancel it with scancel <jobid>_<row>.
#
# Be fair to the other cluster users: 3 tasks at a time, 2 cores each. Change MAXJOBS only if the
# cluster is quiet (sinfo -p standard -o "%C").
#
# Run from the project root on the cluster (after git pull):
#   bash scripts/09_dataset_pipeline/submit_random_pilot.sh                 # submit
#   DRY=1 bash scripts/09_dataset_pipeline/submit_random_pilot.sh           # only print the sbatch lines
# Options (environment variables):
#   ROWS=142-171     which table rows (default: the 30 pilot rows)
#   MAXJOBS=3        most tasks running at the same time, per array
#   SENSOR=g10       g10 = 10 % Gaussian noise + 1 s average (default; the worst case that was safe at margin 0)
#                    ideal = ideal sensor, no noise, no delay
#   MEAS_SEED=1      noise seed of the sensor
# ============================================================================
set -euo pipefail
D=scripts/09_dataset_pipeline
ROWS="${ROWS:-142-171}"
MAXJOBS="${MAXJOBS:-3}"
SENSOR="${SENSOR:-g10}"
MEAS_SEED="${MEAS_SEED:-1}"
DRY="${DRY:-0}"

# ---- checks: the random rows must be on the cluster (git pull) --------------
grep -q "^RT030," ${D}/scenario_table.csv || { echo "scenario_table.csv has no RT030: git pull first." >&2; exit 1; }
grep -q "^RT030," ${D}/staged_caps.csv    || { echo "staged_caps.csv has no RT030: git pull first." >&2; exit 1; }
grep -q SKIP_BISECT ${D}/run_pipeline_array.sh || { echo "run_pipeline_array.sh has no SKIP_BISECT: git pull first." >&2; exit 1; }
grep -q OFFSETS ${D}/run_staged_array.sh       || { echo "run_staged_array.sh has no OFFSETS: git pull first." >&2; exit 1; }
mkdir -p logs

case "${SENSOR}" in
  g10)   SENS="MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_FILTER=1,MEAS_NOISE=10,MEAS_SEED=${MEAS_SEED}"; TAG="rnd_g10_w1s" ;;
  ideal) SENS=""; TAG="rnd_ideal" ;;
  *) echo "SENSOR must be g10 or ideal" >&2; exit 1 ;;
esac
SETUP_RES="--cpus-per-task=2 --mem=20G"
SWEEP_RES="--cpus-per-task=2 --mem=32G"
COMMON="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10,TAG=${TAG}"
[ -n "${SENS}" ] && COMMON="${COMMON},${SENS}"

CMD1="sbatch --parsable --array=${ROWS}%${MAXJOBS} ${SETUP_RES} --export=ALL,SKIP_BISECT=1 ${D}/run_pipeline_array.sh"
if [ "${DRY}" = "1" ]; then
  echo "[dry run] ${CMD1}"
  echo "[dry run] sbatch --parsable --array=${ROWS}%${MAXJOBS} ${SWEEP_RES} --dependency=aftercorr:<setup job> --export=ALL,${COMMON} ${D}/run_staged_array.sh"
  exit 0
fi

J1=$(${CMD1})
J2=$(sbatch --parsable --array="${ROWS}%${MAXJOBS}" ${SWEEP_RES} --dependency="aftercorr:${J1}" \
       --export=ALL,${COMMON} ${D}/run_staged_array.sh)
echo "setup  (no-shed run + detector)   job ${J1}   rows ${ROWS}   ${MAXJOBS} at a time"
echo "sweep  (staged limiter, tag ${TAG})   job ${J2}   waits for the setup of the same row"
echo
echo "Watch:    squeue -u \$USER"
echo "Options check (once a sweep task has started):  grep Options logs/staged_${J2}_142.log"
echo "Stop all: scancel ${J1} ${J2}      (finished work is kept; re-submitting skips it)"

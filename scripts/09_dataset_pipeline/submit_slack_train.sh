#!/bin/bash
# Margin fit, step 1: offset sweep on TRAINING scenarios (18 scenarios, 3 per cap cell).
# Submits THREE job arrays, one per sensor setting:
#   slacktr_ideal   ideal sensor (no noise, no delay)
#   slacktr_n2f     noise 2 %  + 10 s average
#   slacktr_n10f    noise 10 % + 10 s average
# Each task tries offsets +15, +10, +5, 0, -5, -10 kW on all six caps (most permissive first) and
# stops at the first one that keeps all 10 buses alive. Network caps, sensor period 1 s, noise seed 1.
# A scenario is skipped (with a message) if its no-shed collapse file or detector-trigger file is missing.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_slack_train.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_slack_margin.py
set -euo pipefail
# table rows (= scenario_table.csv row numbers) and their ids:
#  cell (1 bus, no E/F)  S003=3  S006=6  S030=30        cell (1 bus, E/F)  S010=10 S031=31 T024=84
#  cell (2 bus, no E/F)  S002=2  S011=11 S027=27        cell (2 bus, E/F)  S009=9  S026=26 T004=64
#  cell (3 bus, no E/F)  S001=1  S016=16 T008=68        cell (3 bus, E/F)  S023=23 T005=65 T048=108
IDS="S003 S006 S030 S010 S031 T024 S002 S011 S027 S009 S026 T004 S001 S016 T008 S023 T005 T048"
ROWNUM="3 6 30 10 31 84 2 11 27 9 26 64 1 16 68 23 65 108"
ROOT="${DATASET_OUTPUT_ROOT:-model_outputs/dataset_pipeline}"
ROWS=""
set -- ${ROWNUM}
for ID in ${IDS}; do
  R="$1"; shift
  if [ -f "${ROOT}/${ID}/noshed_${ID}_collapse.csv" ] && [ -f "${ROOT}/${ID}/detector_triggers_${ID}.json" ]; then
    ROWS="${ROWS:+${ROWS},}${R}"
  else
    echo "SKIPPED ${ID} (row ${R}): noshed_${ID}_collapse.csv or detector_triggers_${ID}.json not found in ${ROOT}/${ID}"
  fi
done
[ -n "${ROWS}" ] || { echo "No scenario has the needed files." >&2; exit 1; }
echo "Rows submitted: ${ROWS}"
COMMON="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10"
SCRIPT=scripts/09_dataset_pipeline/run_staged_array.sh
grep -q OFFSETS "$SCRIPT" || { echo "run_staged_array.sh has no OFFSETS option: git pull first." >&2; exit 1; }

J1=$(sbatch --parsable --array=${ROWS}%5 --export=ALL,${COMMON},TAG=slacktr_ideal "$SCRIPT")
J2=$(sbatch --parsable --array=${ROWS}%5 --export=ALL,${COMMON},TAG=slacktr_n2f,MEAS_NOISE=2,MEAS_FILTER=10 "$SCRIPT")
J3=$(sbatch --parsable --array=${ROWS}%5 --export=ALL,${COMMON},TAG=slacktr_n10f,MEAS_NOISE=10,MEAS_FILTER=10 "$SCRIPT")
FIRST="${ROWS%%,*}"
echo "ideal sensor        job ${J1}   (tag slacktr_ideal)"
echo "noise 2 % + avg     job ${J2}   (tag slacktr_n2f)"
echo "noise 10 % + avg    job ${J3}   (tag slacktr_n10f)"
echo "Check the settings, once the first task of each job has started:"
echo "  grep Options logs/staged_${J1}_${FIRST}.log logs/staged_${J2}_${FIRST}.log logs/staged_${J3}_${FIRST}.log"

#!/bin/bash
# Margin fit, step 2: two more checks. Submits FOUR job arrays:
#   slacktr_n5f       18 training scenarios, noise 5 %  + 10 s average, seed 1   (offsets 15..-10)
#   slack_n5f         12 held-out/example scenarios, same setting                (offsets 10..-10)
#   slacktr_n10f_s2   18 training scenarios, noise 10 % + 10 s average, SEED 2   (offsets 10..-10)
#   slack_n10f_s2     12 held-out/example scenarios, same setting                (offsets 10..-10)
# Each task tries the offsets most permissive first and stops at the first cap that keeps all 10 buses alive.
# Network caps, sensor period 1 s. A scenario is skipped (with a message) if its no-shed collapse file or
# detector-trigger file is missing. At most 5 tasks per array run at once.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_slack_extra.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_slack_margin2.py
set -euo pipefail
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
    echo "SKIPPED ${ID} (row ${R}): files not found in ${ROOT}/${ID}"
  fi
done
[ -n "${ROWS}" ] || { echo "No training scenario has the needed files." >&2; exit 1; }
TEST_ROWS=12,13,14,19,20,24,32,34,35,127,128,129
echo "Training rows: ${ROWS}"
echo "Held-out/example rows: ${TEST_ROWS}"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv"
SCRIPT=scripts/09_dataset_pipeline/run_staged_array.sh
grep -q OFFSETS "$SCRIPT" || { echo "run_staged_array.sh has no OFFSETS option: git pull first." >&2; exit 1; }

J1=$(sbatch --parsable --array=${ROWS}%5 --export=ALL,${BASE},OFFSETS=15:10:5:0:-5:-10,TAG=slacktr_n5f,MEAS_NOISE=5,MEAS_FILTER=10 "$SCRIPT")
J2=$(sbatch --parsable --array=${TEST_ROWS}%5 --export=ALL,${BASE},OFFSETS=10:5:0:-5:-10,TAG=slack_n5f,MEAS_NOISE=5,MEAS_FILTER=10 "$SCRIPT")
J3=$(sbatch --parsable --array=${ROWS}%5 --export=ALL,${BASE},OFFSETS=10:5:0:-5:-10,TAG=slacktr_n10f_s2,MEAS_NOISE=10,MEAS_FILTER=10,MEAS_SEED=2 "$SCRIPT")
J4=$(sbatch --parsable --array=${TEST_ROWS}%5 --export=ALL,${BASE},OFFSETS=10:5:0:-5:-10,TAG=slack_n10f_s2,MEAS_NOISE=10,MEAS_FILTER=10,MEAS_SEED=2 "$SCRIPT")
FIRST="${ROWS%%,*}"
echo "training  5 % + avg          job ${J1}   (tag slacktr_n5f)"
echo "held-out  5 % + avg          job ${J2}   (tag slack_n5f)"
echo "training 10 % + avg, seed 2  job ${J3}   (tag slacktr_n10f_s2)"
echo "held-out 10 % + avg, seed 2  job ${J4}   (tag slack_n10f_s2)"
echo "Check the settings, once the first task of each job has started:"
echo "  grep Options logs/staged_${J1}_${FIRST}.log logs/staged_${J2}_12.log logs/staged_${J3}_${FIRST}.log logs/staged_${J4}_12.log"

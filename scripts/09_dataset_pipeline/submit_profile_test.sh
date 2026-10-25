#!/bin/bash
# Profile-shaped test, all in one go. Submits SIX job arrays, chained so each waits for the one before:
#   P  pipeline steps A + B (no-shed run, detector triggers; no bisection) for rows 130-141 = H04..H15
#   then five offset sweeps on those 12 scenarios, one per sensor setting:
#       slackh_ideal    ideal sensor                      offsets +15 ... -10 kW
#       slackh_n2f      noise 2 %  + 10 s average
#       slackh_n5f      noise 5 %  + 10 s average
#       slackh_n10f     noise 10 % + 10 s average
#       slackh_n10f_s2  noise 10 % + 10 s average, noise seed 2
# (H01-H03 = rows 127-129 already have all of these results.)
# Network caps (staged_cap_groups_nn.csv), sensor period 1 s. The sweeps start when the pipeline job has ended
# (whatever its outcome); a scenario whose files are missing just fails its sweep task and is skipped by the summary.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_profile_test.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_slack_profile.py
set -euo pipefail
D=scripts/09_dataset_pipeline
[ "$(wc -l < ${D}/scenario_table.csv)" -ge 142 ] || { echo "scenario_table.csv has no H04-H15 rows: git pull first." >&2; exit 1; }
grep -q SKIP_BISECT ${D}/run_pipeline_array.sh || { echo "run_pipeline_array.sh has no SKIP_BISECT option: git pull first." >&2; exit 1; }
grep -q OFFSETS ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no OFFSETS option: git pull first." >&2; exit 1; }
for ID in H04 H15; do grep -q "^${ID}," ${D}/staged_caps.csv || { echo "staged_caps.csv has no ${ID} row: git pull first." >&2; exit 1; }; done
mkdir -p logs
ROWS=130-141
JP=$(sbatch --parsable --array=${ROWS}%4 --export=ALL,SKIP_BISECT=1 ${D}/run_pipeline_array.sh)
echo "pipeline A+B   job ${JP}   (rows ${ROWS})"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10"
DEP="--dependency=afterany:${JP}"
SCRIPT=${D}/run_staged_array.sh
J1=$(sbatch --parsable ${DEP} --array=${ROWS}%4 --export=ALL,${BASE},TAG=slackh_ideal "$SCRIPT")
J2=$(sbatch --parsable ${DEP} --array=${ROWS}%4 --export=ALL,${BASE},TAG=slackh_n2f,MEAS_NOISE=2,MEAS_FILTER=10 "$SCRIPT")
J3=$(sbatch --parsable ${DEP} --array=${ROWS}%4 --export=ALL,${BASE},TAG=slackh_n5f,MEAS_NOISE=5,MEAS_FILTER=10 "$SCRIPT")
J4=$(sbatch --parsable ${DEP} --array=${ROWS}%4 --export=ALL,${BASE},TAG=slackh_n10f,MEAS_NOISE=10,MEAS_FILTER=10 "$SCRIPT")
J5=$(sbatch --parsable ${DEP} --array=${ROWS}%4 --export=ALL,${BASE},TAG=slackh_n10f_s2,MEAS_NOISE=10,MEAS_FILTER=10,MEAS_SEED=2 "$SCRIPT")
echo "ideal sensor       job ${J1}   (tag slackh_ideal)"
echo "noise 2 % + avg    job ${J2}   (tag slackh_n2f)"
echo "noise 5 % + avg    job ${J3}   (tag slackh_n5f)"
echo "noise 10 % + avg   job ${J4}   (tag slackh_n10f)"
echo "noise 10 % s2      job ${J5}   (tag slackh_n10f_s2)"
echo "Watch:  squeue -u \$USER      Pipeline check (first task):  tail -n 5 logs/pipeline_${JP}_130.log"
echo "Once the sweeps have started:  grep Options logs/staged_${J1}_130.log logs/staged_${J2}_130.log logs/staged_${J3}_130.log logs/staged_${J4}_130.log logs/staged_${J5}_130.log"

#!/bin/bash
# Low-noise slack test: how fast does the cap room (+10 kW, ideal sensor) shrink as sensor noise grows?
# Submits FOUR job arrays (12 scenarios each):
#   slack_n1    noise 1 %            slack_n1f   noise 1 % + 10 s average
#   slack_n2    noise 2 %            slack_n2f   noise 2 % + 10 s average
# Each task tries offsets +10, +5, 0, -5, -10 kW on all six caps (most permissive first) and
# stops at the first one that keeps all 10 buses alive. Network caps, sensor period 1 s, noise seed 1.
# At most 3 tasks per array run at once.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_slack_lownoise.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_slack_curve.py
set -euo pipefail
ROWS=12,13,14,19,20,24,32,34,35,127,128,129
COMMON="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=10:5:0:-5:-10"
SCRIPT=scripts/09_dataset_pipeline/run_staged_array.sh
grep -q OFFSETS "$SCRIPT" || { echo "run_staged_array.sh has no OFFSETS option: git pull first." >&2; exit 1; }

J1=$(sbatch --parsable --array=${ROWS}%3 --export=ALL,${COMMON},TAG=slack_n1,MEAS_NOISE=1 "$SCRIPT")
J2=$(sbatch --parsable --array=${ROWS}%3 --export=ALL,${COMMON},TAG=slack_n2,MEAS_NOISE=2 "$SCRIPT")
J3=$(sbatch --parsable --array=${ROWS}%3 --export=ALL,${COMMON},TAG=slack_n1f,MEAS_NOISE=1,MEAS_FILTER=10 "$SCRIPT")
J4=$(sbatch --parsable --array=${ROWS}%3 --export=ALL,${COMMON},TAG=slack_n2f,MEAS_NOISE=2,MEAS_FILTER=10 "$SCRIPT")
echo "noise 1 %           job ${J1}   (tag slack_n1)"
echo "noise 2 %           job ${J2}   (tag slack_n2)"
echo "noise 1 % + avg     job ${J3}   (tag slack_n1f)"
echo "noise 2 % + avg     job ${J4}   (tag slack_n2f)"
echo "Check the settings, once the first task of each job has started:"
echo "  grep Options logs/staged_${J1}_12.log logs/staged_${J2}_12.log logs/staged_${J3}_12.log logs/staged_${J4}_12.log"

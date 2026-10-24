#!/bin/bash
# Noise-aware slack test: how much of the cap room (+10 kW with an ideal sensor)
# survives a noisy sensor?  Submits THREE job arrays (12 scenarios each), one per setting:
#   slack_n5     sensor noise 5 %
#   slack_n10    sensor noise 10 %
#   slack_n10f   sensor noise 10 % + the limiter averages the last 10 s of readings
# Each task tries offsets +10, +5, 0, -5, -10 kW on all six caps (most permissive first) and
# stops at the first one that keeps all 10 buses alive. Network caps (staged_cap_groups_nn.csv),
# sensor period 1 s, noise seed 1. At most 4 tasks per array run at once.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_slack_noise.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_slack_noise.py
set -euo pipefail
ROWS=12,13,14,19,20,24,32,34,35,127,128,129
COMMON="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=10:5:0:-5:-10"
SCRIPT=scripts/09_dataset_pipeline/run_staged_array.sh
grep -q OFFSETS "$SCRIPT" || { echo "run_staged_array.sh has no OFFSETS option: git pull first." >&2; exit 1; }

J1=$(sbatch --parsable --array=${ROWS}%4 --export=ALL,${COMMON},TAG=slack_n5,MEAS_NOISE=5 "$SCRIPT")
J2=$(sbatch --parsable --array=${ROWS}%4 --export=ALL,${COMMON},TAG=slack_n10,MEAS_NOISE=10 "$SCRIPT")
J3=$(sbatch --parsable --array=${ROWS}%4 --export=ALL,${COMMON},TAG=slack_n10f,MEAS_NOISE=10,MEAS_FILTER=10 "$SCRIPT")
echo "noise 5 %          job ${J1}   (tag slack_n5)"
echo "noise 10 %         job ${J2}   (tag slack_n10)"
echo "noise 10 % + filt  job ${J3}   (tag slack_n10f)"
echo "Check the setting went through, once the first task of each job has started:"
echo "  grep Options logs/staged_${J1}_12.log logs/staged_${J2}_12.log logs/staged_${J3}_12.log"

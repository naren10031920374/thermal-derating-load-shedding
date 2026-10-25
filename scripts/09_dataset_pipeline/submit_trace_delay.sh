#!/bin/bash
# Two trace runs for the delay figure: H02 (row 128), 10 % noise + 10 s average, present 5 kW margin (offset 0),
# once with no delay and once with a 5 s delay. Network caps, sensor period 1 s, noise seed 1.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_trace_delay.sh
set -euo pipefail
D=scripts/09_dataset_pipeline
mkdir -p logs
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,SAVE_TRACE=1,OFFSETS=0,MEAS_NOISE=10,MEAS_FILTER=10"
J1=$(sbatch --parsable --array=128 --export=ALL,${BASE},TAG=tr_dly0 ${D}/run_staged_array.sh)
J2=$(sbatch --parsable --array=128 --export=ALL,${BASE},MEAS_DELAY=5,TAG=tr_dly5s ${D}/run_staged_array.sh)
echo "tr_dly0  job ${J1}   no delay"
echo "tr_dly5s job ${J2}   5 s delay"
echo "Watch: squeue -u \$USER   (about 15-25 min)"

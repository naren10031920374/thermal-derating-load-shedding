#!/bin/bash
# Trace runs for the sensor-margin figures (bus voltages and load over time, like the S012/S020 figure).
# 10 single simulations of the staged limiter (network caps) with SAVE_TRACE=1, one cap offset per run.
# Nothing is swept: each run is one case that is shown in one figure panel.
# Offset = 5 - margin, so offset +5 = margin 0, offset 0 = margin 5, offset +6 = margin -1.
#   H01 = row 127, S035 = row 35, H09 = row 135
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_trace_sensor.sh
# Then:  python3 scripts/09_dataset_pipeline/make_sensor_graphs.py --dir model_outputs/dataset_pipeline --out sensor_graphs
set -euo pipefail
D=scripts/09_dataset_pipeline
grep -q SAVE_TRACE ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no SAVE_TRACE option: git pull first." >&2; exit 1; }
grep -q MEAS_MODE ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
mkdir -p logs
SCRIPT=${D}/run_staged_array.sh
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,SAVE_TRACE=1"
FAST="MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_SEED=1"   # one reading per 10 ms step, 50 ns delay (rounds to 0 steps)
sub() { # tag row offset extra
  J=$(sbatch --parsable --array="$2" --export=ALL,${BASE},TAG="$1",OFFSETS="$3",$4 "$SCRIPT")
  printf "%-24s job %s   row %-4s offset %-3s %s\n" "$1" "$J" "$2" "$3" "$4"
}
# Test 1 figure: H01 at margin 0, old test against 1 s and 100 ms averages
sub tr_s_h01_old        127 5  "MEAS_NOISE=10,MEAS_FILTER=10"                  # old test: 1 s sampling, 10 s average
sub tr_s_h01_g10w1      127 5  "MEAS_NOISE=10,MEAS_FILTER=1,${FAST}"
sub tr_s_h01_g10w100ms  127 5  "MEAS_NOISE=10,MEAS_FILTER=0.1,${FAST}"
# Test 2 figure: S035 (a tight scenario) at margin 0
sub tr_s_s035_old       35  5  "MEAS_NOISE=10,MEAS_FILTER=10"
sub tr_s_s035_g10w1     35  5  "MEAS_NOISE=10,MEAS_FILTER=1,${FAST}"
sub tr_s_s035_b05w1     35  5  "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_FILTER=1,${FAST}"
# Test 3 and Test 4 figures: H09
sub tr_s_h09_g10w1_o5     135 5 "MEAS_NOISE=10,MEAS_FILTER=1,${FAST}"
sub tr_s_h09_g10w500ms_o5 135 5 "MEAS_NOISE=10,MEAS_FILTER=0.5,${FAST}"
sub tr_s_h09_g10w500ms_o0 135 0 "MEAS_NOISE=10,MEAS_FILTER=0.5,${FAST}"
sub tr_s_h09_g10w1_o6     135 6 "MEAS_NOISE=10,MEAS_FILTER=1,${FAST}"
echo "Watch: squeue -u \$USER     (10 simulations, about 15 minutes each)"

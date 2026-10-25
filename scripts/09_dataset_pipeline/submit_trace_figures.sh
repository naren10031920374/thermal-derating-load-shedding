#!/bin/bash
# Trace runs for the result figures (voltage + load over time, like the S012/S020 figure).
# 12 single simulations of the staged limiter (network caps, sensor period 1 s) with SAVE_TRACE=1,
# one cap offset per run. Nothing is swept: each run is one case that is shown in one figure panel.
#   H01 = row 127, H04 = row 130, H09 = row 135, H10 = row 136, H12 = row 138
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_trace_figures.sh
# Then:  python3 scripts/09_dataset_pipeline/make_result_graphs.py --dir model_outputs/dataset_pipeline --out result_graphs
set -euo pipefail
D=scripts/09_dataset_pipeline
grep -q SAVE_TRACE ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no SAVE_TRACE option: git pull first." >&2; exit 1; }
grep -q "^H12," ${D}/staged_caps.csv || { echo "staged_caps.csv has no H12 row: git pull first." >&2; exit 1; }
mkdir -p logs
SCRIPT=${D}/run_staged_array.sh
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,SAVE_TRACE=1"
sub() { # name rows offsets extra
  J=$(sbatch --parsable --array="$2" --export=ALL,${BASE},TAG="$1",OFFSETS="$3"${4:+,$4} "$SCRIPT")
  printf "%-12s job %s   rows %-18s offset %-4s %s\n" "$1" "$J" "$2" "$3" "${4:-ideal sensor}"
}
sub tr_idp10    127            10                                          # ideal sensor, +10 kW room
sub tr_idp15    127            15                                          # ideal sensor, +15 kW (too much)
sub tr_raw10    127            -5  "MEAS_NOISE=10"                         # 10 % noise, readings not averaged
sub tr_n10fm5   127,135        -5  "MEAS_NOISE=10,MEAS_FILTER=10"          # 10 % noise, 10 s average, 10 kW margin
sub tr_n2f0     127            0   "MEAS_NOISE=2,MEAS_FILTER=10"           # 2 % noise, 10 s average, present 5 kW margin
sub tr_n10f0    127,130,135,136,138 0 "MEAS_NOISE=10,MEAS_FILTER=10"      # 10 % noise, 10 s average, present 5 kW margin
sub tr_n10f0s2  130            0   "MEAS_NOISE=10,MEAS_FILTER=10,MEAS_SEED=2"   # same, noise seed 2
echo "Watch: squeue -u \$USER     (12 simulations in total)"

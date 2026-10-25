#!/bin/bash
# Sensor delay test, 12 test/example scenarios (S012 S013 S014 S019 S020 S024 S032 S034 S035 H01 H02 H03).
# Staged limiter, network caps (staged_cap_groups_nn.csv), sensor period 1 s, 10 s average of the readings, noise seed 1.
# Offsets (kW added to all six caps, most permissive first, stop at the first safe one): 10:5:0:-5:-10:-15.
#   dly100ns_ideal   delay 100 ns, no noise       -> compare tag slack_f10  (ideal sensor + 10 s average)
#   dly100ns_n10f    delay 100 ns, noise 10 %     -> compare tag slack_n10f (noise 10 % + 10 s average)
#   dly1s_n10f       delay 1 s,   noise 10 %
#   dly5s_n10f       delay 5 s,   noise 10 %
#   dly10s_n10f      delay 10 s,  noise 10 %
# The simulation step is 10 ms, so a 100 ns delay is rounded to zero steps: those two runs must reproduce the
# no-delay results exactly. They are run to show it, not assumed.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_delay_test.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_delay.py   (after  module load python/3.11.5)
set -euo pipefail
D=scripts/09_dataset_pipeline
grep -q MEAS_DELAY ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no MEAS_DELAY option: git pull first." >&2; exit 1; }
mkdir -p logs
ROWS="12,13,14,19,20,24,32,34,35,127,128,129"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=10:5:0:-5:-10:-15,MEAS_FILTER=10"
SCRIPT=${D}/run_staged_array.sh
sub() { J=$(sbatch --parsable --array="${ROWS}%4" --export=ALL,${BASE},TAG="$1",$2 "$SCRIPT"); printf "%-16s job %s   (%s)\n" "$1" "$J" "$2"; }
sub dly100ns_ideal "MEAS_DELAY=0.0000001"
sub dly100ns_n10f  "MEAS_DELAY=0.0000001,MEAS_NOISE=10"
sub dly1s_n10f     "MEAS_DELAY=1,MEAS_NOISE=10"
sub dly5s_n10f     "MEAS_DELAY=5,MEAS_NOISE=10"
sub dly10s_n10f    "MEAS_DELAY=10,MEAS_NOISE=10"
echo "Watch: squeue -u \$USER    First check:  grep Options logs/staged_<jobid>_12.log"

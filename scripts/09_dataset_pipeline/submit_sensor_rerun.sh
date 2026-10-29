#!/bin/bash
# Sensor re-run with Don's numbers (2026-10-29), two scenarios: H01 (row 127, single bus C) and H09 (row 135, pair H+K).
# Sensor model: sampled at EVERY simulation step (0.01 s; 10 ns cannot be simulated with a 10 ms step), delay 50 ns
# (= 0 steps at 10 ms), noise seed 1, network caps (staged_cap_groups_nn.csv), staged limiter.
# Offsets (kW added to all six caps, most permissive first, stop at the first safe one): 15:10:5:0:-5:-10.
#
# Averaging windows ("w"): none, 100 ms, 1 s, 10 s.  Noise models: "b" = bounded (Uniform, every reading within +/- pct),
# "g" = Gaussian (pct is a standard deviation, stricter than a bound).
#   A  0.5 % bounded:   sr_b05_w0   sr_b05_w100ms   sr_b05_w1s   sr_b05_w10s
#   B  0.5 % Gaussian:  sr_g05_w0   sr_g05_w100ms   sr_g05_w1s   sr_g05_w10s
#   C  10 % Gaussian (safety check, the hardest old case):  sr_g10_w100ms   sr_g10_w1s   sr_g10_w10s
# Each tag = 2 tasks (H01, H09), each task at most 6 simulations of about 11 minutes. 11 job arrays, 22 tasks in total.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_sensor_rerun.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_sensor_rerun.py
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
grep -q MEAS_MODE "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
grep -q meas_noise_mode ${D}/run_staged_limiter.m || { echo "run_staged_limiter.m has no meas_noise_mode option: git pull first." >&2; exit 1; }
for ID in H01 H09; do grep -q "^${ID}," ${D}/staged_caps.csv || { echo "staged_caps.csv has no ${ID} row: git pull first." >&2; exit 1; }; done
mkdir -p logs
ROWS="127,135"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_SEED=1"
sub() { J=$(sbatch --parsable --array="${ROWS}" --export=ALL,${BASE},TAG="$1",$2 "$SCRIPT"); printf "%-16s job %s   (%s)\n" "$1" "$J" "$2"; }

# A. 0.5 % bounded
sub sr_b05_w0      "MEAS_NOISE=0.5,MEAS_MODE=uniform"
sub sr_b05_w100ms  "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_FILTER=0.1"
sub sr_b05_w1s     "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_FILTER=1"
sub sr_b05_w10s    "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_FILTER=10"
# B. 0.5 % Gaussian (stricter)
sub sr_g05_w0      "MEAS_NOISE=0.5"
sub sr_g05_w100ms  "MEAS_NOISE=0.5,MEAS_FILTER=0.1"
sub sr_g05_w1s     "MEAS_NOISE=0.5,MEAS_FILTER=1"
sub sr_g05_w10s    "MEAS_NOISE=0.5,MEAS_FILTER=10"
# C. 10 % Gaussian safety check
sub sr_g10_w100ms  "MEAS_NOISE=10,MEAS_FILTER=0.1"
sub sr_g10_w1s     "MEAS_NOISE=10,MEAS_FILTER=1"
sub sr_g10_w10s    "MEAS_NOISE=10,MEAS_FILTER=10"
echo "Watch: squeue -u \$USER    First check (once a task has started):  grep Options logs/staged_<jobid>_127.log"

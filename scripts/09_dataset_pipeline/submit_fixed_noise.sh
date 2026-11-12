#!/bin/bash
# Fixed-size noise check (2026-11-11, approved by Prof. Bui): ONE scenario, H09 (row 135, pair H+K, the tightest one).
# Why: Phil's scope files (29 Oct test) suggest the swings in the current reading have a fixed size (about 3 A)
# whenever the converter switches, and do not grow with the load. Earlier tests used noise as a percent of the reading.
# Noise here: Normal, standard deviation 2400 W (3 A x 800 V), added to every reading, the same size at any load.
# This is a ROUGH check: the volts-to-amps conversion and the source of the swings are not confirmed by Phil.
#
# Same sensor model as the earlier sensor re-run: one reading per simulation step (0.01 s), delay 50 ns (= 0 steps),
# noise seed 1, network caps (staged_cap_groups_nn.csv), staged limiter.
# Offsets (kW added to all six caps, most permissive first, stop at the first safe one): 15:10:5:0:-5:-10.
#   fx_a2400_w1s     fixed 2400 W noise, 1 s average      (compare with sr_g10_w1s:    10 % noise, 1 s)
#   fx_a2400_w100ms  fixed 2400 W noise, 100 ms average   (compare with sr_g10_w100ms: 10 % noise, 100 ms)
# 2 tasks in total, each at most 6 simulations of about 11 minutes.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_fixed_noise.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_fixed_noise.py
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
grep -q MEAS_ABS_W "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_ABS_W option: git pull first." >&2; exit 1; }
grep -q meas_noise_abs_w ${D}/run_staged_limiter.m || { echo "run_staged_limiter.m has no meas_noise_abs_w option: git pull first." >&2; exit 1; }
grep -q "^H09," ${D}/staged_caps.csv || { echo "staged_caps.csv has no H09 row: git pull first." >&2; exit 1; }
mkdir -p logs
ROWS="135"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_SEED=1,MEAS_ABS_W=2400"
sub() { J=$(sbatch --parsable --array="${ROWS}" --export=ALL,${BASE},TAG="$1",$2 "$SCRIPT"); printf "%-16s job %s   (%s)\n" "$1" "$J" "$2"; }
sub fx_a2400_w1s     "MEAS_FILTER=1"
sub fx_a2400_w100ms  "MEAS_FILTER=0.1"
echo "Watch: squeue -u \$USER    First check (once a task has started):  grep 'fixed noise' logs/staged_<jobid>_135.log"

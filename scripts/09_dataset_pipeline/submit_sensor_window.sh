#!/bin/bash
# Shortest averaging window that still gives margin 0 (2026-10-29), 24 test + profile scenarios, noise seed 1.
# Idea: what matters is the noise left after averaging = noise / sqrt(readings in the window), at a 10 ms reading step.
#   known:  10 % + 1 s    -> 1.0 % left  -> margin 0 on 24/24 scenarios
#           10 % + 100 ms -> 3.2 % left  -> margin 10 (only H01 and H09 were tested)
#   now:    sr_w_g10_w500ms   10 % noise, 500 ms average -> 1.4 % left
#           sr_w_g05_w100ms    5 % noise, 100 ms average -> 1.6 % left   (same noise left as the line above, almost)
#           sr_w_g10_w300ms   10 % noise, 300 ms average -> 1.8 % left
# If the answers follow the noise left (and not the noise or the window alone), the rule holds.
# Sensor: read at EVERY 10 ms step, delay 50 ns (= 0 steps), Gaussian noise, staged limiter, network caps.
# Offsets (kW added to all six caps, most permissive first, stop at the first safe one): 10:5:0:-5:-10.
# The ideal sensor and the 1 s result never went above +10 except H06 and H07, and margin 0 is offset +5, so +15 is not needed.
# 3 arrays x 24 scenarios = 72 tasks, each up to 5 simulations of about 11 minutes. At most 6 tasks per array at once.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_sensor_window.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_sensor_next.py     (after  module load python/3.11.5)
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
grep -q MEAS_MODE "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
grep -q meas_noise_mode ${D}/run_staged_limiter.m || { echo "run_staged_limiter.m has no meas_noise_mode option: git pull first." >&2; exit 1; }

ROOT="${DATASET_OUTPUT_ROOT:-model_outputs/dataset_pipeline}"
for ID in S012 S013 S014 S019 S020 S024 S032 S034 S035 H01 H02 H03 H04 H05 H06 H07 H08 H09 H10 H11 H12 H13 H14 H15; do
  if [ ! -f "${ROOT}/${ID}/noshed_${ID}_collapse.csv" ] || [ ! -f "${ROOT}/${ID}/detector_triggers_${ID}.json" ]; then
    echo "WARNING: ${ID}: no-shed collapse file or detector-trigger file not found in ${ROOT}/${ID}"
  fi
done

mkdir -p logs
ROWS="12,13,14,19,20,24,32,34,35,127-141"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=10:5:0:-5:-10,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_SEED=1"
sub() { J=$(sbatch --parsable --array="${ROWS}%6" --export=ALL,${BASE},TAG="$1",$2 "$SCRIPT"); printf "%-18s job %s   (%s)\n" "$1" "$J" "$2"; }

sub sr_w_g10_w500ms "MEAS_NOISE=10,MEAS_FILTER=0.5"
sub sr_w_g05_w100ms "MEAS_NOISE=5,MEAS_FILTER=0.1"
sub sr_w_g10_w300ms "MEAS_NOISE=10,MEAS_FILTER=0.3"
echo "Watch: squeue -u \$USER    First check (once a task has started):  grep Options logs/staged_<jobid>_12.log"
